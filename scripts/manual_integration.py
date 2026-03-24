#!/usr/bin/env python3
"""Run the cs-email LangGraph against the real Anthropic API (manual / integration test).

From the repository root:

  cp .env.example .env   # set ANTHROPIC_API_KEY
  uv sync
  uv run python scripts/manual_integration.py

Examples:

  uv run python scripts/manual_integration.py \\
    --email "How do I reset my password?"

  uv run python scripts/manual_integration.py \\
    --email "I was charged twice for my subscription!" \\
    --auto-approve

When the graph stops at human review, you can approve interactively or pass --auto-approve
to approve immediately using the draft (or an empty body if there was no draft yet).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    # Also load from cwd if user runs from elsewhere
    load_dotenv()


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and set the key, "
            "or export ANTHROPIC_API_KEY in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)


def _initial_state(args: argparse.Namespace) -> dict:
    return {
        "email_content": args.email,
        "sender_email": args.sender,
        "email_id": args.email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [HumanMessage(content=args.email)],
    }


def _print_result(title: str, result: dict, *, include_interrupt: bool) -> None:
    print(f"\n--- {title} ---")
    if include_interrupt:
        print(json.dumps(result, indent=2, default=str))
        return
    safe = {k: v for k, v in result.items() if k != "__interrupt__"}
    print(json.dumps(safe, indent=2, default=str))
    if "__interrupt__" in result:
        print("\n(__interrupt__ present — graph paused for human review)")


def _interactive_resume() -> dict:
    """Prompt for approve/reject and optional edited reply."""
    ans = input("\nApprove and send this draft? [y/N/reject]: ").strip().lower()
    if ans in ("y", "yes"):
        edited = input("Edited reply (empty = keep draft as-is): ").strip()
        # Omit edited_response so the node keeps the existing draft from state
        if edited:
            return {"approved": True, "edited_response": edited}
        return {"approved": True}
    if ans in ("r", "reject", "n", "no", ""):
        return {"approved": False}
    return {"approved": False}


def main() -> None:
    _load_env()
    _require_api_key()

    parser = argparse.ArgumentParser(
        description="Invoke the cs-email graph with real Anthropic (integration test).",
    )
    parser.add_argument(
        "--email",
        # default="How do I reset my password?",
        default="I was charged twice for my subscription!",
        help="Customer email body (default: simple product question).",
    )
    parser.add_argument(
        "--sender",
        default="customer@example.com",
        help="Sender email address.",
    )
    parser.add_argument(
        "--email-id",
        default="manual_run_1",
        help="Opaque id for this message (logging / threading).",
    )
    parser.add_argument(
        "--thread",
        default="manual-integration-thread",
        help="LangGraph thread_id (must be stable across resume).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="If the graph hits human_review, resume immediately with approved=True.",
    )
    parser.add_argument(
        "--json-full",
        action="store_true",
        help="Print full result dict including __interrupt__ (can be large).",
    )
    args = parser.parse_args()

    # Import after env is loaded so ChatAnthropic sees the key
    from cs_email.graph import app

    config = {"configurable": {"thread_id": args.thread}}
    initial = _initial_state(args)

    print("Invoking graph (real API calls)...")
    result = app.invoke(initial, config)
    _print_result("First invoke", result, include_interrupt=args.json_full)

    if "__interrupt__" not in result:
        print("\nDone (no interrupt). Final draft_response is in the JSON above.")
        return

    if not args.json_full:
        intr = result.get("__interrupt__")
        print("\n--- interrupt summary ---")
        print(json.dumps(intr, indent=2, default=str))

    resume_payload: dict
    if args.auto_approve:
        # Let human_review use draft_response from graph state (omit edited_response)
        resume_payload = {"approved": True}
        print("\n--auto-approve: resuming with approved=True.")
    else:
        resume_payload = _interactive_resume()

    final = app.invoke(Command(resume=resume_payload), config)
    _print_result("After resume", final, include_interrupt=args.json_full)

    if "__interrupt__" in final:
        print(
            "\nStill interrupted (unexpected). Check LangGraph / resume payload.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("\nDone. Final state printed above.")


if __name__ == "__main__":
    main()
