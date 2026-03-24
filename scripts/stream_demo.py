#!/usr/bin/env python3
"""Stream the cs-email graph to stdout, or launch the FastAPI SSE server.

Direct mode (default) prints LangGraph stream chunks (updates + messages) without HTTP.
Requires ANTHROPIC_API_KEY for real LLM calls (see scripts/manual_integration.py).

Server mode:

  uv run python scripts/stream_demo.py --server

Then POST to http://127.0.0.1:8000/email/stream (SSE) from a client or curl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and set the key, "
            "or export ANTHROPIC_API_KEY in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_stream_chunk(chunk: tuple) -> bool:
    """Print one stream chunk; return True if graph hit interrupt (in updates)."""
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        print(f"(unexpected chunk type: {type(chunk)})", file=sys.stderr)
        return False
    mode, data = chunk[0], chunk[1]
    if mode == "messages":
        msg_chunk, metadata = data
        content = getattr(msg_chunk, "content", None)
        if content:
            print(content, end="", flush=True)
        return False
    if mode == "updates" and isinstance(data, dict):
        if "__interrupt__" in data:
            print("\n--- interrupt ---", file=sys.stderr)
            print(json.dumps(data, default=str, indent=2), file=sys.stderr)
            return True
        for node_name, update in data.items():
            if node_name == "__interrupt__":
                continue
            print(f"\n--- node update: {node_name} ---", file=sys.stderr)
            print(json.dumps(update, default=str, indent=2), file=sys.stderr)
    return False


def _interactive_resume() -> dict:
    ans = input("\nApprove and send this draft? [y/N/reject]: ").strip().lower()
    if ans in ("y", "yes"):
        edited = input("Edited reply (empty = keep draft as-is): ").strip()
        if edited:
            return {"approved": True, "edited_response": edited}
        return {"approved": True}
    if ans in ("r", "reject", "n", "no", ""):
        return {"approved": False}
    return {"approved": False}


def _run_direct_stream(args: argparse.Namespace) -> None:
    _load_env()
    _require_api_key()

    from cs_email.graph import app

    config = {"configurable": {"thread_id": args.thread}}
    initial = {
        "email_content": args.email,
        "sender_email": args.sender,
        "email_id": args.email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [],
    }

    print("Streaming graph (messages + updates)...", file=sys.stderr)
    interrupted = False
    for chunk in app.stream(initial, config, stream_mode=["messages", "updates"]):
        if _print_stream_chunk(chunk):
            interrupted = True
            break

    if not interrupted:
        print("\nDone (no interrupt).", file=sys.stderr)
        return

    resume_payload: dict
    if args.auto_approve:
        resume_payload = {"approved": True}
        print("--auto-approve: resuming with approved=True.", file=sys.stderr)
    else:
        resume_payload = _interactive_resume()

    print("\nResuming stream...", file=sys.stderr)
    for chunk in app.stream(
        Command(resume=resume_payload), config, stream_mode=["messages", "updates"]
    ):
        _print_stream_chunk(chunk)
    print("\nDone after resume.", file=sys.stderr)


def _run_server(args: argparse.Namespace) -> None:
    host = args.host
    port = args.port
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "cs_email.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if args.reload:
        cmd.append("--reload")
    print("Starting:", " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream cs-email graph or run SSE server.")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run FastAPI + uvicorn (SSE API) instead of direct stream to stdout.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="With --server, pass --reload to uvicorn.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for --server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --server.")
    parser.add_argument(
        "--email",
        default="How do I reset my password?",
        help="Customer email body (direct mode only).",
    )
    parser.add_argument("--sender", default="customer@example.com", help="Sender (direct mode).")
    parser.add_argument(
        "--email-id", default="stream_demo_1", dest="email_id", help="Email id (direct)."
    )
    parser.add_argument(
        "--thread",
        default="stream-demo-thread",
        help="LangGraph thread_id (direct mode).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="If interrupt, resume immediately with approved=True (direct mode).",
    )
    args = parser.parse_args()

    if args.server:
        _run_server(args)
    else:
        _run_direct_stream(args)


if __name__ == "__main__":
    main()
