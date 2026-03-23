#!/usr/bin/env python3
"""Multi-turn integration check: same thread_id accumulates `messages` (LLM mocked).

Run from repo root (no ANTHROPIC_API_KEY required):

  uv run python scripts/test_multiturn.py
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from langgraph.types import Command


def _first_turn_state(*, email_content: str, email_id: str) -> dict:
    """First invoke on a thread: seed reducer with an empty messages list."""
    return {
        "email_content": email_content,
        "sender_email": "customer@example.com",
        "email_id": email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [],
    }


def _next_turn_state(*, email_content: str, email_id: str) -> dict:
    """Later turns: omit `messages` so prior thread history is not reset."""
    return {
        "email_content": email_content,
        "sender_email": "customer@example.com",
        "email_id": email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
    }


def main() -> None:
    from cs_email.graph import build_compiled_graph

    classification_turn1 = {
        "intent": "question",
        "urgency": "low",
        "topic": "password",
        "summary": "How to reset password",
    }
    classification_turn2 = {
        "intent": "billing",
        "urgency": "high",
        "topic": "double charge",
        "summary": "Charged twice",
    }

    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = [classification_turn1, classification_turn2]

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_mock
    draft_msg = MagicMock()
    draft_msg.content = "Draft reply turn 1."
    llm.invoke.return_value = draft_msg

    with patch("cs_email.nodes.get_chat_model", return_value=llm):
        app = build_compiled_graph()
        thread_id = "multiturn-test-thread"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1: question -> search -> draft -> send (no interrupt)
        initial1 = _first_turn_state(
            email_content="How do I reset my password?",
            email_id="turn1",
        )
        result1 = app.invoke(initial1, config)

        assert "__interrupt__" not in result1, "Turn 1 should complete without interrupt"
        msgs1 = result1.get("messages") or []
        print("--- After turn 1 (messages) ---")
        for i, m in enumerate(msgs1):
            print(f"  [{i}] {m}")

        assert any("Processing email" in m for m in msgs1)
        assert any(m.startswith("Classified intent=") for m in msgs1)
        assert any("Searched documentation" in m for m in msgs1)
        assert any(m == "Drafted response." for m in msgs1)
        assert any(m == "Sent reply." for m in msgs1)

        # Turn 2: billing -> human_review -> interrupt (same thread_id)
        initial2 = _next_turn_state(
            email_content="I was charged twice for my subscription!",
            email_id="turn2",
        )
        result2 = app.invoke(initial2, config)

        assert "__interrupt__" in result2, "Turn 2 should pause at human_review"
        msgs2 = result2.get("messages") or []
        print("\n--- After turn 2 invoke (interrupt; messages) ---")
        for i, m in enumerate(msgs2):
            print(f"  [{i}] {m}")

        assert len(msgs2) > len(msgs1), "messages should grow across turns on the same thread"
        assert any("turn2" in m or "charged twice" in m.lower() for m in msgs2), (
            "Turn 2 read_email should appear in messages"
        )

        # Resume approval
        final = app.invoke(Command(resume={"approved": True}), config)

        assert "__interrupt__" not in final, "Resume should finish the graph"
        msgs_final = final.get("messages") or []
        print("\n--- After resume (final messages) ---")
        for i, m in enumerate(msgs_final):
            print(f"  [{i}] {m}")

        assert any(m == "Human review: approved" for m in msgs_final)
        assert msgs_final.count("Sent reply.") >= 2, "Should have sent after turn 1 and turn 2"

    print("\nAll multi-turn assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"Assertion failed: {e}", file=sys.stderr)
        sys.exit(1)
