"""Smoke tests for the compiled LangGraph (LLM calls mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cs_email.graph import app, build_compiled_graph


def _make_llm_mocks(classification: dict, draft_content: str = "Here is our reply.") -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = classification
    llm.with_structured_output.return_value = structured
    draft_msg = MagicMock()
    draft_msg.content = draft_content
    llm.invoke.return_value = draft_msg
    return llm


@patch("cs_email.nodes.get_chat_model")
def test_question_email_routes_search_then_send(mock_get_llm: MagicMock) -> None:
    """Low-urgency question: classify -> search -> draft -> send (no interrupt)."""
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "question",
            "urgency": "low",
            "topic": "password",
            "summary": "How to reset password",
        }
    )

    initial = {
        "email_content": "How do I reset my password?",
        "sender_email": "customer@example.com",
        "email_id": "email_123",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": None,
    }
    config = {"configurable": {"thread_id": "test_thread_question"}}
    result = app.invoke(initial, config)

    assert result.get("draft_response")
    assert "__interrupt__" not in result
    mock_get_llm.assert_called()


@patch("cs_email.nodes.get_chat_model")
def test_billing_email_pauses_at_human_review(mock_get_llm: MagicMock) -> None:
    """Billing intent routes to human_review and hits interrupt()."""
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "billing",
            "urgency": "high",
            "topic": "double charge",
            "summary": "Charged twice",
        }
    )

    initial = {
        "email_content": "I was charged twice!",
        "sender_email": "customer@example.com",
        "email_id": "email_billing",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": None,
    }
    config = {"configurable": {"thread_id": "test_thread_billing"}}
    result = app.invoke(initial, config)

    assert "__interrupt__" in result
    mock_get_llm.assert_called()


def test_build_compiled_graph_returns_compiled() -> None:
    g = build_compiled_graph()
    assert g is not None
    assert callable(getattr(g, "invoke", None))


@patch("cs_email.nodes.get_chat_model")
def test_bug_flow_goes_through_draft(mock_get_llm: MagicMock) -> None:
    """Bug intent: classify -> bug_tracking -> draft -> send (low urgency)."""
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "bug",
            "urgency": "low",
            "topic": "export",
            "summary": "PDF export crashes",
        }
    )

    initial = {
        "email_content": "Export crashes when I pick PDF.",
        "sender_email": "customer@example.com",
        "email_id": "email_bug",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": None,
    }
    config = {"configurable": {"thread_id": "test_thread_bug"}}
    result = app.invoke(initial, config)

    assert result.get("search_results")
    assert "BUG-" in str(result.get("search_results"))
    assert result.get("draft_response")
