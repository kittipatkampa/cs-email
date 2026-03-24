"""Tests for SSE streaming API and graph stream chunks."""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command

from cs_email.graph import build_compiled_graph
from cs_email.server import app as fastapi_app


def _msg_content(msg: BaseMessage | str) -> str:
    """Extract text content from a message (BaseMessage or legacy str)."""
    if isinstance(msg, str):
        return msg
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _make_llm_mocks(classification: dict, draft_content: str = "Here is our reply.") -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = classification
    llm.with_structured_output.return_value = structured
    draft_msg = MagicMock()
    draft_msg.content = draft_content
    llm.invoke.return_value = draft_msg
    return llm


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Parse raw SSE text into (event_name, data_json_string) pairs."""
    events: list[tuple[str, str]] = []
    text = body.replace("\r\n", "\n")
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events


def test_health_endpoint() -> None:
    client = TestClient(fastapi_app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@patch("cs_email.nodes.get_chat_model")
def test_stream_question_email_sse(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "question",
            "urgency": "low",
            "topic": "password",
            "summary": "How to reset password",
        }
    )
    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/email/stream",
        json={
            "email_content": "How do I reset my password?",
            "sender_email": "customer@example.com",
            "email_id": "email_123",
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    events = _parse_sse(body)
    names = [e[0] for e in events]
    assert names[0] == "thread_id"
    tid = json.loads(events[0][1])["thread_id"]
    assert uuid.UUID(tid)  # server-generated UUID

    assert "node_update" in names
    assert names[-1] == "done"
    assert "error" not in names


@patch("cs_email.nodes.get_chat_model")
def test_stream_billing_email_hits_interrupt(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "billing",
            "urgency": "high",
            "topic": "double charge",
            "summary": "Charged twice",
        }
    )
    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/email/stream",
        json={
            "email_content": "I was charged twice!",
            "sender_email": "customer@example.com",
            "email_id": "email_billing",
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()

    events = _parse_sse(body)
    names = [e[0] for e in events]
    assert "interrupt" in names
    intr = next(json.loads(d) for n, d in events if n == "interrupt")
    assert "interrupts" in intr
    assert len(intr["interrupts"]) >= 1
    assert intr["interrupts"][0]["value"]["intent"] == "billing"


@patch("cs_email.nodes.get_chat_model")
def test_resume_after_interrupt(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "billing",
            "urgency": "high",
            "topic": "x",
            "summary": "y",
        }
    )
    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/email/stream",
        json={
            "email_content": "Billing issue",
            "sender_email": "a@b.com",
            "email_id": "e1",
        },
    ) as response:
        body1 = response.read().decode()

    events1 = _parse_sse(body1)
    tid = json.loads(events1[0][1])["thread_id"]

    with client.stream(
        "POST",
        "/email/resume",
        json={"thread_id": tid, "approved": True},
    ) as response2:
        assert response2.status_code == 200
        body2 = response2.read().decode()

    events2 = _parse_sse(body2)
    names2 = [e[0] for e in events2]
    assert names2[0] == "thread_id"
    assert json.loads(events2[0][1])["thread_id"] == tid
    assert "node_update" in names2 or "done" in names2
    assert names2[-1] == "done"


@patch("cs_email.nodes.get_chat_model")
def test_thread_id_returned_when_not_provided(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "question",
            "urgency": "low",
            "topic": "t",
            "summary": "s",
        }
    )
    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/email/stream",
        json={
            "email_content": "Q?",
            "sender_email": "a@b",
            "email_id": "id1",
        },
    ) as response:
        body = response.read().decode()
    events = _parse_sse(body)
    tid = json.loads(events[0][1])["thread_id"]
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", tid, flags=re.I
    )


@patch("cs_email.nodes.get_chat_model")
def test_thread_id_preserved_when_provided(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "question",
            "urgency": "low",
            "topic": "t",
            "summary": "s",
        }
    )
    fixed = "my-fixed-thread-id-001"
    client = TestClient(fastapi_app)
    with client.stream(
        "POST",
        "/email/stream",
        json={
            "email_content": "Q?",
            "sender_email": "a@b",
            "email_id": "id1",
            "thread_id": fixed,
        },
    ) as response:
        body = response.read().decode()
    events = _parse_sse(body)
    tid = json.loads(events[0][1])["thread_id"]
    assert tid == fixed


@patch("cs_email.nodes.get_chat_model")
def test_graph_stream_updates_order(mock_get_llm: MagicMock) -> None:
    """Direct graph.stream yields updates in node order for a question email."""
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "question",
            "urgency": "low",
            "topic": "password",
            "summary": "x",
        }
    )
    g = build_compiled_graph()
    cfg = {"configurable": {"thread_id": "test_stream_order"}}
    initial = {
        "email_content": "How do I reset my password?",
        "sender_email": "customer@example.com",
        "email_id": "e1",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [HumanMessage(content="How do I reset my password?")],
    }
    nodes_seen: list[str] = []
    for chunk in g.stream(initial, cfg, stream_mode=["updates"]):
        mode, data = chunk
        assert mode == "updates"
        assert isinstance(data, dict)
        for name in data:
            if name != "__interrupt__":
                nodes_seen.append(name)
    assert nodes_seen == [
        "read_email",
        "classify_intent",
        "search_documentation",
        "draft_response",
        "send_reply",
    ]


@patch("cs_email.nodes.get_chat_model")
def test_resume_uses_command(mock_get_llm: MagicMock) -> None:
    """Resume path accepts Command(resume=...) on the compiled graph."""
    mock_get_llm.return_value = _make_llm_mocks(
        {
            "intent": "billing",
            "urgency": "high",
            "topic": "t",
            "summary": "s",
        }
    )
    g = build_compiled_graph()
    cfg = {"configurable": {"thread_id": "test_resume_cmd"}}
    initial = {
        "email_content": "Bill",
        "sender_email": "a@b",
        "email_id": "e1",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [HumanMessage(content="Bill")],
    }
    r1 = g.invoke(initial, cfg)
    assert "__interrupt__" in r1
    final = g.invoke(Command(resume={"approved": True}), cfg)
    assert "__interrupt__" not in final
    assert any("Sent reply." in _msg_content(m) for m in (final.get("messages") or []))
