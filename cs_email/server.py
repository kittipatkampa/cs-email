"""FastAPI server exposing the email agent as an SSE stream for frontend clients."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


def _load_env() -> None:
    """Load `.env` from repo root so `ANTHROPIC_API_KEY` is set when uvicorn imports this app."""
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


_load_env()

from cs_email.graph import app as compiled_graph  # noqa: E402 — after load_dotenv for API keys

app = FastAPI(title="cs-email", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    email_content: str = Field(..., min_length=1)
    sender_email: str = Field(..., min_length=1)
    email_id: str = Field(..., min_length=1)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    approved: bool
    edited_response: str | None = None


def _json_default(obj: Any) -> Any:
    """Fallback for json.dumps; avoid leaking raw email bodies in error paths."""
    return str(obj)


def _interrupt_payload(raw: Any) -> list[dict[str, Any]]:
    """Serialize LangGraph interrupt tuple to JSON-friendly dicts."""
    out: list[dict[str, Any]] = []
    if raw is None:
        return out
    items = raw if isinstance(raw, (list, tuple)) else (raw,)
    for item in items:
        if hasattr(item, "id") and hasattr(item, "value"):
            out.append({"id": getattr(item, "id", ""), "value": getattr(item, "value")})
        else:
            out.append({"id": "", "value": item})
    return out


def _iter_sse_chunks(
    input_data: dict[str, Any] | Command,
    config: dict[str, Any],
    thread_id: str,
) -> Iterator[dict[str, str]]:
    """Map LangGraph stream chunks to SSE event dicts."""
    yield {"event": "thread_id", "data": json.dumps({"thread_id": thread_id})}
    try:
        for chunk in compiled_graph.stream(
            input_data,
            config,
            stream_mode=["messages", "updates"],
        ):
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                continue
            mode, data = chunk[0], chunk[1]
            if mode == "messages":
                msg_chunk, metadata = data
                content = getattr(msg_chunk, "content", None)
                if content:
                    node = ""
                    if isinstance(metadata, dict):
                        node = str(metadata.get("langgraph_node", ""))
                    else:
                        node = str(getattr(metadata, "langgraph_node", "") or "")
                    payload = {"content": content, "node": node}
                    yield {"event": "token", "data": json.dumps(payload, default=_json_default)}
            elif mode == "updates" and isinstance(data, dict):
                if "__interrupt__" in data:
                    interrupts = _interrupt_payload(data["__interrupt__"])
                    yield {
                        "event": "interrupt",
                        "data": json.dumps({"interrupts": interrupts}, default=_json_default),
                    }
                else:
                    for node_name, update in data.items():
                        if node_name == "__interrupt__":
                            continue
                        payload = {"node": node_name, "update": update}
                        yield {
                            "event": "node_update",
                            "data": json.dumps(payload, default=_json_default),
                        }
        yield {"event": "done", "data": "{}"}
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"error": str(exc)}, default=_json_default)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/email/stream")
def stream_email(req: EmailRequest) -> EventSourceResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "email_content": req.email_content,
        "sender_email": req.sender_email,
        "email_id": req.email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": [],
    }
    return EventSourceResponse(_iter_sse_chunks(initial_state, config, thread_id))


@app.post("/email/resume")
def resume_email(req: ResumeRequest) -> EventSourceResponse:
    thread_id = req.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    resume: dict[str, Any] = {"approved": req.approved}
    if req.edited_response is not None:
        resume["edited_response"] = req.edited_response
    cmd = Command(resume=resume)
    return EventSourceResponse(_iter_sse_chunks(cmd, config, thread_id))
