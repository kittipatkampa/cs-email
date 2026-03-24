#!/usr/bin/env python3
"""HTTP client for the local SSE API (run server first: scripts/stream_demo.py --server).

Reads the HTTP response **incrementally** so tokens appear as the server streams them (not after
the full body is buffered).

Requires dev deps (httpx): `uv sync --extra dev`

If stdout still looks batched, run unbuffered: `python -u scripts/test_sse_client.py` or
`PYTHONUNBUFFERED=1 uv run python scripts/test_sse_client.py`.

Examples:

  uv run python scripts/test_sse_client.py
  uv run python scripts/test_sse_client.py --email "How do I reset my password?"
  uv run python scripts/test_sse_client.py --email "I was charged twice!" --resume

When the stream hits an interrupt, use --resume to POST /email/resume (prompts if omitted).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    import httpx
except ImportError as e:
    print("Install httpx: uv sync --extra dev", file=sys.stderr)
    raise SystemExit(1) from e


def _parse_sse_block(block: str) -> tuple[str | None, str | None]:
    """Return (event_name, data) from one SSE block (newline-separated lines)."""
    event_name: str | None = None
    data_lines: list[str] = []
    for line in block.replace("\r\n", "\n").split("\n"):
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if not data_lines:
        return event_name, None
    return event_name, "\n".join(data_lines)


def _iter_sse_events(text: str) -> list[tuple[str, str]]:
    """Parse full SSE body into (event, data) pairs (used for tests / batch parsing)."""
    normalized = text.replace("\r\n", "\n")
    events: list[tuple[str, str]] = []
    for block in normalized.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev, data = _parse_sse_block(block)
        if data is not None and ev is not None:
            events.append((ev, data))
    return events


def _normalize_stream_line(line: str) -> str:
    """Strip CR from CRLF lines from the wire."""
    return line.rstrip("\r")


def _iter_sse_from_streaming_response(response: httpx.Response):
    """Yield (event, data) as each SSE event arrives (do not buffer the full body)."""
    response.raise_for_status()
    buf_lines: list[str] = []
    for line in response.iter_lines():
        if line is None:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = _normalize_stream_line(line)
        if line == "":
            if buf_lines:
                block = "\n".join(buf_lines)
                buf_lines = []
                ev, data = _parse_sse_block(block)
                if ev is not None and data is not None:
                    yield (ev, data)
        else:
            buf_lines.append(line)
    if buf_lines:
        block = "\n".join(buf_lines)
        ev, data = _parse_sse_block(block)
        if ev is not None and data is not None:
            yield (ev, data)


def _stream_request_iter(
    client: httpx.Client,
    method: str,
    url: str,
    json_body: dict[str, Any],
):
    """POST and yield SSE events as the HTTP response streams (real-time)."""
    with client.stream(method, url, json=json_body, timeout=300.0) as response:
        yield from _iter_sse_from_streaming_response(response)


def _stringify_message_content(content: Any) -> str:
    """Turn LangChain/Anthropic message `content` into a string (may be str or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Common: {"type": "text", "text": "..."}
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _print_event(event: str, data: str) -> None:
    if event == "token":
        try:
            payload = json.loads(data)
            content = payload.get("content", "")
            sys.stdout.write(_stringify_message_content(content))
            sys.stdout.flush()
            return
        except json.JSONDecodeError:
            pass
    print(f"\n[{event}] {data}", file=sys.stderr)


def run_stream(
    base: str,
    *,
    email: str,
    sender: str,
    email_id: str,
    thread_id: str | None,
) -> tuple[str | None, bool]:
    """POST /email/stream; print tokens to stdout, other events to stderr."""
    url = base.rstrip("/") + "/email/stream"
    payload: dict[str, Any] = {
        "email_content": email,
        "sender_email": sender,
        "email_id": email_id,
    }
    if thread_id:
        payload["thread_id"] = thread_id

    print("--- POST /email/stream ---", file=sys.stderr)
    tid: str | None = None
    draft_tokens_printed = False
    had_interrupt = False
    with httpx.Client() as client:
        for event, data in _stream_request_iter(client, "POST", url, payload):
            if event == "thread_id":
                tid = json.loads(data)["thread_id"]
                print(f"thread_id: {tid}", file=sys.stderr)
            elif event == "token":
                if not draft_tokens_printed:
                    print("\n--- draft (tokens) ---", file=sys.stderr)
                    draft_tokens_printed = True
                _print_event(event, data)
            elif event == "interrupt":
                had_interrupt = True
                print(f"\n[{event}] {data}", file=sys.stderr)
            elif event == "done":
                print("\n--- done ---", file=sys.stderr)
            elif event == "error":
                print(f"error: {data}", file=sys.stderr)
            else:
                _print_event(event, data)

    if draft_tokens_printed:
        print(file=sys.stderr)
    return tid, had_interrupt


def run_resume(
    base: str,
    *,
    thread_id: str,
    approved: bool,
    edited_response: str | None,
) -> None:
    url = base.rstrip("/") + "/email/resume"
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "approved": approved,
    }
    if edited_response is not None:
        payload["edited_response"] = edited_response

    print("--- POST /email/resume ---", file=sys.stderr)
    draft_tokens_printed = False
    with httpx.Client() as client:
        for event, data in _stream_request_iter(client, "POST", url, payload):
            if event == "thread_id":
                print(f"thread_id: {json.loads(data)['thread_id']}", file=sys.stderr)
            elif event == "token":
                if not draft_tokens_printed:
                    print("\n--- draft (tokens) ---", file=sys.stderr)
                    draft_tokens_printed = True
                _print_event(event, data)
            elif event == "done":
                print("\n--- done ---", file=sys.stderr)
            elif event == "error":
                print(f"error: {data}", file=sys.stderr)
            else:
                _print_event(event, data)
    if draft_tokens_printed:
        print(file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the cs-email SSE API (local server).")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--email",
        default="How do I reset my password?",
        help="Customer email body for /email/stream.",
    )
    parser.add_argument("--sender", default="customer@example.com")
    parser.add_argument("--email-id", default="sse_test_1", dest="email_id")
    parser.add_argument("--thread-id", default=None, dest="thread_id", help="Optional thread id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only call /email/resume (use --thread-id from a prior interrupt).",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="With --resume, set approved=True (otherwise prompts).",
    )
    parser.add_argument(
        "--reject",
        action="store_true",
        help="With --resume, set approved=False.",
    )
    parser.add_argument(
        "--edited",
        default=None,
        help="With --resume and approve, optional edited_response text.",
    )
    args = parser.parse_args()

    health = args.base_url.rstrip("/") + "/health"
    try:
        r = httpx.get(health, timeout=10.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(
            f"Server not reachable at {args.base_url} ({e}). "
            "Start it with: uv run python scripts/stream_demo.py --server",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    if args.resume:
        if not args.thread_id:
            print("--resume requires --thread-id from an interrupted run.", file=sys.stderr)
            raise SystemExit(1)
        if args.reject:
            approved = False
        elif args.approve:
            approved = True
        else:
            ans = input("Approve? [y/N]: ").strip().lower()
            approved = ans in ("y", "yes")
        run_resume(
            args.base_url,
            thread_id=args.thread_id,
            approved=approved,
            edited_response=args.edited,
        )
        return

    tid, had_interrupt = run_stream(
        args.base_url,
        email=args.email,
        sender=args.sender,
        email_id=args.email_id,
        thread_id=args.thread_id,
    )

    if had_interrupt and tid:
        print("Resume after human review:", file=sys.stderr)
        print(
            f"  uv run python scripts/test_sse_client.py --resume --thread-id {tid} --approve",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
