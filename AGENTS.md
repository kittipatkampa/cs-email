# AGENTS.md — working in cs-email

This file is the **single entrypoint** for humans and AI agents: what the repo is, how to run things, and non-negotiable boundaries.

## What this repo is

**cs-email** is a customer-support email AI assistant built with **LangGraph** and **Anthropic** (via `langchain-anthropic`).

## Architecture

The intended design follows the **LangGraph** walkthrough for a customer-support email agent:

- **Reference**: [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph) — nodes (read → classify → doc search / bug track / human review → draft → send), **shared state** with raw data (not prompt text), **`Command`**-based routing, **retries** on flaky I/O, **`interrupt()`** for human review with a **checkpointer** for durable runs.
- **Stack**: pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) (`langgraph`, `langchain`, `langchain-anthropic`).

Cursor rule for graph code: [`.cursor/rules/langgraph.mdc`](.cursor/rules/langgraph.mdc).

## Repository layout

| Path | Purpose |
|------|---------|
| [`pyproject.toml`](pyproject.toml) | Project metadata and dependencies |
| [`uv.lock`](uv.lock) | Locked dependency versions (commit this) |
| [`langgraph.json`](langgraph.json) | LangGraph app config (graph entrypoint for LangGraph CLI / deployment) |
| [`.env.example`](.env.example) | Environment variable names (copy to `.env`) |
| [`cs_email/state.py`](cs_email/state.py) | `TypedDict` state and classification types |
| [`cs_email/nodes.py`](cs_email/nodes.py) | Node functions (`read_email`, `classify_intent`, …) |
| [`cs_email/graph.py`](cs_email/graph.py) | `StateGraph` wiring, `MemorySaver`, exported `app` |
| [`tests/`](tests/) | `pytest` smoke tests (LLM calls mocked) |
| [`scripts/manual_integration.py`](scripts/manual_integration.py) | Real Anthropic integration (optional) |
| [`scripts/test_multiturn.py`](scripts/test_multiturn.py) | Mocked multi-turn check on one `thread_id` (no API key) |

## Commands

| Task | Command |
|------|---------|
| Install deps (including dev) | `uv sync --extra dev` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format (apply) | `uv run ruff format .` |
| Format (check only) | `uv run ruff format --check .` |
| Smoke-load compiled graph | `uv run python -m cs_email.graph` |
| Multi-turn (mocked) | `uv run python scripts/test_multiturn.py` |
| LangGraph CLI | Install `langgraph-cli` if needed, then run from repo root using [`langgraph.json`](langgraph.json) |

Requires **Python 3.12+** (`requires-python` in [`pyproject.toml`](pyproject.toml)).

## Environment variables

Copy [`.env.example`](.env.example) to `.env` (never commit `.env`).

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes for real LLM calls | Anthropic API authentication |
| `ANTHROPIC_MODEL` | No | Override default chat model (see `.env.example`) |

## Safety

- **Secrets**: use environment variables only; document variable **names** here, never values.
- **PII**: treat email content and addresses as sensitive; avoid logging or storing them unless required and documented.

## Cursor / AI

- Project rules live in [`.cursor/rules/`](.cursor/rules/).
- Indexing exclusions: see [`.cursorignore`](.cursorignore).
