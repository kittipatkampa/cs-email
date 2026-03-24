"""Compile the customer-support email LangGraph.

``app`` is exported **without** a checkpointer so ``langgraph dev`` (which
provides its own persistence) can load it via ``langgraph.json``.

Use ``build_compiled_graph()`` when you need a standalone graph with an
in-memory checkpointer (custom SSE server, tests, scripts).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from cs_email.nodes import (
    bug_tracking,
    classify_intent,
    draft_response,
    human_review,
    read_email,
    search_documentation,
    send_reply,
)
from cs_email.state import EmailAgentState


def _build_workflow() -> StateGraph:
    """Return the (uncompiled) StateGraph with all nodes and edges."""
    workflow = StateGraph(EmailAgentState)

    workflow.add_node("read_email", read_email)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    workflow.add_node("bug_tracking", bug_tracking)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_reply", send_reply)

    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")
    workflow.add_edge("send_reply", END)

    return workflow


def build_compiled_graph():
    """Build and compile the graph with an in-memory checkpointer.

    Used by the custom SSE server, tests, and scripts that run outside
    the LangGraph Platform.
    """
    return _build_workflow().compile(checkpointer=MemorySaver())


# Exported for langgraph.json → ``langgraph dev``.
# No checkpointer: the LangGraph Platform provides its own persistence.
app = _build_workflow().compile()


if __name__ == "__main__":
    print("Compiled graph:", app)
