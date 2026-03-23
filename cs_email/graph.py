"""Compile the customer-support email LangGraph."""

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


def build_compiled_graph():
    """Build and compile the graph with an in-memory checkpointer."""
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

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


app = build_compiled_graph()


if __name__ == "__main__":
    print("Compiled graph:", app)
