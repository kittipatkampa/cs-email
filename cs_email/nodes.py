"""LangGraph node functions for the customer-support email workflow."""

from __future__ import annotations

import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt

from cs_email.state import EmailAgentState, EmailClassification


class SearchAPIError(Exception):
    """Raised when documentation search fails (transient or otherwise)."""


def get_chat_model() -> ChatAnthropic:
    """Return the chat model (patch in tests).

    ``streaming=True`` so LangGraph ``stream_mode="messages"`` receives token chunks (SSE + UI).
    """
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        temperature=0,
        streaming=True,
    )


def _extract_email_content(state: EmailAgentState) -> str:
    """Return email_content from state, falling back to the last HumanMessage.

    When invoked via Agent Chat UI the graph input is ``{"messages": [HumanMessage(...)]}``,
    so ``email_content`` may be empty.  This helper bridges both entry paths.
    """
    if state.get("email_content"):
        return state["email_content"]
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content (production would connect to your email service)."""
    email_text = _extract_email_content(state)
    snippet = email_text[:200]
    update: dict = {"messages": [AIMessage(content=f"Processing email: {snippet}")]}
    if not state.get("email_content"):
        update["email_content"] = email_text
    if not state.get("sender_email"):
        update["sender_email"] = "customer@chat-ui.local"
    if not state.get("email_id"):
        update["email_id"] = "chat-ui"
    return update


def classify_intent(
    state: EmailAgentState,
) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Classify intent and urgency, then route to the appropriate node."""
    llm = get_chat_model()
    structured_llm = llm.with_structured_output(EmailClassification)

    classification_prompt = f"""
Analyze this customer email and classify it:

Email: {state["email_content"]}
From: {state["sender_email"]}

Provide classification including intent, urgency, topic, and summary.
"""
    classification = structured_llm.invoke(classification_prompt)

    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        next_node: Literal[
            "search_documentation", "human_review", "draft_response", "bug_tracking"
        ] = "human_review"
    elif classification["intent"] in ("question", "feature"):
        next_node = "search_documentation"
    elif classification["intent"] == "bug":
        next_node = "bug_tracking"
    else:
        next_node = "draft_response"

    cls_msg = (
        f"Classified intent={classification['intent']}, "
        f"urgency={classification['urgency']}, topic={classification['topic']}"
    )
    return Command(
        update={
            "classification": classification,
            "messages": [AIMessage(content=cls_msg)],
        },
        goto=next_node,
    )


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information."""
    classification = state.get("classification") or {}
    _ = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols",
        ]
    except SearchAPIError as e:
        search_results = [f"Search temporarily unavailable: {e!s}"]

    topic = classification.get("topic", "unknown")
    return Command(
        update={
            "search_results": search_results,
            "messages": [AIMessage(content=f"Searched documentation (topic: {topic})")],
        },
        goto="draft_response",
    )


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update a bug tracking ticket."""
    _ = state
    ticket_id = "BUG-12345"
    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "messages": [AIMessage(content=f"Bug tracking: created ticket {ticket_id}")],
        },
        goto="draft_response",
    )


def draft_response(
    state: EmailAgentState,
) -> Command[Literal["human_review", "send_reply"]]:
    """Generate a response and route to human review or send."""
    llm = get_chat_model()
    classification = state.get("classification") or {}

    context_sections: list[str] = []
    if state.get("search_results"):
        formatted_docs = "\n".join(f"- {doc}" for doc in state["search_results"])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get("customer_history"):
        context_sections.append(
            f"Customer tier: {state['customer_history'].get('tier', 'standard')}"
        )

    draft_prompt = f"""
Draft a response to this customer email:
{state["email_content"]}

Email intent: {classification.get("intent", "unknown")}
Urgency level: {classification.get("urgency", "medium")}

{chr(10).join(context_sections)}

Guidelines:
- Be professional and helpful
- Address their specific concern
- Use the provided documentation when relevant
"""

    response = llm.invoke(draft_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    needs_review = (
        classification.get("urgency") in ("high", "critical")
        or classification.get("intent") == "complex"
    )
    goto: Literal["human_review", "send_reply"] = "human_review" if needs_review else "send_reply"

    return Command(
        update={
            "draft_response": content,
            "messages": [AIMessage(content="Drafted response.")],
        },
        goto=goto,
    )


def human_review(state: EmailAgentState) -> Command:
    """Pause for human review; resume with approval or rejection.

    Agent Chat UI requires at least one AIMessage in state before the interrupt renders.
    The ``draft_response`` node emits one; for the classify->human_review shortcut
    (billing/critical without a draft), we emit a status AIMessage here.
    """
    if not state.get("draft_response"):
        pass  # AIMessage from classify_intent is already in messages

    human_decision = interrupt(
        {
            "email_id": state.get("email_id", ""),
            "original_email": state.get("email_content", ""),
            "draft_response": state.get("draft_response", ""),
            "urgency": (state.get("classification") or {}).get("urgency"),
            "intent": (state.get("classification") or {}).get("intent"),
            "action": "Please review and approve/edit this response",
        }
    )

    if isinstance(human_decision, str):
        # Agent Chat UI sends a plain string when the user types in the interrupt input
        approved = human_decision.strip().lower() in ("approve", "approved", "yes", "y", "ok")
        edited = None if approved else None
        if not approved:
            return Command(
                update={"messages": [AIMessage(content="Human review: rejected")]},
                goto=END,
            )
        return Command(
            update={"messages": [AIMessage(content="Human review: approved")]},
            goto="send_reply",
        )

    if human_decision.get("approved"):
        edited = human_decision.get("edited_response", state.get("draft_response", ""))
        return Command(
            update={
                "draft_response": edited,
                "messages": [AIMessage(content="Human review: approved")],
            },
            goto="send_reply",
        )
    return Command(
        update={"messages": [AIMessage(content="Human review: rejected")]},
        goto=END,
    )


def send_reply(state: EmailAgentState) -> dict:
    """Send the email response (integrate with your email service in production)."""
    draft = state.get("draft_response") or ""
    _ = draft[:100]
    return {"messages": [AIMessage(content="Sent reply.")]}
