from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ToolName = Literal["rag", "web"]
TOOL_NAMES: tuple[ToolName, ...] = ("rag", "web")


class Document(TypedDict):
    id: str
    title: str
    snippet: str
    source: str   # URI or URL shown in the citation
    origin: str   # "rag" | "web"
    score: float


_RESET_KEY = "__reset__"


def reset_to(items: Optional[list] = None) -> dict:
    """Update value that clears a per-turn list channel, then seeds it with `items`."""
    return {_RESET_KEY: True, "items": list(items or [])}


def _split_update(update: Any) -> tuple[bool, list]:
    if isinstance(update, dict) and update.get(_RESET_KEY):
        return True, list(update.get("items") or [])
    return False, list(update or [])


def merge_documents(existing: Optional[list[Document]], update: Any) -> list[Document]:
    """Append documents, de-duplicating by id and keeping the higher score."""
    reset, new = _split_update(update)
    by_id: dict[str, Document] = {} if reset else {d["id"]: d for d in (existing or [])}
    for doc in new:
        current = by_id.get(doc["id"])
        if current is None or doc.get("score", 0.0) > current.get("score", 0.0):
            by_id[doc["id"]] = doc
    return list(by_id.values())


def merge_errors(existing: Optional[list[dict]], update: Any) -> list[dict]:
    reset, new = _split_update(update)
    return new if reset else list(existing or []) + new


class AgentState(TypedDict, total=False):
    # Conversation scope: accumulates across turns through the checkpointer.
    messages: Annotated[list[AnyMessage], add_messages]

    # Turn scope: reset by analyze_query at the start of every turn.
    question: str                  # standalone question (follow-ups resolved against history)
    query: str                     # current search query (may be refined between tool calls)
    intent: Literal["search", "direct", "invalid"]
    invalid_reason: Optional[str]
    plan: list[ToolName]           # tools the analysis asked for, in order
    attempted: list[ToolName]      # tools called so far this turn, in order
    documents: Annotated[list[Document], merge_documents]
    tool_call_count: int
    tool_failures: dict[str, int]  # consecutive failures per tool
    last_call_failed: bool
    sufficient: bool
    next_step: str                 # decided by evaluate_results, read by the conditional edge
    breaker_tripped: bool
    errors: Annotated[list[dict], merge_errors]
    answer: Optional[str]
    citations: list[dict]
