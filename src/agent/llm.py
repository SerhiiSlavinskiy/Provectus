"""The three LLM decisions the graph needs, behind one small protocol."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional, Protocol, Sequence

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.state import Document
from agent.tools import STOPWORDS, tokenize


class Analysis(BaseModel):
    standalone_query: str = Field(description="The user's question rewritten to stand on its own.")
    intent: Literal["search", "direct"] = Field(
        description="'direct' for chitchat or questions answerable from the conversation alone."
    )
    plan: list[Literal["rag", "web"]] = Field(
        default_factory=list,
        description="Tools to call, in order: 'rag' for internal docs, 'web' for external or recent info.",
    )


class Grade(BaseModel):
    sufficient: bool = Field(description="Whether the documents are enough to answer the question.")
    refined_query: Optional[str] = Field(
        default=None, description="A better search query if the documents are not sufficient."
    )


class AgentLLM(Protocol):
    def analyze(self, query: str, history: Sequence[AnyMessage]) -> Analysis: ...

    def grade(self, query: str, documents: Sequence[Document]) -> Grade: ...

    def synthesize(
        self, question: str, documents: Sequence[Document], history: Sequence[AnyMessage]
    ) -> str:
        """Answer using only `documents`, citing them as [1], [2], ... by list position."""
        ...


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# --------------------------------------------------------------------------- heuristic

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|good (morning|afternoon|evening)|bye|goodbye)\b[\s!.,]*$",
    re.IGNORECASE,
)
_FOLLOW_UP_START = re.compile(r"^\s*(and|also|what about|how about|but)\b", re.IGNORECASE)
_PRONOUN = re.compile(r"\b(it|its|that|those|they|them|these)\b", re.IGNORECASE)
_WEB_HINTS = frozenset(
    """latest news today current recent trend market competitor industry web internet online
    external public price weather release announced""".split()
)
_INTERNAL_HINTS = frozenset(
    """our internal company policy handbook doc docs documentation runbook onboarding vacation
    expense deployment incident retention remote password""".split()
)


class HeuristicLLM:
    """Deterministic stand-in for a chat model, so the graph runs offline and tests are stable."""

    def __init__(self, sufficient_score: float = 0.5) -> None:
        self.sufficient_score = sufficient_score

    def analyze(self, query: str, history: Sequence[AnyMessage]) -> Analysis:
        if _GREETING.match(query):
            return Analysis(standalone_query=query, intent="direct", plan=[])

        standalone = query
        is_follow_up = (
            bool(_FOLLOW_UP_START.match(query)) or bool(_PRONOUN.search(query)) or len(tokenize(query)) <= 2
        )
        if is_follow_up:
            previous = self._previous_question(history)
            if previous:
                standalone = f"{previous} {query}"

        words = set(re.findall(r"[a-z]+", standalone.lower()))
        plan: list[Literal["rag", "web"]] = []
        if words & _INTERNAL_HINTS:
            plan.append("rag")
        if words & _WEB_HINTS:
            plan.append("web")
        if not plan:
            plan = ["rag"]  # internal docs first; evaluation escalates to web if they are weak
        return Analysis(standalone_query=standalone, intent="search", plan=plan)

    @staticmethod
    def _previous_question(history: Sequence[AnyMessage]) -> Optional[str]:
        for message in reversed(history):
            if isinstance(message, HumanMessage):
                text = message_text(message).strip()
                if text and not _GREETING.match(text):
                    return text
        return None

    def grade(self, query: str, documents: Sequence[Document]) -> Grade:
        best = max((d.get("score", 0.0) for d in documents), default=0.0)
        if best >= self.sufficient_score:
            return Grade(sufficient=True)
        keywords = " ".join(
            w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOPWORDS
        )
        refined = keywords if keywords and keywords != query.lower().strip() else None
        return Grade(sufficient=False, refined_query=refined)

    def synthesize(
        self, question: str, documents: Sequence[Document], history: Sequence[AnyMessage]
    ) -> str:
        if not documents:
            return (
                "Hello! I can answer questions using the internal documentation and web search. "
                "What would you like to know?"
            )
        lines = [f'Here is what I found for "{question}":']
        lines += [f"- {doc['snippet']} [{i}]" for i, doc in enumerate(documents, start=1)]
        return "\n".join(lines)


# --------------------------------------------------------------------------- chat model

_ANALYZE_PROMPT = (
    "You route questions for an assistant that has two tools: 'rag' searches internal company "
    "documentation, 'web' searches the public internet. Rewrite the latest user message as a "
    "standalone question using the conversation, decide whether it needs a search at all, and "
    "list the tools to call. Prefer 'rag' for anything about the company; add 'web' only for "
    "external or time-sensitive information."
)
_GRADE_PROMPT = (
    "Decide whether the documents below are sufficient to answer the question. If not, propose "
    "a better search query."
)
_SYNTHESIZE_PROMPT = (
    "Answer the question using only the numbered sources provided. Cite every claim with the "
    "source number in square brackets, e.g. [1]. If the sources do not contain the answer, say "
    "so. Do not add a source list; it is appended automatically."
)


def format_documents(documents: Sequence[Document]) -> str:
    return "\n".join(
        f"[{i}] {doc['title']} ({doc['source']}): {doc['snippet']}"
        for i, doc in enumerate(documents, start=1)
    )


class ChatModelLLM:
    """Adapter for any LangChain chat model that supports `with_structured_output`."""

    def __init__(self, model: Any, history_window: int = 8) -> None:
        self._model = model
        self._analyzer = model.with_structured_output(Analysis)
        self._grader = model.with_structured_output(Grade)
        self._history_window = history_window

    def analyze(self, query: str, history: Sequence[AnyMessage]) -> Analysis:
        recent = list(history)[-self._history_window :]
        return self._analyzer.invoke([SystemMessage(_ANALYZE_PROMPT), *recent, HumanMessage(query)])

    def grade(self, query: str, documents: Sequence[Document]) -> Grade:
        body = f"Question: {query}\n\nDocuments:\n{format_documents(documents) or '(none)'}"
        return self._grader.invoke([SystemMessage(_GRADE_PROMPT), HumanMessage(body)])

    def synthesize(
        self, question: str, documents: Sequence[Document], history: Sequence[AnyMessage]
    ) -> str:
        recent = list(history)[-self._history_window :]
        body = f"Question: {question}\n\nSources:\n{format_documents(documents) or '(none)'}"
        reply = self._model.invoke([SystemMessage(_SYNTHESIZE_PROMPT), *recent, HumanMessage(body)])
        return message_text(reply)
