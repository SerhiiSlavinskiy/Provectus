"""Graph nodes. None of them raise: failures are recorded in state and routed around."""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.config import AgentConfig
from agent.llm import AgentLLM, Analysis, Grade, message_text
from agent.resilience import call_with_retry
from agent.routing import decide_next_step
from agent.state import TOOL_NAMES, AgentState, Document, reset_to
from agent.tools import SearchTool, ToolError

logger = logging.getLogger("agent.nodes")

_MARKER = re.compile(r"\[(\d+)\]")


def _thread_id(config: Optional[RunnableConfig]) -> str:
    return str(((config or {}).get("configurable") or {}).get("thread_id", "-"))


def _error(node: str, kind: str, detail: Any, **extra: Any) -> dict:
    return {"node": node, "kind": kind, "detail": str(detail), **extra}


def validate_citations(text: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """Strip [n] markers that do not point at a source.

    Returns (clean_text, valid marker numbers in order of first use, invalid numbers).
    """
    cited: list[int] = []
    invalid: list[int] = []

    def check(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            if n not in cited:
                cited.append(n)
            return match.group(0)
        invalid.append(n)
        return ""

    clean = _MARKER.sub(check, text)
    clean = re.sub(r"[ \t]+([.,;:])", r"\1", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, cited, invalid


class AgentNodes:
    def __init__(
        self, config: AgentConfig, llm: AgentLLM, rag_tool: SearchTool, web_tool: SearchTool
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools: dict[str, SearchTool] = {"rag": rag_tool, "web": web_tool}

    # ------------------------------------------------------------------ analyze

    def analyze_query(self, state: AgentState, config: RunnableConfig) -> dict:
        thread = _thread_id(config)
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        text = message_text(last).strip() if isinstance(last, HumanMessage) else ""

        def turn(intent: str, *, question: str = text, plan=(), errors=(), invalid_reason=None) -> dict:
            # Every per-turn field is rewritten here so nothing leaks from the previous turn.
            return {
                "question": question,
                "query": question,
                "intent": intent,
                "invalid_reason": invalid_reason,
                "plan": list(plan),
                "attempted": [],
                "documents": reset_to(),
                "tool_call_count": 0,
                "tool_failures": {},
                "last_call_failed": False,
                "sufficient": False,
                "next_step": "",
                "breaker_tripped": False,
                "errors": reset_to(list(errors)),
                "answer": None,
                "citations": [],
            }

        if not text:
            logger.warning("thread=%s node=analyze_query outcome=invalid reason=empty_query", thread)
            return turn("invalid", invalid_reason="empty_query",
                        errors=[_error("analyze_query", "invalid_input", "empty query")])
        if len(text) > self.config.max_query_chars:
            logger.warning("thread=%s node=analyze_query outcome=invalid reason=query_too_long", thread)
            return turn("invalid", invalid_reason="query_too_long",
                        errors=[_error("analyze_query", "invalid_input", f"{len(text)} chars")])

        errors: list[dict] = []
        try:
            analysis = self.llm.analyze(text, messages[:-1])
        except Exception as exc:  # noqa: BLE001
            logger.exception("thread=%s node=analyze_query llm_error; defaulting to rag", thread)
            errors.append(_error("analyze_query", "llm_error", exc))
            analysis = Analysis(standalone_query=text, intent="search", plan=["rag"])

        question = (analysis.standalone_query or "").strip() or text
        plan = [t for t in dict.fromkeys(analysis.plan) if t in TOOL_NAMES]
        intent = analysis.intent
        if intent == "search" and not plan:
            plan = ["rag"]
        if intent == "direct":
            plan = []
        logger.info("thread=%s node=analyze_query intent=%s plan=%s question=%r", thread, intent, plan, question)
        return turn(intent, question=question, plan=plan, errors=errors)

    # ------------------------------------------------------------------ tools

    def rag_search(self, state: AgentState, config: RunnableConfig) -> dict:
        return self._search("rag", state, config)

    def web_search(self, state: AgentState, config: RunnableConfig) -> dict:
        return self._search("web", state, config)

    def _search(self, name: str, state: AgentState, config: RunnableConfig) -> dict:
        thread = _thread_id(config)
        tool = self.tools[name]
        query = state.get("query", "")
        count = state.get("tool_call_count", 0) + 1
        failures = dict(state.get("tool_failures") or {})
        update: dict = {
            "tool_call_count": count,
            "attempted": [*(state.get("attempted") or []), name],
        }
        try:
            documents: list[Document] = call_with_retry(
                lambda: tool.search(query, self.config.top_k),
                name=name,
                retries=self.config.max_retries,
                timeout_s=self.config.tool_timeout_s,
                backoff_base_s=self.config.backoff_base_s,
                backoff_max_s=self.config.backoff_max_s,
            )
        except ToolError as exc:
            failures[name] = failures.get(name, 0) + 1
            logger.error("thread=%s node=%s_search call=%d outcome=failed failures=%d error=%s",
                         thread, name, count, failures[name], exc)
            update.update(
                tool_failures=failures,
                last_call_failed=True,
                errors=[_error(f"{name}_search", "tool_error", exc, tool=name,
                               error_type=type(exc).__name__)],
            )
            return update

        failures[name] = 0
        logger.info("thread=%s node=%s_search call=%d outcome=ok documents=%d query=%r",
                    thread, name, count, len(documents), query)
        update.update(tool_failures=failures, last_call_failed=False, documents=list(documents))
        return update

    # ------------------------------------------------------------------ evaluate

    def evaluate_results(self, state: AgentState, config: RunnableConfig) -> dict:
        thread = _thread_id(config)
        documents = state.get("documents") or []
        update: dict = {}
        try:
            grade = self.llm.grade(state.get("query", ""), documents)
        except Exception as exc:  # noqa: BLE001
            logger.exception("thread=%s node=evaluate_results llm_error", thread)
            update["errors"] = [_error("evaluate_results", "llm_error", exc)]
            grade = Grade(sufficient=bool(documents))

        decision = decide_next_step(state, self.config, grade.sufficient, grade.refined_query)
        update.update(
            sufficient=grade.sufficient,
            next_step=decision.step,
            breaker_tripped=decision.breaker_tripped,
        )
        if decision.query:
            update["query"] = decision.query
        level = logging.WARNING if decision.breaker_tripped else logging.INFO
        logger.log(level, "thread=%s node=evaluate_results sufficient=%s tool_calls=%d/%d next=%s breaker_tripped=%s",
                   thread, grade.sufficient, state.get("tool_call_count", 0),
                   self.config.max_tool_calls, decision.step, decision.breaker_tripped)
        return update

    # ------------------------------------------------------------------ synthesize

    def _caveats(self, state: AgentState) -> list[str]:
        notes = []
        if state.get("breaker_tripped"):
            notes.append(
                f"Note: the tool-call limit ({self.config.max_tool_calls}) was reached before enough "
                "evidence was gathered, so this answer may be incomplete."
            )
        failed = sorted({e["tool"] for e in state.get("errors") or [] if e.get("kind") == "tool_error"})
        if failed:
            names = {"rag": "internal documentation search", "web": "web search"}
            notes.append(
                "Note: " + " and ".join(names.get(t, t) for t in failed)
                + " failed during this request, so this answer may be incomplete."
            )
        return notes

    def synthesize(self, state: AgentState, config: RunnableConfig) -> dict:
        thread = _thread_id(config)
        messages = state.get("messages") or []
        question = state.get("question") or state.get("query", "")
        documents = sorted(state.get("documents") or [], key=lambda d: d.get("score", 0.0), reverse=True)
        documents = documents[: self.config.max_sources]
        caveats = self._caveats(state)
        errors: list[dict] = []

        if state.get("intent") == "search" and not documents:
            # Never ask the model to answer a factual question with no evidence.
            body = "I couldn't find information to answer this in the internal documentation or on the web."
            cited: list[int] = []
        else:
            try:
                draft = self.llm.synthesize(question, documents, messages[:-1])
            except Exception as exc:  # noqa: BLE001
                logger.exception("thread=%s node=synthesize llm_error", thread)
                return {"answer": None, "errors": [_error("synthesize", "llm_error", exc)]}
            if not isinstance(draft, str) or not draft.strip():
                logger.error("thread=%s node=synthesize outcome=empty_answer", thread)
                return {"answer": None, "errors": [_error("synthesize", "empty_answer", "")]}
            body, cited, invalid = validate_citations(draft, len(documents))
            if invalid:
                errors.append(_error("synthesize", "invalid_citation", f"removed markers {invalid}"))
            if documents and not cited:
                errors.append(_error("synthesize", "uncited_answer", "answer cites no sources"))

        citations = [
            {
                "n": n,
                "id": documents[n - 1]["id"],
                "title": documents[n - 1]["title"],
                "source": documents[n - 1]["source"],
                "origin": documents[n - 1]["origin"],
            }
            for n in sorted(cited)
        ]
        parts = [body, *caveats]
        if citations:
            parts.append("Sources:\n" + "\n".join(f"[{c['n']}] {c['title']} - {c['source']}" for c in citations))
        answer = "\n\n".join(parts)
        logger.info("thread=%s node=synthesize documents=%d citations=%d degraded=%s",
                    thread, len(documents), len(citations), bool(caveats))
        update: dict = {"answer": answer, "citations": citations, "messages": [AIMessage(answer)]}
        if errors:
            update["errors"] = errors
        return update

    # ------------------------------------------------------------------ fallback

    def fallback(self, state: AgentState, config: RunnableConfig) -> dict:
        reason = state.get("invalid_reason")
        if reason == "empty_query":
            answer = "I didn't receive a question. Please type what you'd like to know."
        elif reason == "query_too_long":
            answer = (
                f"That message is too long (limit: {self.config.max_query_chars} characters). "
                "Please shorten it and try again."
            )
        else:
            answer = "Sorry, I couldn't produce an answer because of an internal error. Please try again."
        logger.warning("thread=%s node=fallback reason=%s", _thread_id(config), reason or "synthesis_failed")
        return {"answer": answer, "citations": [], "messages": [AIMessage(answer)]}
