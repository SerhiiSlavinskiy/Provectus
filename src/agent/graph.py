from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agent.config import AgentConfig
from agent.llm import AgentLLM, HeuristicLLM
from agent.nodes import AgentNodes
from agent.routing import make_route_after_analysis, route_after_evaluation, route_after_synthesis
from agent.state import AgentState
from agent.tools import MockRAGTool, MockWebSearchTool, SearchTool

logger = logging.getLogger("agent.graph")


def build_graph(
    config: Optional[AgentConfig] = None,
    llm: Optional[AgentLLM] = None,
    rag_tool: Optional[SearchTool] = None,
    web_tool: Optional[SearchTool] = None,
    checkpointer: Any = None,
):
    """Compile the agent graph. Pass a checkpointer to get multi-turn memory per thread_id."""
    config = config or AgentConfig()
    nodes = AgentNodes(config, llm or HeuristicLLM(), rag_tool or MockRAGTool(), web_tool or MockWebSearchTool())

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("rag_search", nodes.rag_search)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("evaluate_results", nodes.evaluate_results)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("fallback", nodes.fallback)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        make_route_after_analysis(config),
        {"rag_search": "rag_search", "web_search": "web_search", "synthesize": "synthesize", "fallback": "fallback"},
    )
    graph.add_edge("rag_search", "evaluate_results")
    graph.add_edge("web_search", "evaluate_results")
    graph.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {"rag_search": "rag_search", "web_search": "web_search", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges("synthesize", route_after_synthesis, {"end": END, "fallback": "fallback"})
    graph.add_edge("fallback", END)
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def open_sqlite_checkpointer(path: str) -> Iterator[Any]:
    """SQLite-backed checkpointer; conversations survive process restarts."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


@dataclass
class TurnResult:
    answer: str
    ok: bool = True
    citations: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    breaker_tripped: bool = False


def run_turn(graph: Any, text: str, thread_id: str, config: Optional[AgentConfig] = None) -> TurnResult:
    """Run one conversation turn. Never raises; failures come back as ok=False."""
    config = config or AgentConfig()
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.recursion_limit}
    try:
        state = graph.invoke({"messages": [HumanMessage(str(text or ""))]}, run_config)
    except GraphRecursionError as exc:
        logger.error("thread=%s recursion limit reached: %s", thread_id, exc)
        return TurnResult(
            answer="I stopped because this request took too many steps. Please try rephrasing it.",
            ok=False,
            errors=[{"node": "graph", "kind": "recursion_limit", "detail": str(exc)}],
        )
    except Exception as exc:  # noqa: BLE001 - last line of defence
        logger.exception("thread=%s unhandled graph error", thread_id)
        return TurnResult(
            answer="Sorry, something went wrong while handling your request. Please try again.",
            ok=False,
            errors=[{"node": "graph", "kind": "unhandled", "detail": f"{type(exc).__name__}: {exc}"}],
        )
    return TurnResult(
        answer=state.get("answer") or "",
        citations=state.get("citations") or [],
        errors=state.get("errors") or [],
        tool_calls=state.get("tool_call_count", 0),
        tools_used=state.get("attempted") or [],
        breaker_tripped=bool(state.get("breaker_tripped")),
    )
