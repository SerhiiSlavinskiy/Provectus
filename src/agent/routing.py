"""Routing decisions. The `route_*` functions are the graph's conditional edges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.config import AgentConfig
from agent.state import TOOL_NAMES, AgentState

NODE_FOR_TOOL = {"rag": "rag_search", "web": "web_search"}


def tool_is_open(state: AgentState, config: AgentConfig, tool: str) -> bool:
    """True when the tool's breaker is open (too many consecutive failures this turn)."""
    return (state.get("tool_failures") or {}).get(tool, 0) >= config.failure_threshold


@dataclass(frozen=True)
class Decision:
    step: str                      # "rag_search" | "web_search" | "synthesize"
    breaker_tripped: bool = False
    query: Optional[str] = None    # replacement search query, if refined


def decide_next_step(
    state: AgentState, config: AgentConfig, sufficient: bool, refined_query: Optional[str]
) -> Decision:
    """Pick the step after a tool call. Used by the evaluate_results node."""
    plan = list(state.get("plan") or [])
    attempted = list(state.get("attempted") or [])
    usable = [t for t in dict.fromkeys([*plan, *TOOL_NAMES]) if not tool_is_open(state, config, t)]

    wanted: Optional[str] = None
    new_query: Optional[str] = None

    planned_pending = [t for t in usable if t in plan and t not in attempted]
    if planned_pending:
        wanted = planned_pending[0]                 # finish the plan first
    elif sufficient:
        return Decision("synthesize")
    else:
        untried = [t for t in usable if t not in attempted]
        if untried:
            wanted = untried[0]                     # escalate to the other tool
        elif usable and refined_query and refined_query != state.get("query"):
            wanted, new_query = usable[0], refined_query
        elif usable and state.get("last_call_failed"):
            last = attempted[-1] if attempted else usable[0]
            wanted = last if last in usable else usable[0]

    if wanted is None:
        return Decision("synthesize")               # nothing left worth trying
    if state.get("tool_call_count", 0) >= config.max_tool_calls:
        return Decision("synthesize", breaker_tripped=True)
    return Decision(NODE_FOR_TOOL[wanted], query=new_query)


def make_route_after_analysis(config: AgentConfig):
    def route_after_analysis(state: AgentState) -> str:
        intent = state.get("intent")
        if intent == "invalid":
            return "fallback"
        plan = state.get("plan") or []
        if intent == "direct" or not plan or config.max_tool_calls <= 0:
            return "synthesize"
        return NODE_FOR_TOOL[plan[0]]

    return route_after_analysis


def route_after_evaluation(state: AgentState) -> str:
    step = state.get("next_step")
    return step if step in ("rag_search", "web_search") else "synthesize"


def route_after_synthesis(state: AgentState) -> str:
    return "end" if state.get("answer") else "fallback"
