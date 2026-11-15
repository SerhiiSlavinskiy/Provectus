import pytest

from agent import AgentConfig, run_turn
from agent.routing import decide_next_step


@pytest.mark.parametrize(
    "query, expected_tools",
    [
        ("What is our vacation policy?", ["rag"]),
        ("What is the latest news on GPU prices?", ["web"]),
        ("Compare our deployment runbook with the latest industry practices", ["rag", "web"]),
        ("hello", []),
        # No routing hints and nothing in the corpus: RAG first, then escalate to web.
        ("How do quokkas sleep?", ["rag", "web"]),
    ],
)
def test_tools_selected_per_query(make_graph, config, query, expected_tools):
    result = run_turn(make_graph(), query, "t", config)
    assert result.ok
    assert result.tools_used == expected_tools
    assert result.tool_calls == len(expected_tools)
    assert result.answer


@pytest.mark.parametrize("text", ["", "   ", "x" * 5000])
def test_invalid_input_goes_to_fallback(make_graph, config, text):
    result = run_turn(make_graph(), text, "t", config)
    assert result.tool_calls == 0
    assert result.citations == []
    assert "question" in result.answer or "too long" in result.answer
    assert result.errors[0]["kind"] == "invalid_input"


def test_decide_finishes_plan_before_synthesizing():
    state = {"plan": ["rag", "web"], "attempted": ["rag"], "tool_call_count": 1, "query": "q"}
    assert decide_next_step(state, AgentConfig(), sufficient=True, refined_query=None).step == "web_search"


def test_decide_skips_open_tool():
    state = {
        "plan": ["rag"], "attempted": ["rag", "web", "web"], "tool_call_count": 3, "query": "q",
        "tool_failures": {"web": 2}, "last_call_failed": True,
    }
    decision = decide_next_step(state, AgentConfig(), sufficient=False, refined_query="better q")
    assert decision.step == "rag_search" and decision.query == "better q"


def test_decide_stops_when_nothing_new_to_try():
    state = {"plan": ["rag"], "attempted": ["rag", "web"], "tool_call_count": 2, "query": "q"}
    decision = decide_next_step(state, AgentConfig(), sufficient=False, refined_query="q")
    assert decision.step == "synthesize" and not decision.breaker_tripped
