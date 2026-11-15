from dataclasses import replace

import pytest

from agent import run_turn
from agent.tools import FaultInjector, MockRAGTool, MockWebSearchTool
from conftest import FAST, CountingTool, NeverSatisfiedLLM


@pytest.mark.parametrize("budget", [1, 2, 4, 7])
def test_stops_at_exactly_max_tool_calls(make_graph, budget):
    config = replace(FAST, max_tool_calls=budget)
    rag, web = CountingTool("rag"), CountingTool("web")
    graph = make_graph(config, llm=NeverSatisfiedLLM(), rag_tool=rag, web_tool=web)

    result = run_turn(graph, "What is our vacation policy?", "t", config)

    assert result.ok
    assert rag.calls + web.calls == budget
    assert result.tool_calls == budget
    assert result.breaker_tripped
    assert "tool-call limit" in result.answer
    assert result.citations  # still answers with what it gathered


def test_zero_budget_calls_no_tools(make_graph):
    config = replace(FAST, max_tool_calls=0)
    rag = CountingTool("rag")
    result = run_turn(make_graph(config, rag_tool=rag), "What is our vacation policy?", "t", config)
    assert result.ok and rag.calls == 0 and result.tool_calls == 0


def test_breaker_not_tripped_on_normal_run(make_graph, config):
    result = run_turn(make_graph(), "What is our vacation policy?", "t", config)
    assert not result.breaker_tripped
    assert "tool-call limit" not in result.answer


def test_failing_tool_is_skipped_after_threshold(make_graph, config):
    # Both tools fail on every attempt: each is tried until its breaker opens, then the run ends.
    rag_faults, web_faults = FaultInjector(always_fail=True, permanent=True), FaultInjector(always_fail=True, permanent=True)
    graph = make_graph(rag_tool=MockRAGTool(faults=rag_faults), web_tool=MockWebSearchTool(faults=web_faults))

    result = run_turn(graph, "What is our vacation policy?", "t", config)

    assert result.ok
    assert rag_faults.calls == config.failure_threshold
    assert web_faults.calls == config.failure_threshold
    assert result.citations == []
    assert "couldn't find" in result.answer
    assert "failed during this request" in result.answer


def test_breaker_state_resets_next_turn(make_graph):
    config = replace(FAST, max_tool_calls=2)
    graph = make_graph(config, llm=NeverSatisfiedLLM(), rag_tool=CountingTool("rag"), web_tool=CountingTool("web"))
    first = run_turn(graph, "first question about policy", "t", config)
    second = run_turn(graph, "second question about policy", "t", config)
    assert first.breaker_tripped and second.breaker_tripped
    assert second.tool_calls == 2  # budget is per turn, not cumulative
