from dataclasses import replace

import pytest

from agent import run_turn
from agent.llm import HeuristicLLM
from agent.resilience import call_with_retry
from agent.tools import (
    FaultInjector,
    MockRAGTool,
    MockWebSearchTool,
    PermanentToolError,
    ToolTimeoutError,
    TransientToolError,
)
from conftest import FAST

QUESTION = "What is our vacation policy?"


def test_transient_failure_recovers_via_retry(make_graph, config):
    faults = FaultInjector(fail_times=1)
    result = run_turn(make_graph(rag_tool=MockRAGTool(faults=faults)), QUESTION, "t", config)
    assert faults.calls == 2          # one failure, one retry
    assert result.tool_calls == 1     # still one logical tool call
    assert result.errors == []
    assert result.citations[0]["id"] == "kb-001"


def test_rag_down_falls_back_to_web(make_graph, config):
    faults = FaultInjector(always_fail=True)
    result = run_turn(make_graph(rag_tool=MockRAGTool(faults=faults)), QUESTION, "t", config)
    assert result.ok
    assert result.tools_used == ["rag", "web"]
    assert faults.calls == config.max_retries + 1
    assert result.citations and all(c["origin"] == "web" for c in result.citations)
    assert any(e["kind"] == "tool_error" and e["tool"] == "rag" for e in result.errors)
    assert "internal documentation search failed" in result.answer


def test_both_tools_down_gives_degraded_answer(make_graph, config):
    graph = make_graph(
        rag_tool=MockRAGTool(faults=FaultInjector(always_fail=True)),
        web_tool=MockWebSearchTool(faults=FaultInjector(always_fail=True)),
    )
    result = run_turn(graph, QUESTION, "t", config)
    assert result.ok
    assert result.tool_calls <= config.max_tool_calls
    assert result.citations == []
    assert "couldn't find" in result.answer


def test_permanent_error_is_not_retried(make_graph, config):
    faults = FaultInjector(fail_times=1, permanent=True)
    result = run_turn(make_graph(rag_tool=MockRAGTool(faults=faults)), QUESTION, "t", config)
    assert result.tools_used[:2] == ["rag", "web"]
    assert result.errors[0]["error_type"] == "PermanentToolError"
    assert faults.calls == 1


def test_slow_tool_times_out(make_graph):
    config = replace(FAST, tool_timeout_s=0.05, max_retries=0)
    graph = make_graph(config, rag_tool=MockRAGTool(faults=FaultInjector(latency_s=0.3)))
    result = run_turn(graph, QUESTION, "t", config)
    assert result.ok
    assert result.errors[0]["error_type"] == "ToolTimeoutError"
    assert "web" in result.tools_used


def test_tool_bug_does_not_crash_graph(make_graph, config):
    class BrokenTool:
        name = "rag"

        def search(self, query, top_k):
            raise KeyError("boom")

    result = run_turn(make_graph(rag_tool=BrokenTool()), QUESTION, "t", config)
    assert result.ok and result.answer
    assert result.errors[0]["error_type"] == "PermanentToolError"


class _BrokenLLM(HeuristicLLM):
    def __init__(self, broken: str) -> None:
        super().__init__()
        self.broken = broken

    def analyze(self, *a):
        if self.broken == "analyze":
            raise RuntimeError("llm down")
        return super().analyze(*a)

    def grade(self, *a):
        if self.broken == "grade":
            raise RuntimeError("llm down")
        return super().grade(*a)

    def synthesize(self, *a):
        if self.broken == "synthesize":
            raise RuntimeError("llm down")
        return super().synthesize(*a)


@pytest.mark.parametrize("stage", ["analyze", "grade"])
def test_llm_failure_in_planning_still_answers(make_graph, config, stage):
    result = run_turn(make_graph(llm=_BrokenLLM(stage)), QUESTION, "t", config)
    assert result.ok
    assert result.citations[0]["id"] == "kb-001"
    assert any(e["kind"] == "llm_error" for e in result.errors)


def test_llm_failure_in_synthesis_uses_fallback(make_graph, config):
    result = run_turn(make_graph(llm=_BrokenLLM("synthesize")), QUESTION, "t", config)
    assert "internal error" in result.answer
    assert result.citations == []
    assert any(e["node"] == "synthesize" and e["kind"] == "llm_error" for e in result.errors)


def test_unhandled_graph_error_is_contained(config):
    class ExplodingGraph:
        def invoke(self, *a, **k):
            raise RuntimeError("checkpointer unavailable")

    result = run_turn(ExplodingGraph(), QUESTION, "t", config)
    assert not result.ok
    assert result.errors[0]["kind"] == "unhandled"


# ---------------------------------------------------------------- call_with_retry


def _retry(fn, **kwargs):
    kwargs.setdefault("retries", 2)
    kwargs.setdefault("timeout_s", 1.0)
    kwargs.setdefault("sleep", lambda s: None)
    return call_with_retry(fn, name="t", **kwargs)


def test_retry_backoff_grows_and_is_capped():
    delays = []

    def fail():
        raise TransientToolError("x")

    with pytest.raises(TransientToolError):
        _retry(fail, retries=4, backoff_base_s=1.0, backoff_max_s=3.0, sleep=delays.append, rng=lambda: 1.0)
    assert delays == [1.0, 2.0, 3.0, 3.0]


def test_retry_wraps_unexpected_exceptions():
    calls = []

    def bug():
        calls.append(1)
        raise ValueError("bad")

    with pytest.raises(PermanentToolError):
        _retry(bug)
    assert len(calls) == 1


def test_retry_timeout_is_transient():
    import time

    with pytest.raises(ToolTimeoutError):
        _retry(lambda: time.sleep(0.3), retries=0, timeout_s=0.05)
