from __future__ import annotations

import itertools

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent import AgentConfig, build_graph
from agent.llm import Grade, HeuristicLLM
from agent.state import Document

FAST = AgentConfig(backoff_base_s=0.0, tool_timeout_s=2.0)


@pytest.fixture
def config() -> AgentConfig:
    return FAST


@pytest.fixture
def make_graph():
    def _make(config: AgentConfig = FAST, **kwargs):
        kwargs.setdefault("checkpointer", InMemorySaver())
        return build_graph(config=config, **kwargs)

    return _make


class CountingTool:
    """Returns weak, unique documents and counts calls."""

    def __init__(self, name: str, score: float = 0.1) -> None:
        self.name = name
        self.score = score
        self.calls = 0

    def search(self, query: str, top_k: int) -> list[Document]:
        self.calls += 1
        return [
            Document(
                id=f"{self.name}-{self.calls}",
                title=f"{self.name} doc {self.calls}",
                snippet=f"weak result {self.calls}",
                source=f"mock://{self.name}/{self.calls}",
                origin=self.name,
                score=self.score,
            )
        ]


class NeverSatisfiedLLM(HeuristicLLM):
    """Always asks for another search with a fresh query."""

    def __init__(self) -> None:
        super().__init__()
        self._counter = itertools.count(1)

    def grade(self, query, documents) -> Grade:
        return Grade(sufficient=False, refined_query=f"refined query {next(self._counter)}")
