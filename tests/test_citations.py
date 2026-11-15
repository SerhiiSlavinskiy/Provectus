import re

from agent import run_turn
from agent.llm import Analysis, ChatModelLLM, Grade, HeuristicLLM
from agent.nodes import validate_citations


def test_every_marker_maps_to_a_listed_source(make_graph, config):
    result = run_turn(make_graph(), "Compare our deployment runbook with the latest industry practices", "t", config)
    body, _, sources = result.answer.partition("Sources:")
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    listed = {int(n) for n in re.findall(r"^\[(\d+)\]", sources, flags=re.MULTILINE)}

    assert markers and markers == listed == {c["n"] for c in result.citations}
    assert {c["origin"] for c in result.citations} == {"rag", "web"}
    for citation in result.citations:
        assert citation["source"] in sources


def test_validate_citations_strips_unknown_markers():
    text, cited, invalid = validate_citations("Fact one [2]. Made up [7]. Fact two [1] [2].", n_sources=2)
    assert text == "Fact one [2]. Made up. Fact two [1] [2]."
    assert cited == [2, 1]
    assert invalid == [7]


class _HallucinatingLLM(HeuristicLLM):
    def synthesize(self, question, documents, history):
        return "Employees get 25 days [1] and a free car [9]."


def test_hallucinated_citation_is_removed_and_recorded(make_graph, config):
    result = run_turn(make_graph(llm=_HallucinatingLLM()), "What is our vacation policy?", "t", config)
    assert "[9]" not in result.answer
    assert [c["n"] for c in result.citations] == [1]
    assert any(e["kind"] == "invalid_citation" for e in result.errors)


def test_direct_reply_has_no_sources(make_graph, config):
    result = run_turn(make_graph(), "hello", "t", config)
    assert result.citations == [] and "Sources:" not in result.answer


class _FakeStructured:
    def __init__(self, value):
        self.value = value

    def invoke(self, messages):
        return self.value


class _FakeChatModel:
    """Minimal stand-in for a LangChain chat model."""

    def with_structured_output(self, schema):
        if schema is Analysis:
            return _FakeStructured(Analysis(standalone_query="What is our vacation policy?", intent="search", plan=["rag"]))
        return _FakeStructured(Grade(sufficient=True))

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content=[{"type": "text", "text": "You get 25 days per year [1]."}])


def test_chat_model_adapter_drives_the_graph(make_graph, config):
    result = run_turn(make_graph(llm=ChatModelLLM(_FakeChatModel())), "vacation?", "t", config)
    assert result.tools_used == ["rag"]
    assert result.answer.startswith("You get 25 days per year [1].")
    assert result.citations[0]["id"] == "kb-001"
