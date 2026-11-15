from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from agent import build_graph, open_sqlite_checkpointer, run_turn
from conftest import FAST


def _messages(graph, thread_id):
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])


def test_conversation_survives_restart(tmp_path):
    config = replace(FAST, db_path=str(tmp_path / "checkpoints.sqlite"))

    with open_sqlite_checkpointer(config.db_path) as saver:
        graph = build_graph(config=config, checkpointer=saver)
        first = run_turn(graph, "What is our vacation policy?", "alice", config)
        assert first.citations[0]["id"] == "kb-001"

    # New connection and a freshly built graph, as after a process restart.
    with open_sqlite_checkpointer(config.db_path) as saver:
        graph = build_graph(config=config, checkpointer=saver)
        second = run_turn(graph, "and what about its limits?", "alice", config)

        assert second.citations[0]["id"] == "kb-001"   # follow-up resolved against turn 1
        assert "carry-over limit" in second.answer
        messages = _messages(graph, "alice")
        assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage]


def test_threads_are_isolated(make_graph, config):
    graph = make_graph()
    run_turn(graph, "What is our vacation policy?", "alice", config)
    run_turn(graph, "and what about its limits?", "bob", config)
    run_turn(graph, "and what about its limits?", "alice", config)

    def question(thread_id):
        return graph.get_state({"configurable": {"thread_id": thread_id}}).values["question"]

    assert len(_messages(graph, "alice")) == 4
    assert len(_messages(graph, "bob")) == 2
    assert "vacation" in question("alice")          # resolved against alice's history
    assert "vacation" not in question("bob")        # bob has no vacation context


def test_per_turn_state_does_not_leak(make_graph, config):
    graph = make_graph()
    first = run_turn(graph, "What is our expense policy?", "t", config)
    second = run_turn(graph, "What is the latest news on GPU prices?", "t", config)

    assert first.citations[0]["id"] == "kb-002"
    assert second.tools_used == ["web"] and second.tool_calls == 1
    assert all(c["origin"] == "web" for c in second.citations)   # no documents from turn 1


def test_thread_usable_after_failed_turn(make_graph, config):
    graph = make_graph()
    bad = run_turn(graph, "", "t", config)
    good = run_turn(graph, "What is our vacation policy?", "t", config)
    assert bad.errors and good.errors == []
    assert good.citations[0]["id"] == "kb-001"
