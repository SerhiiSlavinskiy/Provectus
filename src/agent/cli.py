"""Multi-turn command line chat: python -m agent.cli --thread-id demo"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace

from agent.config import AgentConfig
from agent.graph import build_graph, open_sqlite_checkpointer, run_turn
from agent.llm import ChatModelLLM, HeuristicLLM
from agent.tools import FaultInjector, MockRAGTool, MockWebSearchTool


def _build_llm(kind: str, model: str):
    if kind == "heuristic":
        return HeuristicLLM()
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        sys.exit('The anthropic LLM needs the extra: pip install -e ".[anthropic]"')
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    return ChatModelLLM(ChatAnthropic(model=model))


def _print_result(result) -> None:
    print(f"\nagent> {result.answer}")
    flags = [f"tools: {', '.join(result.tools_used) or 'none'}", f"tool calls: {result.tool_calls}"]
    if result.breaker_tripped:
        flags.append("circuit breaker tripped")
    if result.errors:
        flags.append(f"errors: {len(result.errors)}")
    print(f"       ({'; '.join(flags)})\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG + web search demo")
    parser.add_argument("--thread-id", default="default", help="conversation id; reuse it to resume")
    parser.add_argument("--db", default=AgentConfig().db_path, help="SQLite checkpoint file")
    parser.add_argument("--max-tool-calls", type=int, default=AgentConfig().max_tool_calls)
    parser.add_argument("--llm", choices=["heuristic", "anthropic"], default="heuristic")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "claude-opus-5-5"))
    parser.add_argument("--rag-fail", action="store_true", help="make the RAG tool always fail")
    parser.add_argument("--web-fail", action="store_true", help="make the web tool always fail")
    parser.add_argument("-q", "--query", action="append", help="ask non-interactively (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show node-level logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.CRITICAL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = replace(AgentConfig(), max_tool_calls=args.max_tool_calls, db_path=args.db)

    with open_sqlite_checkpointer(config.db_path) as checkpointer:
        graph = build_graph(
            config=config,
            llm=_build_llm(args.llm, args.model),
            rag_tool=MockRAGTool(faults=FaultInjector(always_fail=args.rag_fail)),
            web_tool=MockWebSearchTool(faults=FaultInjector(always_fail=args.web_fail)),
            checkpointer=checkpointer,
        )
        if args.query:
            for query in args.query:
                print(f"you> {query}")
                _print_result(run_turn(graph, query, args.thread_id, config))
            return 0

        print(f"Thread '{args.thread_id}' (history in {config.db_path}). Type 'exit' to quit.")
        while True:
            try:
                query = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if query.strip().lower() in {"exit", "quit"}:
                return 0
            _print_result(run_turn(graph, query, args.thread_id, config))


if __name__ == "__main__":
    raise SystemExit(main())
