# Plan: LangGraph agentic RAG + web search orchestrator

## Context

`project-plan.md` specifies an agentic system that analyzes a query, calls a RAG tool over internal docs, optionally calls web search, and synthesizes a cited answer, with multi-turn state, a circuit breaker, checkpointing and production-grade error handling. RAG and web search are mocked; the focus is orchestration.

The folder contains only the spec, so this is a greenfield build. The spec says "implement all TODO methods", but no skeleton file exists, so I will design the class/method layout myself (if you have a skeleton with the TODOs, drop it in the folder and I will fit the implementation to it instead).

Environment found: Python 3.13.7, no `uv`/`poetry`, no LangGraph installed, no `ANTHROPIC_API_KEY`. So: plain `venv` + `pip`, and the system must run fully offline.

## Decisions

- **LLM is injectable, mock by default.** Nodes depend on a small `AgentLLM` protocol (`analyze`, `grade`, `synthesize`). The default `HeuristicLLM` is deterministic (keyword rules + templates), so the demo and tests need no API key. An optional `ChatModelLLM` adapter wraps any LangChain chat model with structured output (Claude via `langchain-anthropic` when `ANTHROPIC_API_KEY` is set).
- **Circuit breaker lives in graph state**, so it is checkpointed and enforced by conditional edges, not by ad hoc checks inside tools.
- **Checkpointer: `SqliteSaver`** (survives process restarts) for the CLI, `InMemorySaver` for tests.

## Layout

```
pyproject.toml            # langgraph, langgraph-checkpoint-sqlite, langchain-core, pydantic; extras: anthropic, dev (pytest)
README.md                 # run instructions + graph diagram
src/agent/
  config.py               # AgentConfig: max_tool_calls, max_retries, tool_timeout_s, failure_threshold, db path
  state.py                # AgentState TypedDict + reducers
  tools.py                # MockRAGTool, MockWebSearchTool, ToolError hierarchy
  llm.py                  # AgentLLM protocol, HeuristicLLM, ChatModelLLM
  resilience.py           # call_with_retry (timeout + exponential backoff), breaker helpers
  nodes.py                # node functions
  routing.py              # conditional-edge functions
  graph.py                # build_graph(config, llm, tools, checkpointer)
  cli.py                  # multi-turn REPL with --thread-id
tests/
  test_routing.py  test_circuit_breaker.py  test_errors.py  test_persistence.py  test_citations.py
```

## State (`state.py`)

| Field | Type / reducer | Scope |
|---|---|---|
| `messages` | `Annotated[list[AnyMessage], add_messages]` | conversation (persists across turns) |
| `query` | `str` (standalone, rewritten from history) | per turn |
| `plan` | `list["rag" \| "web"]` | per turn |
| `documents` | list of `{id, title, snippet, source, origin}`, dedup-by-id reducer | per turn |
| `tool_call_count` | `int` | per turn |
| `tool_failures` | `dict[str, int]` consecutive failures per tool | per turn |
| `errors` | `Annotated[list[dict], operator.add]` | per turn |
| `breaker_tripped` | `bool` | per turn |
| `answer`, `citations` | `str`, `list[dict]` | per turn |

`analyze_query` resets the per-turn fields at the start of every turn; `messages` accumulates through the checkpointer.

## Graph (`graph.py`)

```
START -> analyze_query -> [route_after_analysis]
            |- "rag"        -> rag_search  -> evaluate_results
            |- "web"        -> web_search  -> evaluate_results
            |- "direct"     -> synthesize            (chitchat / answerable from history)
            '- "invalid"    -> fallback
evaluate_results -> [route_after_evaluation]
            |- breaker tripped / budget exhausted -> synthesize (degraded, says so)
            |- sufficient                         -> synthesize
            |- need web (planned, or RAG weak/failed) -> web_search
            '- retry rag with refined query       -> rag_search
synthesize -> [route_after_synthesis] -> END | fallback
fallback -> END
```

All branching uses `add_conditional_edges`. The evaluate → tool loop is what the breaker bounds.

## Nodes (`nodes.py`)

- `analyze_query`: validate input (empty/oversized → `invalid`), rewrite follow-ups into a standalone query using `messages`, produce `plan`.
- `rag_search` / `web_search`: call the mock tool via `call_with_retry`; increment `tool_call_count`; on success append documents and zero the tool's failure count; on failure append a structured error and bump `tool_failures`. Never raise out of the node.
- `evaluate_results`: grade document sufficiency, optionally refine the query, set `breaker_tripped`.
- `synthesize`: build the answer with `[n]` markers and a Sources list; verify every marker maps to a retrieved document and drop/flag any that do not; append `AIMessage`.
- `fallback`: safe user-facing message when input is invalid or synthesis fails.

## Circuit breaker and error handling

- **Call budget:** `tool_call_count >= max_tool_calls` (default 4) → `breaker_tripped`, route to `synthesize` with whatever was gathered, and the answer states it is partial.
- **Per-tool breaker:** `tool_failures[tool] >= failure_threshold` (default 2) marks that tool open for the turn; routing skips it (RAG open → try web; both open → degraded synthesis).
- **Retries:** `call_with_retry` applies a per-call timeout and exponential backoff with jitter, only for `TransientToolError`; `PermanentToolError` fails fast.
- **Backstop:** `recursion_limit` set on invoke; `GraphRecursionError` and any uncaught exception are caught in the CLI/entry point and returned as a clean error without corrupting the thread.
- **Mocks support fault injection:** `failure_rate`, `latency_s`, `fail_times`, `always_fail` on both mock tools, plus a small in-memory corpus with keyword scoring for RAG and canned results for web.
- Structured `logging` per node (thread id, node, tool, attempt, outcome).

## Build order

1. `pyproject.toml`, venv, install deps; `config.py`, `state.py`.
2. `tools.py` + `resilience.py` with unit tests.
3. `llm.py` (`HeuristicLLM` first, `ChatModelLLM` adapter after).
4. `nodes.py`, `routing.py`, `graph.py`.
5. `cli.py` with `SqliteSaver`.
6. Remaining tests, README.

## Verification

- `pytest`, covering:
  - routing: internal-doc query → RAG only; "latest/news" query → web; mixed → RAG then web; greeting → direct.
  - circuit breaker: always-insufficient results stop at exactly `max_tool_calls` and still return an answer flagged partial.
  - errors: transient failure recovers via retry; RAG always failing falls back to web; both failing yields a degraded answer, no exception.
  - persistence: turn 1, rebuild graph with the same SQLite file and `thread_id`, turn 2 follow-up ("and what about its limits?") resolves using history; different `thread_id` is isolated.
  - citations: every `[n]` in the answer maps to a retrieved document.
- Manual: `python -m agent.cli --thread-id demo`, ask two related questions, quit, restart with the same thread id and confirm history is kept; rerun with `--rag-fail` to watch the fallback path in logs.
- Print `graph.get_graph().draw_mermaid()` and check it matches the diagram above.
