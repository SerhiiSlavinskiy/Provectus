from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # Circuit breaker
    max_tool_calls: int = 4        # tool-call budget per turn
    failure_threshold: int = 2     # consecutive failures before a tool is skipped for the turn

    # Retry / timeout for a single tool call
    max_retries: int = 2           # extra attempts after the first, transient errors only
    tool_timeout_s: float = 5.0
    backoff_base_s: float = 0.2
    backoff_max_s: float = 2.0

    # Retrieval / synthesis
    top_k: int = 3
    max_sources: int = 5
    max_query_chars: int = 2000

    # Persistence
    db_path: str = "checkpoints.sqlite"

    @property
    def recursion_limit(self) -> int:
        # Each tool call costs two supersteps (tool + evaluate); leave headroom
        # so the tool-call budget, not the recursion limit, ends a normal run.
        return max(25, 2 * self.max_tool_calls + 10)
