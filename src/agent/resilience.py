"""Timeout + retry wrapper for tool calls."""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from agent.tools import PermanentToolError, ToolTimeoutError, TransientToolError

logger = logging.getLogger("agent.resilience")

T = TypeVar("T")

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tool")


def _call_with_timeout(fn: Callable[[], T], timeout_s: float, name: str) -> T:
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as exc:
        # A running thread cannot be killed; the call is abandoned and its result ignored.
        future.cancel()
        raise ToolTimeoutError(f"{name}: timed out after {timeout_s}s") from exc


def call_with_retry(
    fn: Callable[[], T],
    *,
    name: str,
    retries: int,
    timeout_s: float,
    backoff_base_s: float = 0.2,
    backoff_max_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Run `fn` with a timeout, retrying transient errors with jittered exponential backoff.

    Raises TransientToolError when retries are exhausted and PermanentToolError for
    anything not worth retrying, including unexpected exception types.
    """
    last_error: TransientToolError | None = None
    for attempt in range(retries + 1):
        try:
            return _call_with_timeout(fn, timeout_s, name)
        except TransientToolError as exc:
            last_error = exc
            logger.warning("tool=%s attempt=%d/%d transient_error=%s", name, attempt + 1, retries + 1, exc)
            if attempt < retries:
                delay = min(backoff_max_s, backoff_base_s * (2**attempt)) * (0.5 + rng() / 2)
                sleep(delay)
        except PermanentToolError as exc:
            logger.error("tool=%s attempt=%d permanent_error=%s", name, attempt + 1, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - tool bugs must not crash the graph
            logger.exception("tool=%s attempt=%d unexpected_error", name, attempt + 1)
            raise PermanentToolError(f"{name}: unexpected {type(exc).__name__}: {exc}") from exc
    assert last_error is not None
    raise last_error
