"""Fault tolerance: circuit breakers, rate limiting, step/depth budgets, loop detection."""
from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Hashable, Iterator

from agi.domain.exceptions import BudgetExceeded, CircuitOpen, LoopDetected


class CircuitBreaker:
    """Stops calling a failing dependency, then probes it again after a cool-down.

    closed -> (``failure_threshold`` consecutive failures) -> open
    open -> (``reset_after`` seconds) -> half-open: one trial call
    half-open -> success: closed / failure: open again

    Works for sync and async callables and is safe to share between threads.
    """

    def __init__(self, failure_threshold: int = 5, reset_after: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, name: str = "breaker",
                 is_failure: Callable[[BaseException], bool] | None = None):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self.failure_threshold, self.reset_after = failure_threshold, reset_after
        self.clock, self.name = clock, name
        self.is_failure = is_failure or (lambda exc: True)
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False
        self._lock = threading.Lock()
        self.rejected = 0
        self.last_error: str | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self._opened_at >= self.reset_after else "open"

    def _before(self) -> None:
        with self._lock:
            state = self._state()
            if state == "open" or (state == "half-open" and self._probing):
                self.rejected += 1
                wait = max(0.0, self.reset_after - (self.clock() - (self._opened_at or 0.0)))
                raise CircuitOpen(f"{self.name} is open after {self._failures} consecutive failures "
                                  f"(last: {self.last_error}); retry in {wait:.1f}s")
            if state == "half-open":
                self._probing = True

    def _after(self, error: BaseException | None) -> None:
        with self._lock:
            self._probing = False
            if error is None:
                self._failures, self._opened_at = 0, None
            elif self.is_failure(error):
                self._failures += 1
                self.last_error = f"{type(error).__name__}: {error}"[:200]
                if self._failures >= self.failure_threshold or self._opened_at is not None:
                    self._opened_at = self.clock()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._before()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            self._after(exc)
            raise
        self._after(None)
        return result

    async def acall(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        self._before()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            self._after(exc)
            raise
        self._after(None)
        return result

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` behind this breaker (e.g. ``llm = breaker.wrap(llm)``)."""
        def guarded(*args: Any, **kwargs: Any) -> Any:
            return self.call(fn, *args, **kwargs)
        guarded.breaker = self                         # type: ignore[attr-defined]
        return guarded

    def reset(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._probing = 0, None, False


class RateLimiter:
    """Token bucket: at most ``rate`` calls per second, with bursts up to ``burst``."""

    def __init__(self, rate: float, burst: int = 1, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be positive and burst at least 1")
        self.rate, self.burst, self.clock, self.sleep = rate, burst, clock, sleep
        self._tokens = float(burst)
        self._stamp = clock()
        self._lock = threading.Lock()
        self.waited = 0.0

    def _take(self) -> float:
        """Take a token; returns how long the caller must wait first (0 if none)."""
        with self._lock:
            now = self.clock()
            self._tokens = min(self.burst, self._tokens + (now - self._stamp) * self.rate)
            self._stamp = now
            self._tokens -= 1.0
            return 0.0 if self._tokens >= 0 else -self._tokens / self.rate

    def allow(self) -> bool:
        """Take a token if one is available, without waiting and without running up a debt.

        For servers that reject rather than queue: a refused request costs the caller nothing
        further, so a burst of rejections does not lock them out afterwards.
        """
        with self._lock:
            now = self.clock()
            self._tokens = min(self.burst, self._tokens + (now - self._stamp) * self.rate)
            self._stamp = now
            if self._tokens < 1.0:
                return False
            self._tokens -= 1.0
            return True

    def acquire(self) -> float:
        wait = self._take()
        if wait > 0:
            self.waited += wait
            self.sleep(wait)
        return wait

    async def aacquire(self) -> float:
        wait = self._take()
        if wait > 0:
            self.waited += wait
            await asyncio.sleep(wait)
        return wait

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def limited(*args: Any, **kwargs: Any) -> Any:
            self.acquire()
            return fn(*args, **kwargs)
        return limited


class StepBudget:
    """Hard ceilings on steps, nesting depth and wall-clock time for one task."""

    def __init__(self, max_steps: int = 50, max_depth: int = 4, max_seconds: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.max_steps, self.max_depth, self.max_seconds, self.clock = max_steps, max_depth, max_seconds, clock
        self.steps = 0
        self.depth = 0
        self.deepest = 0
        self._started = clock()

    def tick(self, what: str = "step") -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"step limit of {self.max_steps} exceeded at {what}")
        if self.max_seconds is not None and self.clock() - self._started > self.max_seconds:
            raise BudgetExceeded(f"time limit of {self.max_seconds:g}s exceeded at {what}")

    @contextmanager
    def nested(self, what: str = "sub-task") -> Iterator[int]:
        """Enter one level of recursion (an agent delegating to an agent, say)."""
        if self.depth >= self.max_depth:
            raise BudgetExceeded(f"recursion depth limit of {self.max_depth} exceeded at {what}")
        self.depth += 1
        self.deepest = max(self.deepest, self.depth)
        try:
            yield self.depth
        finally:
            self.depth -= 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.steps)


class LoopDetector:
    """Notices when an agent is going round in circles.

    Catches both the same state repeated (A A A) and cycles (A B A B A B) by
    counting how often a state signature occurs within a sliding window.
    """

    def __init__(self, window: int = 12, max_repeats: int = 3):
        self.window: deque[Hashable] = deque(maxlen=window)
        self.max_repeats = max_repeats

    def observe(self, signature: Hashable) -> int:
        """Record a state; returns how many times it now appears in the window."""
        self.window.append(signature)
        return Counter(self.window)[signature]

    def check(self, signature: Hashable) -> None:
        count = self.observe(signature)
        if count >= self.max_repeats:
            raise LoopDetected(f"state repeated {count} times within the last {len(self.window)} steps")

    def reset(self) -> None:
        self.window.clear()
