"""A tool registry with the production wrapping on: tracing, guardrails, repair, circuit breakers."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

from agi.application.cortex.resilience import CircuitBreaker
from agi.application.tools.healing import repair_arguments
from agi.domain.entities.tool import Tool, ToolRegistry, ToolResult
from agi.domain.exceptions import CircuitOpen

#: Failures that say "the dependency is unhealthy", as opposed to "the call was wrong".
INFRASTRUCTURE_ERRORS = frozenset({"Timeout", "ConnectionError", "HTTPError", "URLError", "TimeoutError",
                                   "RateLimited", "ServiceUnavailable", "OSError"})


class ManagedRegistry(ToolRegistry):
    """Everything :class:`ToolRegistry` does, plus:

    * **argument repair** - a call rejected for a misnamed argument is retried once with the name fixed;
    * **circuit breakers** - a tool that keeps failing for infrastructure reasons is short-circuited
      for a cool-down instead of being hammered;
    * **guardrails** - tool output is treated as untrusted data (secrets redacted, instructions fenced);
    * **tracing** - every call is a span.
    """

    def __init__(self, tools: Iterable[Tool | Callable] = (), tracer=None, guardrails=None,
                 repair: bool = True, breaker_threshold: int = 3, breaker_reset_s: float = 30.0,
                 clock: Callable[[], float] | None = None, **kwargs):
        super().__init__(tools, **kwargs)
        self.tracer, self.guardrails, self.repair = tracer, guardrails, repair
        self.breaker_threshold, self.breaker_reset_s = breaker_threshold, breaker_reset_s
        self._clock = clock
        self.breakers: dict[str, CircuitBreaker] = {}
        self.repaired: list[tuple[str, str]] = []           # (tool, what was changed)

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self.breakers:
            extra = {"clock": self._clock} if self._clock else {}
            self.breakers[name] = CircuitBreaker(self.breaker_threshold, self.breaker_reset_s,
                                                 name=f"tool '{name}'", **extra)
        return self.breakers[name]

    def _guarded(self, name: str, args: Mapping | None) -> ToolResult:
        breaker = self._breaker(name)
        try:
            breaker._before()
        except CircuitOpen as exc:
            return self._fail(name, args or {}, "CircuitOpen",
                              f"{exc}. Use a different tool or approach for now.")
        result = super().call(name, args)
        breaker._after(RuntimeError(result.output) if result.error_type in INFRASTRUCTURE_ERRORS else None)
        return result

    def call(self, name: str, args: Mapping | None = None) -> ToolResult:
        if self.tracer is None:
            return self._call(name, args, None)
        with self.tracer.span("action", name, args=dict(args) if isinstance(args, Mapping) else args) as span:
            result = self._call(name, args, span)
            span.update(ok=result.ok, error_type=result.error_type, seconds=round(result.seconds, 4),
                        status="ok" if result.ok else "error")
            self.tracer.event("observation", name, status="ok" if result.ok else "error", output=result.output)
            return result

    def _call(self, name: str, args: Mapping | None, span: dict | None) -> ToolResult:
        result = self._guarded(name, args)
        tool = self.get(name)
        if (self.repair and not result.ok and result.error_type == "BadArguments" and tool is not None
                and isinstance(args, Mapping)):
            fixed = repair_arguments(tool, args)
            if fixed is not None:
                retry = self._guarded(name, fixed[0])
                if retry.ok or retry.error_type != "BadArguments":
                    self.repaired.append((name, fixed[1]))
                    retry.output = f"[auto-repair: {fixed[1]}]\n{retry.output}"
                    if span is not None:
                        span["repaired"] = fixed[1]
                    result = retry
        if self.guardrails is not None and result.ok:
            verdict = self.guardrails.check_observation(result.output, source=name)
            result.output = verdict.text
            if span is not None and (verdict.flags or verdict.redactions):
                span["guardrail"] = verdict.flags + [f"redacted:{r}" for r in verdict.redactions]
        return result
