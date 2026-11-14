"""What the server measures about itself and the agent, as Prometheus metrics."""
from __future__ import annotations

import collections
import threading
import time
from typing import Any

from agi.domain.value_objects.records import EngineResult
from agi.infrastructure.adapters.observability.metrics import Registry

RUN_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def outcome_of(result: EngineResult) -> str:
    if result.blocked:
        return "blocked"
    if result.stop_reason in ("timeout", "error"):
        return result.stop_reason
    return "failure" if result.success is False else "success"


class AgentMetrics:
    """Counters fed by every finished run and every HTTP request, plus gauges read at scrape time."""

    def __init__(self, service: Any, server: Any = None, registry: Registry | None = None):
        from agi import __version__
        self.service, self.server = service, server
        self.registry = r = registry or Registry()
        self.http_requests = r.counter("agi_http_requests_total", "HTTP requests handled.",
                                       ("method", "route", "status"))
        self.http_seconds = r.histogram("agi_http_request_duration_seconds", "Time to answer an HTTP request.",
                                        ("route",), HTTP_BUCKETS)
        self.http_rejected = r.counter("agi_http_rejected_total", "Requests refused before reaching the agent.",
                                       ("reason",))
        self.http_active = r.gauge("agi_http_connections_active", "Connections being served right now.")
        self.http_refused = r.counter("agi_http_connections_refused_total",
                                      "Connections refused because the server was at its connection limit.")
        self.runs = r.counter("agi_runs_total", "Tasks handled, by the mode that answered and the outcome.",
                              ("mode", "outcome"))
        self.run_seconds = r.histogram("agi_run_duration_seconds", "Time to handle one task.", ("mode",), RUN_BUCKETS)
        self.llm_calls = r.counter("agi_llm_calls_total", "Calls made to the language model.")
        self.tool_calls = r.counter("agi_tool_calls_total", "Tool calls made.")
        self.tool_errors = r.counter("agi_tool_errors_total", "Tool calls that failed.")
        self.repairs = r.counter("agi_repairs_total", "Failed tool calls that self-healing repaired.")
        self.escalations = r.counter("agi_escalations_total", "Tasks that had to move from a fast mode to a slower one.")
        self.compactions = r.counter("agi_context_compactions_total",
                                     "Times a run condensed earlier steps to stay inside its context window.")
        self.ungrounded = r.counter("agi_ungrounded_answers_total",
                                    "Answers containing figures that no tool result supports.")
        self.in_flight = r.gauge("agi_runs_in_flight", "Tasks being worked on right now.")
        self.sessions = r.gauge("agi_sessions", "Conversations the service is keeping.")
        self.circuit = r.gauge("agi_model_circuit_open", "1 while the circuit breaker around the model is open.")
        self.model = r.gauge("agi_model_configured", "1 if the service has a language model.")
        self.uptime = r.gauge("agi_uptime_seconds", "Seconds since the service started.")
        self.tools = r.gauge("agi_tools", "Tools the agent has.")
        self.drift = r.gauge("agi_drift_alarms", "Drift alarms raised since the service started.")
        self.spans = r.counter("agi_trace_spans_exported_total", "Spans handed to a trace collector, by result.",
                               ("result",))
        r.gauge("agi_build_info", "Always 1; the version is in the label.", ("version",)).set(1, version=__version__)
        r.on_collect(self._collect)

    # -- fed as things happen ------------------------------------------------ #
    def observe_result(self, result: EngineResult) -> None:
        self.runs.inc(mode=result.mode, outcome=outcome_of(result))
        self.run_seconds.observe(result.seconds, mode=result.mode)
        self.llm_calls.inc(result.llm_calls)
        self.tool_calls.inc(result.tool_calls)
        self.tool_errors.inc(result.tool_errors)
        self.repairs.inc(result.repairs)
        self.compactions.inc(result.compactions)
        if result.escalated:
            self.escalations.inc()
        if result.grounded is False:
            self.ungrounded.inc()

    def observe_http(self, method: str, route: str, status: int, seconds: float) -> None:
        self.http_requests.inc(method=method, route=route, status=str(status))
        self.http_seconds.observe(seconds, route=route)
        if status in (401, 429, 503):
            self.http_rejected.inc(reason={401: "unauthorized", 429: "rate_limited", 503: "overloaded"}[status])

    # -- read at scrape time --------------------------------------------------- #
    def _collect(self) -> None:
        service = self.service
        health = service.health()
        self.in_flight.set(health["in_flight"])
        self.circuit.set(1 if health["model_circuit"] == "open" else 0)
        self.model.set(1 if health["model"] else 0)
        self.uptime.set(health["uptime_s"])
        self.tools.set(health["tools"])
        store = getattr(service, "session_store", None)
        self.sessions.set(len(store) if store is not None else len(getattr(service, "_sessions", ())))
        self.drift.set(len(getattr(service, "alarms", ())))
        for exporter in getattr(service.engine.tracer, "exporters", ()):
            if hasattr(exporter, "stats"):
                numbers = exporter.stats()
                self.spans.at_least(numbers.get("sent", 0), result="sent")
                self.spans.at_least(numbers.get("dropped", 0), result="dropped")
        if self.server is not None:
            http = self.server.http_stats()
            self.http_active.set(http["active_connections"])
            self.http_refused.at_least(http["refused_connections"])

    def render(self) -> str:
        return self.registry.render()


class RunHistory:
    """The latest runs of each caller, for the console's "recent" list. Bounded, in memory only."""

    def __init__(self, per_caller: int = 100, callers: int = 1000, redact=None):
        """``redact`` (text -> text) is applied to each task before it is kept."""
        self._per_caller, self._callers, self._redact = per_caller, callers, redact
        self._runs: "collections.OrderedDict[str, collections.deque]" = collections.OrderedDict()
        self._lock = threading.Lock()

    def add(self, caller: str, result: EngineResult, session: str | None) -> None:
        task = result.task if self._redact is None else self._redact(result.task)
        entry = {"trace_id": result.trace_id, "task": task[:300], "mode": result.mode, "success": result.success,
                 "blocked": result.blocked, "stop_reason": result.stop_reason, "seconds": round(result.seconds, 3),
                 "tool_calls": result.tool_calls, "llm_calls": result.llm_calls, "session": session,
                 "at": round(time.time(), 3)}
        with self._lock:
            runs = self._runs.setdefault(caller, collections.deque(maxlen=self._per_caller))
            runs.appendleft(entry)
            self._runs.move_to_end(caller)
            while len(self._runs) > self._callers:
                self._runs.popitem(last=False)

    def of(self, caller: str) -> list[dict]:
        with self._lock:
            return list(self._runs.get(caller, ()))
