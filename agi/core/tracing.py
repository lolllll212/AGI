"""Execution tracing: every thought, action, observation and reward, as structured spans.

A :class:`Tracer` records :class:`~agi.core.types.TraceEvent` spans in memory
and, optionally, appends them to a JSONL file as they finish. ``to_otlp``
renders a trace in the OpenTelemetry OTLP/JSON shape, so it can be posted to any
collector without this package depending on the OpenTelemetry SDK.
"""
from __future__ import annotations

import contextvars
import json
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .types import TraceEvent

_current: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("agi_span", default=None)


def _clean(value: Any, limit: int = 2000) -> Any:
    """Attributes must be JSON-safe and bounded."""
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[+{len(value) - limit} chars]"
    if isinstance(value, (list, tuple)):
        return [_clean(v, limit) for v in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): _clean(v, limit) for k, v in list(value.items())[:50]}
    return _clean(str(value), limit)


class Tracer:
    def __init__(self, path: str | None = None, service: str = "agi", clock=time.time, max_events: int = 50_000):
        self.path, self.service, self.clock, self.max_events = path, service, clock, max_events
        self.events: list[TraceEvent] = []
        self._lock = threading.Lock()

    # -- recording --------------------------------------------------------- #
    def _record(self, event: TraceEvent) -> TraceEvent:
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(event.model_dump_json() + "\n")
        return event

    @contextmanager
    def span(self, kind: str, name: str, **attributes: Any) -> Iterator[dict]:
        """Time a block. Yields a dict of attributes that can be added to while it runs."""
        parent = _current.get()
        trace_id = parent[0] if parent else secrets.token_hex(16)
        span_id = secrets.token_hex(8)
        token = _current.set((trace_id, span_id))
        live: dict[str, Any] = dict(attributes, trace_id=trace_id, span_id=span_id)
        start, status = self.clock(), "ok"
        try:
            yield live
        except BaseException as exc:
            status = "error"
            live["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _current.reset(token)
            attrs = {k: v for k, v in live.items() if k not in ("trace_id", "span_id")}
            if attrs.pop("status", None) == "error":
                status = "error"
            self._record(TraceEvent(trace_id=trace_id, span_id=span_id, parent_id=parent[1] if parent else None,
                                    kind=kind, name=name, start=start, end=self.clock(), status=status,
                                    attributes=_clean(attrs)))

    def event(self, kind: str, name: str, status: str = "ok", **attributes: Any) -> TraceEvent:
        """A point-in-time event inside the current span (or a trace of its own)."""
        parent = _current.get()
        now = self.clock()
        return self._record(TraceEvent(
            trace_id=parent[0] if parent else secrets.token_hex(16), span_id=secrets.token_hex(8),
            parent_id=parent[1] if parent else None, kind=kind, name=name, start=now, end=now,
            status=status, attributes=_clean(attributes)))

    def reward(self, value: float, reason: str = "", **attributes: Any) -> TraceEvent:
        return self.event("reward", "reward", value=float(value), reason=reason, **attributes)

    @staticmethod
    def current_trace_id() -> str | None:
        current = _current.get()
        return current[0] if current else None

    # -- reading ----------------------------------------------------------- #
    def trace(self, trace_id: str) -> list[TraceEvent]:
        with self._lock:
            return sorted((e for e in self.events if e.trace_id == trace_id), key=lambda e: (e.start, e.end))

    def trace_ids(self) -> list[str]:
        with self._lock:
            return list(dict.fromkeys(e.trace_id for e in self.events))

    def rewards(self, trace_id: str | None = None) -> list[float]:
        with self._lock:
            return [e.attributes["value"] for e in self.events
                    if e.kind == "reward" and (trace_id is None or e.trace_id == trace_id)]

    def export_jsonl(self, path: str, trace_id: str | None = None) -> int:
        events = self.trace(trace_id) if trace_id else list(self.events)
        with open(path, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(event.model_dump_json() + "\n")
        return len(events)

    def to_otlp(self, trace_id: str | None = None) -> dict:
        """The trace(s) as an OTLP/JSON ``ExportTraceServiceRequest`` body."""
        def attribute(key: str, value: Any) -> dict:
            if isinstance(value, bool):
                typed = {"boolValue": value}
            elif isinstance(value, int):
                typed = {"intValue": str(value)}
            elif isinstance(value, float):
                typed = {"doubleValue": value}
            else:
                typed = {"stringValue": value if isinstance(value, str) else json.dumps(value, default=str)}
            return {"key": key, "value": typed}

        events = self.trace(trace_id) if trace_id else list(self.events)
        spans = [{
            "traceId": e.trace_id, "spanId": e.span_id, **({"parentSpanId": e.parent_id} if e.parent_id else {}),
            "name": f"{e.kind}:{e.name}", "kind": 1,
            "startTimeUnixNano": str(int(e.start * 1e9)), "endTimeUnixNano": str(int(e.end * 1e9)),
            "attributes": [attribute("agi.kind", e.kind)] + [attribute(k, v) for k, v in e.attributes.items()],
            "status": {"code": 2 if e.status == "error" else 1},
        } for e in events]
        return {"resourceSpans": [{
            "resource": {"attributes": [attribute("service.name", self.service)]},
            "scopeSpans": [{"scope": {"name": "agi.core.tracing"}, "spans": spans}]}]}


def load_jsonl(path: str) -> list[TraceEvent]:
    with open(path, encoding="utf-8") as handle:
        return [TraceEvent.model_validate_json(line) for line in handle if line.strip()]
