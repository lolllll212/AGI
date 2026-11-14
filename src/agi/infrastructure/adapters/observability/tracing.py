"""Execution tracing: every thought, action, observation and reward, as structured spans.

A :class:`Tracer` records :class:`~agi.core.types.TraceEvent` spans in memory
and, optionally, appends them to a JSONL file as they finish. ``to_otlp``
renders a trace in the OpenTelemetry OTLP/JSON shape, and :class:`OTLPExporter`
posts spans to a collector as they finish - without this package depending on
the OpenTelemetry SDK.
"""
from __future__ import annotations

import contextvars
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

from agi.domain.value_objects.records import TraceEvent
from agi.infrastructure.adapters.observability.otlp import _clean, otlp_payload

_current: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("agi_span", default=None)


class Tracer:
    def __init__(self, path: str | None = None, service: str = "agi", clock=time.time, max_events: int = 50_000,
                 exporters: Iterable[Callable[[TraceEvent], None]] = ()):
        """``exporters`` are called with every finished span (for example an :class:`OTLPExporter`)."""
        self.path, self.service, self.clock, self.max_events = path, service, clock, max_events
        self.events: list[TraceEvent] = []
        self.exporters: list[Callable[[TraceEvent], None]] = list(exporters)
        self._lock = threading.Lock()

    def add_exporter(self, exporter: Callable[[TraceEvent], None]) -> None:
        self.exporters.append(exporter)

    # -- recording --------------------------------------------------------- #
    def _record(self, event: TraceEvent) -> TraceEvent:
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(event.model_dump_json() + "\n")
        for exporter in self.exporters:
            try:
                exporter(event)
            except Exception:                               # an exporter must never break the run it observes
                pass
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
        return otlp_payload(self.trace(trace_id) if trace_id else list(self.events), self.service)


def load_jsonl(path: str) -> list[TraceEvent]:
    with open(path, encoding="utf-8") as handle:
        return [TraceEvent.model_validate_json(line) for line in handle if line.strip()]
