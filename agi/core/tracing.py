"""Execution tracing: every thought, action, observation and reward, as structured spans.

A :class:`Tracer` records :class:`~agi.core.types.TraceEvent` spans in memory
and, optionally, appends them to a JSONL file as they finish. ``to_otlp``
renders a trace in the OpenTelemetry OTLP/JSON shape, and :class:`OTLPExporter`
posts spans to a collector as they finish - without this package depending on
the OpenTelemetry SDK.
"""
from __future__ import annotations

import contextvars
import json
import queue
import secrets
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

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


def otlp_payload(events: Iterable[TraceEvent], service: str = "agi") -> dict:
    """Events as an OTLP/JSON ``ExportTraceServiceRequest`` body."""
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

    spans = [{
        "traceId": e.trace_id, "spanId": e.span_id, **({"parentSpanId": e.parent_id} if e.parent_id else {}),
        "name": f"{e.kind}:{e.name}", "kind": 1,
        "startTimeUnixNano": str(int(e.start * 1e9)), "endTimeUnixNano": str(int(e.end * 1e9)),
        "attributes": [attribute("agi.kind", e.kind)] + [attribute(k, v) for k, v in e.attributes.items()],
        "status": {"code": 2 if e.status == "error" else 1},
    } for e in events]
    return {"resourceSpans": [{
        "resource": {"attributes": [attribute("service.name", service)]},
        "scopeSpans": [{"scope": {"name": "agi.core.tracing"}, "spans": spans}]}]}


class OTLPExporter:
    """Posts finished spans to an OpenTelemetry collector over OTLP/HTTP (JSON).

    ``endpoint`` is the full traces URL, for example
    ``http://localhost:4318/v1/traces``. Spans are queued and sent in batches by
    a background thread, so tracing never waits on the network. A collector
    that is down costs spans, not runs: after ``retries`` attempts a batch is
    dropped and counted in :attr:`dropped`; nothing is ever raised to the agent.
    """

    def __init__(self, endpoint: str, headers: dict[str, str] | None = None, service: str = "agi",
                 batch_size: int = 64, interval: float = 2.0, timeout: float = 5.0, retries: int = 2,
                 max_queue: int = 10_000, start: bool = True):
        if urlsplit(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"the OTLP endpoint must be an http(s) URL, got {endpoint!r}")
        self.endpoint, self.headers, self.service = endpoint, dict(headers or {}), service
        self.batch_size, self.interval, self.timeout, self.retries = batch_size, interval, timeout, retries
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._wake, self._stop = threading.Event(), threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self.sent = self.dropped = self.errors = 0
        self.last_error: str | None = None
        self._sleep: Callable[[float], None] = time.sleep
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._loop, name="agi-otlp-export", daemon=True)
            self._thread.start()

    def __call__(self, event: TraceEvent) -> None:
        """Queue one span. Called by the tracer; never blocks and never raises."""
        try:
            self._queue.put_nowait(event)
            self._idle.clear()
        except queue.Full:
            self.dropped += 1
            return
        if self._queue.qsize() >= self.batch_size:
            self._wake.set()

    def _post(self, events: list[TraceEvent]) -> bool:
        body = json.dumps(otlp_payload(events, self.service)).encode()
        request = urllib.request.Request(self.endpoint, data=body, method="POST",
                                         headers={"Content-Type": "application/json", **self.headers})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response.read()
                self.sent += len(events)
                return True
            except urllib.error.HTTPError as exc:
                self.errors += 1
                self.last_error = f"HTTP {exc.code}"
                if exc.code < 500 and exc.code != 429:      # the collector rejected it: retrying cannot help
                    break
            except Exception as exc:                        # refused, timed out, DNS, TLS ...
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                self._sleep(min(2.0, 0.2 * 2 ** attempt))
        self.dropped += len(events)
        return False

    def _drain(self) -> None:
        while True:
            batch: list[TraceEvent] = []
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                self._idle.set()
                return
            self._post(batch)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.interval)
            self._wake.clear()
            self._drain()

    def flush(self, timeout: float = 10.0) -> bool:
        """Send what is queued now. Returns False if it did not finish in time."""
        if self._thread is None or not self._thread.is_alive():
            self._drain()
            return True
        self._wake.set()
        return self._idle.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def stats(self) -> dict:
        return {"endpoint": self.endpoint, "sent": self.sent, "dropped": self.dropped, "errors": self.errors,
                "queued": self._queue.qsize(), "last_error": self.last_error}


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
