"""OpenTelemetry export: events as OTLP/JSON, sent to a collector in the background."""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from agi.domain.value_objects.records import TraceEvent


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
