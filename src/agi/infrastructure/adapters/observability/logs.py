"""Structured logs: one JSON object per line, which is what Promtail and Loki want."""
from __future__ import annotations

import datetime as _dt
import json
import sys
import threading
from typing import Any, TextIO


class JsonLog:
    """Write events as JSON lines to a stream. Never raises: a full disk must not take a request down."""

    def __init__(self, stream: TextIO | None = None, service: str = "agi", clock=None):
        self.stream = stream if stream is not None else sys.stdout
        self.service = service
        self._clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))
        self._lock = threading.Lock()
        self._owned = False

    @classmethod
    def open(cls, target: str, service: str = "agi") -> "JsonLog":
        """``"-"`` logs to standard output; anything else is a file to append to."""
        if target == "-":
            return cls(sys.stdout, service)
        log = cls(open(target, "a", encoding="utf-8", buffering=1), service)
        log._owned = True
        return log

    def __call__(self, event: str, level: str = "info", **fields: Any) -> None:
        record = {"ts": self._clock().isoformat(timespec="milliseconds"), "level": level,
                  "service": self.service, "event": event, **fields}
        try:
            line = json.dumps(record, default=str)
            with self._lock:
                self.stream.write(line + "\n")
                self.stream.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._owned:
            try:
                self.stream.close()
            except Exception:
                pass
