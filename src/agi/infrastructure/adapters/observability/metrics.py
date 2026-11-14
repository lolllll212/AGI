"""Prometheus metrics in the text exposition format (standard library only).

    registry = Registry()
    requests = registry.counter("agi_http_requests_total", "Requests handled.", ("method", "status"))
    requests.inc(method="GET", status="200")
    print(registry.render())

Counters, gauges and histograms with labels are supported; that is all the
server needs. ``on_collect`` registers a function that runs just before every
scrape, for values that are cheaper to read on demand than to keep up to date.
"""
from __future__ import annotations

import math
import re
import threading
from typing import Callable, Iterable, Sequence

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _number(value: float) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        if math.isnan(value):
            return "NaN"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names: Sequence[str], values: Sequence[str], extra: tuple[str, str] | None = None) -> str:
    pairs = [f'{name}="{_escape(value)}"' for name, value in zip(names, values)]
    if extra:
        pairs.append(f'{extra[0]}="{extra[1]}"')
    return "{" + ",".join(pairs) + "}" if pairs else ""


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help: str, labelnames: Iterable[str], lock: threading.Lock):
        labelnames = tuple(labelnames)
        if not _NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a valid metric name")
        for label in labelnames:
            if not _LABEL.fullmatch(label) or label.startswith("__") or label == "le":
                raise ValueError(f"{label!r} is not a valid label name")
        self.name, self.help, self.labelnames = name, help, labelnames
        self._lock = lock
        self._values: dict[tuple[str, ...], float] = {}

    def _key(self, labels: dict) -> tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name} takes the labels {self.labelnames}, got {tuple(sorted(labels))}")
        return tuple(str(labels[name]) for name in self.labelnames)

    def value(self, **labels: object) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def samples(self) -> list[str]:
        return [f"{self.name}{_labels(self.labelnames, key)} {_number(value)}"
                for key, value in sorted(self._values.items())]


class Counter(_Metric):
    """Only goes up. The name should end in ``_total``."""

    kind = "counter"

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        if amount < 0:
            raise ValueError("a counter cannot go down")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def at_least(self, total: float, **labels: object) -> None:
        """Raise the counter to ``total`` if it is below it (for totals kept elsewhere)."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = max(self._values.get(key, 0.0), float(total))


class Gauge(_Metric):
    """Goes up and down."""

    kind = "gauge"

    def set(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: object) -> None:
        self.inc(-amount, **labels)


class Histogram(_Metric):
    """Counts observations in cumulative buckets, and keeps their sum."""

    kind = "histogram"

    def __init__(self, name: str, help: str, labelnames: Iterable[str], lock: threading.Lock,
                 buckets: Sequence[float] = DEFAULT_BUCKETS):
        super().__init__(name, help, labelnames, lock)
        bounds = sorted(float(b) for b in buckets if not math.isinf(b))
        if not bounds:
            raise ValueError("a histogram needs at least one bucket")
        self.buckets = tuple(bounds)
        self._series: dict[tuple[str, ...], list[float]] = {}     # counts per bucket, then +Inf, then the sum

    def observe(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            series = self._series.setdefault(key, [0.0] * (len(self.buckets) + 2))
            index = next((i for i, bound in enumerate(self.buckets) if value <= bound), len(self.buckets))
            series[index] += 1
            series[-1] += value

    def count(self, **labels: object) -> int:
        with self._lock:
            series = self._series.get(self._key(labels))
            return int(sum(series[:-1])) if series else 0

    def samples(self) -> list[str]:
        lines = []
        for key, series in sorted(self._series.items()):
            running = 0.0
            for bound, count in zip(self.buckets, series):
                running += count
                lines.append(f"{self.name}_bucket{_labels(self.labelnames, key, ('le', _number(bound)))} "
                             f"{_number(running)}")
            running += series[len(self.buckets)]
            lines.append(f"{self.name}_bucket{_labels(self.labelnames, key, ('le', '+Inf'))} {_number(running)}")
            lines.append(f"{self.name}_sum{_labels(self.labelnames, key)} {_number(series[-1])}")
            lines.append(f"{self.name}_count{_labels(self.labelnames, key)} {_number(running)}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, _Metric] = {}
        self._collectors: list[Callable[[], None]] = []

    def _add(self, metric: _Metric) -> _Metric:
        if metric.name in self._metrics:
            raise ValueError(f"the metric {metric.name!r} is already registered")
        self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help: str, labelnames: Iterable[str] = ()) -> Counter:
        return self._add(Counter(name, help, labelnames, self._lock))

    def gauge(self, name: str, help: str, labelnames: Iterable[str] = ()) -> Gauge:
        return self._add(Gauge(name, help, labelnames, self._lock))

    def histogram(self, name: str, help: str, labelnames: Iterable[str] = (),
                  buckets: Sequence[float] = DEFAULT_BUCKETS) -> Histogram:
        return self._add(Histogram(name, help, labelnames, self._lock, buckets))

    def on_collect(self, collector: Callable[[], None]) -> None:
        self._collectors.append(collector)

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def render(self) -> str:
        """Every metric in the Prometheus text format. A collector that fails is skipped, not fatal."""
        for collector in self._collectors:
            try:
                collector()
            except Exception:
                pass
        lines = []
        with self._lock:
            for name in sorted(self._metrics):
                metric = self._metrics[name]
                help_text = metric.help.replace("\\", "\\\\").replace("\n", "\\n")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric.kind}")
                lines.extend(metric.samples())
        return "\n".join(lines) + "\n"
