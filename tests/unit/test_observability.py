"""Metrics in the Prometheus text format, and the JSON-lines log."""
import datetime
import io
import json
import threading

import pytest

from agi.infrastructure.adapters.observability import JsonLog, Registry
from agi.infrastructure.adapters.observability.metrics import CONTENT_TYPE


def test_counters_gauges_and_labels_render_in_the_text_format():
    registry = Registry()
    requests = registry.counter("app_requests_total", "Requests served.", ["method", "status"])
    active = registry.gauge("app_active", "Requests in progress.")
    requests.inc(method="GET", status=200)
    requests.inc(2, method="GET", status=200)
    requests.inc(method="POST", status=500)
    active.set(3)
    active.dec()
    assert requests.value(method="GET", status=200) == 3 and active.value() == 2
    assert registry.render() == (
        "# HELP app_active Requests in progress.\n"
        "# TYPE app_active gauge\n"
        "app_active 2\n"
        "# HELP app_requests_total Requests served.\n"
        "# TYPE app_requests_total counter\n"
        'app_requests_total{method="GET",status="200"} 3\n'
        'app_requests_total{method="POST",status="500"} 1\n')
    assert CONTENT_TYPE.startswith("text/plain; version=0.0.4")


def test_label_values_and_help_text_are_escaped():
    registry = Registry()
    seen = registry.counter("seen_total", 'Line one\nline "two" \\ end', ["path"])
    seen.inc(path='a"b\\c\nd')
    text = registry.render()
    assert '# HELP seen_total Line one\\nline "two" \\\\ end\n' in text
    assert 'seen_total{path="a\\"b\\\\c\\nd"} 1\n' in text
    assert len(text.splitlines()) == 3                      # the raw newline did not split a sample in two


def test_a_histogram_is_cumulative_and_keeps_sum_and_count():
    registry = Registry()
    took = registry.histogram("took_seconds", "How long.", ["route"], buckets=(0.1, 1, 10, float("inf")))
    for value in (0.05, 0.1, 0.5, 7, 99):
        took.observe(value, route="/run")
    assert took.buckets == (0.1, 1.0, 10.0) and took.count(route="/run") == 5 and took.count(route="/other") == 0
    lines = [line for line in registry.render().splitlines() if not line.startswith("#")]
    assert lines == ['took_seconds_bucket{route="/run",le="0.1"} 2',        # a value on the boundary counts
                     'took_seconds_bucket{route="/run",le="1"} 3',
                     'took_seconds_bucket{route="/run",le="10"} 4',
                     'took_seconds_bucket{route="/run",le="+Inf"} 5',
                     'took_seconds_sum{route="/run"} 106.65',
                     'took_seconds_count{route="/run"} 5']


def test_misuse_is_refused():
    registry = Registry()
    counter = registry.counter("jobs_total", "Jobs.", ["kind"])
    with pytest.raises(ValueError, match="already registered"):
        registry.gauge("jobs_total", "Again.")
    with pytest.raises(ValueError, match="not a valid metric name"):
        registry.counter("9lives", "Bad name.")
    for label in ("le", "__reserved", "has space"):
        with pytest.raises(ValueError, match="not a valid label name"):
            registry.counter(f"x_{len(label)}_total", "Bad label.", [label])
    with pytest.raises(ValueError, match="takes the labels"):
        counter.inc()
    with pytest.raises(ValueError, match="takes the labels"):
        counter.inc(kind="a", extra="b")
    with pytest.raises(ValueError):
        counter.inc(-1, kind="a")                            # a counter never goes down
    with pytest.raises(ValueError, match="at least one bucket"):
        registry.histogram("h_seconds", "No buckets.", buckets=(float("inf"),))


def test_at_least_follows_a_total_kept_elsewhere_without_going_backwards():
    counter = Registry().counter("calls_total", "Calls.")
    counter.at_least(5)
    counter.at_least(3)                                      # the source was reset: keep what was reported
    assert counter.value() == 5
    counter.at_least(9)
    assert counter.value() == 9


def test_collectors_run_at_render_time_and_a_broken_one_is_skipped():
    registry = Registry()
    queue = registry.gauge("queue_depth", "Waiting jobs.")
    depth = [4]
    registry.on_collect(lambda: 1 / 0)
    registry.on_collect(lambda: queue.set(depth[0]))
    assert "queue_depth 4" in registry.render()
    depth[0] = 7
    assert "queue_depth 7" in registry.render()
    assert registry.names() == ["queue_depth"]


def test_counting_from_many_threads_loses_nothing():
    registry = Registry()
    hits = registry.counter("hits_total", "Hits.", ["worker"])
    took = registry.histogram("work_seconds", "Work.")

    def work(worker):
        for _ in range(2000):
            hits.inc(worker=worker % 2)
            took.observe(0.01)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert hits.value(worker=0) == hits.value(worker=1) == 8000 and took.count() == 16000


def test_json_log_writes_one_object_per_line():
    stream = io.StringIO()
    clock = lambda: datetime.datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=datetime.timezone.utc)  # noqa: E731
    log = JsonLog(stream, service="api", clock=clock)
    log("request", method="GET", status=200, note='quote " and\nnewline')
    log("crash", level="error", error=ValueError("boom"))
    first, second = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert first == {"ts": "2026-01-02T03:04:05.678+00:00", "level": "info", "service": "api", "event": "request",
                     "method": "GET", "status": 200, "note": 'quote " and\nnewline'}
    assert second["level"] == "error" and second["error"] == "boom"       # anything not JSON is written as text


def test_json_log_appends_to_a_file_and_never_raises(tmp_path):
    target = tmp_path / "access.log"
    target.write_text('{"event": "earlier"}\n')
    log = JsonLog.open(str(target))
    log("request", status=200)
    log.close()
    log("after close", status=500)                           # the stream is gone: dropped, not raised
    events = [json.loads(line)["event"] for line in target.read_text().splitlines()]
    assert events == ["earlier", "request"]

    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("disk full")
    JsonLog(Broken())("request")
