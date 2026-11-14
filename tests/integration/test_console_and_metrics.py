"""The routes added in 0.5: the web console's files, /metrics, /v1/info, /v1/runs and the access log."""
import asyncio
import io
import json
import re

import pytest

from agi.application.interfaces.service import AgentService
from agi.infrastructure.adapters.observability import JsonLog
from agi.infrastructure.api.routes import API_ROUTES, UI_ROUTES
from agi.infrastructure.api.server import AgentServer
from agi.infrastructure.api.telemetry import RunHistory, outcome_of
from tests.fakes import fetch, get_json, sales_service, send_json

ALICE, BOB = "alice-key-0123456789abcdef", "bob-key-0123456789abcdef"
NORTH = "What is the total amount for North in sales.csv?"


@pytest.fixture
def service(tmp_path):
    made = sales_service(tmp_path)
    yield made
    made.close()


@pytest.fixture
def console(tmp_path):
    """A stand-in for the built console: an index page, one script, and a file outside it."""
    root = tmp_path / "console"
    (root / "assets" / "deep").mkdir(parents=True)
    (root / "index.html").write_text('<!doctype html><div id="root"></div><script type="module" src="./assets/app.js"></script>')
    (root / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (root / "assets" / "app.js").write_text("console.log('hi')")
    (root / "assets" / "deep" / "chunk.css").write_text("body{}")
    (tmp_path / "secret.txt").write_text("not for the web")
    return root


SAMPLE_LINE = re.compile(r'[a-z_]+(\{([a-z_]+="(?:[^"\\\n]|\\.)*",?)+\})? (-?[0-9][0-9.e+-]*|[+-]Inf|NaN)')


def sample(text: str, name: str) -> float:
    """The value of one sample line in a metrics page."""
    found = re.search(rf"^{re.escape(name)} (\S+)$", text, re.MULTILINE)
    assert found, f"{name} is not in the metrics page"
    return float(found.group(1))


# ------------------------------------------------------------- the console --- #
def test_console_files_are_served_without_a_token_and_with_a_strict_policy(service, console):
    with AgentServer(service, port=0, token="the-shared-token-1", ui_dir=str(console)) as server:
        for path in ("/", "/index.html"):
            status, body, headers = fetch(server, "GET", path)
            assert status == 200 and b'<div id="root">' in body
            assert headers["Content-Type"] == "text/html; charset=utf-8" and headers["Cache-Control"] == "no-cache"
            policy = headers["Content-Security-Policy"]
            assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy and "unsafe-eval" not in policy
            assert "'unsafe-inline'" not in policy.split("script-src")[1].split(";")[0]
            assert headers["X-Frame-Options"] == "DENY" and headers["X-Content-Type-Options"] == "nosniff"
        status, body, headers = fetch(server, "GET", "/assets/app.js")
        assert (status, body) == (200, b"console.log('hi')")
        assert headers["Content-Type"].startswith("text/javascript") and "immutable" in headers["Cache-Control"]
        status, _, headers = fetch(server, "GET", "/assets/deep/chunk.css")
        assert status == 200 and headers["Content-Type"].startswith("text/css")
        assert fetch(server, "GET", "/favicon.svg")[2]["Content-Type"] == "image/svg+xml"
        # ...but the API behind it still needs the token.
        assert fetch(server, "GET", "/v1/info")[0] == 401
        assert fetch(server, "POST", "/", {})[0] == 405


@pytest.mark.parametrize("path", [
    "/assets/../../secret.txt", "/assets/..%2f..%2fsecret.txt", "/assets/%2e%2e/%2e%2e/secret.txt",
    "/assets/..\\..\\secret.txt", "/assets//etc/passwd", "/assets/", "/assets/missing.js", "/assets/deep",
    "/secret.txt", "/assets/../index.html",
])
def test_nothing_outside_the_console_folder_can_be_fetched(service, console, path):
    with AgentServer(service, port=0, ui_dir=str(console)) as server:
        import http.client
        connection = http.client.HTTPConnection(server.host, server.port, timeout=10)
        connection.request("GET", path)                      # sent as written: no client-side normalising
        response = connection.getresponse()
        body = response.read()
        connection.close()
        assert response.status == 404 and b"not for the web" not in body and b"root:" not in body


def test_console_can_be_switched_off_framed_or_missing(service, console, tmp_path):
    with AgentServer(service, port=0, ui=False, ui_dir=str(console)) as server:
        assert fetch(server, "GET", "/")[0] == 404 and fetch(server, "GET", "/assets/app.js")[0] == 404
        assert get_json(server, "/health")[0] == 200
    with AgentServer(service, port=0, ui_dir=str(console), allow_framing=True) as server:
        headers = fetch(server, "GET", "/")[2]
        assert "X-Frame-Options" not in headers and "frame-ancestors" not in headers["Content-Security-Policy"]
    with AgentServer(service, port=0, ui_dir=str(tmp_path / "empty")) as server:
        status, body, _ = fetch(server, "GET", "/")
        assert status == 404 and "not built" in json.loads(body)["error"]["message"]


def test_the_console_committed_to_the_repository_is_a_complete_build():
    from agi.infrastructure.api.routes.ui import STATIC
    index = (STATIC / "index.html").read_text()
    referenced = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', index)
    assert len(referenced) >= 2, "index.html should load a script and a stylesheet from assets/"
    for name in referenced:
        assert (STATIC / name).is_file(), f"{name} is referenced by index.html but missing: rebuild web/"
    assert "<script>" not in index and "onload=" not in index       # nothing inline: the policy would block it


# ------------------------------------------------------------ info and runs --- #
def test_info_describes_the_server_to_its_caller(service):
    service.label, service.demo, service.examples = "test-model", True, ["What is 2 + 2?"]
    with AgentServer(service, port=0, keys={"alice": ALICE}) as server:
        status, info = get_json(server, "/v1/info", ALICE)
        assert status == 200
        assert info == {"version": "0.5.0", "model": "test-model", "demo": True, "examples": ["What is 2 + 2?"],
                        "caller": "alice", "sandbox": "Sandbox", "metrics": True}
        health = get_json(server, "/health")[1]
        assert health["auth_required"] is True and "caller" not in health
    with AgentServer(service, port=0, metrics=False) as server:
        assert get_json(server, "/v1/info")[1]["metrics"] is False
        assert get_json(server, "/health")[1]["auth_required"] is False
        assert get_json(server, "/metrics")[0] == 404


def test_recent_runs_are_listed_newest_first_and_only_to_their_owner(service):
    with AgentServer(service, port=0, keys={"alice": ALICE, "bob": BOB}) as server:
        first = send_json(server, "/v1/run", {"task": NORTH, "session": "s1"}, ALICE)[1]
        second = send_json(server, "/v1/run", {"task": "What is 6 * 7?"}, ALICE)[1]
        send_json(server, "/v1/batch", {"tasks": ["What is 1 + 1?", "What is 2 + 2?"]}, BOB)
        runs = get_json(server, "/v1/runs", ALICE)[1]["runs"]
        assert [run["task"] for run in runs] == ["What is 6 * 7?", NORTH]
        assert runs[1]["trace_id"] == first["trace_id"] and runs[0]["trace_id"] == second["trace_id"]
        assert runs[1]["session"] == "s1" and runs[0]["session"] is None
        assert (runs[1]["mode"], runs[1]["success"], runs[1]["tool_calls"]) == ("react", True, 1)
        assert runs[0]["mode"] == "reflex" and runs[0]["at"] > 1_700_000_000
        assert sorted(run["task"] for run in get_json(server, "/v1/runs", BOB)[1]["runs"]) == ["What is 1 + 1?", "What is 2 + 2?"]
        assert get_json(server, "/v1/runs")[0] == 401


def test_run_history_is_bounded_and_redacts(service):
    result = asyncio.run(service.ask("What is 3 * 3?"))
    history = RunHistory(per_caller=3, callers=2, redact=lambda text: text.replace("3", "#"))
    for _ in range(5):
        history.add("a", result, None)
    history.add("b", result, "s")
    assert len(history.of("a")) == 3 and history.of("a")[0]["task"] == "What is # * #?"
    history.add("c", result, None)                           # a third caller pushes out the least recent one
    assert history.of("a") == [] and len(history.of("b")) == 1 and len(history.of("c")) == 1
    assert history.of("nobody") == []


def test_secrets_in_a_task_do_not_reach_the_run_list(service):
    with AgentServer(service, port=0) as server:
        send_json(server, "/v1/run", {"task": "My key is sk-abcdefghijklmnopqrstuvwxyz123456, what is 2 + 2?"})
        listed = get_json(server, "/v1/runs")[1]["runs"][0]["task"]
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in listed and "what is 2 + 2?" in listed


# ------------------------------------------------------------------ metrics --- #
def test_metrics_count_runs_requests_and_rejections(service):
    with AgentServer(service, port=0, token="the-shared-token-1") as server:
        token = "the-shared-token-1"
        assert fetch(server, "GET", "/metrics")[0] == 401                      # behind the token like the rest
        send_json(server, "/v1/run", {"task": NORTH}, token)
        send_json(server, "/v1/run", {"task": "What is 6 * 7?"}, token)
        send_json(server, "/v1/run", {"task": ""}, token)                      # 422
        get_json(server, "/v1/traces/does-not-exist", token)                   # 404
        get_json(server, "/v1/nothing-here", token)                            # no such route
        status, body, headers = fetch(server, "GET", "/metrics", token=token)
        text = body.decode()
        assert status == 200 and headers["Content-Type"].startswith("text/plain; version=0.0.4")

        assert sample(text, 'agi_runs_total{mode="react",outcome="success"}') == 1
        assert sample(text, 'agi_runs_total{mode="reflex",outcome="success"}') == 1
        assert sample(text, 'agi_run_duration_seconds_count{mode="react"}') == 1
        assert sample(text, "agi_llm_calls_total") == 2 and sample(text, "agi_tool_calls_total") == 2
        assert sample(text, 'agi_http_requests_total{method="POST",route="/v1/run",status="200"}') == 2
        assert sample(text, 'agi_http_requests_total{method="POST",route="/v1/run",status="422"}') == 1
        # The route label is the pattern, never the raw path: an attacker cannot mint endless label values.
        assert sample(text, 'agi_http_requests_total{method="GET",route="/v1/traces/{trace_id}",status="404"}') == 1
        assert sample(text, 'agi_http_requests_total{method="GET",route="other",status="404"}') == 1
        assert "does-not-exist" not in text and "nothing-here" not in text
        assert sample(text, 'agi_http_rejected_total{reason="unauthorized"}') == 1
        assert sample(text, "agi_model_configured") == 1 and sample(text, "agi_model_circuit_open") == 0
        assert sample(text, "agi_runs_in_flight") == 0 and sample(text, "agi_tools") >= 10
        assert sample(text, 'agi_build_info{version="0.5.0"}') == 1 and sample(text, "agi_uptime_seconds") >= 0
        assert sample(text, 'agi_http_request_duration_seconds_bucket{route="/v1/run",le="+Inf"}') == 3
        for line in text.splitlines():                       # every line is a comment or "name{labels} number"
            assert line.startswith("# ") or SAMPLE_LINE.fullmatch(line), line


def test_outcomes_are_told_apart(service):
    class Result:
        def __init__(self, **fields):
            self.blocked, self.success, self.stop_reason, self.error = False, True, "answer", None
            self.__dict__.update(fields)

    assert outcome_of(Result()) == "success"
    assert outcome_of(Result(blocked=True, success=False)) == "blocked"
    assert outcome_of(Result(success=False, stop_reason="timeout")) == "timeout"
    assert outcome_of(Result(success=False, stop_reason="max_steps")) == "failure"
    assert outcome_of(Result(success=False, stop_reason="error", error="boom")) == "error"


def test_a_failing_observer_does_not_break_a_run(service):
    seen = []
    service.observers.append(lambda result: 1 / 0)
    service.observers.append(seen.append)
    result = asyncio.run(service.ask("What is 6 * 7?"))
    assert result.answer and "42" in result.answer and seen == [result]


# --------------------------------------------------------------- access log --- #
def test_every_request_is_logged_once_with_its_id_and_never_with_the_token_or_task(service):
    stream = io.StringIO()
    token = "the-shared-token-1"
    with AgentServer(service, port=0, token=token, access_log=JsonLog(stream)) as server:
        _, _, headers = fetch(server, "POST", "/v1/run", {"task": "What is 6 * 7? (private detail)"}, token,
                              headers={"X-Request-Id": "trace-me-123"})
        assert headers["X-Request-Id"] == "trace-me-123"
        generated = fetch(server, "GET", "/v1/info")[2]["X-Request-Id"]              # no token: 401
        hostile = fetch(server, "GET", "/health", headers={"X-Request-Id": 'x" onload="alert(1)'})[2]["X-Request-Id"]
        assert re.fullmatch(r"[0-9a-f]{16}", generated) and re.fullmatch(r"[0-9a-f]{16}", hostile)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [(line["method"], line["route"], line["status"], line["caller"]) for line in lines] == [
        ("POST", "/v1/run", 200, "default"), ("GET", "/v1/info", 401, "(unauthenticated)"), ("GET", "/health", 200, "default")]
    assert lines[0]["request_id"] == "trace-me-123" and lines[1]["request_id"] == generated
    assert all(line["event"] == "request" and line["level"] == "info" and line["duration_ms"] >= 0 for line in lines)
    assert token not in stream.getvalue() and "private detail" not in stream.getvalue()


def test_route_tables_have_no_duplicates_and_only_the_expected_public_routes():
    routes = API_ROUTES + UI_ROUTES
    keys = [(route.method, route.pattern) for route in routes]
    assert len(keys) == len(set(keys))
    public = sorted(route.pattern for route in routes if route.public)
    assert public == ["/", "/assets/{path*}", "/favicon.svg", "/health", "/index.html"]
    assert isinstance(AgentService, type)
