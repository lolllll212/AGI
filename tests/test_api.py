"""The async service and the HTTP front end."""
import asyncio
import json
import re
import threading
import time
import urllib.error
import urllib.request

import pytest

from agi.api import AgentServer, AgentService, build_engine
from agi.core.engine import EngineConfig
from agi.core.llm import ScriptedLLM
from agi.evaluation import DriftMonitor, write_fixtures

QUERY = ('Thought: query the file.\nAction: query_data\nAction Input: {"file": "sales.csv", "sql": '
         '"SELECT ROUND(SUM(amount), 2) FROM data WHERE region = \'%s\'"}')


def sales_model(prompt):
    """Answers regional totals from sales.csv by querying it; splits and combines when asked to."""
    if "Split this task" in prompt:
        return '{"subtasks": ["What is the total amount for North in sales.csv?", "What is the total amount for East in sales.csv?"]}'
    if "Combine the results" in prompt:
        return "East (3,739.10) sold more than North (3,090.75)."
    task = prompt.rsplit("\n\nTask: ", 1)[-1]
    region = next((r for r in ("North", "South", "East") if r in task.split("\n")[0]), None)
    if region is None:
        return "Final Answer: I can only report regional totals."
    if "Observation:" not in task:
        return QUERY % region
    figure = float(re.findall(r"\d+\.\d+", task.rsplit("Observation:", 1)[1])[-1])
    return f"Thought: the query returned the total.\nFinal Answer: The {region} total is {figure:,.2f}."


@pytest.fixture
def service(tmp_path):
    made = AgentService.create(sales_model, workspace=write_fixtures(tmp_path / "ws"), concurrency=3)
    yield made
    made.close()


# ---------------------------------------------------------------- service --- #
def test_build_engine_wires_tools_memory_and_healing(tmp_path):
    engine = build_engine(ScriptedLLM([]), workspace=write_fixtures(tmp_path / "ws"),
                          memory_dir=str(tmp_path / "state"), toolsmith=True)
    names = [tool.name for tool in engine.tools]
    assert {"calculator", "python", "query_data", "read_file", "remember", "recall", "find_tools", "create_tool"} <= set(names)
    assert engine.healer is not None and engine.memory.directory == str(tmp_path / "state")
    # The python tool is the self-healing one: a missing import is repaired and the repair is reported.
    result = engine.tools.call("python", {"code": "print(math.sqrt(81))"})
    assert result.ok and "9.0" in result.output and "import math" in result.output and engine.healer.repairs == 1
    engine.memory.close()
    bare = build_engine(None, memory=None, heal=False)
    assert bare.memory is None and bare.healer is None and "query_data" not in bare.tools
    with pytest.raises(ValueError, match="unknown sandbox"):
        build_engine(None, sandbox="vm")


def test_ask_learns_and_a_similar_task_becomes_a_reflex(service):
    async def scenario():
        first = await service.ask("What is the total amount for North in sales.csv?")
        second = await service.ask("What is the total amount for South in sales.csv?")
        return first, second

    first, second = asyncio.run(scenario())
    assert (first.mode, first.success, first.answer) == ("react", True, "The North total is 3,090.75.")
    assert (second.mode, second.llm_calls, second.answer) == ("reflex", 0, "The South total is 2,491.15.")
    stats = service.stats()
    assert stats["engine"]["runs"] == 2 and stats["engine"]["reflex"] == 1 and stats["memory"] is not None
    assert service.health() == {"status": "ok", "model": True, "model_circuit": "closed",
                                "uptime_s": service.health()["uptime_s"], "in_flight": 0,
                                "tools": len(service.engine.tools)}
    trace = service.trace(first.trace_id)
    assert trace[-1]["kind"] == "run" or any(e["kind"] == "run" for e in trace)
    assert any(e["kind"] == "action" and e["name"] == "query_data" for e in trace)
    assert service.trace("nope") == []


def test_sessions_carry_context_and_are_bounded(tmp_path):
    seen = []

    def model(prompt):
        seen.append(prompt)
        return "Final Answer: noted, the project is called Atlas." if "Atlas" in prompt.rsplit("Task: ", 1)[-1] \
            else "Final Answer: you called it Atlas."

    service = AgentService.create(model, memory=None, heal=False)
    service._max_sessions = 2

    async def scenario():
        await service.ask("The project is called Atlas, token sk-abcdefghijklmnopqrstuvwxyz123456.", session="s1")
        answer = await service.ask("What did I call the project?", session="s1")
        other = await service.ask("What did I call the project?", session="s2")
        await service.ask("What did I call the project?", session="s3")
        return answer, other

    answer, _ = asyncio.run(scenario())
    assert answer.answer == "you called it Atlas."
    assert "EARLIER IN THIS CONVERSATION\nUser: The project is called Atlas" in seen[1]
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in seen[1]         # secrets are not replayed from history
    assert "EARLIER IN THIS CONVERSATION" not in seen[2]               # another session sees nothing
    assert list(service._sessions) == ["s2", "s3"]                     # the oldest session was dropped
    assert service.end_session("s2") and not service.end_session("s2")


def test_concurrency_is_bounded_and_timeouts_come_back_as_results():
    running = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow(prompt):
        with lock:
            running["now"] += 1
            running["peak"] = max(running["peak"], running["now"])
        time.sleep(0.15)
        with lock:
            running["now"] -= 1
        return "Final Answer: done"

    service = AgentService.create(slow, memory=None, heal=False, concurrency=2,
                                  config=EngineConfig(learn=False))
    started = time.perf_counter()
    results = asyncio.run(service.ask_many([f"Summarise report number {i}" for i in range(6)]))
    elapsed = time.perf_counter() - started
    assert [r.answer for r in results] == ["done"] * 6 and running["peak"] == 2 and 0.4 < elapsed < 1.5
    late = asyncio.run(service.ask("Summarise the annual report", timeout=0.05))
    assert (late.success, late.stop_reason, late.error) == (False, "timeout", "did not finish within 0.05s")
    assert service.in_flight == 0


def test_solve_parallel_splits_runs_and_combines(service):
    result = asyncio.run(service.solve_parallel("Compare the North and East totals in sales.csv"))
    assert result.success and result.decomposed and result.answer == "East (3,739.10) sold more than North (3,090.75)."
    assert [p.answer for p in result.parts] == ["The North total is 3,090.75.", "The East total is 3,739.10."]


def test_the_service_feeds_a_drift_monitor():
    healthy = {"on": True}

    def model(prompt):
        if not healthy["on"]:
            raise ConnectionError("upstream model is down")
        return "Final Answer: done"

    service = AgentService.create(model, memory=None, heal=False, config=EngineConfig(learn=False),
                                  monitor=DriftMonitor(reference=10, window=5, metrics=()))

    async def scenario():
        for i in range(15):
            await service.ask(f"Summarise report number {i}", check=lambda a: a == "done")
        healthy["on"] = False
        for i in range(5):
            await service.ask(f"Summarise report number {i}", check=lambda a: a == "done")

    asyncio.run(scenario())
    stats = service.stats()
    assert stats["drift"]["reference_success"] == 1.0 and stats["drift"]["recent_success"] < 0.5
    assert stats["drift_alarms"] and stats["drift_alarms"][0]["metric"] == "success_rate"
    assert stats["drift_alarms"][0]["direction"] == "worse"
    assert service.health()["status"] == "degraded" and service.health()["model_circuit"] == "open"


# ------------------------------------------------------------------- http --- #
def call(server, method, path, body=None, token=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(server.url + path, data=data, method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_run_batch_solve_trace_and_stats(service):
    with AgentServer(service, port=0) as server:
        assert call(server, "GET", "/health")[1]["status"] == "ok"
        status, tools = call(server, "GET", "/v1/tools")
        query = next(t for t in tools["tools"] if t["name"] == "query_data")
        assert status == 200 and query["parameters"]["file"]["type"] == "string"

        status, run = call(server, "POST", "/v1/run", {"task": "What is the total amount for North in sales.csv?"})
        assert status == 200 and run["answer"] == "The North total is 3,090.75." and run["success"] is True
        assert run["steps"] == 1 and run["mode"] == "react" and "route" not in run
        status, detailed = call(server, "POST", "/v1/run", {"task": "What is the total amount for East in sales.csv?",
                                                            "steps": True, "mode": "react"})
        assert detailed["steps"][0]["action"] == "query_data" and detailed["route"]["mode"] == "react"

        status, trace = call(server, "GET", f"/v1/traces/{run['trace_id']}")
        assert status == 200 and {"run", "action"} <= {e["kind"] for e in trace["events"]}

        status, batch = call(server, "POST", "/v1/batch", {"tasks": [
            "What is the total amount for South in sales.csv?", "What is the weather?"]})
        assert [r["answer"] for r in batch["results"]] == ["The South total is 2,491.15.",
                                                           "I can only report regional totals."]
        assert batch["results"][0]["mode"] == "reflex"                 # learned from the earlier runs

        status, solved = call(server, "POST", "/v1/solve", {"task": "Compare the North and East totals in sales.csv"})
        assert status == 200 and solved["decomposed"] and len(solved["parts"]) == 2

        status, stats = call(server, "GET", "/v1/stats")
        assert stats["engine"]["runs"] == 6 and stats["sessions"] == 0

        call(server, "POST", "/v1/run", {"task": "What is the weather?", "session": "abc"})
        assert call(server, "DELETE", "/v1/sessions/abc") == (200, {"ended": "abc"})
        assert call(server, "DELETE", "/v1/sessions/abc")[0] == 404


def test_http_rejects_bad_requests_without_leaking_details(service):
    with AgentServer(service, port=0) as server:
        assert call(server, "GET", "/nope") == (404, {"error": {"type": "not_found",
                                                                 "message": "no such endpoint: GET /nope"}})
        assert call(server, "GET", "/v1/run")[1]["error"] == {"type": "method_not_allowed",
                                                              "message": "use POST for /v1/run"}
        assert call(server, "GET", "/v1/traces/unknown")[0] == 404
        assert call(server, "POST", "/v1/run", raw=b"{not json")[1]["error"]["message"] == "the body is not valid JSON"
        assert call(server, "POST", "/v1/run", raw=b"")[0] == 400
        status, error = call(server, "POST", "/v1/run", {"task": "", "mode": "psychic", "extra": 1})
        assert status == 422 and error["error"]["type"] == "invalid_request"
        assert "task: String should have at least 1 character" in error["error"]["message"]
        assert "mode:" in error["error"]["message"] and "extra: Extra inputs are not permitted" in error["error"]["message"]
        assert call(server, "POST", "/v1/batch", {"tasks": []})[0] == 422
        assert call(server, "POST", "/v1/batch", {"tasks": ["x"] * 33})[0] == 422
        assert call(server, "POST", "/v1/run", raw=b"x" * 1_000_001)[0] == 413

        def broken(*args, **kwargs):
            raise RuntimeError("database password is hunter2")

        service.stats = broken
        assert call(server, "GET", "/v1/stats") == (500, {"error": {"type": "internal_error",
                                                                    "message": "RuntimeError"}})


def test_http_authentication_and_rate_limit(service):
    with pytest.raises(ValueError, match="refusing to listen on 0.0.0.0 without a token"):
        AgentServer(service, host="0.0.0.0", port=0)
    with AgentServer(service, port=0, token="s3cret", rate_limit=0.001, burst=2) as server:
        assert call(server, "GET", "/health")[0] == 200                # liveness needs no token
        assert call(server, "GET", "/v1/tools")[0] == 401
        assert call(server, "GET", "/v1/tools", token="wrong")[1]["error"]["type"] == "unauthorized"
        assert call(server, "GET", "/v1/tools", token="s3cret")[0] == 200
        assert call(server, "GET", "/v1/stats", token="s3cret")[0] == 200
        assert call(server, "GET", "/v1/stats", token="s3cret") == (
            429, {"error": {"type": "rate_limited", "message": "too many requests; slow down"}})
        assert call(server, "GET", "/v1/stats", token="wrong")[0] == 401   # bad tokens do not use up the allowance
        assert call(server, "GET", "/health")[0] == 200


def test_the_server_stops_cleanly(service):
    server = AgentServer(service, port=0).start()
    url = server.url
    assert call(server, "GET", "/health")[0] == 200
    server.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url + "/health", timeout=2)
