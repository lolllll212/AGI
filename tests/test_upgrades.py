"""Version 0.4: the context window in the loop, learned routing, concept matching, trace export,
the extended benchmark, and the hardened HTTP server."""
import asyncio
import json
import re

import pytest

from agi.api import AgentService
from agi.core.engine import CognitiveEngine, EngineConfig
from agi.core.llm import ScriptedLLM
from agi.core.react import ReActAgent
from agi.memory.working import WorkingMemory
from agi.tools import Tool, ToolRegistry
from agi.tools.base import Param


# ------------------------------------------------- context window in the loop --- #
def page_tools():
    def read_page(number: int) -> str:
        return f"page {number}: " + " ".join(f"entry{number}_{i} value {number * 1000 + i}" for i in range(150))
    return ToolRegistry([Tool("read_page", "Read one page of the ledger.", read_page,
                              {"number": Param("integer", "the page number")})])


def page_reader(pages):
    """A stand-in model that reads ``pages`` pages one by one and then reports the last value it saw."""
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        done = len(re.findall(r"^Action: read_page", prompt.rsplit("\n\nTask: ", 1)[-1], re.M))
        condensed = re.search(r"\[Summary of (\d+) earlier messages\]", prompt)
        done += int(condensed.group(1)) // 2 if condensed else 0
        if done < pages:
            return f'Thought: read page {done + 1}.\nAction: read_page\nAction Input: {{"number": {done + 1}}}'
        return f"Thought: all pages read.\nFinal Answer: The last value on page {pages} is {pages * 1000 + 149}."
    return model, prompts


def test_a_long_run_stays_inside_its_token_budget():
    model, prompts = page_reader(8)
    engine = CognitiveEngine(model, page_tools(), config=EngineConfig(context_tokens=1500, learn=False))
    result = engine.run("Read all 8 pages of the ledger and report the last value on page 8.", mode="react")
    assert result.success and result.answer == "The last value on page 8 is 8149." and result.tool_calls == 8
    assert result.compactions >= 1 and result.grounded is True
    window = WorkingMemory(budget=1500)
    assert all(window.count(p.rsplit("\n\nTask: ", 1)[-1]) < 1500 + 200 for p in prompts)
    assert "[Summary of" in prompts[-1] and "condensed to save space" in prompts[-1]
    assert "entry8_149 value 8149" in prompts[-1]                      # the newest output is whole
    assert "entry1_149 value 1149" not in prompts[-1]                  # the oldest is not
    event = next(e for e in engine.tracer.trace(result.trace_id) if e.kind == "memory" and e.name == "context")
    assert event.attributes["compactions"] >= 1 and event.attributes["budget"] == 1500

    # Without a window nothing is condensed and the same run's last prompt is several times larger.
    plain_model, plain_prompts = page_reader(8)
    plain = CognitiveEngine(plain_model, page_tools(), config=EngineConfig(context_tokens=0, learn=False))
    unbounded = plain.run("Read all 8 pages of the ledger and report the last value on page 8.", mode="react")
    assert unbounded.success and unbounded.compactions == 0 and unbounded.prompt_reuse is None
    assert len(plain_prompts[-1]) > 3 * len(prompts[-1])


def test_prompt_reuse_is_reported_and_high_when_nothing_is_condensed():
    model, _ = page_reader(4)
    engine = CognitiveEngine(model, page_tools(), config=EngineConfig(context_tokens=20_000, learn=False))
    result = engine.run("Read all 4 pages of the ledger and report the last value on page 4.", mode="react")
    assert result.success and result.compactions == 0
    # Each prompt only appends to the one before, so all of the previous prompt is reused. (The share is
    # of the new prompt, and every step here adds a page as large as everything before it.)
    assert 0.6 < result.prompt_reuse < 0.7
    single = CognitiveEngine(ScriptedLLM(["Final Answer: nothing to do"]), page_tools(),
                             config=EngineConfig(learn=False)).run("Say there is nothing to do.", mode="react")
    assert single.prompt_reuse is None                                 # one prompt: nothing to compare


def test_the_model_can_write_the_summaries():
    def model(prompt):
        if prompt.startswith("Summarise this part of an agent's work"):
            return "Pages 1-4 were read; nothing unusual. Highest value so far: 4149."
        return page_reader(6)[0](prompt)

    engine = CognitiveEngine(model, page_tools(),
                             config=EngineConfig(context_tokens=1500, summarize_context=True, learn=False))
    result = engine.run("Read all 6 pages of the ledger and report the last value on page 6.", mode="react")
    assert result.success and result.compactions >= 1
    assert result.llm_calls > 7                                        # the summaries are model calls, and counted


def test_react_agent_accepts_a_window_directly_and_resets_it():
    model, prompts = page_reader(5)
    window = WorkingMemory(budget=1200)
    agent = ReActAgent(model, page_tools(), working=window, verifier=None)
    first = agent.run("Read all 5 pages of the ledger and report the last value on page 5.")
    assert first.finished and first.context.compactions >= 1 and first.context.tokens <= 1200
    again = ReActAgent(ScriptedLLM(["Final Answer: fine"]), page_tools(), working=window, verifier=None).run("Say fine.")
    assert again.context.messages == 0 and again.answer == "fine"      # the window was emptied for the new run


def test_long_sessions_are_condensed_not_cut():
    seen = []

    def model(prompt):
        seen.append(prompt)
        return f"Final Answer: noted item {len(seen)}, " + "with a long explanation " * 12

    service = AgentService.create(model, memory=None, heal=False, config=EngineConfig(learn=False))
    service._session_tokens = 400

    async def chat():
        for i in range(1, 9):
            await service.ask(f"Please record inventory item number {i} for the audit", session="audit")

    asyncio.run(chat())
    history = seen[-1].split("EARLIER IN THIS CONVERSATION\n", 1)[1].split("\n\nTask: ")[0]
    assert history.startswith("Earlier: [Summary of") and "User asked: Please record inventory item number 1" in history
    assert "User: Please record inventory item number 7 for the audit" in history       # recent turns are verbatim
    assert "User: Please record inventory item number 2 for the audit" not in history   # old ones are in the summary
    assert WorkingMemory().count(history) <= 400
    assert service.stats()["sessions"] == 1 and service.end_session("audit")


# ------------------------------------------------------------ learned routing --- #
from agi.core.router import RouteLearner, Router, RouterConfig          # noqa: E402
from agi.memory import HierarchicalMemory                               # noqa: E402

PLAN = json.dumps({"plans": [{"steps": ["inspect what changed before the crash", "state the cause"], "tools": []}]})


def planner_needed(prompt):
    """A stand-in that only finds the cause of a crash when it has a plan to follow."""
    if "Do not solve" in prompt:
        return PLAN
    task = prompt.rsplit("\n\nTask: ", 1)[-1]
    if "crashed" not in task:
        return "Final Answer: summary written"
    return "Final Answer: the cause is a full disk" if "PLAN TO FOLLOW" in prompt else "Final Answer: unknown"


def test_the_learner_keeps_an_outcome_record_per_kind_of_task(tmp_path):
    learner = RouteLearner()
    assert RouteLearner.features(["analysis", "long"], 70) == ["analysis", "length:long"]
    assert RouteLearner.features([], 5) == ["length:short"]
    why, plain = ["analysis", "length:short"], ["length:short"]
    assert learner.risk(why) == 0.5 and learner.runs(why) == 0          # no evidence: no opinion
    for _ in range(6):
        learner.update(why, react_failed=True)
        learner.update(plain, react_failed=False)
    assert learner.risk(why) == 0.875 and learner.risk(plain) == 0.125  # (6+1)/(6+2) and (0+1)/(6+2)
    assert learner.risk(["coding", "length:short"]) == 0.5              # evidence does not leak between kinds
    assert learner.explain() == [("analysis + length:short", 0.875, 6), ("length:short", 0.125, 6)]
    assert learner.observations == 12

    path = str(tmp_path / "routing.json")
    saved = RouteLearner(path)
    saved.update(["coding", "length:medium"], react_failed=True)
    again = RouteLearner(path)
    assert again.profiles == {"coding + length:medium": [1, 1]} and again.observations == 1
    (tmp_path / "routing.json").write_text("{not json")
    assert RouteLearner(path).observations == 0                         # a damaged file is ignored, not fatal


def test_experience_overrules_the_rules_in_both_directions():
    router = Router(ToolRegistry(), config=RouterConfig(min_observations=6))
    why = "Explain why the nightly export was late"
    both = "Explain why we should compare these two vendors"            # two weak signals: the rules say plan
    assert router.route(why).mode == "react" and router.route(both).mode == "deliberate"
    for _ in range(5):
        router.learn(why, react_failed=True)
    assert router.route(why).mode == "react"                            # five outcomes are not yet enough
    router.learn(why, react_failed=True)
    planned = router.route(why)
    assert planned.mode == "deliberate" and "experience" in planned.signals
    assert planned.reason == ("experience: the reason-act loop failed on most earlier tasks like this "
                              "(estimated failure 88% over 6 runs)")
    assert router.route("Summarise the weekly report").mode == "react"  # other kinds of task are unaffected
    for _ in range(10):                                                 # the world changes: react now works
        router.learn(why, react_failed=False)
    assert router.route(why).mode == "react"                            # 6 failures in 16: back below the bar

    relaxed = Router(ToolRegistry(), config=RouterConfig(min_observations=6))
    for _ in range(6):
        relaxed.learn(both, react_failed=False)
    easy = relaxed.route(both)
    assert easy.mode == "react" and easy.reason.startswith("experience: tasks like this have been solved without")
    assert relaxed.route("Write a python function that parses dates and then explain why it works").mode == "deliberate"
    off = Router(ToolRegistry(), config=RouterConfig(learn_routes=False))
    off.learn(why, react_failed=True)
    assert off.learner is None and off.route(why).mode == "react"


def test_exploration_keeps_the_record_of_planned_tasks_fresh():
    router = Router(ToolRegistry(), config=RouterConfig(min_observations=4, explore_every=3))
    hard = "Explain why we should compare these two vendors"           # two weak signals: the rules say plan
    routes = [router.route(hard) for _ in range(6)]
    assert [r.mode for r in routes] == ["deliberate", "deliberate", "react", "deliberate", "deliberate", "react"]
    assert "exploring" in routes[2].signals and routes[2].reason.startswith("exploring: trying the cheaper path")
    for _ in range(6):
        router.learn(hard, react_failed=False)                          # the explorations all succeeded ...
    settled = router.route(hard)
    assert settled.mode == "react" and "experience" in settled.signals  # ... so planning is no longer the default

    # Tasks moved to planning by experience are re-tried too; otherwise that record could never recover.
    why = "Explain why the nightly export was late"
    for _ in range(4):
        router.learn(why, react_failed=True)
    modes = [router.route(why).mode for _ in range(6)]
    assert modes.count("deliberate") == 4 and modes.count("react") == 2
    assert Router(ToolRegistry()).config.explore_every == 0             # off unless asked for


def test_the_engine_learns_to_plan_first_and_remembers_across_restarts(tmp_path):
    state = str(tmp_path / "state")
    engine = CognitiveEngine(planner_needed, memory=HierarchicalMemory(state), config=EngineConfig(learn=False))
    check = lambda answer: "full disk" in answer                        # noqa: E731
    costs = []
    for i in range(1, 11):
        result = engine.run(f"Explain why service number {i} crashed last night", check=check)
        assert result.success and result.mode == "deliberate"
        costs.append((result.escalated, result.llm_calls))
    assert costs[0] == (True, 3)                                        # tried react, failed, then planned
    assert costs[-1] == (False, 2)                                      # now it plans straight away
    assert [escalated for escalated, _ in costs].index(False) == 8      # after the 8 outcomes it needs
    assert engine.run("Summarise the weekly report").mode == "react"    # other tasks are not dragged along
    engine.memory.close()

    restarted = CognitiveEngine(planner_needed, memory=HierarchicalMemory(state), config=EngineConfig(learn=False))
    result = restarted.run("Explain why service number 99 crashed last night", check=check)
    assert (result.escalated, result.llm_calls, result.mode) == (False, 2, "deliberate")
    assert "experience" in result.route.signals
    assert restarted.router.learner.explain()[0] == ("analysis + length:short", 0.9, 8)
    restarted.memory.close()


def test_a_model_outage_does_not_teach_the_router_anything():
    def down(prompt):
        raise ConnectionError("offline")

    engine = CognitiveEngine(down, config=EngineConfig(learn=False))
    for i in range(10):
        engine.run(f"Explain why service number {i} crashed last night")
    assert engine.router.learner.observations == 0


# ------------------------------------------------------------ concept matching --- #
from agi.api import build_engine                                        # noqa: E402
from agi.memory import MemoryEngine                                     # noqa: E402
from agi.memory.embed import ConceptEmbedder, HashingEmbedder, cosine   # noqa: E402
from agi.tools import SemanticToolRouter, standard_tools                # noqa: E402

REQUESTS = [  # the same needs in a user's words, which rarely repeat the tool's description
    ("How many days are left until the 1st of March?", "current_time"),
    ("What weekday is it?", "current_time"),
    ("What is 15 percent of 2,340?", "calculator"),
    ("Work out the compound interest on 5000 at 4% for 3 years", "calculator"),
    ("Which region had the highest revenue in the spreadsheet?", "query_data"),
    ("Average order size per customer from the orders table", "query_data"),
    ("Show me what is inside notes.txt", "read_file"),
    ("Open the README and tell me what it says", "read_file"),
    ("Which lines of the logs mention a timeout?", "search_files"),
    ("Find every occurrence of TODO in the project", "search_files"),
    ("What documents do we have in the folder?", "list_files"),
    ("Save the summary as a report", "write_file"),
    ("Produce an output file with the results", "write_file"),
    ("Simulate 1000 dice rolls and report the distribution", "python"),
    ("Download the pricing page from the website", "http_get"),
    ("Keep in mind that the client prefers weekly updates", "remember"),
    ("What did we learn previously about the billing export?", "recall"),
    ("Grab the JSON from that API link", "http_get"),
]


def hits(router, k):
    return sum(want in [m.name for m in router.rank(request)][:k] for request, want in REQUESTS)


def test_related_words_match_even_when_no_word_is_shared():
    embedder, plain = ConceptEmbedder(), HashingEmbedder()
    ask, tool = "how many days until the deadline", "the current date and clock reading"
    assert cosine(plain.embed(ask), plain.embed(tool)) < 0.1            # no word in common (only stray letters)
    assert cosine(embedder.embed(ask), embedder.embed(tool)) > 0.5      # same concept: time
    assert embedder.concepts(ask) == ["time"] and embedder.concepts("save the csv totals")[:1] == ["aggregate"]
    unrelated = cosine(embedder.embed(ask), embedder.embed("download the web page"))
    assert unrelated < 0.1
    domain = ConceptEmbedder(extra={"billing": "invoice refund chargeback"})
    assert cosine(domain.embed("issue a refund"), domain.embed("the invoice tool")) > 0.1
    assert cosine(embedder.embed("issue a refund"), embedder.embed("the invoice tool")) == 0.0


def test_tool_matching_on_paraphrased_requests_improves():
    tools = standard_tools("/tmp", network=True, memory=MemoryEngine())
    before = SemanticToolRouter(tools, embed=HashingEmbedder().embed)
    before.hints = {}
    for tool in tools:
        before.index(tool)                                              # the 0.3 behaviour: words only, no hints
    after = SemanticToolRouter(tools)
    assert (hits(before, 1), hits(before, 3)) == (6, 9)                 # of 18
    assert (hits(after, 1), hits(after, 3)) == (14, 17)
    assert after.best("How many days are left until the 1st of March?").name == "current_time"
    # Word groups are still only words: "3 years" pulls this request towards the clock.
    assert after.best("Work out the compound interest on 5000 at 4% for 3 years").name == "current_time"
    # Your own vocabulary can be added per tool, on top of the built-in hints.
    weak = after.rank("What was our turnover in Q3?")[0]
    custom = SemanticToolRouter(tools, hints={"query_data": "turnover revenue quarter figures"})
    strong = custom.rank("What was our turnover in Q3?")[0]
    assert strong.name == "query_data" and strong.score > 2 * weak.score
    assert "spreadsheet" in custom.hints["query_data"]


def test_an_embedding_model_can_be_plugged_into_tools_and_memory(tmp_path):
    class FakeEmbedder:                                                 # stands in for OpenAICompatEmbedder
        def __init__(self):
            self.calls = 0

        def embed(self, text, weight=None):
            self.calls += 1
            topics = ("refund", "weather", "invoice", "forecast")
            return {i: float(t in text.lower()) + 0.01 for i, t in enumerate(topics)}

    embedder = FakeEmbedder()
    engine = build_engine(None, embed=embedder, heal=False, extra_tools=[
        Tool("billing", "Handle a refund or an invoice.", lambda: "ok", {}),
        Tool("meteo", "Give the weather forecast.", lambda: "sunny", {})])
    assert engine.tool_router.best("refund for order 7").name == "billing"
    assert engine.tool_router.best("weather tomorrow").name == "meteo"
    engine.memory.episodic.add("note", "the customer asked for a refund")
    engine.memory.episodic.add("note", "the forecast was wrong")
    assert engine.memory.episodic.search("refund status", k=1)[0].record.text == "the customer asked for a refund"
    assert engine.memory.episodic.dim == 4 and embedder.calls > 4


# --------------------------------------------------------------- trace export --- #
import threading                                                        # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer     # noqa: E402

from agi.core.tracing import OTLPExporter, Tracer, otlp_payload         # noqa: E402


class Collector:
    """A stand-in OpenTelemetry collector: records what is posted to /v1/traces."""

    def __init__(self, statuses=()):
        self.bodies, self.headers, self.statuses = [], [], list(statuses)
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                status = collector.statuses.pop(0) if collector.statuses else 200
                if status == 200:
                    collector.bodies.append(json.loads(body))
                    collector.headers.append(dict(self.headers))
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/traces"

    def spans(self):
        return [span for body in self.bodies for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def test_spans_reach_a_collector_in_otlp_json():
    collector = Collector()
    exporter = OTLPExporter(collector.url, headers={"X-Team": "agents"}, service="billing-agent", batch_size=3)
    tracer = Tracer(service="billing-agent", exporters=[exporter])
    with tracer.span("run", "task", task="total by region") as run:
        with tracer.span("action", "query_data", rows=10, cached=False, share=0.5):
            pass
        tracer.event("observation", "query_data", status="error", detail="no such column")
        tracer.reward(1.0, "verified")
    assert exporter.flush() and exporter.stats()["sent"] == 4 and exporter.stats()["dropped"] == 0
    spans = collector.spans()
    assert sorted(s["name"] for s in spans) == ["action:query_data", "observation:query_data", "reward:reward", "run:task"]
    root = next(s for s in spans if s["name"] == "run:task")
    child = next(s for s in spans if s["name"] == "action:query_data")
    assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16 and "parentSpanId" not in root
    assert child["traceId"] == run["trace_id"] and child["parentSpanId"] == root["spanId"]
    assert {"key": "rows", "value": {"intValue": "10"}} in child["attributes"]
    assert {"key": "cached", "value": {"boolValue": False}} in child["attributes"]
    assert {"key": "share", "value": {"doubleValue": 0.5}} in child["attributes"]
    assert next(s for s in spans if s["name"] == "observation:query_data")["status"] == {"code": 2}
    assert int(root["endTimeUnixNano"]) >= int(root["startTimeUnixNano"]) > 10 ** 18
    resource = collector.bodies[0]["resourceSpans"][0]["resource"]["attributes"]
    assert resource == [{"key": "service.name", "value": {"stringValue": "billing-agent"}}]
    assert collector.headers[0]["X-Team"] == "agents" and collector.headers[0]["Content-Type"] == "application/json"
    assert len(collector.bodies) >= 2                                   # batches of three
    assert tracer.to_otlp() == otlp_payload(tracer.events, "billing-agent")
    exporter.close()
    collector.stop()


def test_a_failing_collector_costs_spans_not_runs():
    flaky = Collector(statuses=[503, 200])
    exporter = OTLPExporter(flaky.url, start=False, retries=2)
    exporter._sleep = lambda seconds: None
    tracer = Tracer(exporters=[exporter])
    tracer.event("action", "one")
    exporter.flush()
    assert exporter.stats()["sent"] == 1 and exporter.errors == 1 and exporter.last_error == "HTTP 503"   # retried

    rejecting = Collector(statuses=[400, 400, 400])
    strict = OTLPExporter(rejecting.url, start=False, retries=2)
    strict(tracer.events[0])
    strict.flush()
    assert (strict.sent, strict.dropped, strict.errors) == (0, 1, 1)    # a 4xx is not retried
    assert len(rejecting.statuses) == 2

    flaky.stop(), rejecting.stop()
    dead = OTLPExporter(flaky.url, start=False, retries=1, timeout=0.5)
    dead._sleep = lambda seconds: None
    engine = CognitiveEngine(ScriptedLLM(["Final Answer: done"]), tracer=Tracer(exporters=[dead]),
                             config=EngineConfig(learn=False))
    result = engine.run("Summarise the weekly report")                  # the collector is gone: the run is unaffected
    dead.flush()
    assert result.success and dead.sent == 0 and dead.dropped >= 3 and "refused" in dead.last_error.lower()

    full = OTLPExporter(flaky.url, start=False, max_queue=2)
    for _ in range(5):
        full(tracer.events[0])
    assert full.stats()["queued"] == 2 and full.dropped == 3            # a full queue drops, it never blocks

    def broken(event):
        raise RuntimeError("exporter bug")

    safe = Tracer(exporters=[broken])
    assert safe.event("action", "still recorded").name == "still recorded" and len(safe.events) == 1
    with pytest.raises(ValueError, match="http"):
        OTLPExporter("file:///etc/passwd")


def test_the_service_exports_traces_and_reports_on_it():
    collector = Collector()
    service = AgentService.create(ScriptedLLM(["Final Answer: done"]), memory=None, heal=False,
                                  otlp_endpoint=collector.url, config=EngineConfig(learn=False))
    result = asyncio.run(service.ask("Summarise the weekly report"))
    service.close()                                                     # flushes what is queued
    names = [s["name"] for s in collector.spans()]
    assert "run:task" in names and "llm:completion" in names and "reward:reward" in names
    assert {s["traceId"] for s in collector.spans()} == {result.trace_id}
    export = service.stats()["trace_export"][0]
    assert export["sent"] == len(names) and export["dropped"] == 0 and export["endpoint"] == collector.url
    collector.stop()


# ------------------------------------------------------- the hardened server --- #
import shutil                                                           # noqa: E402
import socket                                                           # noqa: E402
import ssl                                                              # noqa: E402
import subprocess                                                       # noqa: E402
import time                                                             # noqa: E402
import urllib.error                                                     # noqa: E402
import urllib.request                                                   # noqa: E402
import warnings                                                         # noqa: E402

from agi.api import AgentServer, SessionStore                           # noqa: E402
from agi.core.resilience import RateLimiter                             # noqa: E402

ALICE, BOB = "alice-key-0123456789abcdef", "bob-key-0123456789abcdef"


def echo_context(prompt):
    """A stand-in model that answers with the names it can see in the conversation so far."""
    task = prompt.rsplit("\n\nTask: ", 1)[-1]
    seen = sorted(set(re.findall(r"project (\w+)", prompt.replace(task, ""))))
    return "Final Answer: known projects: " + (", ".join(seen) or "none")


def http(server, method, path, body=None, token=None, headers=None, context=None):
    request = urllib.request.Request(server.url + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
    for key, value in {**({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})}.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), dict(error.headers)


def plain_service(llm=echo_context, **options):
    return AgentService.create(llm, memory=None, heal=False, config=EngineConfig(learn=False), **options)


def test_each_caller_has_their_own_key_sessions_and_rate_limit():
    service = plain_service()
    with AgentServer(service, port=0, keys={"alice": ALICE, "bob": BOB}, rate_limit=0.001, burst=3) as server:
        assert http(server, "GET", "/v1/tools")[0] == 401 and http(server, "GET", "/v1/tools", token="wrong-key")[0] == 401
        assert http(server, "GET", "/health")[0] == 200                 # liveness needs no key
        ask = {"task": "Remember project Atlas", "session": "main"}
        assert http(server, "POST", "/v1/run", ask, ALICE)[1]["success"]
        mine = http(server, "POST", "/v1/run", {"task": "Which do you know?", "session": "main"}, ALICE)[1]
        theirs = http(server, "POST", "/v1/run", {"task": "Which do you know?", "session": "main"}, BOB)[1]
        assert mine["answer"] == "known projects: Atlas"
        assert theirs["answer"] == "known projects: none"               # same session name, different caller
        assert http(server, "DELETE", "/v1/sessions/main", token=BOB)[0] == 200     # bob ends bob's, not alice's
        assert sorted(service._sessions) == ["alice/main"]
        # Each caller has a burst of three. Alice uses her third and is then refused; Bob still has one.
        assert http(server, "GET", "/v1/tools", token=ALICE)[0] == 200
        status, body, headers = http(server, "GET", "/v1/tools", token=ALICE)
        assert (status, body["error"]["type"], headers["Retry-After"]) == (429, "rate_limited", "1")
        status, stats, _ = http(server, "GET", "/v1/stats", token=BOB)
        assert status == 200 and stats["http"]["requests"] == {"alice": 3, "bob": 3}
        assert stats["http"]["rejected"] == {"(unauthenticated)": 2, "alice": 1} and stats["http"]["tls"] is False
        assert http(server, "GET", "/v1/stats", token=BOB)[0] == 429

    for bad, message in (({"keys": {"alice": "short"}}, "too short"),
                         ({"keys": {"a b": ALICE}}, "key name"),
                         ({"keys": {"alice": ALICE, "bob": ALICE}}, "different key"),
                         ({"host": "0.0.0.0"}, "without a token")):
        with pytest.raises(ValueError, match=message):
            AgentServer(service, port=0, **bad)
    AgentServer(service, host="0.0.0.0", port=0, keys={"alice": ALICE}).stop()      # a key is enough to go public


def test_wrong_tokens_are_slowed_down_and_rejections_cost_nothing_later():
    with AgentServer(plain_service(), port=0, token="the-shared-token") as server:
        codes = [http(server, "GET", "/v1/tools", token=f"guess-{i}")[0] for i in range(14)]
        assert codes[:10] == [401] * 10 and set(codes[10:]) == {429}    # ten wrong guesses, then a brake
        assert http(server, "GET", "/v1/tools", token="the-shared-token")[0] == 200      # the right token still works

    clock = [0.0]
    limiter = RateLimiter(rate=1.0, burst=2, clock=lambda: clock[0])
    assert [limiter.allow() for _ in range(50)] == [True, True] + [False] * 48
    clock[0] = 1.0                                                      # 48 refusals later, one second is enough
    assert limiter.allow() and not limiter.allow()


def test_request_ids_are_echoed_or_issued():
    with AgentServer(plain_service(), port=0) as server:
        assert http(server, "GET", "/health", headers={"X-Request-Id": "job-42.a"})[2]["X-Request-Id"] == "job-42.a"
        issued = http(server, "GET", "/health")[2]["X-Request-Id"]
        assert re.fullmatch(r"[0-9a-f]{16}", issued)
        hostile = http(server, "GET", "/nowhere", headers={"X-Request-Id": "x" * 500})
        assert hostile[0] == 404 and re.fullmatch(r"[0-9a-f]{16}", hostile[2]["X-Request-Id"])


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command to make a certificate")
def test_tls_serves_https_and_a_stalled_handshake_blocks_nobody(tmp_path):
    cert, key = str(tmp_path / "cert.pem"), str(tmp_path / "key.pem")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost", "-keyout", key, "-out", cert],
                   check=True, capture_output=True)
    trusted = ssl.create_default_context(cafile=cert)
    with AgentServer(plain_service(), port=0, token="the-shared-token", certfile=cert, keyfile=key) as server:
        assert server.url.startswith("https://") and server.tls
        stalled = socket.create_connection((server.host, server.port))  # connects, then never starts a handshake
        status, body, headers = http(server, "POST", "/v1/run", {"task": "Which do you know?"},
                                     "the-shared-token", context=trusted)
        assert status == 200 and body["answer"] == "known projects: none"
        assert headers["Strict-Transport-Security"] == "max-age=31536000"
        with pytest.raises(urllib.error.URLError):                      # an untrusting client refuses the certificate
            http(server, "GET", "/health", context=ssl.create_default_context())
        with socket.create_connection((server.host, server.port)) as raw:           # plain HTTP to the TLS port
            raw.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            raw.settimeout(5)
            try:
                assert b"200 OK" not in raw.recv(200)
            except OSError:
                pass
        old = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        old.check_hostname, old.verify_mode = False, ssl.CERT_NONE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)         # asking for old TLS is the point here
            try:
                old.minimum_version, old.maximum_version = ssl.TLSVersion.TLSv1, ssl.TLSVersion.TLSv1_1
            except ValueError:
                old = None                                              # this OpenSSL cannot even offer TLS 1.1
        if old is not None:
            with pytest.raises((ssl.SSLError, OSError)):
                with socket.create_connection((server.host, server.port)) as raw:
                    old.wrap_socket(raw).do_handshake()
        assert http(server, "GET", "/health", context=trusted)[0] == 200            # still serving after all that
        stalled.close()
    with pytest.raises(ValueError, match="keyfile was given without certfile"):
        AgentServer(plain_service(), port=0, keyfile=key)


def test_connections_beyond_the_cap_get_503_at_once():
    release = threading.Event()

    def slow(prompt):
        release.wait(20)
        return "Final Answer: done"

    with AgentServer(plain_service(slow), port=0, max_connections=2) as server:
        results = []
        workers = [threading.Thread(target=lambda: results.append(http(server, "POST", "/v1/run", {"task": "Go on"})))
                   for _ in range(2)]
        for worker in workers:
            worker.start()
        deadline = time.time() + 10
        while server.service.in_flight < 2 and time.time() < deadline:
            time.sleep(0.01)
        started = time.time()
        status, body, headers = http(server, "GET", "/health")
        assert (status, body["error"]["type"], headers["Retry-After"]) == (503, "overloaded", "1")
        assert time.time() - started < 2                                # refused, not queued behind the slow ones
        release.set()
        for worker in workers:
            worker.join(20)
        assert [r[0] for r in results] == [200, 200]
        status, body, _ = http(server, "GET", "/v1/stats")
        assert status == 200 and body["http"]["refused_connections"] == 1


def test_stop_lets_requests_in_progress_finish():
    release = threading.Event()

    def slow(prompt):
        release.wait(20)
        return "Final Answer: finished properly"

    server = AgentServer(plain_service(slow), port=0).start()
    results = []
    worker = threading.Thread(target=lambda: results.append(http(server, "POST", "/v1/run", {"task": "Go on"})))
    worker.start()
    while server.service.in_flight < 1:
        time.sleep(0.01)
    threading.Timer(0.3, release.set).start()
    server.stop(drain=10)                                               # returns once the request is done
    worker.join(10)
    assert results and results[0][0] == 200 and results[0][1]["answer"] == "finished properly"
    with pytest.raises(OSError):
        socket.create_connection((server.host, server.port), timeout=1).close()


def test_conversations_survive_a_restart(tmp_path):
    state = str(tmp_path / "state")

    def run(*turns, **options):
        service = AgentService.create(echo_context, memory_dir=state, heal=False, config=EngineConfig(learn=False),
                                      **options)
        answers = [asyncio.run(service.ask(task, session="s1")).answer for task in turns]
        return service, answers

    service, _ = run("Remember project Atlas", "Remember project Borealis")
    assert service.stats()["sessions"] == 1
    service.close()
    service, answers = run("Which do you know?")                        # a new process, in effect
    assert answers == ["known projects: Atlas, Borealis"]
    assert service.end_session("s1") and not service.end_session("s1")
    service.close()
    service, answers = run("Which do you know?")
    assert answers == ["known projects: none"]                          # ended means gone from disk too
    service.close()

    clock = [1000.0]
    store = SessionStore(str(tmp_path / "sessions.db"), ttl_s=60, clock=lambda: clock[0])
    store.save("old", [{"role": "user", "content": "hello"}])
    clock[0] += 30
    store.save("new", [{"role": "user", "content": "hi"}])
    assert len(store) == 2 and store.load("old") == [{"role": "user", "content": "hello"}]
    clock[0] += 45                                                      # "old" is now 75 s idle, "new" 45 s
    assert store.load("old") is None and store.prune() == 1 and len(store) == 1
    # RAM holds only max_sessions conversations; the rest are read back from disk when needed.
    bounded = AgentService(service.engine.__class__(echo_context, config=EngineConfig(learn=False)),
                           max_sessions=1, session_store=store)
    for name in ("a", "b"):
        asyncio.run(bounded.ask(f"Remember project {name.upper()}x", session=name))
    assert list(bounded._sessions) == ["b"] and bounded.stats()["sessions"] == 3
    assert asyncio.run(bounded.ask("Which do you know?", session="a")).answer == "known projects: Ax"
    store.close()
