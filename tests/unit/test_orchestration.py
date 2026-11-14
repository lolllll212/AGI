"""Graph of Thoughts, plan search, and asynchronous multi-agent messaging."""
import asyncio
import heapq
import json
import time

import pytest

from agi.application.reasoning.got import GraphOfThoughts, LLMOperations
from agi.application.reasoning.planner import Execution, Plan, PlanProblem, parse_plans, search_plans
from agi.application.swarm.orchestrator import Actor, Message, MessageBus, Orchestrator
from agi.domain.exceptions import LoopDetected
from agi.domain.value_objects.records import EngineResult
from agi.infrastructure.adapters.llm.scripted import ScriptedLLM

# ------------------------------------------------------ graph of thoughts --- #
DATA = [9, 3, 7, 1, 8, 2, 6, 4, 5, 0, 11, 10]


def sortedness(xs):
    return sum(a <= b for a, b in zip(xs, xs[1:])) / max(1, len(xs) - 1)


def test_split_solve_merge_is_a_graph_not_a_tree():
    got = GraphOfThoughts(generate=lambda xs, k: [xs[i::k] for i in range(k)], refine=sorted,
                          aggregate=lambda parts: list(heapq.merge(*parts)), score=sortedness,
                          solved=lambda xs: xs == sorted(DATA))
    result = got.run(DATA, [("generate", 4), ("refine", 1), ("aggregate", 2), ("aggregate",)])
    assert result.solved and result.best.content == sorted(DATA) and result.operations == 8
    assert result.best.op == "aggregate" and len(result.best.parents) == 2          # two parents: a merge
    lineage = result.lineage()
    assert lineage[0].op == "input" and len(lineage) == 12                          # 1 + 4 + 4 + 2 + 1
    assert [t.op for t in lineage].count("aggregate") == 3
    assert result.log == ["generate(4,): 4 thought(s)", "refine(1,): 4 thought(s)",
                          "aggregate(2,): 2 thought(s)", "aggregate: 1 thought(s)"]
    assert result.to_dot().count("->") == 4 + 4 + 4 + 2


def test_keep_prunes_and_refinement_never_makes_things_worse():
    got = GraphOfThoughts(generate=lambda x, k: [x + d for d in range(1, k + 1)],
                          refine=lambda x: x - 5, score=lambda x: -abs(10 - x))
    result = got.run(4, [("generate", 4), ("keep", 2), ("refine", 2)])
    assert result.best.content == 8                                    # 8 and 7 kept; "refining" to 3 was refused
    assert sum(1 for t in result.thoughts if not t.alive) == 2 + 4     # two pruned, four rejected refinements
    improving = GraphOfThoughts(refine=lambda x: x + 1, score=lambda x: -abs(10 - x))
    assert improving.run(7, [("refine", 3)]).best.content == 10


def test_operation_budget_and_plan_errors():
    endless = GraphOfThoughts(generate=lambda x, k: [x + 1] * k, score=float, max_operations=5)
    result = endless.run(0, [("generate", 3)] * 6)
    assert result.operations == 5 and any("operation limit of 5" in line for line in result.log)
    with pytest.raises(ValueError, match="no aggregate operation"):
        GraphOfThoughts(generate=lambda x, k: [x, x]).run(1, [("generate", 2), ("aggregate",)])
    with pytest.raises(ValueError, match="unknown operation"):
        GraphOfThoughts().run(1, [("meditate",)])


def test_model_backed_operations():
    llm = ScriptedLLM(['["use SQL", "use pandas", "read by hand"]', '{"score": 9}', '{"score": 4}', "not json",
                       "Use SQL, then check with pandas.", "Use SQL with GROUP BY, then check with pandas."])
    ops = LLMOperations(llm, "total sales by region")
    got = GraphOfThoughts(ops.generate, ops.aggregate, ops.refine, ops.score)
    result = got.run("start", [("generate", 3), ("keep", 2), ("aggregate",), ("refine", 1)])
    assert [t.score for t in result.thoughts[1:4]] == [0.9, 0.4, 0.0]  # an unparseable score counts as zero
    assert result.best.content == "Use SQL with GROUP BY, then check with pandas."
    assert "[1] use SQL\n\n[2] use pandas" in llm.prompts[4]
    assert LLMOperations(ScriptedLLM(["garbage"]), "t").generate("x", 2) == []


# ------------------------------------------------------------ plan search --- #
PLANS = json.dumps({"plans": [{"steps": ["use magic"], "tools": ["magic_tool"]},
                              {"steps": ["query the csv", "answer"], "tools": ["query_data"], "rationale": "direct"},
                              {"steps": ["read file", "sum by hand"], "tools": ["read_file"]}]})
REVISION = json.dumps({"plans": [{"steps": ["query the csv with group by", "answer"], "tools": ["query_data"]}]})


def test_plans_are_parsed_leniently():
    plans = parse_plans("Here you go:\n```json\n" + PLANS + "\n```")
    assert [p.steps[0] for p in plans] == ["use magic", "query the csv", "read file"] and plans[1].rationale == "direct"
    assert plans[1].render() == "1. query the csv\n2. answer"
    assert [p.steps for p in parse_plans('[["a", "b"], {"steps": []}, 7, {"steps": ["A", "b"]}]')] == [["a", "b"]]
    assert parse_plans("no json here") == [] and parse_plans('{"plans": "nope"}') == []
    assert Plan(steps=["Query  the CSV!"]) == Plan(steps=["query the csv"])        # same plan, different spelling


def test_search_refutes_executes_backtracks_and_revises():
    executed = []

    def execute(plan, earlier):
        executed.append((plan.steps[0], len(earlier)))
        ok = "group by" in plan.steps[0]
        return Execution(success=ok, score=0.3, detail="" if ok else "no such column: total")

    llm = ScriptedLLM([PLANS, REVISION, REVISION, REVISION])
    problem = PlanProblem("total by region", llm, execute, "tools...", ["query_data", "read_file"])
    found = search_plans(problem, max_executions=4)
    assert found.solved and found.plan.steps[0] == "query the csv with group by"
    assert executed == [("query the csv", 0), ("read file", 0), ("query the csv with group by", 1)]
    assert found.plans_tried == 3 and found.plans_refuted == 1         # "use magic" was never executed
    assert "use magic" not in [step for step, _ in executed]
    assert "WHAT ACTUALLY HAPPENED\nno such column: total" in llm.prompts[1]
    assert found.trace[0] == "refute: use magic - the plan relies on tools that do not exist: magic_tool"


def test_search_respects_its_execution_budget_and_survives_a_dead_model():
    count = []

    def execute(plan, earlier):
        count.append(plan)
        return Execution(success=False, detail="still wrong")

    problem = PlanProblem("t", ScriptedLLM(lambda p: PLANS if "Do not solve" in p else REVISION), execute,
                          tool_names=["query_data", "read_file"])
    found = search_plans(problem, max_executions=2)
    assert not found.solved and len(count) == 2 and found.plans_tried == 2

    def down(prompt):
        raise ConnectionError("model offline")

    dead = PlanProblem("t", down, execute)
    assert not search_plans(dead).solved and dead.llm_errors == ["ConnectionError: model offline"]
    with pytest.raises(ValueError):
        search_plans(problem, strategy="astar")


def test_a_critic_decides_which_plan_is_tried_first():
    order = []

    def execute(plan, earlier):
        order.append(plan.steps[0])
        return Execution(success=plan.steps[0] == "read file")

    problem = PlanProblem("t", ScriptedLLM([PLANS]), execute, tool_names=["query_data", "read_file"],
                          critic=lambda plan: 0.9 if "read_file" in plan.tools else 0.1)
    assert search_plans(problem).solved and order == ["read file"]


# ------------------------------------------------------------ message bus --- #
def run(coroutine):
    return asyncio.run(coroutine)


def test_direct_mail_topics_and_request_reply():
    async def scenario():
        bus = MessageBus()
        heard = []

        async def listener(message, actor):
            heard.append((actor.name, message.kind, message.content))

        async def calculator(message, actor):
            return message.content["a"] + message.content["b"]

        a = Actor("a", bus, listener, topics=["news"]).start()
        b = Actor("b", bus, listener, topics=["news"]).start()
        calc = Actor("calc", bus, calculator).start()
        client = Actor("client", bus, listener)
        assert await client.send("a", "hello") == 1
        assert await client.publish("news", "release 2.0") == 2
        assert await client.send("ghost", "anyone?") == 0 and await client.publish("empty-topic", "x") == 0
        assert await client.ask("calc", {"a": 2, "b": 40}) == 42
        await asyncio.sleep(0.01)
        for actor in (a, b, calc, client):
            await actor.stop()
        return bus, heard

    bus, heard = run(scenario())
    assert sorted(heard) == [("a", "event", "release 2.0"), ("a", "message", "hello"), ("b", "event", "release 2.0")]
    assert bus.stats.dropped == 2 and bus.stats.sent == 6 and bus.agents == []
    assert [m.kind for m in bus.log] == ["message", "event", "message", "event", "request", "reply"]


def test_a_failing_handler_reports_an_error_and_keeps_running():
    async def scenario():
        bus = MessageBus()

        async def fragile(message, actor):
            if message.content == "bad":
                raise ValueError("cannot handle that")
            return message.content.upper()

        worker = Actor("worker", bus, fragile).start()
        client = Actor("client", bus, fragile)
        with pytest.raises(RuntimeError, match="worker failed: ValueError: cannot handle that"):
            await client.ask("worker", "bad")
        assert await client.ask("worker", "still alive") == "STILL ALIVE"
        with pytest.raises(KeyError):
            await client.ask("nobody", "x")
        await worker.stop()
        return worker

    worker = run(scenario())
    assert worker.handled == 2 and worker.errors == ["ValueError: cannot handle that"]


def test_requests_time_out_and_names_are_unique():
    async def scenario():
        bus = MessageBus()

        async def silent(message, actor):
            await asyncio.sleep(5)

        slow = Actor("slow", bus, silent).start()
        client = Actor("client", bus, silent)
        with pytest.raises(TimeoutError, match="'slow' did not reply to 'client' within 0.05s"):
            await client.ask("slow", "hurry", timeout=0.05)
        with pytest.raises(ValueError, match="already on the bus"):
            Actor("slow", bus, silent)
        with pytest.raises(KeyError):
            bus.subscribe("stranger", "news")
        await slow.stop()

    run(scenario())


def test_two_agents_cannot_ping_pong_forever():
    async def scenario():
        bus = MessageBus(max_hops=6)
        stopped = []

        async def echo(message, actor):
            other = "pong" if actor.name == "ping" else "ping"
            try:
                await actor.send(other, message.content, cause=message)
            except LoopDetected as exc:
                stopped.append(str(exc))

        ping, pong = Actor("ping", bus, echo).start(), Actor("pong", bus, echo).start()
        await bus.send(Message(sender="test", recipient="ping", content="ball"))
        await asyncio.sleep(0.05)
        await ping.stop(), await pong.stop()
        return bus, stopped, ping.handled + pong.handled

    bus, stopped, handled = run(scenario())
    assert handled == 7 and len(stopped) == 1 and "exceeded 6 hops" in stopped[0]
    assert bus.stats.hop_limited == 1 and max(m.hops for m in bus.log) == 7


def test_planner_worker_critic_pipeline_over_the_bus():
    """Three agents cooperating by message: a planner fans work out, workers answer, a critic checks."""
    async def scenario():
        bus = MessageBus()

        async def worker(message, actor):
            await asyncio.sleep(0.1)
            return {"region": message.content, "total": {"North": 3090.75, "East": 3739.10}[message.content]}

        async def critic(message, actor):
            totals = message.content
            return {"approved": all(t["total"] > 0 for t in totals), "checked": len(totals)}

        async def planner(message, actor):
            parts = await asyncio.gather(*(actor.ask(f"worker-{i}", region, cause=message)
                                           for i, region in enumerate(message.content)))
            review = await actor.ask("critic", parts, cause=message)
            return {"best": max(parts, key=lambda p: p["total"])["region"], "review": review}

        actors = [Actor("planner", bus, planner).start(), Actor("critic", bus, critic).start(),
                  Actor("worker-0", bus, worker).start(), Actor("worker-1", bus, worker).start()]
        user = Actor("user", bus, worker)
        started = time.perf_counter()
        answer = await user.ask("planner", ["North", "East"])
        elapsed = time.perf_counter() - started
        for actor in actors:
            await actor.stop()
        return answer, elapsed

    answer, elapsed = run(scenario())
    assert answer == {"best": "East", "review": {"approved": True, "checked": 2}}
    assert elapsed < 0.19                                              # the two workers ran at the same time


# ------------------------------------------------------------ orchestrator --- #
def test_map_runs_in_parallel_in_order_with_failures_isolated():
    async def worker(task):
        await asyncio.sleep(0.2)
        if "explode" in task:
            raise ValueError("boom")
        if "engine" in task:
            return EngineResult(task=task, answer=None, success=False, stop_reason="budget", error="step limit")
        return task.upper()

    async def scenario():
        started = time.perf_counter()
        results = await Orchestrator(worker, concurrency=4).map(["a", "explode", "c", "engine"])
        return results, time.perf_counter() - started

    results, elapsed = run(scenario())
    assert elapsed < 0.5 and [r.task for r in results] == ["a", "explode", "c", "engine"]
    assert [(r.ok, r.answer, r.error) for r in results] == [
        (True, "A", None), (False, None, "ValueError: boom"), (True, "C", None), (False, None, "step limit")]


def test_concurrency_is_bounded_and_slow_parts_time_out():
    running = {"now": 0, "peak": 0}

    async def worker(task):
        running["now"] += 1
        running["peak"] = max(running["peak"], running["now"])
        await asyncio.sleep(1.0 if task == "slow" else 0.05)
        running["now"] -= 1
        return task

    results = run(Orchestrator(worker, concurrency=2, timeout=0.3).map(["a", "b", "slow", "c", "d"]))
    assert running["peak"] == 2
    assert [r.ok for r in results] == [True, True, False, True, True] and results[2].error == "timed out after 0.3s"


def test_solve_decomposes_fans_out_and_synthesises():
    llm = ScriptedLLM(['{"subtasks": ["Total for North in sales.csv", "Total for East in sales.csv", '
                       '"Total for North in sales.csv"]}', "East (3,739.10) is ahead of North (3,090.75)."])

    async def worker(task):
        return "3,090.75" if "North" in task else "3,739.10"

    result = run(Orchestrator(worker, llm).solve("Compare North and East totals in sales.csv"))
    assert result.success and result.decomposed and result.answer == "East (3,739.10) is ahead of North (3,090.75)."
    assert [p.task for p in result.parts] == ["Total for North in sales.csv", "Total for East in sales.csv"]
    assert "[1] Total for North in sales.csv\n    result: 3,090.75" in llm.prompts[1]


def test_solve_degrades_gracefully():
    async def worker(task):
        if "bad" in task:
            raise RuntimeError("no data")
        return f"done {task}"

    # No model: no decomposition, results are listed as they are.
    plain = run(Orchestrator(worker).solve("one thing"))
    assert plain.answer == "done one thing" and plain.success and not plain.decomposed
    listed = run(Orchestrator(worker).solve("x", subtasks=["good", "bad"]))
    assert not listed.success and listed.answer == "- good: done good\n- bad: FAILED (RuntimeError: no data)"
    # A model that cannot split the task: the task runs whole.
    whole = run(Orchestrator(worker, ScriptedLLM(["I cannot split this."])).solve("indivisible"))
    assert whole.answer == "done indivisible" and len(whole.parts) == 1
    # A failed part is passed to the synthesiser as a failure, and the result is not a success.
    llm = ScriptedLLM(['{"subtasks": ["good", "bad"]}', "Only the first part is known."])
    partial = run(Orchestrator(worker, llm).solve("both"))
    assert not partial.success and "FAILED: RuntimeError: no data" in llm.prompts[1]
