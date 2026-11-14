"""0.6: a router that generalises across kinds of task, grounding of names, and lesson credit by use."""
import json
import random

import pytest

from agi.application.cortex.engine import EngineConfig
from agi.application.cortex.react import GroundingVerifier, Objection, ReActAgent, ReActStep
from agi.application.cortex.router import RouteLearner, RouteModel, Router, RouterConfig, task_words
from agi.application.memory.credit import lesson_verdict
from agi.application.memory.engine import MemoryEngine
from agi.domain.entities.memory import Lesson
from agi.domain.entities.tool import ToolRegistry
from agi.domain.services.claims import Evidence, names_in, unsupported_names
from agi.infrastructure.adapters.execution.builtin import calculator_tool, standard_tools
from agi.infrastructure.adapters.llm.scripted import ScriptedLLM
from agi.infrastructure.di.container import build_engine


def act(tool, args, thought="next"):
    return f"Thought: {thought}\nAction: {tool}\nAction Input: {json.dumps(args)}"


def final(answer, thought="done"):
    return f"Thought: {thought}\nFinal Answer: {answer}"


# ------------------------------------------------------------------ router --- #
VERBS = ["summarise", "list", "count", "reconcile", "tabulate", "describe", "extract"]
NOUNS = ["ledger", "invoices", "orders", "notes", "tickets", "inventory", "payroll"]
TAILS = ["", " and then email it", " why does it differ?", " step by step", " for the design review"]


def stream(seed, n, rule=True):
    """Tasks whose difficulty depends on their wording ("reconcile", "ledger"), not on their kind."""
    rng = random.Random(seed)
    for _ in range(n):
        task = f"{rng.choice(VERBS)} the {rng.choice(NOUNS)} for {rng.choice(['March', 'April', 'May'])}{rng.choice(TAILS)}"
        hard = "reconcile" in task or "ledger" in task
        failed = rng.random() < ((0.9 if hard else 0.1) if rule else 0.4)
        yield task, hard, failed


def trained(config, seed=3, n=300, rule=True):
    router = Router(ToolRegistry([]), config=config)
    for task, _, failed in stream(seed, n, rule):
        router.learn(task, failed)
    return router


def accuracy(router, seed=99, n=300):
    return sum((router.route(task).mode == "deliberate") == hard for task, hard, _ in stream(seed, n)) / n


def test_task_words_are_distinct_content_words_in_order():
    assert task_words("Please reconcile the Q3 ledger with the bank export, then reconcile again") == [
        "reconcile", "ledger", "bank", "export", "again"]
    assert len(task_words(" ".join(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(200)))) == 40


def test_the_model_learns_which_words_go_with_failure_and_scores_its_own_forecasts():
    model = RouteModel()
    for task, _, failed in stream(1, 300):
        model.update(RouteModel.inputs(["length:short"], task), failed)
    risky = model.predict(RouteModel.inputs(["length:short"], "reconcile the ledger for June"))
    safe = model.predict(RouteModel.inputs(["length:short"], "count the tickets for June"))
    assert risky > 0.85 and safe < 0.2
    assert model.skill > 0.2 and model.seen == 300
    leaned_on = dict(model.strongest(4))
    assert leaned_on["word:reconcile"] > 1 and leaned_on["word:ledger"] > 1
    restored = RouteModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.predict(RouteModel.inputs([], "reconcile it")) == model.predict(RouteModel.inputs([], "reconcile it"))


def test_a_model_fed_noise_does_not_beat_the_average_and_is_never_trusted():
    for seed in (1, 2, 3):
        router = trained(RouterConfig(), seed=seed, n=400, rule=False)
        report = router.learner.report()
        assert report["outcomes"] == 400 and report["skill"] < 0.1 and report["trusted"] is False
    # ...so routing is exactly what the fixed rules say.
    rules = Router(ToolRegistry([]), config=RouterConfig(learn_routes=False))
    for task, _, _ in stream(7, 60):
        assert router.route(task).mode == rules.route(task).mode


def test_the_model_is_not_consulted_before_it_has_seen_enough():
    router = trained(RouterConfig(), n=39)
    assert router.learner.trusted is False
    late = trained(RouterConfig(generalise_after=1000), n=300)
    assert late.learner.model.skill > 0.3 and late.learner.trusted is False


def test_wording_moves_a_task_that_its_kind_alone_would_not():
    router = trained(RouterConfig(recheck_every=0))
    assert router.learner.trusted
    hard = router.route("reconcile the stock ledger for June")
    assert hard.mode == "deliberate" and "generalised" in hard.signals
    assert "experience with similar wording" in hard.reason and "300 outcomes" in hard.reason
    # The rules alone call this one hard ("step by step"); the model has seen such tasks go fine.
    easy = router.route("list the orders for May step by step")
    assert easy.mode == "react" and "generalised" in easy.signals and "without planning" in easy.reason
    # Per-kind counting cannot tell the two apart from others of their kind, and switching it off restores that.
    counts_only = trained(RouterConfig(generalise=False, recheck_every=0))
    assert counts_only.route("reconcile the stock ledger for June").mode == "react"
    assert counts_only.route("list the orders for May step by step").mode == "deliberate"


def test_routing_gets_more_tasks_right_on_a_stream_where_wording_matters():
    rules = accuracy(Router(ToolRegistry([]), config=RouterConfig(learn_routes=False)))
    counts = accuracy(trained(RouterConfig(generalise=False, recheck_every=0)))
    both = accuracy(trained(RouterConfig(recheck_every=0)))
    assert both >= counts + 0.1 and both >= rules + 0.1 and both > 0.75


def test_a_generalised_belief_is_rechecked_now_and_then():
    router = trained(RouterConfig(recheck_every=4))
    modes = [router.route("reconcile the stock ledger for June") for _ in range(8)]
    assert [route.mode for route in modes] == ["deliberate"] * 3 + ["react"] + ["deliberate"] * 3 + ["react"]
    assert "rechecking" in modes[3].signals and "trying the cheaper path again" in modes[3].reason
    # The signal words the router adds never become part of a task's kind.
    assert RouteLearner.features(["analysis", "generalised", "rechecking", "experience"], 5) == ["analysis", "length:short"]


def test_the_model_is_saved_with_the_counts_and_old_or_damaged_files_still_load(tmp_path):
    path = str(tmp_path / "routing.json")
    learner = RouteLearner(path)
    for task, _, failed in stream(3, 120):
        learner.update(RouteLearner.features([], len(task.split())), failed, task)
    again = RouteLearner(path)
    assert again.profiles == learner.profiles and again.model.seen == 120
    assert again.forecast(["length:short"], "reconcile the ledger") == learner.forecast(["length:short"], "reconcile the ledger")

    old = tmp_path / "old.json"                               # written by 0.5: counts only
    old.write_text(json.dumps({"profiles": {"length:short": [9, 8]}, "observations": 9}))
    loaded = RouteLearner(str(old))
    assert loaded.profiles == {"length:short": [9, 8]} and loaded.model.seen == 0 and not loaded.trusted

    data = json.loads((tmp_path / "routing.json").read_text())
    data["model"]["weights"]["word:ledger"] = "not a number"
    (tmp_path / "broken.json").write_text(json.dumps(data))
    broken = RouteLearner(str(tmp_path / "broken.json"))
    assert broken.profiles == learner.profiles and broken.model.seen == 0     # counts kept, model started over


def test_the_model_stays_bounded():
    model = RouteModel(max_features=50)
    for i in range(400):
        model.update(["bias", f"word:w{i}", "word:common"], failed=i % 2 == 0)
    assert len(model.weights) <= 50 and set(model.weights) == set(model.squares)


# ----------------------------------------------------------------- names --- #
@pytest.mark.parametrize("text, expected", [
    ("The North total is 3,090.75.", []),
    ("East (3,739.10) sold more than North (3,090.75).", []),
    ("Python's sorted() is stable. The function returns 9.0 on Monday.", []),
    ("It's the customer's 3rd order, isn't it? The file doesn't exist.", []),
    ("However, the average is 2.5. Therefore the Total is higher. Finally, use SQL.", []),
    ("I saved the summary to report.md and the chart to out/chart.png.", ["report.md", "out/chart.png"]),
    ("The top customer is Acme Corp, contact priya@acme.example.", ["Acme Corp", "priya@acme.example"]),
    ("Order INV-1042 belongs to user_42; status is 'shipped'.", ["INV-1042", "user_42", "shipped"]),
    ("Ask Rahul Sharma in Pune.", ["Rahul Sharma", "Pune"]),
    ("- Manager: Meera Nair\n- Region: East", ["Meera Nair"]),
    ("The best seller is the Widget Pro, made by TechNova.", ["Widget Pro", "TechNova"]),
    ('He said "I will look into it tomorrow and get back to you by the end of the week".', []),   # a sentence, not a value
])
def test_names_are_found_narrowly(text, expected):
    assert names_in(text) == expected


def test_evidence_supports_a_name_in_any_case_and_with_a_plural_or_possessive_ending():
    evidence = Evidence("Which customer spent the most?", "[('acme corp', 9120.5), ('Widgets', 3)]", '{"file": "orders.csv"}')
    assert evidence.supports("Acme Corp") and evidence.supports("Acme Corp's") and evidence.supports("Widget")
    assert evidence.supports("orders.csv") and not evidence.supports("order.csv") and not evidence.supports("Globex")
    assert unsupported_names("Acme Corp spent the most, ahead of Globex, per orders.csv and ledger.csv.", evidence) == [
        "Globex", "ledger.csv"]


GROUNDED = [   # (task, observation, answer): answers a careful agent might give; none should be questioned
    ("Which region sold the most in sales.csv?", "[('East', 3739.1)]", "East sold the most, with 3,739.10 in sales.csv."),
    ("Who manages the Pune office?", "name: Meera Nair, office: Pune", "The Pune office is managed by Meera Nair."),
    ("Summarise notes.txt", "Kickoff with Acme on Monday. Owner: Rahul.", "The notes cover the Acme kickoff on Monday; Rahul owns it."),
    ("How many orders are overdue?", "[(7,)]", "Seven orders are overdue. I used SQL to count them."),
    ("What is the status of INV-1042?", "{'id': 'INV-1042', 'status': 'shipped'}", "INV-1042 has status 'shipped'."),
    ("Write the result to out.json", "wrote 120 bytes to out.json", "Done. The result is in out.json."),
    ("Which product has the best margin?", "Widget Pro | 0.42", "The Widget Pro has the best margin, 42%."),
    ("Compare North and South", "North 3090.75\nSouth 2491.15", "North (3,090.75) is ahead of South (2,491.15). Overall, North leads."),
    ("When is the audit?", "The audit is on Friday 14 March.", "The audit is on Friday, 14 March."),
    ("Find the email of the supplier", "contact: sales@globex.example", "The supplier's email is sales@globex.example."),
    ("What does parse_row do?", "def parse_row(line): return line.split(',')", "The function parse_row splits a line on commas. In Python that returns a list."),
    ("List the files", "sales.csv\nnotes.txt", "There are two files: sales.csv and notes.txt."),
]


def test_plausible_grounded_answers_raise_no_question():
    verifier = GroundingVerifier()
    for task, observation, answer in GROUNDED:
        steps = [ReActStep("look", "read", {}, observation, True)]
        assert verifier(task, answer, steps) is None, answer


def test_a_made_up_name_is_a_soft_objection_and_a_made_up_figure_a_hard_one():
    verifier = GroundingVerifier()
    steps = [ReActStep("query", "query_data", {"file": "sales.csv"}, "[('East', 3739.1)]", True),
             ReActStep("query", "query_data", {"file": "ledger.csv"}, "FileNotFound: ledger.csv", False)]
    task = "Which region sold the most?"
    assert verifier(task, "East sold the most (3,739.10), per sales.csv.", steps) is None
    soft = verifier(task, "East sold the most (3,739.10), managed by Anita Desai.", steps)
    assert isinstance(soft, Objection) and soft.soft and soft.names == ("Anita Desai",) and soft.figures == ()
    assert "'Anita Desai'" in soft and "if you are sure of it, repeat your answer" in soft
    # A failed call vouches for nothing: the file it could not find is still unknown.
    assert verifier(task, "East leads, as ledger.csv confirms.", steps).names == ("ledger.csv",)
    hard = verifier(task, "East sold 4,100.00, managed by Anita Desai.", steps)
    assert not hard.soft and hard.figures == ("4100.00",) and hard.names == ("Anita Desai",)
    assert GroundingVerifier(check_names=False)(task, "East leads, says Anita Desai.", steps) is None
    # Names and figures from earlier turns or memory count as seen.
    assert verifier(task, "Anita Desai's target was 48,250.", steps, evidence="Earlier: Anita Desai set a target of 48,250") is None


def test_a_questioned_name_is_corrected_or_else_accepted_and_reported():
    tools = ToolRegistry([calculator_tool()])
    corrected = ReActAgent(ScriptedLLM([act("calculator", {"expression": "17 * 23"}),
                                        final("It is 391, as Dr Mehta predicted."),
                                        final("It is 391.")]), tools).run("What is 17 * 23?")
    assert corrected.success and corrected.answer == "It is 391." and corrected.unverified == [] and corrected.challenges == 1
    assert "[verifier] The name(s) 'Dr Mehta'" in corrected.steps[1].observation

    insisted = ReActAgent(ScriptedLLM([act("calculator", {"expression": "17 * 23"}),
                                       final("It is 391, as Dr Mehta predicted.")] + [final("It is 391, as Dr Mehta predicted.")]),
                          tools).run("What is 17 * 23?")
    assert insisted.success and insisted.grounded is True and insisted.unverified == ["Dr Mehta"]
    # A figure that stays unsupported is still a failure, exactly as before.
    wrong = ReActAgent(ScriptedLLM([final("It is 400.")] * 2), tools).run("What is 17 * 23?")
    assert wrong.success is False and wrong.grounded is False


def test_what_was_said_earlier_in_a_conversation_counts_as_seen():
    agent = ReActAgent(ScriptedLLM([final("Priya's budget is 48,250.")]), ToolRegistry([calculator_tool()]))
    run = agent.run("What was the budget again?", context="EARLIER: Priya said the budget is 48,250.")
    assert run.success and run.challenges == 0 and run.unverified == []
    # Without that context the same answer is questioned (and, the figure being unsupported, fails).
    cold = ReActAgent(ScriptedLLM([final("Priya's budget is 48,250.")] * 2), ToolRegistry([calculator_tool()]))
    assert cold.run("What was the budget again?").success is False


def test_verifiers_written_for_the_older_three_argument_form_still_work():
    calls = []

    def strict(task, answer, steps):
        calls.append(answer)
        return None if "checked" in answer else "Say that you checked."

    run = ReActAgent(ScriptedLLM([final("42"), final("42, checked")]), ToolRegistry(), verifier=strict).run("x")
    assert run.success and run.answer == "42, checked" and calls == ["42", "42, checked"] and run.unverified == []


def test_the_engine_reports_an_unverified_name_and_does_not_cache_that_run_as_a_procedure(tmp_path):
    (tmp_path / "sales.csv").write_text("region,amount\nNorth,100\nSouth,250\n")
    query = act("query_data", {"file": "sales.csv", "sql": "select sum(amount) from data where region = 'South'"})
    task = "What is the total amount for South in sales.csv?"

    def run_with(answer, directory):
        engine = build_engine(ScriptedLLM([query, final(answer), final(answer)]), workspace=str(tmp_path),
                              memory_dir=str(directory), heal=False)
        result = engine.run(task)
        stored = engine.memory.stats()["procedural"]["trajectories"]
        engine.memory.close()
        return result, stored

    clean, stored = run_with("South sold 250.", tmp_path / "a")
    assert clean.success and clean.flags == [] and stored == 1
    doubtful, stored = run_with("South sold 250, as Vikram Rao expected.", tmp_path / "b")
    assert doubtful.success and doubtful.grounded and doubtful.flags == ["unverified_name:Vikram Rao"] and stored == 0
    off = build_engine(ScriptedLLM([query, final("South sold 250, as Vikram Rao expected.")]), workspace=str(tmp_path),
                       memory=None, heal=False, config=EngineConfig(verify_names=False)).run(task)
    assert off.success and off.flags == [] and off.llm_calls == 2


# ------------------------------------------------------------ lesson credit --- #
def step(action, ok=True, observation="ok", args=None):
    return ReActStep("", action, args or {}, observation, ok)


PROCEDURE = Lesson(kind="procedure", trigger="total for a region", advice="Known-good approach: [\"query_data\"]",
                   payload={"solution": ["query_data"]})
FIX = Lesson(kind="fix", trigger="total for a region (FileNotFound)", advice="'query_data(sale.csv)' failed; 'query_data(sales.csv)' worked",
             payload={"error_type": "FileNotFound", "signature": "no file named '…' in the workspace",
                      "failed": ['query_data({"file": "sale.csv"})'], "worked": 'query_data({"file": "sales.csv"})'})
AVOID = Lesson(kind="avoid", trigger="dates (ValueError)", advice="'python(...)' fails with ValueError",
               payload={"error_type": "ValueError", "signature": "time data '…' does not match format '…'",
                        "failed": ["python(strptime)"], "worked": None})
FACT = Lesson(kind="fact", trigger="currency questions", advice="Convert rupees with the exchange_rate tool before comparing")


@pytest.mark.parametrize("lesson, steps, success, answer, expected", [
    (PROCEDURE, [step("query_data")], True, None, True),                     # followed, and it worked
    (PROCEDURE, [step("query_data")], False, None, False),                   # followed, and it did not
    (PROCEDURE, [step("calculator")], True, None, None),                     # not followed: says nothing about it
    (PROCEDURE, [step("query_data", ok=False, observation="SQLError: bad")], True, None, None),
    (FIX, [step("query_data")], True, None, True),                           # went there, did not fall in
    (FIX, [step("query_data", ok=False, observation="FileNotFound: no file named 'sale.csv' in the workspace"),
           step("query_data")], True, None, False),                          # warned, and it happened again
    (FIX, [step("calculator")], True, None, None),                           # never came near it
    (FIX, [step("query_data")], False, None, None),                          # failed for some other reason
    (AVOID, [step("python", ok=False, observation="ValueError: time data '3/4' does not match format '%Y'")], True, None, False),
    (AVOID, [step("python")], True, None, True),
    (AVOID, [step("read_file")], True, None, None),
    (FACT, [step("exchange_rate", args={"currency": "rupees"})], True, "Converted the rupees first.", True),
    (FACT, [step("calculator")], True, "It is 42.", None),
    (FACT, [step("exchange_rate", args={"currency": "rupees"})], False, "rupees converted", False),
])
def test_a_lesson_is_judged_by_what_the_run_did(lesson, steps, success, answer, expected):
    assert lesson_verdict(lesson, steps, success, answer) is expected


def test_a_lesson_that_was_only_present_earns_nothing(tmp_path):
    (tmp_path / "sales.csv").write_text("region,amount\nNorth,100\nSouth,250\n")
    with MemoryEngine() as memory:
        relevant = memory.learn(Lesson(kind="procedure", trigger="total sales amount for a region in sales.csv",
                                       advice='Known-good approach: ["query_data"]', tags=("sales",),
                                       payload={"solution": ["query_data"]}))
        bystander = memory.learn(Lesson(kind="procedure", trigger="total sales amount for a region, from the web",
                                        advice='Known-good approach: ["http_get"]', tags=("sales",),
                                        payload={"solution": ["http_get"]}))
        llm = ScriptedLLM([act("query_data", {"file": "sales.csv", "sql": "select sum(amount) from data where region = 'South'"}),
                           final("South sold 250.")])
        run = ReActAgent(llm, standard_tools(tmp_path), memory=memory, tags=("sales",)).run(
            "What is the total sales amount for the South region in sales.csv?")
        assert run.success and {lesson.id for lesson in run.recalled} >= {relevant.id, bystander.id}
        assert run.credited[relevant.id] is True and run.credited[bystander.id] is None
        assert (memory.get(relevant.id).helped, memory.get(relevant.id).hurt) == (1, 0)
        assert (memory.get(bystander.id).helped, memory.get(bystander.id).hurt) == (0, 0)   # before 0.6 this was (1, 0)
