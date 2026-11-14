"""Baselines, regression and drift detection, and A/B-tested prompt and policy optimisation."""
import json

import pytest

from agi.core.engine import CognitiveEngine, EngineConfig
from agi.core.llm import ScriptedLLM
from agi.core.react import SYSTEM
from agi.evaluation import (SUITE, DriftMonitor, Evaluation, Evaluator, Mutation, Optimizer, TaskScore, Tolerance,
                            add_instruction, fingerprint, load_config, mcnemar, propose_instructions, reference_llm,
                            save_config, set_field, with_field)
from agi.tools import standard_tools

TASKS = {task.name: task for task in SUITE}


def engine_agent(config=None, model=reference_llm):
    def make(workspace, task):
        return CognitiveEngine(model(task), standard_tools(workspace=workspace), config=config or EngineConfig())
    return make


def lazy_on(*names):
    """A model that gives up on the named tasks and solves the rest properly."""
    def model(task):
        return (lambda prompt: "Final Answer: I am not sure.") if task.name in names else reference_llm(task)
    return model


# -------------------------------------------------------------- evaluator --- #
def test_an_evaluation_records_outcome_and_cost_per_task():
    evaluation = Evaluator(engine_agent()).run("v1", config=EngineConfig())
    assert evaluation.tasks == [t.name for t in SUITE] and evaluation.success_rate == 1.0
    metrics = evaluation.metrics()
    assert metrics["grounded_rate"] == 1.0 and metrics["tool_errors"] == 0
    assert metrics["llm_calls"] == 19 / 8 and metrics["tool_calls"] == 11 / 8 and evaluation.cost() == 3.75
    first = evaluation.scores[0]
    assert (first.task, first.mode, first.stop_reason, first.skill != "") == ("compound_interest", "react", "answer", True)
    assert evaluation.fingerprint == fingerprint(EngineConfig()) and len(evaluation.fingerprint) == 12
    assert "8/8 passed (100%)" in evaluation.summary()


def test_a_crashing_agent_fails_its_task_not_the_evaluation():
    def make(workspace, task):
        if task.name == "top_region":
            raise RuntimeError("could not build the agent")
        return engine_agent()(workspace, task)

    evaluation = Evaluator(make, tasks=[TASKS["top_region"], TASKS["days_between"]], repeats=2).run()
    assert [s.success for s in evaluation.scores] == [False, False, True, True]
    assert evaluation.scores[0].stop_reason == "crash" and "could not build the agent" in evaluation.scores[0].error
    assert evaluation.pass_rates() == {"top_region": 0.0, "days_between": 1.0}
    with pytest.raises(ValueError, match="no make_agent"):
        Evaluator().run()


def test_baselines_are_stored_and_regressions_are_named(tmp_path):
    evaluator = Evaluator(engine_agent(), store=tmp_path / "baselines")
    good = evaluator.run("v1", config=EngineConfig())
    path = evaluator.save_baseline(good, "main")
    assert path.name == "main.json" and evaluator.load_baseline("main").scores == good.scores
    assert evaluator.load_baseline("never-saved") is None

    worse = evaluator.run("v2", make_agent=engine_agent(model=lazy_on("top_region", "log_errors")),
                          config=EngineConfig(max_steps=5))
    comparison = evaluator.compare(worse, evaluator.load_baseline("main"))
    assert comparison.regressed and comparison.newly_failing == ["top_region", "log_errors"]
    assert comparison.config_changed and [c.metric for c in comparison.regressions][0] == "success_rate"
    report = comparison.report()
    assert report.startswith("v2 vs baseline v1: REGRESSION") and "now failing: top_region, log_errors" in report
    assert "success_rate: 1 -> 0.75  << regression" in report

    same = evaluator.compare(evaluator.run("again", config=EngineConfig()), good)
    assert not same.regressed and not same.config_changed and same.report() == "again vs baseline v1: ok"
    for bad in ("", "../escape", ".hidden", "a/b"):
        with pytest.raises(ValueError):
            evaluator.save_baseline(good, bad)
    with pytest.raises(ValueError, match="no store"):
        Evaluator(engine_agent()).save_baseline(good)


def score(task, success=True, llm=2, tools=1, errors=0, seconds=0.1):
    return TaskScore(task=task, success=success, grounded=True, llm_calls=llm, tool_calls=tools,
                     tool_errors=errors, seconds=seconds)


def test_cost_regressions_need_to_be_both_relatively_and_absolutely_large():
    evaluator = Evaluator()
    base = Evaluation(label="base", scores=[score("a"), score("b"), score("c"), score("d")])
    slightly = Evaluation(label="s", scores=[score("a", llm=3), score("b"), score("c"), score("d")])   # +12.5%
    assert not evaluator.compare(slightly, base).regressed
    costly = Evaluation(label="c", scores=[score(t, llm=4) for t in "abcd"])                           # +100%
    assert [c.metric for c in evaluator.compare(costly, base).regressions] == ["llm_calls"]
    assert not evaluator.compare(costly, base, Tolerance(llm_calls=1.5)).regressed
    # Timings are ignored unless a tolerance for them is set.
    slow = Evaluation(label="slow", scores=[score(t, seconds=5.0) for t in "abcd"])
    assert not evaluator.compare(slow, base).regressed
    assert [c.metric for c in evaluator.compare(slow, base, Tolerance(seconds=1.0)).regressions] == ["seconds"]
    # A dropped task is a regression; a new task or a fixed one is not.
    assert evaluator.compare(Evaluation(scores=[score("a"), score("b"), score("c")]), base).missing == ["d"]
    grown = evaluator.compare(Evaluation(scores=base.scores + [score("e")]), base)
    assert grown.added == ["e"] and not grown.regressed
    flaky_before = Evaluation(scores=[score("a"), score("a", success=False)])
    flaky_after = Evaluation(scores=[score("a"), score("a")])
    assert evaluator.compare(flaky_after, flaky_before).newly_passing == ["a"]
    assert evaluator.compare(flaky_before, flaky_after).newly_failing == ["a"]
    errors = Evaluation(scores=[score(t, errors=1) for t in "abcd"])
    assert [c.metric for c in evaluator.compare(errors, base).regressions] == ["tool_errors"]


def test_check_creates_keeps_and_updates_the_baseline(tmp_path):
    tasks = [TASKS["top_region"], TASKS["days_between"]]
    evaluator = Evaluator(engine_agent(), tasks=tasks, store=tmp_path)
    first, comparison = evaluator.check("nightly", label="day 1")
    assert comparison is None and evaluator.load_baseline("nightly").label == "day 1"
    _, comparison = evaluator.check("nightly", label="day 2", update=True)
    assert not comparison.regressed and evaluator.load_baseline("nightly").label == "day 2"
    _, comparison = evaluator.check("nightly", label="day 3", update=True,
                                    make_agent=engine_agent(model=lazy_on("top_region")))
    assert comparison.regressed and comparison.newly_failing == ["top_region"]
    assert evaluator.load_baseline("nightly").label == "day 2"         # a regressed run never becomes the baseline
    history = evaluator.history()
    assert [h["label"] for h in history] == ["day 1", "day 2"] and history[0]["success_rate"] == 1.0
    assert json.loads((tmp_path / "nightly.json").read_text())["label"] == "day 2"


# ------------------------------------------------------------------ drift --- #
def results(n, fail_every=10, llm_calls=2, errors=0):
    return [{"success": (i % fail_every) != 0, "llm_calls": llm_calls, "tool_errors": errors} for i in range(1, n + 1)]


def test_a_stable_agent_raises_no_alarm():
    monitor = DriftMonitor(reference=50, window=30)
    assert not monitor.ready
    for row in results(300):
        assert monitor.observe(row) == []
    status = monitor.status()
    assert status["ready"] and status["seen"] == 300 and status["alarms"] == 0
    assert status["reference_success"] == 0.9 and status["recent_success"] == 0.9


def test_a_drop_in_success_is_detected_once_then_the_alarm_cools_down():
    monitor = DriftMonitor(reference=50, window=30)
    for row in results(100):
        monitor.observe(row)
    raised_at = []
    for i, row in enumerate(results(60, fail_every=2), 1):             # success falls from 90% to 50%
        if monitor.observe(row):
            raised_at.append(i)
    alarm = monitor.alarms[0]
    assert alarm.metric == "success_rate" and alarm.direction == "worse" and alarm.reference == 0.9
    assert alarm.recent < 0.65 and alarm.z <= -3 and alarm.samples == 30
    assert raised_at[0] <= 25                                          # noticed before the window is all bad
    assert all(b - a >= 30 for a, b in zip(raised_at, raised_at[1:]))  # not one alarm per result


def test_cost_drift_improvement_and_trivial_changes():
    monitor = DriftMonitor(reference=40, window=20, metrics=("llm_calls", "tool_errors"))
    for row in results(60):
        monitor.observe(row)
    for row in results(20, llm_calls=5):
        monitor.observe(row)
    assert [(a.metric, a.direction) for a in monitor.alarms] == [("llm_calls", "worse")]
    assert monitor.alarms[0].reference == 2 and monitor.alarms[0].recent > 2.5

    better = DriftMonitor(reference=40, window=20)
    for row in results(40, fail_every=2) + results(20, fail_every=1000):
        better.observe(row)
    assert better.alarms and better.alarms[0].direction == "better"

    # 90% -> 85% is within the tolerance however many results there are.
    trivial = DriftMonitor(reference=2000, window=2000, z_threshold=2.0, min_drop=0.1)
    for row in results(2000) + [{"success": i % 20 not in (0, 1, 2)} for i in range(2000)]:
        trivial.observe(row)
    assert trivial.ready and trivial.alarms == []


def test_a_stored_baseline_can_be_the_reference():
    baseline = Evaluator(engine_agent(), tasks=[TASKS["top_region"], TASKS["days_between"]], repeats=5).run()
    monitor = DriftMonitor(window=10, metrics=()).set_reference(baseline.scores)
    assert monitor.reference_size == 10
    alarms = []
    for _ in range(10):
        alarms += monitor.observe(TaskScore(task="live", success=False))
    assert alarms and alarms[0].metric == "success_rate" and alarms[0].reference == 1.0 and alarms[0].recent <= 0.5
    assert monitor.observe(True) == [] and monitor.seen == 11          # plain booleans are accepted too
    with pytest.raises(ValueError):
        DriftMonitor(window=2)


def test_sign_test_p_values():
    assert mcnemar(0, 0) == 1.0 and mcnemar(3, 3) == 1.0
    assert mcnemar(5, 0) == 0.0625 and mcnemar(0, 5) == 0.0625
    assert mcnemar(10, 0) == pytest.approx(0.001953125) and mcnemar(8, 1) == pytest.approx(0.0390625)


# -------------------------------------------------------------- optimizer --- #
def sloppy(task):
    """A stand-in model whose behaviour depends on its instructions, so that prompt changes have real effects.

    It invents a figure for the unanswerable task unless told never to estimate; told to refuse when unsure,
    it refuses everything that needs a date; told to answer immediately, it skips its tools.
    """
    reference = reference_llm(task)

    def reply(prompt):
        instructions = prompt.split("\n\nTask: ")[0].lower()
        if "answer immediately" in instructions:
            return "Thought: no tools needed.\nFinal Answer: 42"
        if "refuse when unsure" in instructions and task.name in ("unanswerable", "days_between"):
            return "Thought: unsure.\nFinal Answer: That cannot be determined; the information is missing."
        if task.name == "unanswerable" and "never estimate" not in instructions:
            return "Thought: I will estimate.\nFinal Answer: The West region sold about 5,000.00 in total."
        return reference(prompt)
    return reply


def build(config, workspace, task):
    return CognitiveEngine(sloppy(task), standard_tools(workspace=workspace), config=config)


START = EngineConfig(escalate=False)
NEVER_ESTIMATE = add_instruction("Never estimate a figure the data does not contain; say it is missing.")
TUNE = [TASKS["unanswerable"], TASKS["top_region"], TASKS["compound_interest"]]


def test_mutations_change_one_validated_thing():
    changed = set_field("max_steps", 6).apply(START)
    assert changed.max_steps == 6 and START.max_steps == 12 and changed.escalate is False
    assert with_field(START, "router.reflex_confidence", 0.9).router.reflex_confidence == 0.9
    assert set_field("max_steps", 12).apply(START) is None             # no change
    assert set_field("max_steps", 0).apply(START) is None              # fails validation
    assert set_field("no_such_field", 1).apply(START) is None
    prompted = NEVER_ESTIMATE.apply(START)
    assert prompted.system_prompt.startswith(SYSTEM) and prompted.system_prompt.endswith("say it is missing.")
    assert "{tools}" in prompted.system_prompt and NEVER_ESTIMATE.apply(prompted) is None
    assert Mutation(name="empty", kind="instruct", value="  ").apply(START) is None
    with pytest.raises(KeyError):
        with_field(START, "router.nope", 1)


def test_a_change_is_adopted_only_when_measurement_says_so():
    mutations = [add_instruction("Answer immediately.", name="hasty"), set_field("max_steps", 1),
                 set_field("router.small_talk", False), NEVER_ESTIMATE]
    result = Optimizer(build, TUNE, mutations).optimize(START)
    assert result.changed and result.baseline.success_rate == pytest.approx(2 / 3) and result.final.success_rate == 1.0
    assert result.lineage == [NEVER_ESTIMATE.name] and "Never estimate" in result.config.system_prompt
    assert START.system_prompt is None                                 # the starting configuration is untouched
    by_name = {t.mutation: t for t in result.trials if t.generation == 1}
    assert by_name["hasty"].losses == ["top_region", "compound_interest"] and not by_name["hasty"].accepted
    assert by_name["hasty"].reason == "breaks top_region, compound_interest"
    assert by_name["max_steps=1"].reason.startswith("breaks") and by_name["router.small_talk=False"].reason == "no measurable improvement"
    winner = by_name[NEVER_ESTIMATE.name]
    assert winner.accepted and winner.wins == ["unanswerable"] and winner.p_value == 1.0   # one task: not significant
    assert winner.reason == "fixes unanswerable, breaks nothing"
    # The rejected ideas are measured again on top of the new champion (they might combine well); none does.
    assert [t.accepted for t in result.trials if t.generation == 2] == [False, False, False]
    report = result.report()
    assert report.startswith("success 67% -> 100%") and "ACCEPT  instruct: Never estimate" in report


def test_a_significance_bar_blocks_a_single_task_win():
    result = Optimizer(build, TUNE, [NEVER_ESTIMATE], alpha=0.05).optimize(START)
    assert not result.changed and result.config == START and result.final.success_rate == pytest.approx(2 / 3)
    assert "p=1.000 is above alpha=0.05" in result.trials[0].reason
    assert "nothing (the starting configuration stands)" in result.report()


def test_cheaper_at_equal_success_wins_and_the_budget_is_respected():
    optimizer = Optimizer(build, TUNE)
    champion = Evaluation(scores=[score("a", llm=4, tools=2), score("b", llm=4, tools=2)])
    cheaper = Evaluation(scores=[score("a", llm=2, tools=1), score("b", llm=2, tools=1)])
    barely = Evaluation(scores=[score("a", llm=4, tools=2), score("b", llm=4, tools=1.0)])
    accepted, reason, wins, losses, _ = optimizer.ab_test(champion, cheaper)
    assert accepted and reason == "same tasks pass with 3.00 fewer calls per task" and wins == losses == []
    assert optimizer.ab_test(champion, champion)[:2] == (False, "no measurable improvement")
    assert optimizer.ab_test(cheaper, champion)[0] is False
    assert Optimizer(build, TUNE, min_cost_gain=0.2).ab_test(champion, barely)[0] is False
    limited = Optimizer(build, TUNE, [set_field("max_steps", n) for n in range(2, 12)], max_evaluations=4)
    result = limited.optimize(START)
    assert result.evaluations == 4 and len(result.trials) == 3


def test_a_change_that_overfits_the_suite_is_discarded_by_the_holdout():
    refuse = add_instruction("Refuse when unsure.", name="refuse")
    tuned = Optimizer(build, TUNE, [refuse]).optimize(START)
    assert tuned.changed and tuned.final.success_rate == 1.0           # looks like a pure win on the suite ...
    guarded = Optimizer(build, TUNE, [refuse], holdout=[TASKS["days_between"]]).optimize(START)
    assert not guarded.changed and guarded.config == START             # ... but it breaks an unseen task
    assert (guarded.holdout_before, guarded.holdout_after) == (1.0, 0.0)
    assert "all changes discarded as over-fitted" in guarded.trials[-1].reason
    assert not any(t.accepted for t in guarded.trials)
    safe = Optimizer(build, TUNE, [NEVER_ESTIMATE], holdout=[TASKS["days_between"]]).optimize(START)
    assert safe.changed and (safe.holdout_before, safe.holdout_after) == (1.0, 1.0)


def test_a_model_can_propose_instructions_from_the_failures():
    advisor = ScriptedLLM(['{"instructions": ["Never estimate a figure that the data does not hold.", "ok", '
                           '"never estimate a figure that the data does not hold.", "Answer immediately."]}'])
    result = Optimizer(build, TUNE, llm=advisor).optimize(START, generations=2)
    assert result.changed and result.final.success_rate == 1.0 and len(result.lineage) == 1
    assert result.lineage[0].endswith("proposed by the model after reading the failures")
    assert len(result.trials) == 2 and result.trials[1].losses           # the bad suggestion was measured and refused
    assert len(advisor.prompts) == 1                                     # no failures left: the model is not asked again
    assert "task unanswerable" in advisor.prompts[0] and "5,000.00" in advisor.prompts[0]
    assert "top_region" not in advisor.prompts[0]

    passing = Evaluation(scores=[score("a")])
    assert propose_instructions(advisor, START, passing) == []
    failing = Evaluation(scores=[score("a", success=False)])
    assert propose_instructions(ScriptedLLM(["no json"]), START, failing) == []
    assert propose_instructions(ScriptedLLM(['{"instructions": "nope"}']), START, failing) == []


def test_the_winning_configuration_round_trips_through_a_file(tmp_path):
    result = Optimizer(build, TUNE, [NEVER_ESTIMATE]).optimize(START)
    path = save_config(result.config, tmp_path / "configs" / "champion.json")
    loaded = load_config(EngineConfig, path)
    assert loaded == result.config and fingerprint(loaded) == result.final.fingerprint != result.baseline.fingerprint
