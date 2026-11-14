import json
import random

import pytest

from agi import LearningAgent, Task
from agi.application.cortex.react import ReActAgent
from agi.application.memory.engine import MemoryEngine
from agi.application.reasoning.mcts import MCTS
from agi.application.reasoning.problems import OPS, PipelineSynthesis
from agi.domain.entities.tool import ToolRegistry
from agi.infrastructure.adapters.execution.builtin import calculator_tool
from agi.infrastructure.adapters.llm.scripted import ScriptedLLM
from agi.training.dataset import export_jsonl, to_chat_example
from agi.training.policy import SearchPolicy, compare, featurize, make_task, make_tasks, samples_from, train_policy

SECOND_LARGEST = [([3, 1, 2], 2), ([5, 4, 9, 1], 5), ([7, 8, 3], 7), ([1, 2], 1), ([4, 6, 5, 0], 5)]


@pytest.fixture(scope="module")
def trained():
    return train_policy(tasks=4000, seed=0)


def run_program(program, value):
    scope = {}
    exec(PipelineSynthesis([([1], [1])]).source(tuple(program)), scope)
    return scope["solve"](value)


# --------------------------------------------------------------- dreaming --- #
def test_dreamed_tasks_are_consistent_and_non_trivial():
    for task in make_tasks(60, seed=3):
        assert 1 <= len(task.program) <= 4 and len(task.examples) == 5
        assert all(run_program(task.program, list(i)) == o for i, o in task.examples)
        assert len({repr(o) for _, o in task.examples}) >= 3            # not a constant function
        behaviours = [repr([i for i, _ in task.examples])] + [repr(v) for v in task.trace]
        assert len(set(behaviours)) == len(behaviours)                   # no step is a no-op


def test_dreaming_is_deterministic_and_seed_dependent():
    a, b, c = make_tasks(20, seed=1), make_tasks(20, seed=1), make_tasks(20, seed=2)
    assert [t.program for t in a] == [t.program for t in b] != [t.program for t in c]


def test_minimal_tasks_really_need_their_length():
    for task in make_tasks(8, seed=5, lengths=(3,), minimal=True):
        assert len(task.program) == 3
        short = MCTS(PipelineSynthesis(task.examples, max_len=2), seed=0).search(4000)
        assert not short.solved                                          # no 1- or 2-step program fits


def test_samples_follow_the_program():
    examples = [([3, 1, 2], 6), ([5, 4, 9, 1], 18), ([-2, -7], -4)]
    samples = samples_from(examples, ("max", "n_double"))
    assert [s.label for s in samples] == ["max", "n_double"]
    assert "sort" in samples[0].legal and samples[1].legal == ["n_double", "n_increment", "n_square", "n_negate"]
    assert "prev:None" in samples[0].features and "prev:max" in samples[1].features
    assert "shape:list>int" in samples[0].features and "shape:int>int" in samples[1].features
    assert samples_from(examples, ("n_double",)) == []                   # ill-typed programs give nothing


def test_features_describe_the_data_not_candidate_programs():
    f = featurize([[3, 1, 2, 2]], [[1, 2, 3]], 0, None)
    assert {"list>list:all:shorter", "list>list:all:dups_removed", "list>list:all:want_ascending",
            "list>list:all:subset"} <= set(f)
    mixed = featurize([[1, 2, 3], [4, 6]], [[2], [4, 6]], 1, "sort")
    assert "list>list:all:want_all_even" in mixed and "list>list:some:odds_gone" in mixed
    ranked = featurize([[5, 9, 1], [4, 2, 8]], [5, 4], 0, None)
    assert "list>int:all:second_largest" in ranked and "list>int:all:member" in ranked
    # Nothing reveals directly that an op such as first/max/count would be correct.
    names = " ".join(featurize([[5, 9, 1]], [5], 0, None) + featurize([[5, 9, 1]], [9], 0, None)
                     + featurize([[5, 9, 1]], [3], 0, None))
    assert not any(word in names for word in ("is_first", "is_max", "is_count", "at_index_0", "is_sum"))


# --------------------------------------------------------------- learning --- #
def test_training_reduces_loss_and_predicts_held_out_steps(trained):
    policy, report = trained
    assert report.loss[-1] < report.loss[0] * 0.85 and report.samples > 5000
    assert report.accuracy_before < 0.2 and report.accuracy_after > 0.45 and report.top3_after > 0.7
    assert "held-out next-step accuracy" in str(report) and policy.trained_on == report.samples


def test_untrained_policy_is_uniform_and_harmless():
    policy = SearchPolicy()
    probabilities = policy.distribution([[1, 2]], [[2, 1]], 0, None, ["sort", "reverse", "tail"])
    assert probabilities == pytest.approx({"sort": 1 / 3, "reverse": 1 / 3, "tail": 1 / 3})
    examples = [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])]
    plain = MCTS(PipelineSynthesis(examples), seed=0).search(3000)
    guided = MCTS(PipelineSynthesis(examples, policy=policy), seed=0).search(3000)
    assert guided.best_state == plain.best_state and guided.verifications == plain.verifications


def test_trained_policy_speeds_up_search_on_unseen_tasks(trained):
    policy, _ = trained
    tasks = make_tasks(20, seed=4242, lengths=(4,), minimal=True)
    result = compare(policy, tasks, budget=1500)
    s = result.summary
    assert s["solved_trained"] > s["solved_baseline"] and s["solved_trained"] >= 18
    assert s["mean_verifications_trained"] * 2 < s["mean_verifications_baseline"]
    assert "solved" in str(result)


def test_trained_policy_solves_a_task_plain_search_usually_cannot(trained):
    policy, _ = trained
    result = compare(policy, [SECOND_LARGEST], budget=3000, seed=0)
    assert result.baseline == [None] and result.trained[0] is not None and result.trained[0] < 1500


def test_guided_search_still_returns_only_verified_programs(trained):
    policy, _ = trained
    for task in make_tasks(10, seed=77, lengths=(3,)):
        problem = PipelineSynthesis(task.examples, policy=policy)
        result = MCTS(problem, seed=0, max_verifications=1500).search(6000)
        if result.solved:
            assert all(run_program(result.best_state, list(i)) == o for i, o in task.examples)


def test_a_misleading_policy_costs_time_not_correctness():
    policy = SearchPolicy()
    for op in policy.ops:                       # push hard towards useless first moves
        policy.weights[op]["bias"] = 5.0 if op in ("negate", "square", "running_sum") else -5.0
    examples = [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])]
    problem = PipelineSynthesis(examples, policy=policy)
    result = MCTS(problem, seed=0, max_verifications=3000).search(12000)
    assert result.solved and all(run_program(result.best_state, list(i)) == o for i, o in examples)


def test_online_updates_and_persistence(tmp_path):
    policy = SearchPolicy()
    examples = [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])]
    sample = samples_from(examples, ("sort", "reverse"))[0]
    before = policy.probabilities(sample.features, sample.legal)["sort"]
    for _ in range(5):
        policy.update(examples, ("sort", "reverse"))
    after = policy.probabilities(sample.features, sample.legal)["sort"]
    assert after > before * 3 and policy.trained_on == 10
    path = str(tmp_path / "policy.json")
    policy.save(path)
    loaded = SearchPolicy.load(path)
    assert loaded.probabilities(sample.features, sample.legal)["sort"] == pytest.approx(after, abs=1e-3)
    assert loaded.trained_on == 10 and loaded.ops == list(OPS)


def test_learning_agent_trains_its_policy_on_what_it_solves():
    task = Task("sort a list in descending order",
                [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])])
    policy = SearchPolicy()
    first = LearningAgent(MemoryEngine(), policy=policy).solve(task)
    assert first.solved and policy.trained_on == 2
    # Fresh memory, so any speed-up comes from the trained weights alone.
    again = LearningAgent(MemoryEngine(), policy=policy).solve(task)
    assert again.solved and again.how == "search" and again.verifications < first.verifications / 5


# ------------------------------------------------------- fine-tuning data --- #
def _runs():
    registry = lambda: ToolRegistry([calculator_tool()])
    act = 'Thought: compute\nAction: calculator\nAction Input: {"expression": "%s"}'
    good = ReActAgent(ScriptedLLM([act % "2 ^ 8", act % "2 ** 8", "Thought: done\nFinal Answer: It is 256."]),
                      registry()).run("What is 2 to the 8th?")
    made_up = ReActAgent(ScriptedLLM(["Thought: x\nFinal Answer: It is 300."] * 2), registry()).run("2 to the 8th?")
    unfinished = ReActAgent(ScriptedLLM(["?", "?", "?"]), registry()).run("2 to the 8th?")
    return good, made_up, unfinished


def test_chat_example_shape():
    good, _, _ = _runs()
    messages = to_chat_example(good)["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert messages[1]["content"] == "Task: What is 2 to the 8th?"
    assert messages[2]["content"].endswith('Action Input: {"expression": "2 ^ 8"}')
    assert messages[3]["content"].startswith("Observation: ToolError: '^' is not a power operator")
    assert messages[-1]["content"] == "Thought: done\nFinal Answer: It is 256."
    assert len(to_chat_example(good, include_errors=False)["messages"]) == 5


def test_only_verified_runs_are_exported(tmp_path):
    path = str(tmp_path / "data.jsonl")
    assert export_jsonl(_runs(), path) == 1
    (line,) = open(path).read().splitlines()
    assert json.loads(line)["messages"][-1]["content"].endswith("It is 256.")
    assert export_jsonl(_runs(), path, only_successful=False, require_grounded=False) == 2
