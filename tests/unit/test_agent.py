from agi import LearningAgent, Sandbox, Task
from agi.application.memory.engine import MemoryEngine
from agi.domain.entities.memory import Lesson

EVENS = Task("count the distinct even numbers in a list",
             [([2, 2, 4, 3], 2), ([1, 5, 5], 0), ([8, 8, 8], 1), ([6, 4, 2, 1, 1], 3)], ("count",))
POSITIVES = Task("count the distinct positive numbers in a list",
                 [([1, 1, -2, 3], 2), ([0, 5, 5, 5], 1), ([-1, -2], 0), ([4, 3, 2, 1, 1], 4)], ("count",))
DESCENDING = Task("sort a list in descending order",
                  [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])])


def run(source, value):
    scope = {}
    exec(source, scope)
    return scope["solve"](value)


def test_cold_solve_then_instant_recall_in_a_new_session(tmp_path):
    path = str(tmp_path / "agent.db")
    with MemoryEngine(path) as memory:
        first = LearningAgent(memory).solve(EVENS)
    assert first.solved and first.how == "search" and first.recalled == []
    assert first.verifications > 100 and any(l.kind == "procedure" for l in first.learned)
    assert all(run(first.source, i) == o for i, o in EVENS.examples)

    with MemoryEngine(path) as memory:                    # nothing in common but the file
        again = LearningAgent(memory).solve(EVENS)
        assert again.solved and again.how == "recalled" and again.verifications == 1
        assert again.program == first.program
        solution = next(l for l in memory.lessons() if l.kind == "procedure")
        assert solution.helped == 1 and solution.support == 2


def test_experience_transfers_to_a_related_task():
    cold = LearningAgent(MemoryEngine()).solve(POSITIVES)
    memory = MemoryEngine()
    agent = LearningAgent(memory)
    agent.solve(EVENS)
    warm = agent.solve(POSITIVES)
    assert cold.solved and warm.solved and warm.how == "guided search"
    assert warm.verifications * 4 < cold.verifications
    assert all(run(warm.source, i) == o for i, o in POSITIVES.examples)


def test_unrelated_experience_is_not_dragged_in():
    memory = MemoryEngine()
    agent = LearningAgent(memory)
    agent.solve(EVENS)
    attempt = agent.solve(DESCENDING)
    assert attempt.solved and attempt.how == "search"
    assert not any(l.kind == "procedure" for l in attempt.recalled)


def test_wrong_memories_are_reverified_not_trusted():
    memory = MemoryEngine()
    poison = memory.learn(Lesson("procedure", EVENS.description, "Known-good approach: sum",
                                 tags=EVENS.tags, payload={"solution": ["sum"]}))
    gibberish = memory.learn(Lesson("procedure", EVENS.description + " quickly", "use magic",
                                    tags=EVENS.tags, payload={"solution": ["no_such_op"]}))
    attempt = LearningAgent(memory).solve(EVENS)
    assert attempt.solved and attempt.program != ("sum",)
    assert all(run(attempt.source, i) == o for i, o in EVENS.examples)
    assert memory.get(poison.id).hurt == 1 and memory.get(poison.id).utility < 0.5
    assert memory.get(gibberish.id) is not None           # ignored, not crashed on


def test_failure_is_recorded_as_experience_too():
    memory = MemoryEngine()
    impossible = Task("reverse the digits of every number in a list", [([12, 30], [21, 3])])
    attempt = LearningAgent(memory, budget=60).solve(impossible)
    assert not attempt.solved and attempt.program is None and attempt.source is None
    assert "FAILED" in str(attempt)
    (episode,) = memory.episodes()
    assert episode["outcome"] == "failure" and episode["solution"] is None


def test_the_same_loop_works_inside_the_sandbox():
    sandbox = Sandbox()
    memory = MemoryEngine()
    agent = LearningAgent(memory, sandbox)
    first = agent.solve(DESCENDING)
    again = agent.solve(DESCENDING)
    assert first.solved and first.program == ("sort", "reverse")
    assert again.how == "recalled" and again.verifications == 1 and sandbox.runs > 2
