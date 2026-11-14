import itertools
import random

import pytest

from agi.llm import ScriptedLLM
from agi.reasoning import (MCTS, CodeRepair, Game24, PipelineSynthesis, Problem, TreeOfThoughts,
                           Verdict, check_argument, consistent, entails, eval_exact, satisfiable)
from agi.reasoning.logic import LogicSyntaxError, parse
from agi.sandbox import Sandbox, SandboxPolicy


# ------------------------------------------------------------------ logic --- #
def test_valid_and_invalid_inferences():
    assert entails(["rain -> wet", "rain"], "wet")                       # modus ponens
    assert entails(["rain -> wet", "~wet"], "~rain")                     # modus tollens
    fallacy = entails(["rain -> wet", "wet"], "rain")                    # affirming the consequent
    assert not fallacy and fallacy.counterexample == {"rain": False, "wet": True}
    assert entails(["a | b", "~a"], "b") and entails([], "a | ~a")
    assert entails(["a <-> b", "b"], "a") and not entails(["a -> b"], "b -> a")
    assert entails(["a and not b"], "a or b") and entails(["true"], "~false")


def test_consistency_and_models():
    assert not consistent(["a", "a -> b", "~b"])
    assert satisfiable(["(a | b) & ~a", "b <-> c"]) == {"a": False, "b": True, "c": True}
    assert satisfiable(["a & ~a"]) is None


def test_check_argument_pinpoints_the_first_bad_step():
    result = check_argument(["a -> b", "b -> c", "a"], ["b", "c", "d", "c"])
    assert not result.valid and result.failed_step == 2 and result.counterexample["d"] is False
    assert check_argument(["a -> b", "b -> c"], ["a -> c"]).valid
    assert "contradict" in check_argument(["a", "~a"], ["b"]).reason


@pytest.mark.parametrize("bad", ["a &", "(a | b", "a b", "-> a", "a $ b", ""])
def test_malformed_formulas_are_rejected(bad):
    with pytest.raises(LogicSyntaxError):
        parse(bad)


def test_operator_precedence_and_associativity():
    assert parse("~a & b | c") == ("or", ("and", ("not", ("var", "a")), ("var", "b")), ("var", "c"))
    assert parse("a -> b -> c") == ("imp", ("var", "a"), ("imp", ("var", "b"), ("var", "c")))


def _truth(node, world):
    kind = node[0]
    if kind == "var":
        return world[node[1]]
    if kind == "const":
        return node[1]
    if kind == "not":
        return not _truth(node[1], world)
    a, b = _truth(node[1], world), _truth(node[2], world)
    return {"and": a and b, "or": a or b, "imp": (not a) or b, "iff": a == b}[kind]


def _random_formula(rng, depth=3):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice("pqrs")
    if rng.random() < 0.2:
        return f"~({_random_formula(rng, depth - 1)})"
    op = rng.choice(["&", "|", "->", "<->"])
    return f"({_random_formula(rng, depth - 1)} {op} {_random_formula(rng, depth - 1)})"


def test_solver_agrees_with_brute_force_truth_tables():
    rng = random.Random(7)
    for _ in range(150):
        premises = [_random_formula(rng) for _ in range(rng.randint(1, 3))]
        conclusion = _random_formula(rng)
        worlds = [dict(zip("pqrs", bits)) for bits in itertools.product([False, True], repeat=4)]
        expected = all(_truth(parse(conclusion), w) for w in worlds
                       if all(_truth(parse(p), w) for p in premises))
        proof = entails(premises, conclusion)
        assert proof.valid == expected, (premises, conclusion)
        if not proof.valid:                       # the counterexample must be a real one
            world = {**dict.fromkeys("pqrs", False), **proof.counterexample}
            assert all(_truth(parse(p), world) for p in premises)
            assert not _truth(parse(conclusion), world)


# ---------------------------------------------------------------- game 24 --- #
def test_eval_exact_is_exact_and_strict():
    assert eval_exact("(8 / (3 - (8 / 3)))") == 24
    assert eval_exact("-(2 - 5) * 4") == 12
    for bad in ["2 ** 3", "abs(3)", "x + 1", "1.5 + 1"]:
        with pytest.raises(ValueError):
            eval_exact(bad)


@pytest.mark.parametrize("strategy", ["dfs", "bfs"])
@pytest.mark.parametrize("numbers", [(4, 9, 10, 13), (1, 5, 5, 5), (3, 3, 8, 8), (2, 7, 8, 9)])
def test_tree_of_thoughts_solves_24(strategy, numbers):
    result = TreeOfThoughts(Game24(numbers), strategy=strategy, beam=8).search()
    assert result.solved and result.backtracks > 0
    expression = result.best_state[0][1]
    assert eval_exact(expression) == 24
    assert sorted(int(t) for t in expression.replace("(", " ").replace(")", " ").split()
                  if t.isdigit()) == sorted(numbers)          # each number used exactly once


def test_tree_of_thoughts_exhausts_an_impossible_puzzle():
    for strategy in ("dfs", "bfs"):
        result = TreeOfThoughts(Game24((1, 1, 1, 1)), strategy=strategy, beam=100).search()
        assert not result.solved and result.verifications == 0 and result.backtracks > 0
    with pytest.raises(ValueError):
        TreeOfThoughts(Game24((1, 2, 3, 4)), strategy="astar")


def test_mcts_with_rollouts_solves_24():
    result = MCTS(Game24((1, 5, 5, 5)), rollout_depth=3, seed=1).search(4000)
    assert result.solved and "= 24" in result.verdict.detail


# ------------------------------------------------------------------- mcts --- #
class Lock(Problem):
    """Guess a 3-digit code; any prefix that deviates from the code is refuted."""

    def __init__(self, code=(2, 0, 1), feasibility=True):
        self.code, self.feasibility, self.verified = code, feasibility, []

    def initial(self):
        return ()

    def actions(self, state):
        return [] if len(state) == len(self.code) else [0, 1, 2]

    def apply(self, state, action):
        return (*state, action)

    def is_terminal(self, state):
        return len(state) == len(self.code)

    def feasible(self, state):
        if self.feasibility and state != self.code[: len(state)]:
            return f"digit {len(state)} is wrong"
        return None

    def verify(self, state):
        self.verified.append(state)
        right = state == self.code[: len(state)]
        if right and len(state) == len(self.code):
            return Verdict(1.0, solved=True)
        return Verdict(len(state) / 4 if right else 0.0, refuted=not right, detail="wrong prefix")


def test_mcts_backtracks_out_of_refuted_branches():
    problem = Lock()
    result = MCTS(problem, seed=0).search(200)
    assert result.solved and result.best_state == (2, 0, 1)
    assert result.backtracks >= 2 and all(s == (2, 0, 1)[: len(s)] for s in problem.verified)
    assert {e.kind for e in result.trace} >= {"refute", "verify", "solve"}


def test_mcts_never_verifies_a_state_twice_nor_inside_a_dead_subtree():
    problem = Lock(feasibility=False)             # refutation now comes from verification
    result = MCTS(problem, seed=3).search(500)
    assert result.solved and len(problem.verified) == len(set(problem.verified))
    for state in problem.verified:                # nothing was tested beneath a wrong prefix
        assert state[:-1] == (2, 0, 1)[: len(state) - 1]


def test_mcts_reports_exhaustion_when_nothing_works():
    result = MCTS(Lock(code=(9, 9, 9)), seed=0).search(500)
    assert not result.solved and result.iterations < 500 and result.verifications == 0


def test_mcts_respects_the_verification_budget_and_priors():
    problem = Lock(feasibility=False)
    assert MCTS(problem, seed=0, max_verifications=2).search(500).verifications == 2

    class Hinted(Lock):
        def prior(self, state, action):
            return 10.0 if action == self.code[len(state)] else 1.0

    hinted = Hinted(feasibility=False)
    assert MCTS(hinted, seed=0).search(500).verifications == 3      # straight to the answer


# -------------------------------------------------------------- synthesis --- #
SORT_DESC = [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1]), ([2, 2, 1], [2, 2, 1])]
DOUBLE_MAX = [([3, 1, 2], 6), ([5, 4, 9, 1], 18), ([-2, -7], -4)]
COUNT_POS = [([1, 1, -2, 3], 2), ([0, 5, 5, 5], 1), ([-1, -2], 0), ([4, 3, 2, 1, 1], 4)]


def _check(problem, program, examples):
    scope = {}
    exec(problem.source(program), scope)
    return all(scope["solve"](list(i)) == o for i, o in examples)


@pytest.mark.parametrize("examples", [SORT_DESC, DOUBLE_MAX])
def test_synthesis_finds_programs_that_really_satisfy_the_examples(examples):
    problem = PipelineSynthesis(examples)
    result = MCTS(problem, seed=0, max_verifications=1500).search(6000)
    assert result.solved and _check(problem, result.best_state, examples)


def test_synthesis_in_the_sandbox_matches_host_evaluation():
    host = MCTS(PipelineSynthesis(SORT_DESC), seed=0, max_verifications=600).search(3000)
    sandbox = Sandbox(SandboxPolicy(wall_time_s=5))
    boxed = MCTS(PipelineSynthesis(SORT_DESC, sandbox), seed=0, max_verifications=600).search(3000)
    assert boxed.solved and boxed.best_state == host.best_state
    assert boxed.verifications == host.verifications
    assert sandbox.runs < boxed.verifications / 2        # siblings share a sandbox process


def test_synthesis_refutes_crashing_and_redundant_prefixes():
    problem = PipelineSynthesis(COUNT_POS)
    crash = problem.verify(("positives", "max"))          # max([]) on the third example
    assert crash.refuted and "ValueError" in crash.detail
    assert problem.failures[-1][0] == ("positives", "max")
    problem.verify(("sort",))
    redundant = problem.verify(("sort", "sort"))
    assert redundant.refuted and "redundant" in redundant.detail
    assert problem.verify(("reverse", "reverse")).refuted            # same as doing nothing
    assert not problem.verify(("positives",)).refuted


def test_priors_cut_the_search_dramatically():
    cold = MCTS(PipelineSynthesis(COUNT_POS), seed=0, max_verifications=3000).search(12000)
    hint = {"positives": 5.0, "dedupe": 5.0, "count": 5.0}
    warm = MCTS(PipelineSynthesis(COUNT_POS, priors=hint), seed=0,
                max_verifications=3000).search(12000)
    assert cold.solved and warm.solved
    assert warm.verifications * 10 < cold.verifications


def test_avoid_pairs_lower_the_prior():
    problem = PipelineSynthesis(COUNT_POS, avoid=[("positives", "max"), (None, "sum")])
    assert problem.prior(("positives",), "max") < problem.prior(("sort",), "max")
    assert problem.prior((), "sum") < problem.prior((), "max")


def test_actions_are_type_directed():
    problem = PipelineSynthesis(DOUBLE_MAX, max_len=2)
    assert "n_double" not in problem.actions(()) and "sort" in problem.actions(())
    assert set(problem.actions(("max",))) == {"n_double", "n_increment", "n_square", "n_negate"}
    assert problem.actions(("max", "n_double")) == []


# ------------------------------------------------------------ code repair --- #
TESTS = ("def test_basic():\n    assert median([3, 1, 2]) == 2\n"
         "def test_even():\n    assert median([1, 2, 3, 4]) == 2.5\n")
WRONG = "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"
RIGHT = ("def median(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n"
         "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n")


def test_code_repair_feeds_the_real_failure_back_to_the_model():
    llm = ScriptedLLM([f"Here you go:\n```python\n{WRONG}```", f"```python\n{RIGHT}```"])
    problem = CodeRepair("Write median(xs).", TESTS, llm, branching=1, max_depth=3,
                         context="Lessons: remember even-length lists.")
    result = MCTS(problem, seed=0).search(10)
    assert result.solved and result.verifications == 2
    assert result.best_state[-1].strip() == RIGHT.strip()
    first, second = llm.prompts
    assert "Lessons: remember even-length lists." in first and "previous attempt" not in first
    assert "test_even" in second and "sorted(xs)[len(xs) // 2]" in second


def test_code_repair_gives_up_within_its_depth_budget():
    llm = ScriptedLLM(lambda prompt: f"```python\n{WRONG}```")
    result = MCTS(CodeRepair("Write median(xs).", TESTS, llm, branching=2, max_depth=2),
                  seed=0).search(50)
    assert not result.solved and 0 < result.best_score < 1
    assert len(llm.prompts) <= 2 + 4


def test_code_repair_never_runs_unsafe_model_output():
    llm = ScriptedLLM(lambda prompt: "```python\nimport os\ndef median(xs):\n    os.remove('x')\n```")
    problem = CodeRepair("Write median(xs).", TESTS, llm, branching=1, max_depth=1)
    result = MCTS(problem, seed=0).search(5)
    assert not result.solved and "PolicyViolation" in result.verdict.detail
