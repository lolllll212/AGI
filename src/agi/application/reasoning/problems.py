"""Concrete problems for the search procedures.

* :class:`PipelineSynthesis` - find a program (a pipeline of primitives) that
  maps example inputs to outputs. Hypotheses are *executed* in the sandbox.
* :class:`Game24` - the classic Tree-of-Thoughts benchmark.
* :class:`CodeRepair` - a language model proposes programs, the sandbox runs
  the tests, and failures are fed back as the next branch of the tree.
"""
from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from agi.domain.entities.problem import Problem, Verdict
from agi.domain.ports import SandboxPort, make
from agi.domain.ports.llm import LLM
from agi.domain.services.text import extract_code


# --------------------------------------------------------------------------- #
# Program synthesis over a small typed DSL
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Op:
    name: str
    takes: str      # "list" or "int"
    gives: str
    expr: str       # Python expression over ``x``


OPS: dict[str, Op] = {op.name: op for op in [
    Op("sort", "list", "list", "sorted(x)"),
    Op("reverse", "list", "list", "x[::-1]"),
    Op("dedupe", "list", "list", "list(dict.fromkeys(x))"),
    Op("evens", "list", "list", "[v for v in x if v % 2 == 0]"),
    Op("odds", "list", "list", "[v for v in x if v % 2 == 1]"),
    Op("positives", "list", "list", "[v for v in x if v > 0]"),
    Op("double", "list", "list", "[v * 2 for v in x]"),
    Op("increment", "list", "list", "[v + 1 for v in x]"),
    Op("square", "list", "list", "[v * v for v in x]"),
    Op("negate", "list", "list", "[-v for v in x]"),
    Op("tail", "list", "list", "x[1:]"),
    Op("running_sum", "list", "list", "[sum(x[:i + 1]) for i in range(len(x))]"),
    Op("sum", "list", "int", "sum(x)"),
    Op("max", "list", "int", "max(x)"),
    Op("min", "list", "int", "min(x)"),
    Op("count", "list", "int", "len(x)"),
    Op("first", "list", "int", "x[0]"),
    Op("last", "list", "int", "x[-1]"),
    Op("n_double", "int", "int", "x * 2"),
    Op("n_increment", "int", "int", "x + 1"),
    Op("n_square", "int", "int", "x * x"),
    Op("n_negate", "int", "int", "-x"),
]}


def _kind(value: Any) -> str:
    return "list" if isinstance(value, list) else "int"


def similarity(got: Any, want: Any) -> float:
    """Partial credit in [0, 1] for an output that is not (yet) the expected one."""
    if got == want and _kind(got) == _kind(want):
        return 1.0
    if isinstance(got, list) and isinstance(want, list):
        length = 1 - abs(len(got) - len(want)) / max(len(got), len(want), 1)
        pool, shared = list(want), 0
        for item in got:
            if item in pool:
                pool.remove(item)
                shared += 1
        overlap = shared / max(len(got), len(want), 1)
        aligned = sum(a == b for a, b in zip(got, want)) / max(len(got), len(want), 1)
        return 0.15 + 0.15 * length + 0.35 * overlap + 0.3 * aligned
    if isinstance(got, list):          # still a list, but the answer is a number
        return 0.25 + (0.2 if want in got else 0.0)
    if isinstance(want, list) or isinstance(got, bool) or not isinstance(got, (int, float)):
        return 0.0
    return 0.15 + 0.5 / (1 + abs(got - want))


class PipelineSynthesis(Problem):
    """Search for ``op1 | op2 | ...`` consistent with input/output examples.

    A state is a tuple of op names. Every state is a complete program, so every
    node the search creates is compiled to Python and *run*. Two kinds of
    evidence prune the tree, and both are sound:

    * a prefix that crashes on an example (``max`` of an empty list, say) can
      never be repaired by appending more steps;
    * a prefix that behaves exactly like a shorter one already explored is
      redundant.
    """

    def __init__(self, examples: Sequence[tuple], sandbox: SandboxPort | None = None,
                 ops: Mapping[str, Op] | None = None, max_len: int = 4,
                 priors: Mapping[str, float] | None = None, avoid: Sequence[tuple] = (),
                 policy: Any = None):
        """``priors`` weights individual ops; ``avoid`` lists ``(previous_op, op)``
        pairs known to fail (``previous_op`` is ``None`` at the start); ``policy``
        is a trained :class:`agi.training.SearchPolicy` that ranks the next op."""
        self.policy = policy
        self._guidance: dict[tuple, dict] = {}
        self.avoid = {tuple(pair) for pair in avoid}
        self.examples = [(i, o) for i, o in examples]
        self.sandbox = sandbox
        self.ops = dict(ops or OPS)
        self.max_len = max_len
        self.priors = dict(priors or {})
        self.start = _kind(self.examples[0][0])
        self.goal = _kind(self.examples[0][1])
        # Behaviour already seen -> the first program that showed it (identity included).
        self._behaviours: dict[str, tuple] = {repr([i for i, _ in self.examples]): ()}
        self._runs: dict[tuple, list] = {}
        self.failures: list[tuple[tuple, str]] = []     # (program, error) for memory

    def initial(self) -> tuple:
        return ()

    def _type_after(self, state: tuple) -> str:
        return self.ops[state[-1]].gives if state else self.start

    def actions(self, state: tuple) -> list[str]:
        if len(state) >= self.max_len:
            return []
        current = self._type_after(state)
        return [name for name, op in self.ops.items() if op.takes == current]

    def apply(self, state: tuple, action: str) -> tuple:
        return (*state, action)

    def is_terminal(self, state: tuple) -> bool:
        return len(state) >= self.max_len

    def prior(self, state: tuple, action: str) -> float:
        penalty = 0.2 if ((state[-1] if state else None), action) in self.avoid else 1.0
        return self.priors.get(action, 1.0) * penalty * self._learned(state, action)

    def _learned(self, state: tuple, action: str) -> float:
        """The trained policy's opinion, as a multiplier around 1.0 (1.0 = no opinion)."""
        if self.policy is None:
            return 1.0
        guidance = self._guidance.get(state)
        if guidance is None:
            guidance = {}
            # Only values this search has already computed are used: no extra execution.
            results = self._runs.get(state) if state else [
                {"ok": True, "value": i} for i, _ in self.examples]
            if results and all(r["ok"] for r in results):
                legal = self.actions(state)
                probabilities = self.policy.distribution(
                    [r["value"] for r in results], [want for _, want in self.examples],
                    len(state), state[-1] if state else None, legal)
                guidance = {a: min(8.0, max(0.05, p * len(legal))) for a, p in probabilities.items()}
            self._guidance[state] = guidance
        return guidance.get(action, 1.0)

    def describe(self, state: tuple) -> str:
        return " | ".join(state) or "(identity)"

    def source(self, state: tuple, name: str = "solve") -> str:
        """The Python program a state stands for."""
        body = "".join(f"    x = {self.ops[op].expr}\n" for op in state)
        return f"def {name}(x):\n{body}    return x\n"

    def run(self, state: tuple) -> list[dict]:
        """Execute the program on every example input: ``[{"ok", "value", "error"}]``.

        With a sandbox, sibling hypotheses are executed in the same sandbox
        process (one interpreter launch per tree node instead of per program).
        """
        if state not in self._runs:
            batch = [state]
            if self.sandbox is not None and state:
                batch += [s for s in (self.apply(state[:-1], a) for a in self.actions(state[:-1]))
                          if s != state and s not in self._runs]
            for program, results in zip(batch, self._execute(batch)):
                self._runs[program] = results
        return self._runs[state]

    def _execute(self, programs: list[tuple]) -> list[list[dict]]:
        inputs = [[example_in] for example_in, _ in self.examples]
        names = [f"solve_{i}" for i in range(len(programs))]
        code = "\n".join(self.source(p, n) for p, n in zip(programs, names))
        if self.sandbox is not None:
            outcome = self.sandbox.call_each(code, names, inputs)
            if not outcome.ok:
                return [[{"ok": False, "value": None, "error": outcome.failure}] * len(inputs)
                        for _ in programs]
            return [[{"ok": c["ok"], "value": c["value"],
                      "error": None if c["ok"] else f"{c['error_type']}: {c['error']}"}
                     for c in row] for row in outcome.calls]
        scope: dict = {}
        exec(code, scope)                     # trusted: built only from the DSL above
        table = []
        for name in names:
            row = []
            for (example_in,) in inputs:
                try:
                    row.append({"ok": True, "value": scope[name](example_in), "error": None})
                except Exception as exc:
                    row.append({"ok": False, "value": None, "error": f"{type(exc).__name__}: {exc}"})
            table.append(row)
        return table

    def verify(self, state: tuple) -> Verdict:
        results = self.run(state)
        crash = next((r["error"] for r in results if not r["ok"]), None)
        if crash:
            self.failures.append((state, crash))
            return Verdict(0.0, detail=crash, refuted=True)
        outputs = [r["value"] for r in results]
        wanted = [want for _, want in self.examples]
        if outputs == wanted and all(_kind(o) == self.goal for o in outputs):
            return Verdict(1.0, solved=True, detail="all examples pass")
        fingerprint = repr(outputs)
        earlier = self._behaviours.setdefault(fingerprint, state)
        if earlier != state and len(earlier) <= len(state):
            return Verdict(0.0, refuted=True,
                           detail=f"redundant: behaves exactly like '{self.describe(earlier)}'")
        score = sum(similarity(o, w) for o, w in zip(outputs, wanted)) / len(wanted)
        if self.is_terminal(state) and _kind(outputs[0]) != self.goal:
            self.failures.append((state, f"TypeMismatch: produced {_kind(outputs[0])}, "
                                         f"expected {self.goal}"))
        return Verdict(min(score, 0.99), detail=f"{sum(o == w for o, w in zip(outputs, wanted))}"
                                                f"/{len(wanted)} examples pass")


# --------------------------------------------------------------------------- #
# Game of 24
# --------------------------------------------------------------------------- #
def eval_exact(expression: str) -> Fraction:
    """Evaluate ``+ - * /`` arithmetic exactly. Independent of the search's own bookkeeping."""
    def walk(node) -> Fraction:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:40]}")
    return walk(ast.parse(expression, mode="eval"))


class Game24(Problem):
    """Combine the numbers with ``+ - * /`` to reach the target, using each once.

    A state is the multiset of numbers still on the table, each with the
    expression that produced it. A thought is "combine two of them".
    """

    def __init__(self, numbers: Sequence[int], target: int = 24):
        self.numbers = tuple(numbers)
        self.target = Fraction(target)

    def initial(self) -> tuple:
        return tuple(sorted((Fraction(n), str(n)) for n in self.numbers))

    def is_terminal(self, state: tuple) -> bool:
        return len(state) == 1

    def actions(self, state: tuple) -> list[tuple]:
        if len(state) < 2:
            return []
        found: dict[tuple, tuple] = {}
        for i, j in itertools.combinations(range(len(state)), 2):
            for symbol in "+*-/":
                for a, b in ((i, j), (j, i)) if symbol in "-/" else ((i, j),):
                    if symbol == "/" and state[b][0] == 0:
                        continue
                    found.setdefault(self.apply(state, (a, b, symbol)), (a, b, symbol))
        return list(found.values())

    def apply(self, state: tuple, action: tuple) -> tuple:
        a, b, symbol = action
        (x, ex), (y, ey) = state[a], state[b]
        value = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[symbol]
        rest = [item for k, item in enumerate(state) if k not in (a, b)]
        return tuple(sorted([*rest, (value, f"({ex} {symbol} {ey})")]))

    def _reachable(self, x: Fraction, y: Fraction) -> bool:
        options = [x + y, x - y, y - x, x * y]
        options += [x / y] if y else []
        options += [y / x] if x else []
        return self.target in options

    def feasible(self, state: tuple) -> str | None:
        if len(state) == 2 and not self._reachable(state[0][0], state[1][0]):
            return f"no operation on {state[0][0]} and {state[1][0]} gives {self.target}"
        return None

    def value(self, state: tuple) -> float:
        if len(state) == 1:
            return 1.0 if state[0][0] == self.target else 0.0
        if len(state) == 2:
            return 1.0
        # Three or more left: prefer tables holding a factor/neighbour of the target.
        values = [v for v, _ in state]
        handy = sum(1 for v in values if v and (self.target / v).denominator == 1)
        return min(0.9, 0.3 + 0.2 * handy)

    def verify(self, state: tuple) -> Verdict:
        expression = state[0][1]
        try:
            result = eval_exact(expression)
        except (ValueError, ZeroDivisionError, SyntaxError) as exc:
            return Verdict(0.0, detail=f"{type(exc).__name__}: {exc}", refuted=True)
        if result == self.target:
            return Verdict(1.0, solved=True, detail=f"{expression} = {self.target}")
        return Verdict(0.0, detail=f"{expression} = {result}, not {self.target}", refuted=True)

    def describe(self, state: tuple) -> str:
        return ", ".join(expr if expr.startswith("(") else str(value) for value, expr in state)


# --------------------------------------------------------------------------- #
# LLM-in-the-loop: propose code, run tests, branch on the failure
# --------------------------------------------------------------------------- #
_WRITE = """Write Python code for this task. Reply with one fenced code block and nothing else.
Only these modules may be imported: {imports}.
{context}
TASK:
{task}
"""
_REPAIR = _WRITE + """
A previous attempt:
```python
{code}
```
failed verification with:
{failure}
Write a corrected version (variation #{variant}).
"""


class CodeRepair(Problem):
    """Tree search over model-written programs, scored by really running the tests.

    The root is "no code yet". Each action asks the model for one candidate;
    children of a failed candidate are repairs conditioned on its actual error.
    ``context`` is a good place for ``MemoryEngine.context(task)``.
    """

    def __init__(self, task: str, tests: str, llm: LLM, sandbox: SandboxPort | None = None,
                 branching: int = 2, max_depth: int = 3, context: str = ""):
        self.task, self.tests, self.llm = task, tests, llm
        self.sandbox = sandbox or make("sandbox")
        self.branching, self.max_depth, self.context = branching, max_depth, context
        self._failures: dict[tuple, str] = {}

    def initial(self) -> tuple:
        return ()                          # a state is the chain of attempts so far

    def actions(self, state: tuple) -> list[int]:
        return list(range(self.branching)) if len(state) < self.max_depth else []

    def is_terminal(self, state: tuple) -> bool:
        return len(state) >= self.max_depth

    def apply(self, state: tuple, action: int) -> tuple:
        imports = ", ".join(sorted(self.sandbox.policy.allowed_imports))
        if not state:
            prompt = _WRITE.format(task=self.task, imports=imports, context=self.context)
        else:
            prompt = _REPAIR.format(task=self.task, imports=imports, context=self.context,
                                    code=state[-1], failure=self._failures.get(state, "unknown"),
                                    variant=action + 1)
        return (*state, extract_code(self.llm(prompt)))

    def verify(self, state: tuple) -> Verdict:
        if not state:
            return Verdict(0.0, detail="no code yet")
        outcome = self.sandbox.run_tests(state[-1], self.tests)
        if outcome.ok:
            return Verdict(1.0, solved=True, detail=f"{len(outcome.tests)} tests pass")
        self._failures[state] = outcome.failure or "unknown failure"
        passed = sum(1 for v in outcome.tests.values() if v is None)
        return Verdict(0.9 * passed / max(1, len(outcome.tests)), detail=self._failures[state])

    def describe(self, state: tuple) -> str:
        return f"attempt #{len(state)}" if state else "(no code)"
