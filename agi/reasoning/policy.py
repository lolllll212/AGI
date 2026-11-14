"""Training: a policy that learns where program search should look first.

Memory (pillar 1) helps on tasks that resemble ones seen before. This module is
the complementary mechanism: *weights that are actually trained*, so that the
search gets better on tasks it has never seen.

The loop is the one used by DeepCoder/DreamCoder-style systems, in miniature:

1. **Dream** - sample random programs from the DSL, run them on random inputs,
   and keep the (examples, program) pairs. The supervision is free and exact.
2. **Train** - fit a softmax policy ``P(next op | current values, target)`` by
   stochastic gradient descent on those pairs.
3. **Search** - :class:`~agi.reasoning.PipelineSynthesis` uses the policy as its
   prior; every candidate is still executed and verified, so a wrong policy can
   only cost time, never correctness.
4. **Keep learning** - :meth:`SearchPolicy.update` takes gradient steps on each
   task the agent really solves.

The features are properties of the data (lengths, order, parity, sign,
membership, rank). None of them is equivalent to checking a DSL operation, so
"verifications" remains an honest count of the candidates that were tested.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .problems import OPS, Op

_FN = {name: eval(f"lambda x: {op.expr}") for name, op in OPS.items()}     # trusted: DSL constants


def _kind(value: Any) -> str:
    return "list" if isinstance(value, list) else "int"


def _square(n: int) -> bool:
    return n >= 0 and math.isqrt(n) ** 2 == n


def _subsequence(small: list, big: list) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def _pair_properties(cur: Any, want: Any) -> list[str]:
    """Named properties of one (current value, wanted value) pair."""
    p = []
    if isinstance(cur, list) and isinstance(want, list):
        if len(want) == len(cur):
            p.append("same_len")
            if cur and all(w >= c for w, c in zip(want, cur)):
                p.append("pointwise_ge")
            if cur and all(w <= c for w, c in zip(want, cur)):
                p.append("pointwise_le")
        elif len(want) < len(cur):
            p.append("shorter")
            if len(want) == len(cur) - 1:
                p.append("shorter_by_one")
        else:
            p.append("longer")
        if sorted(want) == sorted(cur):
            p.append("same_multiset")
        members = set(cur)
        if all(w in members for w in want):
            p.append("subset")
            if _subsequence(want, cur):
                p.append("subsequence")
        elif want and not any(w in members for w in want):
            p.append("disjoint")
        if len(want) >= 2:
            if all(a <= b for a, b in zip(want, want[1:])):
                p.append("want_ascending")
            if all(a >= b for a, b in zip(want, want[1:])):
                p.append("want_descending")
        if len(cur) >= 2 and all(a <= b for a, b in zip(cur, cur[1:])):
            p.append("cur_ascending")
        if len(cur) >= 2 and all(a >= b for a, b in zip(cur, cur[1:])):
            p.append("cur_descending")
        cur_dups, want_dups = len(members) < len(cur), len(set(want)) < len(want)
        if cur_dups:
            p.append("cur_has_dups")
        if want_dups:
            p.append("want_has_dups")
        if cur_dups and not want_dups:
            p.append("dups_removed")
        if len(want) == len(members):
            p.append("len_is_distinct_count")
        if want:
            if all(w % 2 == 0 for w in want):
                p.append("want_all_even")
                if any(c % 2 for c in cur):
                    p.append("odds_gone")
            if all(w % 2 for w in want):
                p.append("want_all_odd")
                if any(c % 2 == 0 for c in cur):
                    p.append("evens_gone")
            if all(w > 0 for w in want):
                p.append("want_all_positive")
                if any(c <= 0 for c in cur):
                    p.append("nonpositives_gone")
            if all(w <= 0 for w in want):
                p.append("want_all_nonpositive")
            if all(_square(w) for w in want):
                p.append("want_all_squares")
            if max(map(abs, want)) > max(map(abs, cur), default=0):
                p.append("magnitude_grew")
            if cur and want[0] == cur[0]:
                p.append("same_first")
            if cur and want[-1] == cur[-1]:
                p.append("same_last")
            if cur and want[0] == cur[-1]:
                p.append("first_is_last")
        else:
            p.append("want_empty")
    elif isinstance(cur, list):
        if want in cur:
            p.append("member")
            # Where the target sits, by position and by rank. Positions 0/-1 and ranks 0
            # are left out on purpose: they would amount to checking `first`, `last`,
            # `min` or `max` for free.
            if len(cur) >= 2 and cur[1] == want:
                p.append("at_index_1")
            if len(cur) >= 2 and cur[-2] == want:
                p.append("at_index_-2")
            ranked = sorted(set(cur))
            if len(ranked) >= 2:
                if ranked[1] == want:
                    p.append("second_smallest")
                if ranked[-2] == want:
                    p.append("second_largest")
        elif cur:
            if want > max(cur):
                p.append("above_all")
            elif want < min(cur):
                p.append("below_all")
            else:
                p.append("inside_range")
        if 0 <= want <= len(cur):
            p.append("could_be_count")
        if want > sum(abs(c) for c in cur):
            p.append("exceeds_total")
        p += _int_properties(want)
    elif isinstance(want, int):
        p.append("want_gt" if want > cur else "want_lt" if want < cur else "want_eq")
        if (want < 0) != (cur < 0) and want and cur:
            p.append("sign_flipped")
        if abs(want) > 2 * abs(cur) + 1:
            p.append("much_bigger")
        p += _int_properties(want)
    return p


def _int_properties(want: int) -> list[str]:
    p = ["want_even" if want % 2 == 0 else "want_odd"]
    p.append("want_negative" if want < 0 else "want_zero" if want == 0 else "want_positive")
    if _square(want):
        p.append("want_square")
    return p


def featurize(current: Sequence, wanted: Sequence, depth: int, previous: str | None) -> list[str]:
    """Active feature names for one search state.

    ``current`` holds the value the partial program produces on each example,
    ``wanted`` the value it should end with.
    """
    shape = f"{_kind(current[0])}>{_kind(wanted[0])}"
    counts: dict[str, int] = {}
    for cur, want in zip(current, wanted):
        for name in _pair_properties(cur, want):
            counts[name] = counts.get(name, 0) + 1
    total = len(current)
    active = ["bias", f"shape:{shape}", f"depth:{min(depth, 4)}", f"prev:{previous}"]
    for name, count in counts.items():
        active.append(f"{shape}:all:{name}" if count == total else f"{shape}:some:{name}")
    return active


# --------------------------------------------------------------------------- #
# dreaming: free, exact supervision
# --------------------------------------------------------------------------- #
@dataclass
class SynthTask:
    examples: list[tuple]
    program: tuple
    trace: list[list] = field(default_factory=list)     # values after each step, per example


def _random_input(rng: random.Random) -> list[int]:
    pool = [rng.randint(-9, 9) for _ in range(rng.randint(2, 5))]
    return [rng.choice(pool) for _ in range(rng.randint(2, 7))]


def _shortest_lengths(inputs: list, limit: int = 2) -> dict[str, int]:
    """Behaviour fingerprint -> length of the shortest program (up to ``limit``) showing it."""
    seen = {repr(inputs): 0}
    frontier = [("list", inputs)]
    for length in range(1, limit + 1):
        grown = []
        for kind, values in frontier:
            for name, op in OPS.items():
                if op.takes != kind:
                    continue
                try:
                    out = [_FN[name](v) for v in values]
                except Exception:
                    continue
                key = repr(out)
                if key not in seen:
                    seen[key] = length
                    grown.append((op.gives, out))
        frontier = grown
    return seen


def make_task(rng: random.Random, length: int, examples: int = 5, minimal: bool = False,
              tries: int = 200) -> SynthTask | None:
    """Sample a program of exactly ``length`` steps and the examples that define it.

    Rejects programs that crash, repeat an earlier behaviour (``sort | sort``),
    or give the same answer on every example. With ``minimal`` the task is also
    guaranteed not to be solvable by a program of one or two steps.
    """
    for _ in range(tries):
        inputs = [_random_input(rng) for _ in range(examples)]
        values, kind, program, trace = inputs, "list", [], []
        seen = {repr(inputs)}
        ok = True
        for _step in range(length):
            name = rng.choice([n for n, op in OPS.items() if op.takes == kind])
            try:
                values = [_FN[name](v) for v in values]
            except Exception:
                ok = False
                break
            if repr(values) in seen:
                ok = False
                break
            seen.add(repr(values))
            program.append(name)
            trace.append(values)
            kind = OPS[name].gives
        if not ok or len({repr(v) for v in values}) < max(2, examples - 2):
            continue
        if minimal and length > 2 and repr(values) in _shortest_lengths(inputs):
            continue
        return SynthTask(list(zip(inputs, values)), tuple(program), trace)
    return None


def make_tasks(count: int, seed: int = 0, lengths: Sequence[int] = (1, 2, 3, 4),
               minimal: bool = False) -> list[SynthTask]:
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < count:
        task = make_task(rng, rng.choice(list(lengths)), minimal=minimal)
        if task is not None:
            tasks.append(task)
    return tasks


@dataclass
class Sample:
    features: list[str]
    legal: list[str]
    label: str


def samples_from(examples: Sequence[tuple], program: Sequence[str]) -> list[Sample]:
    """One training sample per step of a known-good program."""
    values = [i for i, _ in examples]
    wanted = [o for _, o in examples]
    out, kind = [], "list" if isinstance(values[0], list) else "int"
    for depth, name in enumerate(program):
        legal = [n for n, op in OPS.items() if op.takes == kind]
        if name not in legal:
            break
        out.append(Sample(featurize(values, wanted, depth, program[depth - 1] if depth else None),
                          legal, name))
        try:
            values = [_FN[name](v) for v in values]
        except Exception:
            break
        kind = OPS[name].gives
    return out


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class SearchPolicy:
    """Multinomial logistic regression over DSL operations, trained by SGD."""

    def __init__(self, ops: Mapping[str, Op] | None = None):
        self.ops = list(ops or OPS)
        self.weights: dict[str, dict[str, float]] = {op: {} for op in self.ops}
        self.trained_on = 0

    # -- inference --------------------------------------------------------- #
    def _scores(self, features: Sequence[str], legal: Sequence[str]) -> list[float]:
        scores = []
        for op in legal:
            w = self.weights.get(op, {})
            scores.append(sum(w.get(f, 0.0) for f in features))
        return scores

    def probabilities(self, features: Sequence[str], legal: Sequence[str]) -> dict[str, float]:
        scores = self._scores(features, legal)
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        return {op: e / total for op, e in zip(legal, exps)}

    def distribution(self, current: Sequence, wanted: Sequence, depth: int, previous: str | None,
                     legal: Sequence[str]) -> dict[str, float]:
        return self.probabilities(featurize(current, wanted, depth, previous), legal)

    # -- learning ---------------------------------------------------------- #
    def _step(self, sample: Sample, lr: float, decay: float) -> float:
        probs = self.probabilities(sample.features, sample.legal)
        for op in sample.legal:
            gradient = probs[op] - (1.0 if op == sample.label else 0.0)
            if abs(gradient) < 1e-4:
                continue
            w = self.weights[op]
            for f in sample.features:
                w[f] = w.get(f, 0.0) * (1.0 - lr * decay) - lr * gradient
        return -math.log(max(probs[sample.label], 1e-12))

    def fit(self, samples: Sequence[Sample], epochs: int = 6, lr: float = 0.15,
            decay: float = 1e-4, seed: int = 0) -> list[float]:
        """Train on ``samples``; returns the mean loss of each epoch."""
        rng = random.Random(seed)
        order = list(samples)
        history = []
        for epoch in range(epochs):
            rng.shuffle(order)
            rate = lr / (1 + 0.5 * epoch)
            history.append(sum(self._step(s, rate, decay) for s in order) / max(1, len(order)))
        self.trained_on += len(order)
        return history

    def update(self, examples: Sequence[tuple], program: Sequence[str], passes: int = 3,
               lr: float = 0.1) -> None:
        """Online learning: a few gradient steps on one task that was really solved."""
        samples = samples_from(examples, program)
        for _ in range(passes):
            for sample in samples:
                self._step(sample, lr, 0.0)
        self.trained_on += len(samples)

    def accuracy(self, samples: Sequence[Sample], top: int = 1) -> float:
        hits = 0
        for s in samples:
            probs = self.probabilities(s.features, s.legal)
            best = sorted(probs, key=probs.get, reverse=True)[:top]
            hits += s.label in best
        return hits / max(1, len(samples))

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"ops": self.ops, "trained_on": self.trained_on,
                       "weights": {op: {f: round(v, 5) for f, v in w.items()}
                                   for op, w in self.weights.items()}}, handle)

    @classmethod
    def load(cls, path: str) -> "SearchPolicy":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        policy = cls({op: OPS[op] for op in data["ops"] if op in OPS})
        policy.weights = {op: dict(data["weights"].get(op, {})) for op in policy.ops}
        policy.trained_on = int(data.get("trained_on", 0))
        return policy


# --------------------------------------------------------------------------- #
# train and measure
# --------------------------------------------------------------------------- #
@dataclass
class TrainingReport:
    tasks: int
    samples: int
    loss: list[float]
    accuracy_before: float
    accuracy_after: float
    top3_after: float

    def __str__(self) -> str:
        return (f"{self.tasks} dreamed tasks, {self.samples} samples; loss "
                f"{self.loss[0]:.3f} -> {self.loss[-1]:.3f}; held-out next-step accuracy "
                f"{self.accuracy_before:.0%} -> {self.accuracy_after:.0%} (top-3 {self.top3_after:.0%})")


def train_policy(tasks: int = 1500, seed: int = 0, epochs: int = 6,
                 policy: SearchPolicy | None = None) -> tuple[SearchPolicy, TrainingReport]:
    """Dream ``tasks`` programs, train on them, and score next-step prediction on held-out ones."""
    policy = policy or SearchPolicy()
    train = [s for t in make_tasks(tasks, seed) for s in samples_from(t.examples, t.program)]
    held_out = [s for t in make_tasks(max(100, tasks // 10), seed + 10_007)
                for s in samples_from(t.examples, t.program)]
    before = policy.accuracy(held_out)
    loss = policy.fit(train, epochs=epochs, seed=seed)
    return policy, TrainingReport(tasks, len(train), loss, before, policy.accuracy(held_out),
                                  policy.accuracy(held_out, top=3))


@dataclass
class Comparison:
    budget: int
    baseline: list[int | None]          # verifications per task; None = unsolved within budget
    trained: list[int | None]

    @staticmethod
    def _solved(row: list) -> int:
        return sum(v is not None for v in row)

    def _mean(self, row: list) -> float:
        return sum(self.budget if v is None else v for v in row) / max(1, len(row))

    @property
    def summary(self) -> dict:
        return {"tasks": len(self.baseline), "budget": self.budget,
                "solved_baseline": self._solved(self.baseline), "solved_trained": self._solved(self.trained),
                "mean_verifications_baseline": round(self._mean(self.baseline), 1),
                "mean_verifications_trained": round(self._mean(self.trained), 1)}

    def __str__(self) -> str:
        s = self.summary
        return (f"solved {s['solved_baseline']}/{s['tasks']} -> {s['solved_trained']}/{s['tasks']} "
                f"within {self.budget} verifications; mean programs executed "
                f"{s['mean_verifications_baseline']} -> {s['mean_verifications_trained']} "
                f"(unsolved counted as {self.budget})")


def compare(policy: SearchPolicy, tasks: Sequence, budget: int = 1500, seed: int = 0) -> Comparison:
    """Search every task twice - without and with the policy - under the same budget."""
    from . import MCTS, PipelineSynthesis

    def solve(examples, guide):
        problem = PipelineSynthesis(examples, policy=guide)
        result = MCTS(problem, seed=seed, max_verifications=budget).search(budget * 4)
        return result.verifications if result.solved else None

    rows = [(solve(getattr(t, "examples", t), None), solve(getattr(t, "examples", t), policy))
            for t in tasks]
    return Comparison(budget, [a for a, _ in rows], [b for _, b in rows])
