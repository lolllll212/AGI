"""Causal discovery by experiment.

Watching the world shows what goes together. Only *acting* on it shows what
causes what. :class:`CausalLearner` runs controlled experiments - set one
variable by force, hold the others fixed, see what moves - and builds a causal
graph from the results. It also reports correlations that turned out *not* to
be causal, which is exactly the mistake a purely statistical learner makes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


class SCMWorld:
    """A world defined by structural equations, some of whose variables are hidden.

    ``equations`` maps each variable to ``f(values, rng)`` and must be listed
    causes-first. ``sample(do={...})`` runs the world once, overriding the
    equations of intervened variables (Pearl's do-operator).
    """

    def __init__(self, equations: Mapping[str, Callable], hidden: Sequence[str] = (),
                 seed: int = 0):
        self.equations = dict(equations)
        self.hidden = set(hidden)
        self.rng = random.Random(seed)
        self.samples_drawn = 0

    @property
    def variables(self) -> list[str]:
        return [name for name in self.equations if name not in self.hidden]

    def sample(self, do: Mapping[str, float] | None = None) -> dict:
        do = do or {}
        if self.hidden & set(do):
            raise ValueError("cannot intervene on a hidden variable")
        values: dict = {}
        for name, equation in self.equations.items():
            values[name] = do[name] if name in do else equation(values, self.rng)
        self.samples_drawn += 1
        return {name: values[name] for name in self.variables}


@dataclass
class CausalGraph:
    variables: list
    direct: dict = field(default_factory=dict)       # (cause, effect) -> effect size
    total: dict = field(default_factory=dict)        # (cause, effect) -> effect size, any path
    spurious: list = field(default_factory=list)     # (a, b, correlation) with no causal link

    def parents(self, variable: str) -> list[str]:
        return sorted(c for c, e in self.direct if e == variable)

    def children(self, variable: str) -> list[str]:
        return sorted(e for c, e in self.direct if c == variable)

    def causes(self, cause: str, effect: str) -> bool:
        """True if intervening on ``cause`` changes ``effect`` (through any path)."""
        return (cause, effect) in self.total

    def explain(self) -> str:
        lines = [f"{c} -> {e}   (effect {size:+.2f})" for (c, e), size in sorted(self.direct.items())]
        lines += [f"{a} ~ {b}   correlated (r={r:+.2f}) but NOT causal: forcing either leaves "
                  f"the other unchanged" for a, b, r in self.spurious]
        return "\n".join(lines) or "(no causal structure found)"


def _mean_var(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx, vx = _mean_var(list(xs))
    my, vy = _mean_var(list(ys))
    if vx == 0 or vy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
    return cov / math.sqrt(vx * vy)


class CausalLearner:
    """Learn a causal graph from interventions on anything with ``sample(do=...)``."""

    def __init__(self, world, levels: Mapping[str, tuple] | None = None, trials: int = 120,
                 contexts: int = 6, z_threshold: float = 4.0, min_effect: float = 0.1,
                 seed: int = 0):
        self.world = world
        self.variables = list(world.variables)
        #: The two values each variable is forced to in experiments (default: off / on).
        self.levels = {v: (0, 1) for v in self.variables} | dict(levels or {})
        self.trials, self.contexts = trials, contexts
        self.z_threshold, self.min_effect = z_threshold, min_effect
        self.rng = random.Random(seed)

    def observe(self, n: int = 500) -> list[dict]:
        """Passive observation: no intervention at all."""
        return [self.world.sample() for _ in range(n)]

    def correlations(self, n: int = 500) -> dict:
        rows = self.observe(n)
        return {(a, b): pearson([r[a] for r in rows], [r[b] for r in rows])
                for i, a in enumerate(self.variables) for b in self.variables[i + 1:]}

    def experiment(self, cause: str, effect: str, context: Mapping | None = None) -> tuple[float, float]:
        """Force ``cause`` low then high (holding ``context`` fixed) and compare ``effect``.

        Returns ``(difference_in_means, z_score)``.
        """
        low, high = self.levels[cause]
        fixed = dict(context or {})
        a = [self.world.sample({**fixed, cause: low})[effect] for _ in range(self.trials)]
        b = [self.world.sample({**fixed, cause: high})[effect] for _ in range(self.trials)]
        (mean_a, var_a), (mean_b, var_b) = _mean_var(a), _mean_var(b)
        delta = mean_b - mean_a
        spread = math.sqrt(var_a / len(a) + var_b / len(b))
        return delta, (abs(delta) / spread if spread > 0 else (math.inf if delta else 0.0))

    def _significant(self, delta: float, z: float) -> bool:
        return z >= self.z_threshold and abs(delta) >= self.min_effect

    def discover(self, observations: int = 500) -> CausalGraph:
        graph = CausalGraph(self.variables)
        for cause in self.variables:
            for effect in self.variables:
                if cause == effect:
                    continue
                # Total effect: intervene on the cause only, let everything else run.
                delta, z = self.experiment(cause, effect)
                if self._significant(delta, z):
                    graph.total[(cause, effect)] = delta
                # Direct effect: hold every other variable fixed, in several contexts,
                # so influence that flows through a mediator is blocked.
                others = [v for v in self.variables if v not in (cause, effect)]
                strongest = 0.0
                for _ in range(self.contexts):
                    context = {v: self.rng.choice(self.levels[v]) for v in others}
                    delta, z = self.experiment(cause, effect, context)
                    if self._significant(delta, z) and abs(delta) > abs(strongest):
                        strongest = delta
                if strongest:
                    graph.direct[(cause, effect)] = strongest
        for (a, b), r in self.correlations(observations).items():
            if abs(r) >= 0.3 and (a, b) not in graph.total and (b, a) not in graph.total:
                graph.spurious.append((a, b, r))
        return graph
