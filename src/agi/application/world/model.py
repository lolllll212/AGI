"""A world model learned from consequences.

The model starts from an optional *prior* - what the agent believes before
acting, e.g. a textbook formula or a language model's guess - and learns the
*residual* between that belief and what actually happened. Predictions are
therefore "belief + correction from experience", and every new observation
that surprises the model makes the next prediction better.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from agi.application.world.env import Environment


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` if singular."""
    n = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    solution = [0.0] * n
    for r in range(n - 1, -1, -1):
        rest = sum(rows[r][c] * solution[c] for c in range(r + 1, n))
        solution[r] = (rows[r][n] - rest) / rows[r][r]
    return solution


class LocalRegressor:
    """Locally weighted linear regression over remembered ``(x, y)`` pairs.

    Non-parametric: it assumes nothing about the shape of the world, only that
    nearby actions have similar consequences.
    """

    def __init__(self, neighbours: int = 8, ridge: float = 1e-3):
        self.neighbours, self.ridge = neighbours, ridge
        self.points: list[tuple[tuple[float, ...], float]] = []

    def add(self, x: Sequence[float], y: float) -> None:
        self.points.append((tuple(x), float(y)))

    def predict(self, x: Sequence[float]) -> tuple[float, float]:
        """``(estimate, distance_to_nearest_experience)``; the latter is the uncertainty."""
        if not self.points:
            return 0.0, math.inf
        ranked = sorted((math.dist(x, px), px, py) for px, py in self.points)[: self.neighbours]
        nearest = ranked[0][0]
        bandwidth = max(ranked[-1][0], 1e-6)
        weights = [math.exp(-((d / bandwidth) ** 2) * 2.0) for d, _, _ in ranked]
        dims = len(x) + 1
        if len(ranked) > dims:        # enough neighbours for a local linear fit
            gram = [[self.ridge if i == j else 0.0 for j in range(dims)] for i in range(dims)]
            moment = [0.0] * dims
            for w, (_, px, py) in zip(weights, ranked):
                feats = (1.0, *(a - b for a, b in zip(px, x)))     # centred on the query
                for i in range(dims):
                    moment[i] += w * feats[i] * py
                    for j in range(dims):
                        gram[i][j] += w * feats[i] * feats[j]
            beta = _solve(gram, moment)
            if beta is not None:
                return beta[0], nearest
        total = sum(weights)
        return sum(w * py for w, (_, _, py) in zip(weights, ranked)) / total, nearest


@dataclass
class Experience:
    action: dict
    predicted: float
    observed: float
    target: float | None = None

    @property
    def surprise(self) -> float:
        """How wrong the model was about the consequence of its own action."""
        return abs(self.predicted - self.observed)

    @property
    def miss(self) -> float | None:
        return None if self.target is None else abs(self.observed - self.target)


class WorldModel:
    """Predict one observed quantity from an action; improve by being wrong."""

    def __init__(self, bounds: Mapping[str, tuple[float, float]],
                 prior: Callable[[Mapping[str, float]], float] | None = None,
                 neighbours: int = 8):
        self.bounds = dict(bounds)
        self.prior = prior or (lambda action: 0.0)
        self.residual = LocalRegressor(neighbours)
        self.experience: list[Experience] = []

    def _features(self, action: Mapping[str, float]) -> list[float]:
        return [(action[name] - lo) / (hi - lo) for name, (lo, hi) in self.bounds.items()]

    def predict(self, action: Mapping[str, float]) -> float:
        return self.prior(action) + self.residual.predict(self._features(action))[0]

    def uncertainty(self, action: Mapping[str, float]) -> float:
        return self.residual.predict(self._features(action))[1]

    def update(self, action: Mapping[str, float], observed: float,
               target: float | None = None) -> Experience:
        """Record a real consequence. Returns the experience, including the surprise."""
        record = Experience(dict(action), self.predict(action), float(observed), target)
        self.residual.add(self._features(action), observed - self.prior(action))
        self.experience.append(record)
        return record

    def plan(self, target: float, rng: random.Random | None = None, samples: int = 600,
             caution: float = 1.0) -> dict:
        """Pick the action whose *predicted* consequence is closest to ``target``.

        ``caution`` > 0 prefers actions near past experience, where predictions
        have been corrected by reality; ``0`` trusts the model everywhere.
        """
        rng = rng or random.Random(0)
        scale = caution * (abs(target) or 1.0)

        def cost(action: dict) -> float:
            penalty = scale * min(self.uncertainty(action), 2.0) if caution else 0.0
            return abs(self.predict(action) - target) + penalty

        def clamp(action: dict) -> dict:
            return {k: min(max(v, self.bounds[k][0]), self.bounds[k][1]) for k, v in action.items()}

        pool = [{k: rng.uniform(lo, hi) for k, (lo, hi) in self.bounds.items()}
                for _ in range(samples)]
        pool += [e.action for e in self.experience[-5:]]
        best = min(pool, key=cost)
        step = 0.1
        for _ in range(60):                      # local refinement around the best sample
            trial = clamp({k: v + rng.gauss(0.0, step * (self.bounds[k][1] - self.bounds[k][0]))
                           for k, v in best.items()})
            if cost(trial) < cost(best):
                best = trial
            else:
                step *= 0.93
        return best


class GroundedAgent:
    """The loop that grounds a model: predict -> act -> observe -> correct.

    With no prior belief there is nothing to plan from, so the first
    ``warmup`` actions are random exploration ("motor babbling").
    """

    def __init__(self, env: Environment, model: WorldModel, observe: str, seed: int = 0,
                 warmup: int = 0):
        self.env, self.model, self.observe, self.warmup = env, model, observe, warmup
        self.rng = random.Random(seed)

    def attempt(self, target: float, caution: float = 1.0) -> Experience:
        if len(self.model.experience) < self.warmup:
            action = {k: self.rng.uniform(lo, hi) for k, (lo, hi) in self.model.bounds.items()}
        else:
            action = self.model.plan(target, self.rng, caution=caution)
        outcome = self.env.step(action).observation[self.observe]
        return self.model.update(action, outcome, target)

    def practice(self, target: float, trials: int, caution: float = 1.0) -> list[Experience]:
        return [self.attempt(target, caution) for _ in range(trials)]
