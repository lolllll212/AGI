"""Monte Carlo Tree Search whose rewards come from verification, not from a guess.

Differences from the textbook game-playing version:

* the reward of a simulation is ``problem.verify(...)`` - an actual test run;
* a partial hypothesis refuted by ``problem.feasible`` (or by a verdict with
  ``refuted=True``) is marked *dead*: the search backtracks and never spends
  another simulation beneath it;
* ``problem.prior`` (e.g. lessons recalled from memory) decides which
  hypotheses are tried first and scales their exploration bonus.
"""
from __future__ import annotations

import math
import random
from typing import Any

from .problem import Event, Problem, SearchResult, Verdict


class Node:
    __slots__ = ("state", "parent", "action", "prior", "children", "untried", "visits",
                 "total", "dead")

    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state, self.parent, self.action, self.prior = state, parent, action, prior
        self.children: list[Node] = []
        self.untried: list | None = None      # filled lazily, best prior last (pop())
        self.visits = 0
        self.total = 0.0
        self.dead = False

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, problem: Problem, exploration: float = 0.7, rollout_depth: int = 0,
                 seed: int = 0, max_verifications: int | None = None):
        """``rollout_depth=0`` tests each new node's own hypothesis; a positive
        depth first extends it with that many prior-weighted random steps."""
        self.problem = problem
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.max_verifications = max_verifications
        self.rng = random.Random(seed)
        self._verdicts: dict[Any, Verdict] = {}

    def search(self, iterations: int = 200) -> SearchResult:
        problem = self.problem
        root = Node(problem.initial())
        result = SearchResult(False, root.state, -1.0, None, nodes=1)

        for i in range(iterations):
            result.iterations = i + 1
            if root.dead or (self.max_verifications is not None
                             and result.verifications >= self.max_verifications):
                break
            node = self._select(root)
            child = self._expand(node, result)
            if child is None:
                continue
            state = self._rollout(child.state)
            verdict = self._verify(state, result)
            if verdict.score > result.best_score:
                result.best_state, result.best_score, result.verdict = state, verdict.score, verdict
            if verdict.solved:
                result.solved = True
                result.trace.append(Event("solve", problem.describe(state), verdict.detail, 1.0))
                self._backpropagate(child, verdict.score)
                break
            if verdict.refuted and state == child.state:
                self._kill(child, verdict.detail, result)
            self._backpropagate(child, verdict.score)
        return result

    # -- the four phases ---------------------------------------------------- #
    def _select(self, node: Node) -> Node:
        """Descend by UCT (prior-weighted) to a node that still has untried actions."""
        while True:
            if node.untried is None:
                actions = list(self.problem.actions(node.state))
                self.rng.shuffle(actions)
                weights = [max(1e-6, self.problem.prior(node.state, a)) for a in actions]
                mean = sum(weights) / len(weights) if weights else 1.0
                ranked = sorted(zip(weights, range(len(actions))))        # best prior last
                node.untried = [(actions[i], w / mean) for w, i in ranked]
            alive = [c for c in node.children if not c.dead]
            if not alive:
                return node
            log_n = math.log(node.visits + 1)

            def ucb(mean: float, prior: float, visits: int) -> float:
                bonus = min(4.0, max(0.25, prior)) * math.sqrt(log_n / (visits + 1))
                return mean + self.exploration * bonus

            best = max(alive, key=lambda c: ucb(c.mean, c.prior, c.visits))
            if node.untried:
                # An untried action competes as a virtual child valued like its parent,
                # so a strongly preferred or clearly improving child is followed
                # *before* every sibling has been tried.
                if ucb(node.mean, node.untried[-1][1], 0) >= ucb(best.mean, best.prior, best.visits):
                    return node
            node = best

    def _expand(self, node: Node, result: SearchResult) -> Node | None:
        problem = self.problem
        if not node.untried:                       # exhausted subtree: backtrack
            self._kill(node, "every continuation has been tried", result, kind="backtrack")
            return None
        action, prior = node.untried.pop()
        child = Node(problem.apply(node.state, action), node, action, prior)
        node.children.append(child)
        result.nodes += 1
        reason = problem.feasible(child.state)
        if reason is not None:
            self._kill(child, reason, result)
            self._backpropagate(child, 0.0)
            return None
        return child

    def _rollout(self, state):
        problem = self.problem
        for _ in range(self.rollout_depth):
            if problem.is_terminal(state):
                break
            options = [a for a in problem.actions(state)]
            options = [(a, problem.apply(state, a)) for a in options]
            options = [(a, s) for a, s in options if problem.feasible(s) is None]
            if not options:
                break
            weights = [max(1e-6, problem.prior(state, a)) for a, _ in options]
            state = self.rng.choices([s for _, s in options], weights)[0]
        return state

    def _verify(self, state, result: SearchResult) -> Verdict:
        if state not in self._verdicts:
            self._verdicts[state] = verdict = self.problem.verify(state)
            result.verifications += 1
            result.trace.append(Event("verify", self.problem.describe(state), verdict.detail,
                                      verdict.score))
        return self._verdicts[state]

    def _kill(self, node: Node, reason: str, result: SearchResult, kind: str = "refute") -> None:
        """Mark a hypothesis dead; ancestors left with nothing alive die too."""
        node.dead = True
        node.untried = []
        result.backtracks += 1
        result.trace.append(Event(kind, self.problem.describe(node.state), reason,
                                  action=str(node.action)))
        parent = node.parent
        if parent is not None and parent.untried == [] and all(c.dead for c in parent.children):
            self._kill(parent, "every continuation was refuted", result, kind="backtrack")

    @staticmethod
    def _backpropagate(node: Node | None, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total += reward
            node = node.parent
