"""Tree-of-Thoughts: deliberate search over intermediate reasoning steps.

``dfs`` commits to the most promising thought, and *backtracks* to the last
choice point when a line of reasoning is refuted or fails verification.
``bfs`` keeps the ``beam`` best partial solutions at each depth.
"""
from __future__ import annotations

from .problem import Event, Problem, SearchResult


class TreeOfThoughts:
    def __init__(self, problem: Problem, strategy: str = "dfs", beam: int = 5,
                 max_depth: int = 12, max_verifications: int | None = None):
        if strategy not in ("dfs", "bfs"):
            raise ValueError("strategy must be 'dfs' or 'bfs'")
        self.problem, self.strategy, self.beam = problem, strategy, beam
        self.max_depth, self.max_verifications = max_depth, max_verifications

    def search(self) -> SearchResult:
        root = self.problem.initial()
        self._result = SearchResult(False, root, -1.0, None, nodes=1)
        self._seen = {root}
        (self._dfs(root, 0) if self.strategy == "dfs" else self._bfs(root))
        return self._result

    # -- shared pieces ------------------------------------------------------ #
    def _out_of_budget(self) -> bool:
        limit = self.max_verifications
        return limit is not None and self._result.verifications >= limit

    def _children(self, state) -> list:
        """Feasible, unseen successors of ``state``, most promising first."""
        problem, result, ranked = self.problem, self._result, []
        for action in problem.actions(state):
            child = problem.apply(state, action)
            if child in self._seen:
                continue
            self._seen.add(child)
            result.nodes += 1
            reason = problem.feasible(child)
            if reason is not None:
                result.backtracks += 1
                result.trace.append(Event("refute", problem.describe(child), reason,
                                          action=str(action)))
                continue
            ranked.append((problem.value(child) * problem.prior(state, action), len(ranked), child))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [child for _, _, child in ranked]

    def _check(self, state) -> bool:
        problem, result = self.problem, self._result
        verdict = problem.verify(state)
        result.verifications += 1
        if verdict.score > result.best_score:
            result.best_state, result.best_score, result.verdict = state, verdict.score, verdict
        kind = "solve" if verdict.solved else "verify"
        result.trace.append(Event(kind, problem.describe(state), verdict.detail, verdict.score))
        result.solved = result.solved or verdict.solved
        return verdict.solved

    # -- strategies --------------------------------------------------------- #
    def _dfs(self, state, depth: int) -> bool:
        problem, result = self.problem, self._result
        result.iterations += 1
        if problem.is_terminal(state) or depth >= self.max_depth:
            return self._check(state)
        for child in self._children(state)[: self.beam]:
            if self._out_of_budget():
                return False
            if self._dfs(child, depth + 1):
                return True
        result.backtracks += 1
        result.trace.append(Event("backtrack", problem.describe(state), "no continuation survived"))
        return False

    def _bfs(self, root) -> None:
        problem, result = self.problem, self._result
        frontier = [root]
        for _ in range(self.max_depth):
            result.iterations += 1
            candidates = []
            for state in frontier:
                for child in self._children(state):
                    if problem.is_terminal(child):
                        if self._out_of_budget() or self._check(child):
                            return
                    else:
                        candidates.append(child)
            if not candidates:
                return
            candidates.sort(key=problem.value, reverse=True)
            frontier = candidates[: self.beam]
