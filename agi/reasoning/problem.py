"""The contract between a search procedure and the thing being reasoned about.

A *state* is a (possibly partial) hypothesis. The search proposes ways to
extend it (``actions``), and - crucially - never takes its own word for
whether a hypothesis is right: ``feasible`` is a cheap formal check that can
refute a partial hypothesis early, and ``verify`` is the ground-truth test
(run the code, check the proof, evaluate the expression).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


@dataclass(frozen=True)
class Verdict:
    """Outcome of testing a hypothesis against reality."""

    score: float                 # 0..1, partial credit guides the search
    solved: bool = False
    detail: str = ""             # e.g. the error message; this is what memory learns from
    refuted: bool = False        # the hypothesis *and everything built on it* is dead


class Problem(Generic[S, A]):
    """Subclass and override; defaults make the optional hooks no-ops."""

    def initial(self) -> S:
        raise NotImplementedError

    def actions(self, state: S) -> Sequence[A]:
        """Candidate next steps from ``state``. Empty means dead end."""
        raise NotImplementedError

    def apply(self, state: S, action: A) -> S:
        raise NotImplementedError

    def verify(self, state: S) -> Verdict:
        """Expensive ground-truth check. Called on states the search wants tested."""
        raise NotImplementedError

    def is_terminal(self, state: S) -> bool:
        """True if ``state`` cannot be extended further."""
        return False

    def feasible(self, state: S) -> str | None:
        """Cheap check. Return a reason if ``state`` is already refuted, else ``None``."""
        return None

    def prior(self, state: S, action: A) -> float:
        """Relative plausibility of ``action`` (learned or heuristic). Higher = try sooner."""
        return 1.0

    def value(self, state: S) -> float:
        """Cheap 0..1 estimate of how promising ``state`` is (used by Tree-of-Thoughts)."""
        return 0.5

    def describe(self, state: S) -> str:
        return str(state)


@dataclass
class Event:
    """One entry of the reasoning trace."""

    kind: str          # "verify" | "refute" | "backtrack" | "solve"
    state: str
    detail: str = ""
    score: float = 0.0
    action: str = ""


@dataclass
class SearchResult:
    solved: bool
    best_state: Any
    best_score: float
    verdict: Verdict | None
    iterations: int = 0
    verifications: int = 0
    backtracks: int = 0
    nodes: int = 0
    trace: list = field(default_factory=list)

    def summary(self) -> str:
        status = "SOLVED" if self.solved else f"unsolved (best score {max(0.0, self.best_score):.2f})"
        return (f"{status} after {self.verifications} verifications, {self.backtracks} backtracks, "
                f"{self.nodes} nodes explored")
