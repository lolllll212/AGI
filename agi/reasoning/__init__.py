"""System-2 reasoning: generate hypotheses, verify them for real, backtrack on failure."""
from .logic import ArgumentCheck, Proof, check_argument, consistent, entails, satisfiable
from .mcts import MCTS
from .problem import Event, Problem, SearchResult, Verdict
from .problems import OPS, CodeRepair, Game24, Op, PipelineSynthesis, eval_exact
from .tot import TreeOfThoughts

__all__ = [
    "MCTS", "TreeOfThoughts", "Problem", "Verdict", "SearchResult", "Event",
    "PipelineSynthesis", "Game24", "CodeRepair", "Op", "OPS", "eval_exact",
    "entails", "satisfiable", "consistent", "check_argument", "Proof", "ArgumentCheck",
]
