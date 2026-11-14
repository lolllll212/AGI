"""System-2 reasoning: generate hypotheses, verify them for real, backtrack on failure."""
from .got import GraphOfThoughts, GraphResult, LLMOperations, Thought
from .logic import ArgumentCheck, Proof, check_argument, consistent, entails, satisfiable
from .mcts import MCTS
from .planner import Execution, Plan, PlanOutcome, PlanProblem, parse_plans, search_plans
from .policy import SearchPolicy, compare, featurize, make_task, make_tasks, samples_from, train_policy
from .problem import Event, Problem, SearchResult, Verdict
from .problems import OPS, CodeRepair, Game24, Op, PipelineSynthesis, eval_exact
from .tot import TreeOfThoughts

__all__ = [
    "MCTS", "TreeOfThoughts", "GraphOfThoughts", "GraphResult", "Thought", "LLMOperations",
    "Problem", "Verdict", "SearchResult", "Event",
    "PipelineSynthesis", "Game24", "CodeRepair", "Op", "OPS", "eval_exact",
    "Plan", "PlanProblem", "PlanOutcome", "Execution", "parse_plans", "search_plans",
    "SearchPolicy", "train_policy", "compare", "featurize", "make_task", "make_tasks", "samples_from",
    "entails", "satisfiable", "consistent", "check_argument", "Proof", "ArgumentCheck",
]
