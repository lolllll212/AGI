"""System-2 planning: search over candidate plans, judged by actually carrying them out.

A model proposes several plans. Each is checked cheaply first (does it only
use tools that exist?), then *executed*; a plan that fails is not argued
with - the search backtracks, and its children are revised plans written with
the real failure in hand. The search itself is the existing MCTS or
Tree-of-Thoughts machinery: this module only defines the problem.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

from pydantic import Field

from agi.application.reasoning.mcts import MCTS
from agi.application.reasoning.tot import TreeOfThoughts
from agi.domain.entities.problem import Problem, SearchResult, Verdict
from agi.domain.services.text import extract_json
from agi.domain.value_objects.records import Model


class Plan(Model):
    steps: list[str] = Field(min_length=1)
    rationale: str = ""
    tools: list[str] = Field(default_factory=list)       # tools the plan says it will use

    def render(self) -> str:
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.steps, 1))

    def key(self) -> tuple:
        return tuple(re.sub(r"\W+", " ", s.lower()).strip() for s in self.steps)

    def __hash__(self) -> int:                              # plans are search states: same steps, same plan
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Plan) and self.key() == other.key()


class Execution(Model):
    """What happened when a plan was carried out."""

    success: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: str = ""                    # the failure, in the words the next plan should hear
    payload: dict = Field(default_factory=dict)


PROPOSE = """You are planning how to solve a task. Do not solve it yet.

TASK: {task}

TOOLS AVAILABLE
{tools}
{context}
Write {k} different plans. Each plan is a short list of concrete steps; name the tool each step uses.
Make the plans genuinely different (different tools or a different order), not rewordings.
Reply with JSON only:
{{"plans": [{{"steps": ["..."], "tools": ["tool_name"], "rationale": "why this should work"}}]}}"""

REVISE = """A plan for this task was carried out and failed.

TASK: {task}

TOOLS AVAILABLE
{tools}

THE PLAN THAT FAILED
{plan}

WHAT ACTUALLY HAPPENED
{failure}

Write {k} revised plans that avoid this failure. Reply with JSON only:
{{"plans": [{{"steps": ["..."], "tools": ["tool_name"], "rationale": "what is different this time"}}]}}"""


def parse_plans(reply: str) -> list[Plan]:
    """Plans from a model reply; malformed entries are skipped, not fatal."""
    try:
        data = extract_json(reply)
    except ValueError:
        return []
    items = data.get("plans", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    plans, seen = [], set()
    for item in items:
        if isinstance(item, list):
            item = {"steps": item}
        if not isinstance(item, dict):
            continue
        steps = [str(s).strip() for s in item.get("steps", []) if str(s).strip()]
        if not steps:
            continue
        plan = Plan(steps=steps[:12], rationale=str(item.get("rationale", ""))[:400],
                    tools=[str(t) for t in item.get("tools", []) if isinstance(t, (str, int))])
        if plan.key() not in seen:
            seen.add(plan.key())
            plans.append(plan)
    return plans


class PlanProblem(Problem):
    """States are chains of plans: ``(plan,)`` is a first attempt, ``(plan, revision)`` a repair of it."""

    def __init__(self, task: str, llm: Callable[[str], str], execute: Callable[[Plan, Sequence[Plan]], Execution],
                 tools: str = "", tool_names: Sequence[str] = (), candidates: int = 3, revisions: int = 2,
                 max_depth: int = 2, context: str = "", critic: Callable[[Plan], float] | None = None):
        self.task, self.llm, self.execute = task, llm, execute
        self.tools, self.tool_names = tools, set(tool_names)
        self.candidates, self.revisions, self.max_depth = candidates, revisions, max_depth
        self.context, self.critic = context, critic
        self._children: dict[tuple, list[Plan]] = {}
        self._failures: dict[tuple, str] = {}
        self.executions: dict[tuple, Execution] = {}
        self.llm_errors: list[str] = []

    # -- proposing --------------------------------------------------------- #
    def _propose(self, state: tuple) -> list[Plan]:
        if state in self._children:
            return self._children[state]
        if not state:
            prompt = PROPOSE.format(task=self.task, tools=self.tools or "(none)", k=self.candidates,
                                    context=f"\nWHAT MEMORY SAYS\n{self.context}\n" if self.context else "")
        else:
            prompt = REVISE.format(task=self.task, tools=self.tools or "(none)", plan=state[-1].render(),
                                   failure=self._failures.get(state, "it did not produce a verified answer"),
                                   k=self.revisions)
        try:
            plans = parse_plans(self.llm(prompt))
        except Exception as exc:                            # the model failed: this branch has no children
            self.llm_errors.append(f"{type(exc).__name__}: {exc}")
            plans = []
        tried = {plan.key() for plan in state}
        plans = [p for p in plans if p.key() not in tried][: self.candidates if not state else self.revisions]
        self._children[state] = plans
        return plans

    # -- Problem interface ------------------------------------------------- #
    def initial(self) -> tuple:
        return ()

    def actions(self, state: tuple) -> list[int]:
        if len(state) >= self.max_depth:
            return []
        if state and state not in self._failures:           # only a plan that failed gets revised
            return []
        return list(range(len(self._propose(state))))

    def apply(self, state: tuple, action: int) -> tuple:
        return (*state, self._propose(state)[action])

    def is_terminal(self, state: tuple) -> bool:
        return len(state) >= self.max_depth

    def feasible(self, state: tuple) -> str | None:
        if not state or not self.tool_names:
            return None
        unknown = [t for t in state[-1].tools if t not in self.tool_names]
        return f"the plan relies on tools that do not exist: {', '.join(unknown)}" if unknown else None

    def prior(self, state: tuple, action: int) -> float:
        plan = self._propose(state)[action]
        base = self.critic(plan) if self.critic else 1.0
        return max(0.05, base) * (0.95 ** action)           # the model's own ordering is a weak prior

    def value(self, state: tuple) -> float:
        if not state:
            return 0.5
        return self.critic(state[-1]) if self.critic else 0.5

    def verify(self, state: tuple) -> Verdict:
        if not state:
            return Verdict(0.0, detail="no plan yet")
        if state not in self.executions:
            self.executions[state] = self.execute(state[-1], state[:-1])
        done = self.executions[state]
        if done.success:
            return Verdict(1.0, solved=True, detail=done.detail or "the plan produced a verified answer")
        self._failures[state] = done.detail or "it did not produce a verified answer"
        return Verdict(min(done.score, 0.99), detail=self._failures[state])

    def describe(self, state: tuple) -> str:
        return " => ".join(" / ".join(p.steps)[:80] for p in state) or "(no plan)"


class PlanOutcome(Model):
    solved: bool
    plan: Plan | None = None
    execution: Execution | None = None
    plans_tried: int = 0
    plans_refuted: int = 0
    backtracks: int = 0
    trace: list[str] = Field(default_factory=list)


def search_plans(problem: PlanProblem, max_executions: int = 3, strategy: str = "mcts", seed: int = 0) -> PlanOutcome:
    """Search for a plan that works, executing at most ``max_executions`` of them."""
    if strategy == "mcts":
        result: SearchResult = MCTS(problem, seed=seed, max_verifications=max_executions).search(
            iterations=max_executions * 4 + 4)
    elif strategy == "tot":
        problem.max_depth = 1                               # ToT verifies leaves: one layer of whole plans
        result = TreeOfThoughts(problem, "dfs", beam=problem.candidates, max_depth=1,
                                max_verifications=max_executions).search()
    else:
        raise ValueError("strategy must be 'mcts' or 'tot'")
    best = result.best_state if result.best_state else None
    return PlanOutcome(
        solved=result.solved, plan=best[-1] if best else None,
        execution=problem.executions.get(best) if best else None,
        plans_tried=len(problem.executions), backtracks=result.backtracks,
        plans_refuted=sum(1 for e in result.trace if e.kind == "refute"),
        trace=[f"{e.kind}: {e.state[:100]}" + (f" - {e.detail[:160]}" if e.detail else "") for e in result.trace])
