"""Improve an agent's prompt and settings by measured A/B tests, never by opinion.

The loop is deliberately conservative:

1. Measure the current configuration (the champion) on the task suite.
2. Propose challengers: rule-based :class:`Mutation` objects (change a setting,
   add an instruction) and, optionally, instructions a model writes after
   reading the champion's failures.
3. Measure each challenger on the same tasks. A challenger replaces the champion
   only if it breaks nothing that worked and is strictly better: more tasks
   pass, or the same tasks pass at lower cost.
4. If hold-out tasks were given, the winner must not do worse on them than the
   starting configuration did. That is the guard against tuning to the suite.

Every trial is recorded with its numbers and the reason it was accepted or
rejected, so a change to the prompt can always be traced to its evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import Field

from ..core.llm import extract_json
from ..core.react import SYSTEM
from ..core.types import Model
from .evaluator import Evaluation, Evaluator, fingerprint, mcnemar
from .suite import EvalTask

Build = Callable[[Any, Path, EvalTask], Any]          # (config, workspace, task) -> agent


def get_field(config: Any, path: str) -> Any:
    for part in path.split("."):
        config = getattr(config, part)
    return config


def with_field(config: Any, path: str, value: Any) -> Any:
    """A validated copy of ``config`` with one (possibly nested) field changed."""
    head, _, rest = path.partition(".")
    if head not in type(config).model_fields:
        raise KeyError(f"{type(config).__name__} has no field {head!r}")
    data = config.model_dump()
    data[head] = with_field(getattr(config, head), rest, value).model_dump() if rest else value
    return type(config).model_validate(data)             # validation applies: an out-of-range value raises


class Mutation(Model):
    """One candidate change. ``kind`` is ``set`` (a field) or ``instruct`` (append to the prompt)."""

    name: str
    kind: str = "set"
    field: str = "system_prompt"
    value: Any = None
    why: str = ""

    def apply(self, config: Any) -> Any | None:
        """The changed configuration, or None if this would change nothing or is not valid."""
        try:
            if self.kind == "instruct":
                current = get_field(config, self.field) or SYSTEM
                text = str(self.value).strip()
                if not text or text in current:
                    return None
                return with_field(config, self.field, current.rstrip() + "\n" + text)
            if get_field(config, self.field) == self.value:
                return None
            return with_field(config, self.field, self.value)
        except (KeyError, AttributeError, ValueError):
            return None


def set_field(field: str, value: Any, why: str = "") -> Mutation:
    return Mutation(name=f"{field}={value!r}", kind="set", field=field, value=value, why=why)


def add_instruction(text: str, name: str = "", field: str = "system_prompt", why: str = "") -> Mutation:
    return Mutation(name=name or "instruct: " + text[:48], kind="instruct", field=field, value=text, why=why)


class Trial(Model):
    generation: int
    mutation: str
    accepted: bool
    reason: str
    success_before: float
    success_after: float
    cost_before: float
    cost_after: float
    wins: list[str] = Field(default_factory=list)        # tasks the challenger fixed
    losses: list[str] = Field(default_factory=list)      # tasks the challenger broke
    p_value: float = 1.0                                 # exact sign test on wins vs losses


class Optimization(Model):
    config: Any = None                   # the best configuration found
    changed: bool = False
    baseline: Evaluation
    final: Evaluation
    trials: list[Trial] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)     # accepted changes, in order
    holdout_before: float | None = None
    holdout_after: float | None = None
    evaluations: int = 0

    def report(self) -> str:
        lines = [f"success {self.baseline.success_rate:.0%} -> {self.final.success_rate:.0%}, "
                 f"cost {self.baseline.cost():.2f} -> {self.final.cost():.2f} calls per task, "
                 f"{self.evaluations} evaluation(s)"]
        if self.holdout_before is not None:
            lines.append(f"hold-out {self.holdout_before:.0%} -> {self.holdout_after:.0%}")
        for trial in self.trials:
            lines.append(f"  gen {trial.generation}  {'ACCEPT' if trial.accepted else 'reject'}  {trial.mutation}: "
                         f"{trial.reason}")
        lines.append("kept: " + ("; ".join(self.lineage) if self.lineage else "nothing (the starting configuration stands)"))
        return "\n".join(lines)


REFLECT = """An AI agent failed some tasks. Propose short additions to its instructions that would have
prevented these failures. Each addition must be a general working rule (one sentence), not the answer
to a task, and must not mention the specific files or numbers below.

CURRENT INSTRUCTIONS
{prompt}

FAILURES
{failures}

Reply with JSON only: {{"instructions": ["...", "..."]}}  (at most {limit})"""


def propose_instructions(llm: Callable[[str], str], config: Any, evaluation: Evaluation, limit: int = 3,
                         field: str = "system_prompt") -> list[Mutation]:
    """Ask a model to read the failures and suggest prompt additions. Returns [] on any problem."""
    failed = [s for s in evaluation.scores if not s.success]
    if not failed:
        return []
    listing = "\n".join(f"- task {s.task} ({s.skill or 'general'}): stopped with '{s.stop_reason}', "
                        f"{s.tool_errors} tool error(s); answer: {(s.answer or s.error or 'none')[:200]}"
                        for s in failed[:8])
    try:
        data = extract_json(llm(REFLECT.format(prompt=(get_field(config, field) or SYSTEM)[:3000],
                                               failures=listing, limit=limit)))
    except Exception:
        return []
    items = data.get("instructions", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    seen, mutations = set(), []
    for item in items:
        text = " ".join(str(item).split())
        if 8 <= len(text) <= 400 and text.lower() not in seen:
            seen.add(text.lower())
            mutations.append(add_instruction(text, why="proposed by the model after reading the failures"))
    return mutations[:limit]


class Optimizer:
    def __init__(self, build: Build, tasks: Sequence[EvalTask], mutations: Sequence[Mutation] = (),
                 llm: Callable[[str], str] | None = None, holdout: Sequence[EvalTask] = (),
                 repeats: int = 1, min_cost_gain: float = 0.05, alpha: float | None = None,
                 max_evaluations: int = 40):
        """``alpha``: if set, a success-rate win must also pass an exact sign test at that level.
        On a small suite that is a high bar (five fixed tasks and none broken give p = 0.0625)."""
        self.build, self.tasks, self.holdout = build, list(tasks), list(holdout)
        self.mutations, self.llm = list(mutations), llm
        self.min_cost_gain, self.alpha, self.max_evaluations = min_cost_gain, alpha, max_evaluations
        self.evaluator = Evaluator(tasks=self.tasks, repeats=repeats)
        self.evaluations = 0

    def measure(self, config: Any, label: str = "", tasks: Sequence[EvalTask] | None = None) -> Evaluation:
        self.evaluations += 1
        return self.evaluator.run(label, lambda workspace, task: self.build(config, workspace, task),
                                  config=config, tasks=tasks)

    def ab_test(self, champion: Evaluation, challenger: Evaluation) -> tuple[bool, str, list[str], list[str], float]:
        before, after = champion.pass_rates(), challenger.pass_rates()
        wins = [t for t in before if after.get(t, 0) > before[t] + 1e-9]
        losses = [t for t in before if after.get(t, 0) < before[t] - 1e-9]
        p = mcnemar(len(wins), len(losses))
        if losses:
            return False, f"breaks {', '.join(losses)}", wins, losses, p
        if wins:
            if self.alpha is not None and p > self.alpha:
                return False, f"fixes {', '.join(wins)} but p={p:.3f} is above alpha={self.alpha}", wins, losses, p
            return True, f"fixes {', '.join(wins)}, breaks nothing", wins, losses, p
        saved = champion.cost() - challenger.cost()
        if saved > self.min_cost_gain * max(champion.cost(), 1e-9):
            return True, f"same tasks pass with {saved:.2f} fewer calls per task", wins, losses, p
        return False, "no measurable improvement", wins, losses, p

    def optimize(self, config: Any, generations: int = 3) -> Optimization:
        start = config
        baseline = champion_eval = self.measure(config, "baseline")
        holdout_before = self.measure(config, "holdout", self.holdout).success_rate if self.holdout else None
        trials: list[Trial] = []
        lineage: list[str] = []
        tried: set[str] = {fingerprint(config)}
        for generation in range(1, generations + 1):
            candidates = list(self.mutations)
            if self.llm is not None:
                candidates += propose_instructions(self.llm, config, champion_eval)
            best: tuple[Any, Evaluation, Mutation, Trial] | None = None
            for mutation in candidates:
                if self.evaluations >= self.max_evaluations:
                    break
                challenger = mutation.apply(config)
                if challenger is None or fingerprint(challenger) in tried:
                    continue
                tried.add(fingerprint(challenger))
                evaluation = self.measure(challenger, mutation.name)
                accepted, reason, wins, losses, p = self.ab_test(champion_eval, evaluation)
                trial = Trial(generation=generation, mutation=mutation.name, accepted=False, reason=reason,
                              success_before=champion_eval.success_rate, success_after=evaluation.success_rate,
                              cost_before=champion_eval.cost(), cost_after=evaluation.cost(),
                              wins=wins, losses=losses, p_value=p)
                trials.append(trial)
                if accepted and (best is None or (evaluation.success_rate, -evaluation.cost())
                                 > (best[1].success_rate, -best[1].cost())):
                    best = (challenger, evaluation, mutation, trial)
            if best is None:
                break
            for trial in trials:                         # only the generation's best challenger is adopted
                if trial.generation == generation and trial is not best[3] and "no measurable" not in trial.reason \
                        and not trial.losses and (trial.wins or "fewer calls" in trial.reason):
                    trial.reason += " (a better challenger won this generation)"
            best[3].accepted = True
            config, champion_eval = best[0], best[1]
            lineage.append(best[2].name + (f" - {best[2].why}" if best[2].why else ""))
        holdout_after = None
        if self.holdout:
            holdout_after = self.measure(config, "holdout", self.holdout).success_rate if lineage else holdout_before
            if holdout_after < holdout_before - 1e-9:
                trials.append(Trial(generation=0, mutation="hold-out check", accepted=False,
                                    reason=f"hold-out success fell from {holdout_before:.0%} to {holdout_after:.0%}: "
                                           "all changes discarded as over-fitted to the suite",
                                    success_before=holdout_before, success_after=holdout_after,
                                    cost_before=baseline.cost(), cost_after=champion_eval.cost()))
                for trial in trials:
                    trial.accepted = False
                config, champion_eval, lineage = start, baseline, []
        if lineage and "notes" in getattr(type(config), "model_fields", {}):
            config = with_field(config, "notes", list(config.notes) + lineage)
        return Optimization(config=config, changed=bool(lineage), baseline=baseline, final=champion_eval,
                            trials=trials, lineage=lineage, holdout_before=holdout_before,
                            holdout_after=holdout_after, evaluations=self.evaluations)


def save_config(config: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_config(kind: type, path: str | Path) -> Any:
    return kind.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
