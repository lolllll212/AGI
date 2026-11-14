"""Measure an agent, keep a baseline, and notice when it gets worse.

* :class:`Evaluator` runs an agent over a task suite and records, per task,
  whether it succeeded and what it cost (model calls, tool calls, errors).
* Baselines are stored as JSON. :meth:`Evaluator.compare` reports the tasks
  that used to pass and now fail, and the metrics that moved past a tolerance.
* :class:`DriftMonitor` watches live results: it raises an alarm when the
  recent success rate (or cost) differs from the reference period by more than
  chance would explain.

Timings vary from machine to machine, so they are recorded but only compared
when asked for (``Tolerance(seconds=...)``).
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import Field

from agi.domain.value_objects.records import Model
from agi.eval.suite import SUITE, EvalTask, write_fixtures

LOWER_IS_BETTER = ("llm_calls", "tool_calls", "tool_errors", "seconds")


class TaskScore(Model):
    task: str
    success: bool
    grounded: bool | None = None
    mode: str = ""
    stop_reason: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    seconds: float = 0.0
    answer: str | None = None
    error: str | None = None
    skill: str = ""


class Evaluation(Model):
    label: str = ""
    created: float = Field(default_factory=time.time)
    fingerprint: str = ""                # of the configuration that was measured
    repeats: int = 1
    scores: list[TaskScore] = Field(default_factory=list)

    @property
    def tasks(self) -> list[str]:
        return list(dict.fromkeys(s.task for s in self.scores))

    def pass_rates(self) -> dict[str, float]:
        """Share of runs that passed, per task."""
        totals: dict[str, list[int]] = {}
        for score in self.scores:
            totals.setdefault(score.task, []).append(int(score.success))
        return {task: sum(v) / len(v) for task, v in totals.items()}

    @property
    def success_rate(self) -> float:
        return sum(s.success for s in self.scores) / max(1, len(self.scores))

    def metrics(self) -> dict[str, float]:
        n = max(1, len(self.scores))
        judged = [s for s in self.scores if s.grounded is not None]
        return {"success_rate": self.success_rate,
                "grounded_rate": sum(bool(s.grounded) for s in judged) / len(judged) if judged else 1.0,
                "llm_calls": sum(s.llm_calls for s in self.scores) / n,
                "tool_calls": sum(s.tool_calls for s in self.scores) / n,
                "tool_errors": sum(s.tool_errors for s in self.scores) / n,
                "seconds": sum(s.seconds for s in self.scores) / n}

    def cost(self) -> float:
        """Mean model calls plus tool calls per task: what a run costs, independent of the machine."""
        m = self.metrics()
        return m["llm_calls"] + m["tool_calls"]

    def summary(self) -> str:
        m = self.metrics()
        rows = [f"{'PASS' if s.success else 'FAIL'}  {s.task:<18} {s.mode:<10} {s.llm_calls:>2} model  "
                f"{s.tool_calls:>2} tool  {s.tool_errors} err  {s.stop_reason}" for s in self.scores]
        passed = sum(s.success for s in self.scores)
        return ("\n".join(rows) + f"\n{passed}/{len(self.scores)} passed ({m['success_rate']:.0%}), "
                f"{m['llm_calls']:.1f} model calls and {m['tool_calls']:.1f} tool calls per task")


class Tolerance(Model):
    """How much worse a metric may get before it counts as a regression."""

    success_rate: float = 0.0            # absolute drop allowed
    grounded_rate: float = 0.0
    llm_calls: float = 0.25              # relative increase allowed (25%)
    tool_calls: float = 0.25
    tool_errors: float = 0.5
    seconds: float | None = None         # None = never compare timings
    absolute_slack: float = 0.25         # ignore cost increases smaller than this many calls per task


class Change(Model):
    metric: str
    before: float
    after: float
    regression: bool

    @property
    def delta(self) -> float:
        return self.after - self.before


class Comparison(Model):
    baseline: str
    current: str
    newly_failing: list[str] = Field(default_factory=list)
    newly_passing: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)       # in the baseline, not run now
    added: list[str] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    config_changed: bool = False

    @property
    def regressions(self) -> list[Change]:
        return [c for c in self.changes if c.regression]

    @property
    def regressed(self) -> bool:
        return bool(self.newly_failing or self.missing or self.regressions)

    def report(self) -> str:
        lines = [f"{self.current or 'current'} vs baseline {self.baseline or '(unnamed)'}: "
                 + ("REGRESSION" if self.regressed else "ok")]
        if self.config_changed:
            lines.append("  configuration differs from the baseline's")
        for name, tasks in (("now failing", self.newly_failing), ("now passing", self.newly_passing),
                            ("not run", self.missing), ("new", self.added)):
            if tasks:
                lines.append(f"  {name}: {', '.join(tasks)}")
        for change in self.changes:
            timing_only = change.metric == "seconds" and not change.regression      # noise unless it was asked for
            if abs(change.delta) > 1e-9 and not timing_only:
                lines.append(f"  {change.metric}: {change.before:.3g} -> {change.after:.3g}"
                             + ("  << regression" if change.regression else ""))
        return "\n".join(lines)


def fingerprint(config: Any) -> str:
    """A short stable hash of a configuration, to tell 'the agent changed' from 'the world changed'."""
    if config is None:
        return ""
    data = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]


def score_run(task: EvalTask, run: Any, seconds: float) -> TaskScore:
    """Works for both ``Run`` (ReActAgent) and ``EngineResult`` (CognitiveEngine)."""
    return TaskScore(task=task.name, success=bool(getattr(run, "success", False)),
                     grounded=getattr(run, "grounded", None), mode=str(getattr(run, "mode", "react")),
                     stop_reason=str(getattr(run, "stop_reason", "")), llm_calls=int(getattr(run, "llm_calls", 0)),
                     tool_calls=int(getattr(run, "tool_calls", 0)), tool_errors=int(getattr(run, "tool_errors", 0)),
                     seconds=seconds, answer=getattr(run, "answer", None), error=getattr(run, "error", None),
                     skill=task.skill)


class Evaluator:
    """Runs agents over a suite; stores and compares baselines.

    ``make_agent(workspace, task)`` returns anything with
    ``run(prompt, check=...)``. Every task gets a fresh workspace, so one
    task's files (or memories) cannot help another.
    """

    def __init__(self, make_agent: Callable[[Path, EvalTask], Any] | None = None,
                 tasks: Sequence[EvalTask] = tuple(SUITE), store: str | Path | None = None,
                 repeats: int = 1, tolerance: Tolerance | None = None, workdir: str | Path | None = None):
        self.make_agent, self.tasks, self.repeats = make_agent, list(tasks), max(1, repeats)
        self.store = Path(store) if store else None
        self.tolerance = tolerance or Tolerance()
        self.workdir = workdir

    # -- measuring --------------------------------------------------------- #
    def run(self, label: str = "", make_agent: Callable[[Path, EvalTask], Any] | None = None,
            config: Any = None, tasks: Sequence[EvalTask] | None = None) -> Evaluation:
        build = make_agent or self.make_agent
        if build is None:
            raise ValueError("no make_agent was given")
        scores = []
        for task in (self.tasks if tasks is None else tasks):
            for _ in range(self.repeats):
                with tempfile.TemporaryDirectory(dir=self.workdir) as scratch:
                    workspace = write_fixtures(Path(scratch) / "workspace", getattr(task, "files", None))
                    started = time.perf_counter()
                    try:
                        run = build(workspace, task).run(task.prompt, check=lambda a, w=workspace: task.check(a, w))
                        scores.append(score_run(task, run, time.perf_counter() - started))
                    except Exception as exc:               # a crashing agent fails the task, not the evaluation
                        scores.append(TaskScore(task=task.name, success=False, stop_reason="crash", skill=task.skill,
                                                seconds=time.perf_counter() - started,
                                                error=f"{type(exc).__name__}: {exc}"))
        return Evaluation(label=label, fingerprint=fingerprint(config), repeats=self.repeats, scores=scores)

    # -- baselines --------------------------------------------------------- #
    def _path(self, name: str) -> Path:
        if self.store is None:
            raise ValueError("this evaluator has no store directory for baselines")
        if not name or any(c in name for c in "/\\") or name.startswith("."):
            raise ValueError(f"bad baseline name {name!r}")
        return self.store / f"{name}.json"

    def save_baseline(self, evaluation: Evaluation, name: str = "baseline") -> Path:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)                            # never leave a half-written baseline
        with (self.store / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"name": name, "created": evaluation.created, "label": evaluation.label,
                                  "fingerprint": evaluation.fingerprint, **evaluation.metrics()}) + "\n")
        return path

    def load_baseline(self, name: str = "baseline") -> Evaluation | None:
        path = self._path(name)
        if not path.is_file():
            return None
        return Evaluation.model_validate_json(path.read_text(encoding="utf-8"))

    def history(self) -> list[dict]:
        path = self.store / "history.jsonl" if self.store else None
        if path is None or not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -- comparing --------------------------------------------------------- #
    def compare(self, current: Evaluation, baseline: Evaluation, tolerance: Tolerance | None = None) -> Comparison:
        tol = tolerance or self.tolerance
        before, after = baseline.pass_rates(), current.pass_rates()
        shared = [t for t in before if t in after]
        comparison = Comparison(
            baseline=baseline.label, current=current.label,
            newly_failing=[t for t in shared if after[t] < before[t] - 1e-9 and after[t] < 1.0],
            newly_passing=[t for t in shared if after[t] > before[t] + 1e-9],
            missing=[t for t in before if t not in after], added=[t for t in after if t not in before],
            config_changed=bool(baseline.fingerprint and current.fingerprint
                                and baseline.fingerprint != current.fingerprint))
        old, new = baseline.metrics(), current.metrics()
        for metric in ("success_rate", "grounded_rate"):
            worse = new[metric] < old[metric] - getattr(tol, metric) - 1e-9
            comparison.changes.append(Change(metric=metric, before=old[metric], after=new[metric], regression=worse))
        for metric in LOWER_IS_BETTER:
            allowed = getattr(tol, metric)
            worse = (allowed is not None and new[metric] > old[metric] * (1 + allowed) + 1e-9
                     and new[metric] - old[metric] > (tol.absolute_slack if metric != "seconds" else 0.0))
            comparison.changes.append(Change(metric=metric, before=old[metric], after=new[metric], regression=worse))
        return comparison

    def check(self, name: str = "baseline", label: str = "current", config: Any = None,
              make_agent: Callable[[Path, EvalTask], Any] | None = None,
              update: bool = False) -> tuple[Evaluation, Comparison | None]:
        """Evaluate now and compare with the stored baseline.

        With no baseline yet, the current evaluation becomes it. With
        ``update=True`` a run that did not regress replaces the baseline.
        """
        current = self.run(label, make_agent, config)
        baseline = self.load_baseline(name)
        if baseline is None:
            self.save_baseline(current, name)
            return current, None
        comparison = self.compare(current, baseline)
        if update and not comparison.regressed:
            self.save_baseline(current, name)
        return current, comparison


# --------------------------------------------------------------------------- #
# drift
# --------------------------------------------------------------------------- #
class DriftAlarm(Model):
    metric: str
    reference: float
    recent: float
    z: float
    direction: str                       # "worse" | "better"
    samples: int
    at: int                              # how many results had been seen


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class DriftMonitor:
    """Compare the most recent results with a reference period.

    The first ``reference`` results (or a stored baseline) define normal. After
    that, the last ``window`` results are tested against it: a two-proportion
    z-test for the success rate, and a z-test on the mean for each cost metric.
    An alarm needs both statistical evidence (``z_threshold``) and a difference
    large enough to matter (``min_drop`` / ``min_rise``), so a big sample cannot
    turn a trivial change into an alarm.
    """

    def __init__(self, reference: int = 50, window: int = 30, z_threshold: float = 3.0,
                 min_drop: float = 0.1, min_rise: float = 0.25, metrics: Sequence[str] = ("llm_calls", "tool_errors"),
                 cooldown: int | None = None):
        if window < 5 or reference < 5:
            raise ValueError("reference and window need at least 5 results each")
        self.reference_size, self.window = reference, window
        self.z_threshold, self.min_drop, self.min_rise = z_threshold, min_drop, min_rise
        self.metrics = tuple(metrics)
        self.cooldown = window if cooldown is None else cooldown
        self._reference: list[dict] = []
        self._recent: deque[dict] = deque(maxlen=window)
        self.seen = 0
        self.alarms: list[DriftAlarm] = []
        self._quiet_until: dict[str, int] = {}

    @staticmethod
    def _row(result: Any) -> dict:
        if isinstance(result, dict):
            return result
        if isinstance(result, bool):
            return {"success": result}
        return {"success": bool(getattr(result, "success", False)),
                **{m: float(getattr(result, m, 0) or 0) for m in LOWER_IS_BETTER}}

    def set_reference(self, results: Sequence[Any]) -> "DriftMonitor":
        """Use known-good results (for example a stored baseline's scores) as the reference."""
        self._reference = [self._row(r) for r in results]
        self.reference_size = len(self._reference)
        return self

    @property
    def ready(self) -> bool:
        return len(self._reference) >= self.reference_size and len(self._recent) == self.window

    def observe(self, result: Any) -> list[DriftAlarm]:
        """Add one result. Returns the alarms it triggered (usually none)."""
        row = self._row(result)
        self.seen += 1
        if len(self._reference) < self.reference_size:
            self._reference.append(row)
            return []
        self._recent.append(row)
        if not self.ready:
            return []
        raised = [alarm for alarm in self._test() if self.seen >= self._quiet_until.get(alarm.metric, 0)]
        for alarm in raised:
            self._quiet_until[alarm.metric] = self.seen + self.cooldown
        self.alarms.extend(raised)
        return raised

    def _test(self) -> list[DriftAlarm]:
        alarms = []
        ref, recent = self._reference, list(self._recent)
        n1, n2 = len(ref), len(recent)
        p1 = sum(bool(r.get("success")) for r in ref) / n1
        p2 = sum(bool(r.get("success")) for r in recent) / n2
        pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        spread = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / n1 + 1 / n2))
        z = (p2 - p1) / spread
        if abs(z) >= self.z_threshold and abs(p2 - p1) >= self.min_drop:
            alarms.append(DriftAlarm(metric="success_rate", reference=p1, recent=p2, z=z,
                                     direction="worse" if p2 < p1 else "better", samples=n2, at=self.seen))
        for metric in self.metrics:
            a = [float(r[metric]) for r in ref if metric in r]
            b = [float(r[metric]) for r in recent if metric in r]
            if len(a) < 5 or len(b) < 5:
                continue
            m1, m2 = statistics.fmean(a), statistics.fmean(b)
            spread = math.sqrt(statistics.pvariance(a) / len(a) + statistics.pvariance(b) / len(b))
            if spread == 0:
                z = 0.0 if m1 == m2 else math.copysign(math.inf, m2 - m1)
            else:
                z = (m2 - m1) / spread
            if abs(z) >= self.z_threshold and abs(m2 - m1) >= self.min_rise * max(abs(m1), 1.0):
                alarms.append(DriftAlarm(metric=metric, reference=m1, recent=m2, z=z,
                                         direction="worse" if m2 > m1 else "better", samples=len(b), at=self.seen))
        return alarms

    def status(self) -> dict:
        recent = list(self._recent)
        return {"seen": self.seen, "ready": self.ready,
                "reference_success": (sum(bool(r.get("success")) for r in self._reference) / len(self._reference)
                                      if self._reference else None),
                "recent_success": sum(bool(r.get("success")) for r in recent) / len(recent) if recent else None,
                "alarms": len(self.alarms)}


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided p-value that paired wins and losses differ by chance (sign test)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
