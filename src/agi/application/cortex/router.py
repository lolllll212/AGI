"""System 1: decide, without calling a model, how much thinking a task needs.

The router returns one of three modes:

* ``reflex`` - answer straight away: a canned conversational reply, a direct
  tool call for a recognised pattern (plain arithmetic, the time), or a
  **cached action policy** - a tool trajectory that solved the same kind of task
  before, rewritten for the new values. No reasoning model is consulted.
* ``react`` - an ordinary reason-act-observe loop.
* ``deliberate`` - System 2: search over plans, with execution as the judge.

Escalation is by evidence: signals in the task (several steps, code, a
judgement call, length), and experience (a similar task failed before, or a
cached policy has stopped being reliable).

Experience is kept at two levels. Per kind of task (:class:`RouteLearner`) the
router counts how the reason-act loop did, and follows the count once it has
enough. Across kinds (:class:`RouteModel`) a small model learns which signals
and words go with failure, so that a kind seen only a few times can borrow from
the others; the model is listened to only while its own forecasts, each made
before the outcome was known, have beaten guessing the average.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from typing import Callable, Sequence

from agi.domain.entities.tool import ToolRegistry
from agi.domain.value_objects.records import Model, Route, ToolCall

_SMALL_TALK: list[tuple[str, re.Pattern, str]] = [
    ("greeting", re.compile(r"^(hi|hello|hey|namaste|good (morning|afternoon|evening))[\s!.,]*$", re.I),
     "Hello! Give me a task - a question to answer, data to analyse, or code to write - and I will work on it."),
    ("thanks", re.compile(r"^(thanks|thank you|thx|cheers|much appreciated)[\s!.,a-z]*$", re.I),
     "You're welcome."),
    ("farewell", re.compile(r"^(bye|goodbye|see you|good night)[\s!.,a-z]*$", re.I), "Goodbye."),
    ("acknowledgement", re.compile(r"^(ok|okay|got it|great|cool|nice|understood)[\s!.,]*$", re.I),
     "Understood. What would you like me to do next?"),
]
_IDENTITY = re.compile(r"^(who|what) are you\??$|^what can you do\??$|^help[\s!?.]*$|^what tools do you have\??$", re.I)
_TIME = re.compile(r"^(what(?:'s| is)? (?:the )?(?:current )?(time|date)(?: (?:is it|now|today))?|"
                   r"what day is (?:it|today)|what(?:'s| is) today'?s date)\s*\??$", re.I)
_ARITHMETIC_LEAD = re.compile(r"^(?:what(?:'s| is)|calculate|compute|evaluate|how much is)\s+", re.I)
_EXPRESSION = re.compile(r"^[\d\s.+\-*/%^()]+$")

SIGNALS: dict[str, re.Pattern] = {
    "multi_step": re.compile(r"\b(and then|after that|afterwards|first\b.*\bthen|step by step|followed by|"
                             r"once (?:that|you)|finally)\b", re.I | re.S),
    "coding": re.compile(r"\b(write (?:a|an|the) (?:python )?(?:function|class|script|program)|implement|debug|"
                         r"fix (?:the|this|a) (?:bug|code|error|function)|refactor|traceback|unit tests?|"
                         r"stack trace|algorithm)\b", re.I),
    "judgement": re.compile(r"\b(should (?:i|we)|which (?:is|would be) (?:better|best)|trade-?offs?|compare|"
                            r"recommend|pros and cons|design|strategy|optimi[sz]e|plan (?:for|how|a|the))\b", re.I),
    "ambiguity": re.compile(r"\b(somehow|whatever|appropriate|as needed|if necessary|figure out|not sure|"
                            r"either|it depends)\b", re.I),
    "analysis": re.compile(r"\b(why|explain why|root cause|investigate|diagnose|reconcile|verify that)\b", re.I),
}
_STRONG = {"coding", "multi_step"}


class RouterConfig(Model):
    reflex_confidence: float = 0.55      # minimum confidence to act on a cached policy
    deliberate_signals: int = 2          # this many weak signals -> deliberate
    long_task_words: int = 60
    use_cached_policies: bool = True
    small_talk: bool = True
    learn_routes: bool = True            # adjust react/deliberate choices from how runs actually went
    min_observations: int = 8            # outcomes for a kind of task before experience may overrule the rules
    plan_above: float = 0.65             # estimated chance that react fails, above which to plan first
    relax_below: float = 0.15            # ... below which a task the rules call hard is tried with react
    recheck_every: int = 10              # of tasks experience moved to planning, try every Nth on react again,
                                         # so a record made when things were worse can recover (0 = never)
    explore_every: int = 0               # also try react on every Nth task the *rules* call hard (0 = never)
    generalise: bool = True              # let evidence about other kinds of task inform a kind seen too rarely
    generalise_after: int = 40           # outcomes the cross-kind model must have seen before it is consulted
    min_skill: float = 0.1               # ... and the share of the error of "guess the average" it must remove
    generalise_plan_above: float = 0.8   # the model alone moves a task to planning above this estimate
    generalise_relax_below: float = 0.15 # ... and to the reason-act loop below this one


_WORD = re.compile(r"[a-z][a-z_-]{2,}")
_COMMON = frozenset("""the and for with from that this what which when where who how why are was were will would
can could should does did has have had not but its into over under about than then them they their there here
all any each every some more most other such only also just very please give show tell find make get use using
you your our out per via one two three""".split())
_NO_PROFILE = ("long", "experience", "rechecking", "exploring", "generalised")


def task_words(text: str, limit: int = 40) -> list[str]:
    """The distinct content words of a task, in order of appearance."""
    found: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word not in _COMMON and word not in found:
            found.append(word)
            if len(found) == limit:
                break
    return found


class RouteModel:
    """Estimates the chance that the reason-act loop fails, from a task's signals and words.

    Logistic regression trained one outcome at a time (AdaGrad steps). Unlike the
    per-kind counts it shares evidence: if tasks that mention "reconcile" keep
    failing whatever else they look like, the next one is flagged on first sight.

    It also keeps score on itself. Before learning from an outcome it records the
    loss of the forecast it had just made, next to the loss of forecasting the
    failure rate so far. ``skill`` is the share of that baseline's loss the model
    removes, over roughly the last hundred outcomes. A model that has found
    nothing real scores about zero or below and is then ignored.
    """

    def __init__(self, rate: float = 0.5, decay: float = 0.99, max_features: int = 4000):
        self.rate, self.decay, self.max_features = rate, decay, max_features
        self.weights: dict[str, float] = {}
        self.squares: dict[str, float] = {}                 # per-feature sum of squared gradients
        self.seen = 0
        self.failures = 0
        self.model_loss = 0.0                               # decayed sums of the forecasts' log loss
        self.base_loss = 0.0

    @staticmethod
    def inputs(features: Sequence[str], text: str) -> list[str]:
        return ["bias", *features, *(f"word:{word}" for word in task_words(text))]

    def predict(self, inputs: Sequence[str]) -> float:
        score = sum(self.weights.get(name, 0.0) for name in inputs)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    @property
    def skill(self) -> float:
        return 1.0 - self.model_loss / self.base_loss if self.base_loss > 1e-9 else 0.0

    def update(self, inputs: Sequence[str], failed: bool) -> None:
        target = 1.0 if failed else 0.0
        forecast = self.predict(inputs)
        average = (self.failures + 1) / (self.seen + 2)
        self.model_loss = self.decay * self.model_loss + _log_loss(forecast, target)
        self.base_loss = self.decay * self.base_loss + _log_loss(average, target)
        gradient = forecast - target
        for name in inputs:
            self.squares[name] = self.squares.get(name, 0.0) + gradient * gradient
            self.weights[name] = self.weights.get(name, 0.0) - self.rate * gradient / math.sqrt(self.squares[name] + 1e-8)
        self.seen += 1
        self.failures += int(failed)
        if len(self.weights) > self.max_features:           # keep the features that carry the most
            keep = sorted(self.weights, key=lambda name: -abs(self.weights[name]))[: self.max_features * 3 // 4]
            self.weights = {name: self.weights[name] for name in keep}
            self.squares = {name: self.squares[name] for name in keep}

    def strongest(self, k: int = 8) -> list[tuple[str, float]]:
        """The inputs that move the estimate most (positive: goes with failure)."""
        ranked = sorted(((name, weight) for name, weight in self.weights.items() if name != "bias"),
                        key=lambda item: (-abs(item[1]), item[0]))
        return [(name, round(weight, 3)) for name, weight in ranked[:k]]

    def to_dict(self) -> dict:
        return {"weights": self.weights, "squares": self.squares, "seen": self.seen, "failures": self.failures,
                "model_loss": self.model_loss, "base_loss": self.base_loss}

    @classmethod
    def from_dict(cls, data: dict) -> "RouteModel":
        model = cls()
        model.weights = {str(k): float(v) for k, v in data["weights"].items()}
        model.squares = {str(k): float(v) for k, v in data["squares"].items()}
        model.seen, model.failures = int(data["seen"]), int(data["failures"])
        model.model_loss, model.base_loss = float(data["model_loss"]), float(data["base_loss"])
        if set(model.weights) != set(model.squares) or not all(map(math.isfinite, model.weights.values())):
            raise ValueError("inconsistent route model")
        return model


def _log_loss(forecast: float, target: float) -> float:
    forecast = min(1.0 - 1e-6, max(1e-6, forecast))
    return -math.log(forecast if target else 1.0 - forecast)


class RouteLearner:
    """Learns, from outcomes, which kinds of task the reason-act loop fails on.

    Tasks are grouped by profile: the router's signals plus a length band. For
    each profile it keeps how often a plain reason-act loop was tried and how
    often it failed. A profile's record affects only that profile - evidence
    about "why" questions says nothing about plain summaries - which makes the
    learner slow to generalise and hard to mislead.

    Next to the counts it trains a :class:`RouteModel` on the same outcomes. The
    model generalises where the counts cannot, and is used only for profiles the
    counts know too little about, and only while it is ``trusted``.

    It only ever moves a task between ``react`` and ``deliberate``; the engine
    still escalates when a path fails, so a wrong estimate costs time, never an
    answer.
    """

    def __init__(self, path: str | None = None, generalise_after: int = 40, min_skill: float = 0.1):
        self.path = path
        self.profiles: dict[str, list[int]] = {}           # profile -> [runs, failures]
        self.observations = 0
        self.model = RouteModel()
        self.generalise_after, self.min_skill = generalise_after, min_skill
        self._lock = threading.Lock()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                self.profiles = {str(k): [int(v[0]), int(v[1])] for k, v in data["profiles"].items()}
                self.observations = int(data.get("observations", 0))
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
                self.profiles, self.observations = {}, 0   # a damaged file means starting over
                data = {}
            try:                                            # absent in files written before 0.6
                if "model" in data:
                    self.model = RouteModel.from_dict(data["model"])
            except (ValueError, TypeError, KeyError, AttributeError):
                self.model = RouteModel()

    @staticmethod
    def features(signals: Sequence[str], words: int) -> list[str]:
        size = "short" if words <= 12 else "medium" if words <= 40 else "long"
        return sorted(set(s for s in signals if s not in _NO_PROFILE)) + [f"length:{size}"]

    @staticmethod
    def profile(features: Sequence[str]) -> str:
        return " + ".join(features)

    def runs(self, features: Sequence[str]) -> int:
        return self.profiles.get(self.profile(features), [0, 0])[0]

    def risk(self, features: Sequence[str]) -> float:
        """The estimated chance that a plain reason-act loop fails on this kind of task.

        Laplace-smoothed: with no evidence it is 0.5, and it takes several
        consistent outcomes to move it near either end.
        """
        runs, failures = self.profiles.get(self.profile(features), [0, 0])
        return (failures + 1) / (runs + 2)

    @property
    def trusted(self) -> bool:
        """Whether the cross-kind model has earned a say: enough outcomes, and forecasts that beat the average."""
        return self.model.seen >= self.generalise_after and self.model.skill >= self.min_skill

    def forecast(self, features: Sequence[str], text: str) -> float:
        """The cross-kind model's estimate for this particular task (its signals, length and words)."""
        with self._lock:
            return self.model.predict(RouteModel.inputs(features, text))

    def report(self) -> dict:
        """How the cross-kind model is doing, for people: what it has seen and what it leans on."""
        with self._lock:
            return {"outcomes": self.model.seen, "skill": round(self.model.skill, 3), "trusted": self.trusted,
                    "strongest": self.model.strongest()}

    def update(self, features: Sequence[str], react_failed: bool, text: str | None = None) -> None:
        with self._lock:
            record = self.profiles.setdefault(self.profile(features), [0, 0])
            record[0] += 1
            record[1] += int(bool(react_failed))
            self.observations += 1
            if text is not None:
                self.model.update(RouteModel.inputs(features, text), bool(react_failed))
            if self.path:
                try:
                    temporary = self.path + ".tmp"
                    with open(temporary, "w", encoding="utf-8") as handle:
                        json.dump({"profiles": self.profiles, "observations": self.observations,
                                   "model": self.model.to_dict()}, handle)
                    os.replace(temporary, self.path)
                except OSError:                             # learning is best-effort; routing still works
                    pass

    def explain(self) -> list[tuple[str, float, int]]:
        """Profiles, riskiest first: (profile, estimated failure rate, runs)."""
        rows = [(name, round((failures + 1) / (runs + 2), 3), runs) for name, (runs, failures) in self.profiles.items()]
        return sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))


class Router:
    def __init__(self, tools: ToolRegistry, procedural=None, episodic=None, config: RouterConfig | None = None,
                 evaluate: Callable[[str], object] | None = None, learner: RouteLearner | None = None):
        """``procedural``/``episodic`` are the memory tiers consulted for cached policies and past failures."""
        self.tools, self.procedural, self.episodic = tools, procedural, episodic
        self.config = config or RouterConfig()
        self.learner = learner if learner is not None else (RouteLearner() if self.config.learn_routes else None)
        if self.learner is not None:
            self.learner.generalise_after, self.learner.min_skill = self.config.generalise_after, self.config.min_skill
        self._evaluate = evaluate
        self.decisions: dict[str, int] = {"reflex": 0, "react": 0, "deliberate": 0}
        self._explored = self._rechecked = 0

    # -- System 1 patterns ------------------------------------------------- #
    def _small_talk(self, text: str) -> Route | None:
        for name, pattern, reply in _SMALL_TALK:
            if pattern.match(text):
                return Route(mode="reflex", reason=f"conversational pattern: {name}", confidence=1.0, reply=reply)
        if _IDENTITY.match(text):
            names = ", ".join(self.tools.names[:12]) or "none"
            return Route(mode="reflex", reason="conversational pattern: capabilities", confidence=1.0,
                         reply=f"I am a tool-using agent. I answer by acting, not guessing: every figure I report "
                               f"comes from a tool result. Tools available now: {names}.")
        return None

    def _direct_tool(self, text: str) -> Route | None:
        if "current_time" in self.tools and _TIME.match(text):
            return Route(mode="reflex", reason="pattern: clock question", confidence=0.95,
                         calls=[ToolCall(tool="current_time", args={})], answer_template="{result}")
        if "calculator" in self.tools:
            expression = _ARITHMETIC_LEAD.sub("", text).rstrip(" ?=.").replace("×", "*").replace("÷", "/")
            expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression).replace("^", "**")
            if _EXPRESSION.match(expression) and re.search(r"\d\s*[-+*/%]", expression):
                if self._evaluate is not None:
                    try:
                        self._evaluate(expression)
                    except Exception:
                        return None
                return Route(mode="reflex", reason="pattern: plain arithmetic", confidence=0.95,
                             calls=[ToolCall(tool="calculator", args={"expression": expression})],
                             answer_template=f"{expression.replace('**', '^')} = {{result}}")
        return None

    def _cached_policy(self, text: str) -> Route | None:
        if not self.config.use_cached_policies or self.procedural is None:
            return None
        adapted = self.procedural.adapt(text)
        if adapted is None or adapted.confidence < self.config.reflex_confidence:
            return None
        if any(call.tool not in self.tools for call in adapted.calls):
            return None
        changes = ", ".join(f"{a} -> {b}" for a, b in adapted.substitutions.items()) or "same task as before"
        return Route(mode="reflex", reason=f"cached action policy ({changes}; reliability "
                                           f"{adapted.procedure.reliability:.2f})",
                     confidence=min(1.0, adapted.confidence), calls=adapted.calls,
                     answer_template=adapted.answer_template, expected_figures=adapted.output_figures,
                     policy_id=adapted.procedure.id)

    # -- escalation -------------------------------------------------------- #
    def signals(self, text: str) -> list[str]:
        found = [name for name, pattern in SIGNALS.items() if pattern.search(text)]
        if len(text.split()) > self.config.long_task_words:
            found.append("long")
        if text.count("?") >= 2 or len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+\S", text)) >= 2:
            found.append("several_parts")
        return found

    def _failed_before(self, text: str) -> bool:
        if self.episodic is None:
            return False
        hits = self.episodic.search(text, k=3, kind="trace", min_score=0.6)
        return bool(hits) and hits[0].record.data.get("outcome") == "failure"

    def route(self, task: str) -> Route:
        text = " ".join(task.split())
        route = (self._small_talk(text) if self.config.small_talk else None) \
            or self._direct_tool(text) or self._cached_policy(text)
        if route is None:
            signals = self.signals(text)
            if self._failed_before(text):
                signals.append("failed_before")
            strong = [s for s in signals if s in _STRONG or s == "failed_before"]
            if strong or len(signals) >= self.config.deliberate_signals:
                route = Route(mode="deliberate", signals=signals, confidence=min(1.0, 0.5 + 0.15 * len(signals)),
                              reason="needs deliberate planning: " + ", ".join(signals))
            else:
                route = Route(mode="react", signals=signals, confidence=0.6,
                              reason="no cached policy fits; a single reason-act loop should do"
                                     + (f" (signal: {signals[0]})" if signals else ""))
            route = self._from_experience(text, route)
        self.decisions[route.mode] += 1
        return route

    # -- experience -------------------------------------------------------- #
    def task_features(self, task: str) -> list[str]:
        text = " ".join(task.split())
        return RouteLearner.features(self.signals(text), len(text.split()))

    def _from_experience(self, text: str, route: Route) -> Route:
        """Let measured outcomes overrule the fixed rules, once there are enough of them."""
        learner, config = self.learner, self.config
        if learner is None or not config.learn_routes or "failed_before" in route.signals:
            return route                                    # (a specific memory beats a general statistic)
        features = RouteLearner.features(route.signals, len(text.split()))
        runs = learner.runs(features)
        if config.generalise and learner.trusted:
            # The model looks at this task's own words, not only at its kind, so where it is sure it
            # goes first: two tasks of one kind need not be treated alike.
            risk = learner.forecast(features, text)
            basis = f"estimated failure {risk:.0%}, learned from {learner.model.seen} outcomes across kinds of task"
            if risk >= config.generalise_plan_above:
                if route.mode == "deliberate":
                    return route
                self._rechecked += 1                        # as below: keep testing the belief now and then
                if config.recheck_every and self._rechecked % config.recheck_every == 0:
                    return Route(mode="react", signals=route.signals + ["rechecking"], confidence=1.0 - risk,
                                 reason=f"rechecking: tasks worded like this are usually planned first ({basis}); "
                                        f"trying the cheaper path again")
                return Route(mode="deliberate", signals=route.signals + ["generalised"], confidence=risk,
                             reason=f"experience with similar wording: the reason-act loop tends to fail on "
                                    f"tasks like this ({basis})")
            if risk <= config.generalise_relax_below:
                if route.mode == "react":
                    return route
                return Route(mode="react", signals=route.signals + ["generalised"], confidence=1.0 - risk,
                             reason=f"experience with similar wording: tasks like this have been solved "
                                    f"without planning ({basis})")
        if runs >= config.min_observations:
            risk = learner.risk(features)
            if route.mode == "react" and risk >= config.plan_above:
                # Planning first means the cheap path is no longer tried, so its record would never
                # change. Every so often, try it again; the engine escalates if it still fails.
                self._rechecked += 1
                if config.recheck_every and self._rechecked % config.recheck_every == 0:
                    return Route(mode="react", signals=route.signals + ["rechecking"], confidence=1.0 - risk,
                                 reason=f"rechecking: tasks like this are usually planned first (estimated "
                                        f"failure {risk:.0%} over {runs} runs); trying the cheaper path again")
                return Route(mode="deliberate", signals=route.signals + ["experience"], confidence=risk,
                             reason=f"experience: the reason-act loop failed on most earlier tasks like this "
                                    f"(estimated failure {risk:.0%} over {runs} runs)")
            if route.mode == "deliberate" and risk <= config.relax_below:
                return Route(mode="react", signals=route.signals + ["experience"], confidence=1.0 - risk,
                             reason=f"experience: tasks like this have been solved without planning "
                                    f"(estimated failure {risk:.0%} over {runs} runs)")
        if route.mode == "deliberate" and config.explore_every:
            self._explored += 1
            if self._explored % config.explore_every == 0:
                return Route(mode="react", signals=route.signals + ["exploring"], confidence=0.5,
                             reason="exploring: trying the cheaper path on a task the rules call hard "
                                    "(it escalates if that fails)")
        return route

    def learn(self, task: str, react_failed: bool) -> None:
        """Record how the reason-act loop did on ``task``."""
        if self.learner is not None and self.config.learn_routes:
            self.learner.update(self.task_features(task), react_failed, " ".join(task.split()))
