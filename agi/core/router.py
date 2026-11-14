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
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Callable, Sequence

from ..tools.base import ToolRegistry
from .types import Model, Route, ToolCall

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
    explore_every: int = 0               # try react on every Nth task that would be planned, to keep evidence fresh (0 = never)


class RouteLearner:
    """Learns, from outcomes, which kinds of task the reason-act loop fails on.

    Tasks are grouped by profile: the router's signals plus a length band. For
    each profile it keeps how often a plain reason-act loop was tried and how
    often it failed. A profile's record affects only that profile - evidence
    about "why" questions says nothing about plain summaries - which makes the
    learner slow to generalise and hard to mislead.

    It only ever moves a task between ``react`` and ``deliberate``; the engine
    still escalates when a path fails, so a wrong estimate costs time, never an
    answer.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.profiles: dict[str, list[int]] = {}           # profile -> [runs, failures]
        self.observations = 0
        self._lock = threading.Lock()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                self.profiles = {str(k): [int(v[0]), int(v[1])] for k, v in data["profiles"].items()}
                self.observations = int(data.get("observations", 0))
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
                self.profiles, self.observations = {}, 0   # a damaged file means starting over

    @staticmethod
    def features(signals: Sequence[str], words: int) -> list[str]:
        size = "short" if words <= 12 else "medium" if words <= 40 else "long"
        return sorted(set(s for s in signals if s not in ("long", "experience"))) + [f"length:{size}"]

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

    def update(self, features: Sequence[str], react_failed: bool) -> None:
        with self._lock:
            record = self.profiles.setdefault(self.profile(features), [0, 0])
            record[0] += 1
            record[1] += int(bool(react_failed))
            self.observations += 1
            if self.path:
                try:
                    temporary = self.path + ".tmp"
                    with open(temporary, "w", encoding="utf-8") as handle:
                        json.dump({"profiles": self.profiles, "observations": self.observations}, handle)
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
        self._evaluate = evaluate
        self.decisions: dict[str, int] = {"reflex": 0, "react": 0, "deliberate": 0}
        self._explored = 0

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
        if runs >= config.min_observations:
            risk = learner.risk(features)
            if route.mode == "react" and risk >= config.plan_above:
                route = Route(mode="deliberate", signals=route.signals + ["experience"], confidence=risk,
                              reason=f"experience: the reason-act loop failed on most earlier tasks like this "
                                     f"(estimated failure {risk:.0%} over {runs} runs)")
            elif route.mode == "deliberate" and risk <= config.relax_below:
                return Route(mode="react", signals=route.signals + ["experience"], confidence=1.0 - risk,
                             reason=f"experience: tasks like this have been solved without planning "
                                    f"(estimated failure {risk:.0%} over {runs} runs)")
        if route.mode == "deliberate" and config.explore_every:
            # Planning first means the cheap path is never tried, so its record would never change.
            self._explored += 1
            if self._explored % config.explore_every == 0:
                return Route(mode="react", signals=route.signals + ["exploring"], confidence=0.5,
                             reason="exploring: trying the cheaper path on a task that is normally planned "
                                    "(it escalates if that fails)")
        return route

    def learn(self, task: str, react_failed: bool) -> None:
        """Record how the reason-act loop did on ``task``."""
        if self.learner is not None and self.config.learn_routes:
            self.learner.update(self.task_features(task), react_failed)
