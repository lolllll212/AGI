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

import re
from typing import Callable

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


class Router:
    def __init__(self, tools: ToolRegistry, procedural=None, episodic=None, config: RouterConfig | None = None,
                 evaluate: Callable[[str], object] | None = None):
        """``procedural``/``episodic`` are the memory tiers consulted for cached policies and past failures."""
        self.tools, self.procedural, self.episodic = tools, procedural, episodic
        self.config = config or RouterConfig()
        self._evaluate = evaluate
        self.decisions: dict[str, int] = {"reflex": 0, "react": 0, "deliberate": 0}

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
        self.decisions[route.mode] += 1
        return route
