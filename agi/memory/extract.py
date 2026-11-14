"""Turn raw episodes into lessons.

:class:`HeuristicExtractor` needs no model: it reads the trace, finds what
failed, pairs each failure with whatever worked afterwards, and records
known-good solutions. :class:`LLMExtractor` asks a model to do the same job
with more nuance and falls back to the heuristics if the reply is unusable.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from ..core.llm import LLM, extract_json
from .models import FAILURE, KINDS, SUCCESS, Episode, Lesson

ERROR_HINTS = {
    "ZeroDivisionError": "guard the divisor (zero or empty input) before dividing",
    "IndexError": "check the length first and handle the empty case",
    "KeyError": "check the key exists or use .get() with a default",
    "TypeError": "check operand types and convert or branch before operating",
    "ValueError": "validate the value's range and format before using it",
    "AttributeError": "confirm the object's type; it may be None or a different class",
    "NameError": "define or import the name before use",
    "SyntaxError": "the code does not parse; fix the syntax before anything else",
    "RecursionError": "add a base case or switch to iteration",
    "Timeout": "the approach is too slow or never terminates; bound the loop or change algorithm",
    "MemoryError": "the approach allocates too much; stream or shrink the data",
    "ResourceLimit": "the approach exceeded a resource limit; use a cheaper algorithm",
    "PolicyViolation": "the sandbox forbids this; stay inside the allowed imports and builtins",
    "ImportError": "that module is unavailable; use the allowed standard-library modules",
    "ModuleNotFoundError": "that module is unavailable; use the allowed standard-library modules",
    "AssertionError": "the output did not match the specification; re-read the expected behaviour",
    "TestFailed": "the output did not match the specification; re-read the expected behaviour",
    "TypeMismatch": "the result has the wrong type for this task; end on a step that produces the right type",
    # tool-use failures (agi.tools / agi.react)
    "FileNotFound": "list the files first and use the exact name",
    "PermissionDenied": "stay inside the workspace; that location is off limits",
    "SQLError": "inspect the table's columns (query_data without sql) before writing the query",
    "OnlySelect": "the data tool is read-only; use SELECT",
    "BadArguments": "check the tool's signature and pass exactly those arguments",
    "UnknownTool": "use only the tools that are listed",
    "RepeatedCall": "an identical call gives an identical result; change the call or answer",
    "UrlBlocked": "that address may not be fetched; use a public http(s) URL",
    "HTTPError": "the server refused; check the URL, slow down or try another source",
    "NetworkError": "the network call failed; retry once or use another source",
    "BudgetExceeded": "plan the calls; the tool budget is limited",
    "MaxSteps": "the step limit ran out; take a more direct route to the answer",
    "WrongAnswer": "the answer was wrong or unsupported; get every figure from a tool result",
}

_TYPE = re.compile(r"^\s*((?:[A-Z][a-z0-9]+){2,}(?=:)|[A-Z][A-Za-z]*(?:Error|Exception|Violation|Failed|Limit|Mismatch)"
                   r"|Timeout)\b:?\s*(.*)$", re.S)


def error_signature(error: str | None) -> tuple[str, str]:
    """Split ``"KeyError: 'user_42'"`` into a type and a literal-free message."""
    if not error:
        return "Error", ""
    match = _TYPE.match(error.strip().splitlines()[-1] if "\n" in error else error)
    etype, message = (match.group(1), match.group(2)) if match else ("Error", error)
    message = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", message)
    message = re.sub(r"\b\d+(\.\d+)?\b", "N", message)
    return etype, " ".join(message.split())[:160]


def _brief(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class LessonExtractor(Protocol):
    def extract(self, episode: Episode) -> list[Lesson]: ...


class HeuristicExtractor:
    """Model-free lesson extraction from an episode's trace."""

    def __init__(self, max_examples: int = 4):
        self.max_examples = max_examples

    def extract(self, episode: Episode) -> list[Lesson]:
        situation = _brief(episode.task)
        lessons: list[Lesson] = []

        # Group failed steps by error signature so one noisy search becomes a
        # handful of lessons rather than hundreds.
        groups: dict[tuple[str, str], dict] = {}
        for index, step in enumerate(episode.steps):
            if step.ok:
                continue
            key = error_signature(step.error)
            group = groups.setdefault(key, {"failed": [], "worked": None})
            if step.action not in group["failed"]:
                group["failed"].append(step.action)
            if group["worked"] is None:
                later = [s.action for s in episode.steps[index + 1:] if s.ok]
                tool = step.action.split("(", 1)[0]
                # Prefer the step that fixed *this* call (same tool) over whatever succeeded next.
                group["worked"] = next((a for a in later if "(" in a and a.split("(", 1)[0] == tool),
                                       later[0] if later else None)

        if not groups and episode.outcome == FAILURE and episode.error:
            groups[error_signature(episode.error)] = {"failed": [], "worked": None}

        for (etype, message), group in groups.items():
            hint = ERROR_HINTS.get(etype, "treat this as a known pitfall")
            failed = group["failed"][: self.max_examples]
            what = ", ".join(f"'{_brief(a, 60)}'" for a in failed) or "the attempt"
            detail = f"{etype}: {message}".rstrip(": ")
            if group["worked"]:
                kind = "fix"
                advice = f"{what} failed with {detail}; '{_brief(group['worked'], 80)}' worked instead. Hint: {hint}."
            else:
                kind = "avoid"
                advice = f"{what} fails with {detail}. Hint: {hint}."
            lessons.append(Lesson(
                kind=kind, trigger=f"{situation} ({etype})", advice=advice,
                tags=(*episode.tags, etype), episodes=[episode.id],
                payload={"error_type": etype, "signature": message, "failed": failed,
                         "worked": group["worked"]}))

        if episode.outcome == SUCCESS and episode.solution is not None:
            lessons.append(Lesson(
                kind="procedure", trigger=situation,
                advice=f"Known-good approach: {_brief(json.dumps(episode.solution, default=str), 200)}",
                tags=episode.tags, episodes=[episode.id], payload={"solution": episode.solution}))
        return lessons


_PROMPT = """You are the memory module of an autonomous agent. Below is the trace of one task
attempt. Extract at most {limit} lessons that would help on FUTURE, similar tasks.
Prefer lessons that generalise beyond this exact task. Reply with a JSON array only; each item:
{{"kind": "avoid" | "fix" | "procedure" | "fact", "trigger": "<situation in which it applies>",
  "advice": "<what to do>", "tags": ["<keyword>", ...]}}

EPISODE:
{episode}
"""


class LLMExtractor:
    """Lesson extraction by a language model, with a heuristic safety net."""

    def __init__(self, llm: LLM, limit: int = 4, fallback: LessonExtractor | None = None):
        self.llm = llm
        self.limit = limit
        self.fallback = fallback or HeuristicExtractor()

    def extract(self, episode: Episode) -> list[Lesson]:
        trace = json.dumps(episode.to_dict(), default=str)[:6000]
        try:
            items = extract_json(self.llm(_PROMPT.format(limit=self.limit, episode=trace)))
            lessons = [
                Lesson(kind=item["kind"] if item.get("kind") in KINDS else "fact",
                       trigger=_brief(item["trigger"], 200), advice=_brief(item["advice"], 400),
                       tags=(*episode.tags, *map(str, item.get("tags", []))),
                       episodes=[episode.id])
                for item in items[: self.limit]
                if isinstance(item, dict) and item.get("trigger") and item.get("advice")]
        except Exception:
            lessons = []
        # A solved task always yields its verified solution, whatever the model said.
        proven = [l for l in self.fallback.extract(episode) if l.kind == "procedure"]
        return (lessons + proven) if lessons else self.fallback.extract(episode)
