"""Decide whether a lesson that was shown to the agent had anything to do with how the run went.

Crediting every recalled lesson with every success teaches memory nothing: a
lesson that is merely *present* when easy tasks succeed looks as useful as one
that prevents a failure. So a lesson is judged by what the run did:

* a **procedure** ("these tools solved it before") was followed if the run used
  most of those tools; then it shares the run's outcome. Not followed: no verdict.
* a **fix** or an **avoid** lesson warns about a failure. If the run hit that
  same failure anyway, the warning did not work. If the run went near it (used
  the tool involved) without failing that way and succeeded, it helped.
  Otherwise the run never came close, and there is no verdict.
* anything else (a free-text lesson from a model) is judged by whether the run
  shows its key words; if it does, it shares the outcome.

``None`` means "no evidence either way", and the lesson's record is left alone.
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from agi.application.memory.extract import error_signature
from agi.domain.entities.memory import Lesson

_WORD = re.compile(r"[a-z_][a-z0-9_]{3,}")
_FILLER = frozenset("""with from that this when then them they their there which what should would could about
after before into over under using used uses make sure always never instead because first each every task tasks
tool tools result results answer answers value values call calls data file files step steps hint failed fails
worked error errors known good approach avoid similar""".split())


def _tool(call: str | None) -> str:
    return (call or "").split("(", 1)[0].strip()


def _key_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _FILLER}


def lesson_verdict(lesson: Lesson, steps: Sequence[Any], success: bool, answer: str | None = None) -> bool | None:
    """``True``: the lesson helped. ``False``: it was followed, or should have been, and the run went wrong.
    ``None``: the run does not show whether it mattered.

    ``steps`` are the run's steps: objects with ``action``, ``args``, ``ok`` and ``observation``.
    """
    acted = [step for step in steps if getattr(step, "action", None)]
    used = {step.action for step in acted if step.ok}
    payload = lesson.payload or {}

    if lesson.kind == "procedure":
        wanted = {_tool(str(item)) for item in payload.get("solution") or []} - {""}
        if not wanted:
            return None
        followed = len(wanted & used) / len(wanted) >= 0.5
        return bool(success) if followed else None

    if lesson.kind in ("fix", "avoid") and payload.get("error_type"):
        warned = (payload.get("error_type"), payload.get("signature", ""))
        if any(not step.ok and error_signature(step.observation[:300]) == warned for step in acted):
            return False                                    # warned, and it happened again
        involved = {_tool(call) for call in payload.get("failed") or []} | {_tool(payload.get("worked"))}
        involved.discard("")
        if success and involved & used:
            return True                                     # went there, and did not fall in
        return None

    words = _key_words(lesson.advice)
    if not words:
        return None
    shown = _key_words(" ".join([answer or "", *(f"{step.action} {json.dumps(step.args, default=str)}"
                                                for step in acted)]))
    return bool(success) if len(words & shown) / len(words) >= 0.3 else None
