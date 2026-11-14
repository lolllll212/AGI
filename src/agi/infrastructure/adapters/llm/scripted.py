"""A deterministic stand-in model for tests, demonstrations and offline runs."""
from __future__ import annotations

from typing import Callable, Iterable


class ScriptedLLM:
    """Deterministic stand-in for tests and demos: replays canned replies."""

    def __init__(self, replies: Iterable[str] | Callable[[str], str]):
        self._fn = replies if callable(replies) else None
        self._queue = [] if callable(replies) else list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        if not self._queue:
            raise RuntimeError("ScriptedLLM has no replies left")
        return self._queue.pop(0)
