"""The only place the toolkit touches a language model.

Everything in this package runs offline with deterministic components. Wherever
a model *can* help (extracting lessons, proposing hypotheses, writing or
reviewing tools) the code accepts an ``LLM``: any ``callable(prompt) -> str``.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Iterable

LLM = Callable[[str], str]


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


class OpenAICompatLLM:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with hosted APIs and local servers (Ollama, vLLM, llama.cpp) alike.
    Standard library only.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 60.0, system: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.system = system

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of free-form model output.

    Raises ``ValueError`` if none is found. Tolerates code fences and chatter.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except ValueError:
                continue
    raise ValueError("no JSON value found in model output")


def extract_code(text: str) -> str:
    """Return the first fenced code block of ``text`` (or ``text`` itself)."""
    if "```" not in text:
        return text.strip()
    block = text.split("```", 2)[1]
    first, _, rest = block.partition("\n")
    return (rest if first.strip().isalpha() or not first.strip() else block).strip()
