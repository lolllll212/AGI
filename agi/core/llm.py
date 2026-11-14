"""The only place the toolkit touches a language model.

Everything in this package runs offline with deterministic components. Wherever
a model *can* help (extracting lessons, proposing hypotheses, writing or
reviewing tools) the code accepts an ``LLM``: any ``callable(prompt) -> str``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
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


def _post(url: str, payload: dict, api_key: str, timeout: float, retries: int,
          sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST JSON with retry on rate limits, server errors and dropped connections."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", "replace")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries:
                raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from None
            wait = exc.headers.get("Retry-After", "") if exc.headers else ""
            sleep(min(30.0, float(wait)) if wait.replace(".", "", 1).isdigit() else 2.0 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise LLMError(f"cannot reach {url}: {getattr(exc, 'reason', exc)}") from None
            sleep(2.0 ** attempt)
    raise LLMError(f"no response from {url}")          # pragma: no cover


class LLMError(RuntimeError):
    """The model endpoint could not be used (after retries)."""


class OpenAICompatLLM:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with hosted APIs and local servers (Ollama, vLLM, llama.cpp, LM Studio)
    alike. Standard library only. Transient failures are retried with backoff.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 60.0, system: str | None = None,
                 stop: Iterable[str] | None = None, max_tokens: int | None = None, retries: int = 3):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.system = system
        self.stop = list(stop) if stop else None
        self.max_tokens = max_tokens
        self.retries = retries
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._sleep = time.sleep

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.stop:
            payload["stop"] = self.stop
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        data = _post(f"{self.base_url}/chat/completions", payload, self.api_key, self.timeout,
                     self.retries, self._sleep)
        self.calls += 1
        for key in self.usage:
            self.usage[key] += int((data.get("usage") or {}).get(key) or 0)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from None


class OpenAICompatEmbedder:
    """Dense embeddings from an OpenAI-compatible ``/embeddings`` endpoint.

    Pass it as ``MemoryEngine(embedder=...)`` to make recall semantic rather than
    lexical. Vectors are cached per text, so each lesson is embedded once.
    """

    weighted = False

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 30.0, retries: int = 3):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout, self.retries = timeout, retries
        self._cache: dict[str, dict[int, float]] = {}
        self._sleep = time.sleep

    def embed(self, text: str, weight: Any = None) -> dict[int, float]:
        if text not in self._cache:
            data = _post(f"{self.base_url}/embeddings", {"model": self.model, "input": text},
                         self.api_key, self.timeout, self.retries, self._sleep)
            self._cache[text] = dict(enumerate(data["data"][0]["embedding"]))
        return self._cache[text]


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
