"""Dense embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""
from __future__ import annotations

import os
import time
from typing import Any

from agi.infrastructure.adapters.llm.openai_compat import _post


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
