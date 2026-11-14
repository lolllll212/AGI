"""What the application needs from a language model and from an embedding model.

A model is any ``callable(prompt) -> str``; nothing more is assumed, so a test
can pass a function and production can pass an HTTP client.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

LLM = Callable[[str], str]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a vector: sparse (``{dimension: weight}``) or dense (a list of floats)."""

    def embed(self, text: str) -> Any: ...
