"""Dependency-free text features: tokenizer, keywords and a hashing embedder.

The default embedder is deliberately simple and deterministic. Swap in a neural
embedding model by passing any object with ``embed(text) -> dict[int, float]``
(sparse) to :class:`~agi.memory.engine.MemoryEngine`.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in
into is it its not of on or so than that the then there these this to was were what when
where which while will with would you your then given every each all any return returns
find compute get make use using after before take
""".split())

_WORD = re.compile(r"[a-z][a-z0-9_]+|\d+")


def stem(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def keywords(text: str, limit: int = 8) -> list[str]:
    """Distinct content words in order of appearance."""
    seen: dict[str, None] = {}
    for token in tokens(text):
        if not token.isdigit() and len(token) > 2:
            seen.setdefault(token, None)
    return list(seen)[:limit]


class HashingEmbedder:
    """Signed feature hashing over words and character trigrams, L2-normalised.

    ``embed`` takes an optional ``weight(word)`` function; the memory engine
    passes inverse document frequency so that words every lesson shares
    ("list", "number") count for little and distinctive ones for a lot.
    """

    weighted = True

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._cache: dict[str, tuple] = {}

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        return number % self.dim, 1.0 if (number >> 63) & 1 else -1.0

    def _features(self, word: str) -> tuple:
        cached = self._cache.get(word)
        if cached is None:
            padded = f"^{word}$"
            grams = [self._slot("c:" + padded[i:i + 3]) for i in range(len(padded) - 2)]
            cached = (self._slot("w:" + word), *((i, 0.3 * sign) for i, sign in grams))
            self._cache[word] = cached
        return cached

    def embed(self, text: str, weight: Callable[[str], float] | None = None) -> dict[int, float]:
        vector: dict[int, float] = {}
        for word in tokens(text):
            scale = weight(word) if weight else 1.0
            for index, value in self._features(word):
                vector[index] = vector.get(index, 0.0) + scale * value
        norm = math.sqrt(sum(v * v for v in vector.values()))
        return {i: v / norm for i, v in vector.items()} if norm else {}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(index, 0.0) for index, value in a.items())
