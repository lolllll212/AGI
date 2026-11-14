"""Dependency-free text features: tokenizer, keywords and a hashing embedder.

The default embedder is deliberately simple and deterministic. Swap in a neural
embedding model by passing any object with ``embed(text) -> dict[int, float]``
(sparse) to :class:`~agi.application.memory.engine.MemoryEngine`.
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


# Words that mean roughly the same thing to a tool-using agent. Deliberately small and readable:
# it is a stop-gap for when no embedding model is available, not a substitute for one.
CONCEPTS: dict[str, str] = {
    "arithmetic": "calculate calculation arithmetic add plus subtract minus multiply times divide divided "
                  "percent percentage interest sum product power sqrt square root exactly",
    "aggregate": "total totals sum average mean median count counts group grouped aggregate maximum minimum "
                 "highest lowest most least top largest smallest biggest per breakdown",
    "tabular": "csv table tables spreadsheet column columns row rows sql query dataset data records json "
               "orders sales customers",
    "file": "file files document documents folder directory path workspace txt",
    "read": "read open show view display contents content look inspect print",
    "write": "write save create store export report output produce generate overwrite",
    "search": "search find grep locate lookup match matches pattern contains containing occurrences lines "
              "mention mentions errors error log logs",
    "time": "time date day days today now clock week weeks month months year years deadline tomorrow "
            "yesterday until since ago weekday calendar current",
    "web": "http https url website web page download fetch online internet api site link",
    "code": "python code script run execute program function simulate simulation algorithm loop compute",
    "memory": "remember recall memorise memorize note notes forget earlier previously remind stored fact facts",
    "tools": "tool tools capability capabilities available toolbox",
}


class ConceptEmbedder(HashingEmbedder):
    """A :class:`HashingEmbedder` that also knows a few groups of related words.

    "How many days until the deadline?" and a tool described as "the current
    date and time" share no word; they do share the concept *time*. Each word
    that belongs to a group contributes that group as an extra feature, so
    such pairs get a non-zero similarity. ``extra`` adds groups for your own
    domain: ``ConceptEmbedder(extra={"billing": "invoice refund charge"})``.
    """

    def __init__(self, dim: int = 512, concept_weight: float = 0.9, extra: dict[str, str] | None = None):
        super().__init__(dim)
        self.concept_weight = concept_weight
        self._concepts: dict[str, list[str]] = {}
        for name, words in {**CONCEPTS, **(extra or {})}.items():
            for word in words.split():
                self._concepts.setdefault(stem(word.lower()), []).append(name)

    def concepts(self, text: str) -> list[str]:
        """The concept groups a text touches, most frequent first."""
        counts: dict[str, int] = {}
        for word in tokens(text):
            for name in self._concepts.get(word, ()):
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts, key=lambda name: (-counts[name], name))

    def _features(self, word: str) -> tuple:
        cached = self._cache.get(word)
        if cached is None:
            base = super()._features(word)
            extra = tuple((index, self.concept_weight * sign)
                          for index, sign in (self._slot("k:" + name) for name in self._concepts.get(word, ())))
            cached = self._cache[word] = base + extra
        return cached


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(index, 0.0) for index, value in a.items())
