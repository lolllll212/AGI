"""Episodic memory: record -> distil -> link -> recall -> reinforce -> consolidate."""
from __future__ import annotations

import functools
import math
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from .embed import HashingEmbedder, cosine, keywords, tokens
from .extract import HeuristicExtractor, LessonExtractor
from .graph import KnowledgeGraph
from .models import Episode, Lesson


def _locked(method):
    """Serialise a public method: tools may call the engine from worker threads."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class Recall:
    """A retrieved lesson and why it was retrieved."""

    lesson: Lesson
    score: float
    text_score: float
    graph_score: float


class MemoryEngine:
    """Long-term memory an agent can write to after every task and read before the next.

    * ``record(episode)`` distils lessons, merges them with what is already
      known (compression) and wires them into a knowledge graph.
    * ``recall(query)`` ranks lessons by text similarity *and* by spreading
      activation through shared concepts, weighted by how useful each lesson
      has proven (``feedback``).
    * ``consolidate()`` merges near-duplicates, drops old raw traces and
      retires lessons that keep not helping.

    Pass ``path`` to persist across processes; the default lives in RAM.
    """

    def __init__(self, path: str = ":memory:", extractor: LessonExtractor | None = None,
                 embedder=None, merge_threshold: float = 0.86, min_relevance: float = 0.3):
        self.graph = KnowledgeGraph(path)
        self.extractor = extractor or HeuristicExtractor()
        self.embedder = embedder or HashingEmbedder()
        self.merge_threshold = merge_threshold
        self.min_relevance = min_relevance
        self._vectors: dict[str, tuple[str, dict]] = {}
        self._lock = threading.RLock()

    def __enter__(self) -> "MemoryEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @_locked
    def close(self) -> None:
        self.graph.close()

    # -- write path --------------------------------------------------------- #
    @_locked
    def record(self, episode: Episode) -> list[Lesson]:
        """Store an episode and return the lessons it produced (after merging)."""
        self.graph.put(episode.id, "episode", episode.task[:80], episode.to_dict())
        concepts = self._concepts(episode.task, episode.tags)
        for a in concepts:
            for b in concepts:
                if a < b:
                    self.graph.link(a, "co_occurs", b, 0.25)
        return [self.learn(lesson, episode.id) for lesson in self.extractor.extract(episode)]

    @_locked
    def learn(self, lesson: Lesson, episode_id: str | None = None) -> Lesson:
        """Add one lesson directly, merging it into an equivalent one if present."""
        existing = self._equivalent(lesson)
        if existing is not None:
            lesson = self._absorb(existing, lesson)
        self._save(lesson)
        linked = {other for other, _, _ in self.graph.neighbors(lesson.id, "about")}
        for concept in self._concepts(lesson.trigger, lesson.tags):
            if concept not in linked:
                self.graph.link(lesson.id, "about", concept, 1.0)
        if episode_id:
            self.graph.link(episode_id, "yielded", lesson.id, 1.0)
        return lesson

    @staticmethod
    def _absorb(keeper: Lesson, other: Lesson) -> Lesson:
        """Fold ``other`` into ``keeper``: evidence adds up, the richer advice wins."""
        keeper.support += other.support
        keeper.helped += other.helped
        keeper.hurt += other.hurt
        keeper.tags = tuple(dict.fromkeys((*keeper.tags, *other.tags)))
        keeper.episodes = list(dict.fromkeys((*keeper.episodes, *other.episodes)))[-20:]
        failed = [*keeper.payload.get("failed", []), *other.payload.get("failed", [])]
        if failed:
            keeper.payload["failed"] = list(dict.fromkeys(failed))[:8]
        if keeper.kind == "avoid" and other.kind == "fix":   # we now know what works
            keeper.kind, keeper.advice = "fix", other.advice
            keeper.payload["worked"] = other.payload.get("worked")
        return keeper

    @_locked
    def feedback(self, lesson_id: str, helped: bool) -> None:
        """Tell memory whether a retrieved lesson actually helped."""
        lesson = self.get(lesson_id)
        if lesson is None:
            return
        if helped:
            lesson.helped += 1
        else:
            lesson.hurt += 1
        self._save(lesson)

    @_locked
    def forget(self, lesson_id: str) -> None:
        self.graph.delete(lesson_id)
        self._vectors.pop(lesson_id, None)

    # -- read path ---------------------------------------------------------- #
    @_locked
    def recall(self, query: str, tags: Iterable[str] = (), k: int = 5,
               kinds: Iterable[str] | None = None, touch: bool = True) -> list[Recall]:
        """The ``k`` most relevant lessons for ``query``; empty if nothing is relevant."""
        tags = tuple(tags)
        lessons = [l for l in self.lessons() if kinds is None or l.kind in set(kinds)]
        if not lessons:
            return []
        embed = self._weighted_embedder(lessons)
        query_vector = embed(f"{query} {' '.join(tags)}")
        activation = self._activate(query, tags)
        hits = []
        for lesson in lessons:
            text_score = max(0.0, cosine(query_vector, embed(lesson.text)))
            graph_score = min(1.0, activation.get(lesson.id, 0.0))
            relevance = 0.65 * text_score + 0.35 * graph_score
            if relevance < self.min_relevance:
                continue
            confidence = (0.5 + lesson.utility) * (1 + 0.15 * math.log(lesson.support))
            hits.append(Recall(lesson, relevance * confidence, text_score, graph_score))
        hits.sort(key=lambda h: (-h.score, h.lesson.id))
        hits = hits[:k]
        if touch:
            for hit in hits:
                hit.lesson.last_used = time.time()
                self._save(hit.lesson)
        return hits

    @_locked
    def context(self, query: str, tags: Iterable[str] = (), k: int = 5) -> str:
        """Recalled lessons formatted as a block to prepend to a model prompt."""
        hits = self.recall(query, tags, k)
        if not hits:
            return ""
        lines = ["Lessons from past experience (most relevant first):"]
        for i, hit in enumerate(hits, 1):
            l = hit.lesson
            lines.append(f"{i}. {l.render()} (seen {l.support}x, utility {l.utility:.2f})")
        return "\n".join(lines)

    @_locked
    def get(self, lesson_id: str) -> Lesson | None:
        node = self.graph.get(lesson_id)
        return Lesson.from_dict(node["data"]) if node and node["kind"] == "lesson" else None

    @_locked
    def lessons(self) -> list[Lesson]:
        return [Lesson.from_dict(n["data"]) for n in self.graph.nodes("lesson")]

    @_locked
    def episodes(self) -> list[dict]:
        return [n["data"] for n in self.graph.nodes("episode")]

    @_locked
    def stats(self) -> dict:
        info = self.graph.stats()
        return {"episodes": info["nodes"].get("episode", 0), "lessons": info["nodes"].get("lesson", 0),
                "retired": info["nodes"].get("retired", 0), "concepts": info["nodes"].get("concept", 0),
                "edges": info["edges"]}

    # -- maintenance -------------------------------------------------------- #
    @_locked
    def consolidate(self, keep_traces: int = 50, retire_below: float = 0.3,
                    min_trials: int = 4) -> dict:
        """Compress memory. Nothing is silently lost: weak lessons are retired, not deleted."""
        merged = retired = compressed = 0
        lessons = self.lessons()
        absorbed: set[str] = set()
        for i, keeper in enumerate(lessons):
            if keeper.id in absorbed:
                continue
            for other in lessons[i + 1:]:
                if other.id in absorbed or not self._same(keeper, other):
                    continue
                self._absorb(keeper, other)
                self._save(keeper)
                self.graph.rewire(other.id, keeper.id)
                self._vectors.pop(other.id, None)
                absorbed.add(other.id)
                merged += 1

        for lesson in self.lessons():
            if lesson.helped + lesson.hurt >= min_trials and lesson.utility < retire_below:
                self.graph.put(lesson.id, "retired", lesson.render()[:80], lesson.to_dict())
                self._vectors.pop(lesson.id, None)
                retired += 1

        episodes = sorted(self.graph.nodes("episode"), key=lambda n: n["data"].get("ts", 0))
        for node in episodes[:-keep_traces] if keep_traces else episodes:
            data = node["data"]
            if data.get("steps"):
                data.update(steps=[], compressed=True)
                self.graph.put(node["id"], "episode", node["label"], data)
                compressed += 1
        return {"merged": merged, "retired": retired, "traces_compressed": compressed}

    # -- internals ---------------------------------------------------------- #
    def _save(self, lesson: Lesson) -> None:
        self.graph.put(lesson.id, "lesson", lesson.render()[:80], lesson.to_dict())

    def _weighted_embedder(self, lessons: list[Lesson]):
        """An ``embed(text)`` that down-weights words common to most lessons (IDF)."""
        if not getattr(self.embedder, "weighted", False):
            return self.embedder.embed
        frequency: dict[str, int] = {}
        for lesson in lessons:
            for token in set(tokens(lesson.text)):
                frequency[token] = frequency.get(token, 0) + 1
        count = len(lessons)

        def idf(token: str) -> float:
            return math.log(1.0 + count / (1.0 + frequency.get(token, 0)))

        return lambda text: self.embedder.embed(text, idf)

    def _vector(self, lesson: Lesson) -> dict:
        cached = self._vectors.get(lesson.id)
        if cached is None or cached[0] != lesson.text:
            cached = (lesson.text, self.embedder.embed(lesson.text))
            self._vectors[lesson.id] = cached
        return cached[1]

    def _same(self, a: Lesson, b: Lesson) -> bool:
        """Do two lessons say the same thing? (Same error in a related context,
        or the same solution to near-identical tasks.)"""
        errors = {"avoid", "fix"}
        similarity = cosine(self._vector(a), self._vector(b))
        if a.kind in errors and b.kind in errors:
            same_error = (a.payload.get("error_type"), a.payload.get("signature")) == (
                b.payload.get("error_type"), b.payload.get("signature"))
            if a.payload.get("error_type") and same_error:
                return similarity >= self.merge_threshold * 0.5
        if a.kind != b.kind or a.payload.get("solution") != b.payload.get("solution"):
            return False
        return similarity >= self.merge_threshold

    def _equivalent(self, lesson: Lesson) -> Lesson | None:
        exact = self.get(lesson.id)
        if exact is not None:
            return exact
        return next((old for old in self.lessons() if self._same(old, lesson)), None)

    def _concepts(self, text: str, tags: Iterable[str]) -> list[str]:
        """Ensure concept nodes exist for the keywords/tags and return their ids."""
        names = dict.fromkeys([*(t.lower() for t in tags), *keywords(text)])
        ids = []
        for name in names:
            node_id = f"concept:{name}"
            if self.graph.get(node_id) is None:
                self.graph.put(node_id, "concept", name)
            ids.append(node_id)
        return ids

    def _activate(self, query: str, tags: Iterable[str]) -> dict[str, float]:
        """Seed the query's concepts and spread to lessons.

        Rare concepts weigh more than common ones, explicit tags weigh three
        times a keyword, and concepts memory has never seen dilute the rest.
        A lesson about *every* query concept scores 1.0.
        """
        tags = [t.lower() for t in tags]
        names = list(dict.fromkeys([*tags, *keywords(query)]))
        weights = {}
        for name in names:
            node_id = f"concept:{name}"
            degree = len(self.graph.neighbors(node_id, "about")) if self.graph.get(node_id) else 0
            weights[node_id] = ((3.0 if name in tags else 1.0) / math.log(2 + degree), degree > 0)
        total = sum(w for w, _ in weights.values())
        seeds = {node_id: w / total for node_id, (w, known) in weights.items() if known}
        if not seeds:
            return {}
        return {k: v * 2 for k, v in self.graph.spread(seeds, hops=2, decay=0.5).items()}
