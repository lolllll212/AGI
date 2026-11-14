"""Episodic memory: a persistent vector store of traces, preferences and state snapshots.

Records live in SQLite (so they survive restarts and can be filtered by kind,
tag and time). Vectors are searched with FAISS when it is installed and with a
plain-Python scan otherwise - same results, different speed.

Embeddings come from any ``embed(text) -> list[float]`` callable. The default
(:class:`DenseHashEmbedder`) is lexical: it matches shared words and word
shapes, not meaning. Pass a neural embedder (for example
``OpenAICompatEmbedder(...).embed_dense``) for paraphrase-level recall.
"""
from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Literal, Sequence

from pydantic import Field

from ..core.types import Model
from .embed import HashingEmbedder

try:                                                        # optional accelerators
    import numpy as _np
    import faiss as _faiss
except Exception:                                           # pragma: no cover - depends on the install
    _np = _faiss = None

Kind = Literal["trace", "preference", "snapshot", "note"]


class DenseHashEmbedder:
    """The hashing embedder as a fixed-length dense vector (unit length)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._sparse = HashingEmbedder(dim)

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for index, value in self._sparse.embed(text).items():
            vector[index] = value
        return vector


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else [0.0 for _ in vector]


class PythonIndex:
    """Exact cosine search by linear scan. Fine up to tens of thousands of vectors."""

    name = "python"

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, key: str, vector: Sequence[float]) -> None:
        self.ids.append(key)
        self.vectors.append(list(vector))

    def remove(self, key: str) -> None:
        if key in self.ids:
            index = self.ids.index(key)
            del self.ids[index], self.vectors[index]

    def search(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        scored = [(key, sum(a * b for a, b in zip(vector, stored)))
                  for key, stored in zip(self.ids, self.vectors)]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def __len__(self) -> int:
        return len(self.ids)


class FaissIndex:
    """Exact inner-product search with FAISS (vectors are unit length, so this is cosine)."""

    name = "faiss"

    def __init__(self, dim: int):
        if _faiss is None:
            raise ImportError("faiss is not installed (pip install faiss-cpu)")
        self.dim = dim
        self.ids: list[str] = []
        self._index = _faiss.IndexFlatIP(dim)

    def add(self, key: str, vector: Sequence[float]) -> None:
        self.ids.append(key)
        self._index.add(_np.asarray([vector], dtype="float32"))

    def remove(self, key: str) -> None:
        if key not in self.ids:
            return
        keep = [i for i, existing in enumerate(self.ids) if existing != key]
        vectors = self._index.reconstruct_n(0, len(self.ids)) if self.ids else None
        self._index = _faiss.IndexFlatIP(self.dim)
        self.ids = [self.ids[i] for i in keep]
        if keep:
            self._index.add(_np.ascontiguousarray(vectors[keep]))

    def search(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores, rows = self._index.search(_np.asarray([vector], dtype="float32"), min(k, len(self.ids)))
        return [(self.ids[row], float(score)) for score, row in zip(scores[0], rows[0]) if row >= 0]

    def __len__(self) -> int:
        return len(self.ids)


def make_index(dim: int, backend: str = "auto"):
    if backend == "faiss" or (backend == "auto" and _faiss is not None):
        return FaissIndex(dim)
    if backend in ("auto", "python"):
        return PythonIndex(dim)
    raise ValueError(f"unknown vector backend {backend!r} (use 'auto', 'faiss' or 'python')")


class Record(Model):
    id: str
    kind: Kind
    text: str
    created: float
    tags: list[str] = Field(default_factory=list)
    key: str | None = None               # preferences and snapshots: what this is the value of
    data: dict[str, Any] = Field(default_factory=dict)
    superseded: float | None = None      # when a newer value for the same key replaced this one


class Hit(Model):
    record: Record
    score: float                         # similarity, adjusted for recency if asked
    similarity: float


class EpisodicStore:
    """Execution traces, user preferences and temporal state snapshots, searchable by meaning and time."""

    def __init__(self, path: str = ":memory:", embed: Callable[[str], Sequence[float]] | None = None,
                 backend: str = "auto", clock: Callable[[], float] = time.time):
        self.embed = embed or DenseHashEmbedder()
        self.clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""create table if not exists records (
            id text primary key, kind text not null, text text not null, created real not null,
            tags text not null, key text, data text not null, superseded real, vector blob not null)""")
        self._db.execute("create index if not exists records_kind on records(kind, key, created)")
        self._db.commit()
        self.dim = len(self.embed("dimension probe"))
        self.index = make_index(self.dim, backend)
        for key, blob in self._db.execute("select id, vector from records where superseded is null"):
            if len(blob) == 4 * self.dim:                 # skip vectors from a different embedder
                self.index.add(key, struct.unpack(f"{self.dim}f", blob))

    @property
    def backend(self) -> str:
        return self.index.name

    # -- writing ----------------------------------------------------------- #
    def add(self, kind: Kind, text: str, tags: Iterable[str] = (), key: str | None = None,
            data: dict | None = None, at: float | None = None) -> Record:
        record = Record(id=uuid.uuid4().hex[:12], kind=kind, text=text, tags=sorted(set(tags)), key=key,
                        data=data or {}, created=self.clock() if at is None else at)
        vector = _normalise(self.embed(text))
        if len(vector) != self.dim:
            raise ValueError(f"the embedder returned {len(vector)} dimensions, expected {self.dim}")
        with self._lock:
            self._db.execute("insert into records values (?,?,?,?,?,?,?,?,?)", (
                record.id, kind, text, record.created, json.dumps(record.tags), key,
                json.dumps(record.data, default=str), None, struct.pack(f"{self.dim}f", *vector)))
            self._db.commit()
            self.index.add(record.id, vector)
        return record

    def add_trace(self, task: str, outcome: str, steps: Sequence[dict] = (), answer: str | None = None,
                  tags: Iterable[str] = (), **data: Any) -> Record:
        """One finished run: what was asked, what was done, how it ended."""
        actions = ", ".join(str(s.get("action")) for s in steps if s.get("action"))
        text = f"{task}\noutcome: {outcome}" + (f"\nactions: {actions}" if actions else "") + (
            f"\nanswer: {answer[:300]}" if answer else "")
        return self.add("trace", text, tags, data={"task": task, "outcome": outcome, "steps": list(steps),
                                                   "answer": answer, **data})

    def _supersede(self, kind: str, key: str, now: float) -> None:
        for (old,) in self._db.execute(
                "select id from records where kind=? and key=? and superseded is null", (kind, key)).fetchall():
            self._db.execute("update records set superseded=? where id=?", (now, old))
            self.index.remove(old)

    def set_preference(self, key: str, value: str, tags: Iterable[str] = ()) -> Record:
        """Remember a user preference. A new value for the same key replaces the old one (history is kept)."""
        with self._lock:
            now = self.clock()
            self._supersede("preference", key, now)
            return self.add("preference", f"{key}: {value}", tags, key=key, data={"value": value}, at=now)

    def snapshot(self, key: str, state: dict, tags: Iterable[str] = ()) -> Record:
        """Record the state of something at this moment (a file tree, a plan, an environment)."""
        with self._lock:
            now = self.clock()
            self._supersede("snapshot", key, now)
            text = f"{key}: " + json.dumps(state, sort_keys=True, default=str)[:1000]
            return self.add("snapshot", text, tags, key=key, data={"state": state}, at=now)

    def delete(self, record_id: str) -> None:
        with self._lock:
            self._db.execute("delete from records where id=?", (record_id,))
            self._db.commit()
            self.index.remove(record_id)

    # -- reading ----------------------------------------------------------- #
    @staticmethod
    def _record(row: tuple) -> Record:
        return Record(id=row[0], kind=row[1], text=row[2], created=row[3], tags=json.loads(row[4]),
                      key=row[5], data=json.loads(row[6]), superseded=row[7])

    _COLUMNS = "id, kind, text, created, tags, key, data, superseded"

    def get(self, record_id: str) -> Record | None:
        with self._lock:
            row = self._db.execute(f"select {self._COLUMNS} from records where id=?", (record_id,)).fetchone()
        return self._record(row) if row else None

    def search(self, query: str, k: int = 5, kind: Kind | None = None, tags: Iterable[str] = (),
               since: float | None = None, until: float | None = None, min_score: float = 0.15,
               half_life: float | None = None) -> list[Hit]:
        """The ``k`` current records most similar to ``query``.

        ``half_life`` (seconds) blends recency in: a record that old counts half as much.
        """
        wanted = set(tags)
        vector = _normalise(self.embed(query))
        with self._lock:
            candidates = self.index.search(vector, max(k * 8, 40))
            hits: list[Hit] = []
            now = self.clock()
            for key, similarity in candidates:
                if similarity < min_score:
                    continue
                record = self.get(key)
                if record is None or (kind and record.kind != kind) or not wanted <= set(record.tags):
                    continue
                if (since is not None and record.created < since) or (until is not None and record.created > until):
                    continue
                score = similarity
                if half_life:
                    score *= 0.5 ** (max(0.0, now - record.created) / half_life)
                hits.append(Hit(record=record, score=score, similarity=similarity))
        hits.sort(key=lambda hit: -hit.score)
        return hits[:k]

    def preferences(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("select key, data from records where kind='preference' and "
                                    "superseded is null order by created").fetchall()
        return {key: json.loads(data)["value"] for key, data in rows}

    def state_at(self, key: str, when: float | None = None) -> dict | None:
        """The snapshot of ``key`` that was current at time ``when`` (default: now)."""
        when = self.clock() if when is None else when
        with self._lock:
            row = self._db.execute("select data from records where kind='snapshot' and key=? and created<=? "
                                   "order by created desc, rowid desc limit 1", (key, when)).fetchone()
        return json.loads(row[0])["state"] if row else None

    def history(self, key: str, kind: Kind = "snapshot") -> list[Record]:
        with self._lock:
            rows = self._db.execute(f"select {self._COLUMNS} from records where kind=? and key=? "
                                    "order by created, rowid", (kind, key)).fetchall()
        return [self._record(row) for row in rows]

    def recent(self, kind: Kind | None = None, limit: int = 10) -> list[Record]:
        with self._lock:
            rows = self._db.execute(
                f"select {self._COLUMNS} from records where (? is null or kind=?) "
                "order by created desc, rowid desc limit ?", (kind, kind, limit)).fetchall()
        return [self._record(row) for row in rows]

    def stats(self) -> dict:
        with self._lock:
            counts = dict(self._db.execute("select kind, count(*) from records group by kind").fetchall())
        return {"backend": self.backend, "dim": self.dim, "indexed": len(self.index), "records": counts}

    def close(self) -> None:
        with self._lock:
            self._db.close()
