"""A small typed property graph persisted in SQLite.

Nodes are ``episode``, ``lesson`` and ``concept``; edges are weighted and
typed. Because it is a file, knowledge survives the process that produced it,
which is the whole point: learning without touching model weights.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL,
    data TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL, weight REAL NOT NULL,
    PRIMARY KEY (src, rel, dst));
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
"""


class KnowledgeGraph:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        # The engine serialises access with a lock, so the connection may be shared
        # between threads (agent tools run in worker threads).
        self.db = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # -- nodes -------------------------------------------------------------- #
    def put(self, node_id: str, kind: str, label: str, data: dict | None = None) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "kind=excluded.kind, label=excluded.label, data=excluded.data, updated=excluded.updated",
            (node_id, kind, label, json.dumps(data or {}), now, now))
        self.db.commit()

    def get(self, node_id: str) -> dict | None:
        row = self.db.execute("SELECT id, kind, label, data FROM nodes WHERE id=?",
                              (node_id,)).fetchone()
        return self._row(row) if row else None

    def nodes(self, kind: str | None = None) -> Iterator[dict]:
        query = "SELECT id, kind, label, data FROM nodes"
        rows = self.db.execute(query + (" WHERE kind=?" if kind else "") + " ORDER BY created, id",
                               (kind,) if kind else ())
        return (self._row(row) for row in rows.fetchall())

    def delete(self, node_id: str) -> None:
        self.db.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self.db.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_id, node_id))
        self.db.commit()

    @staticmethod
    def _row(row: tuple) -> dict:
        return {"id": row[0], "kind": row[1], "label": row[2], "data": json.loads(row[3])}

    # -- edges -------------------------------------------------------------- #
    def link(self, src: str, rel: str, dst: str, weight: float = 1.0) -> None:
        """Add an edge; linking the same triple again accumulates its weight."""
        self.db.execute(
            "INSERT INTO edges VALUES (?,?,?,?) ON CONFLICT(src, rel, dst) "
            "DO UPDATE SET weight = weight + excluded.weight", (src, rel, dst, weight))
        self.db.commit()

    def edges(self, node_id: str | None = None, rel: str | None = None) -> list[tuple]:
        """``(src, rel, dst, weight)`` rows, optionally touching ``node_id`` / of type ``rel``."""
        clauses, args = [], []
        if node_id is not None:
            clauses.append("(src=? OR dst=?)")
            args += [node_id, node_id]
        if rel is not None:
            clauses.append("rel=?")
            args.append(rel)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return self.db.execute("SELECT src, rel, dst, weight FROM edges" + where
                               + " ORDER BY src, rel, dst", args).fetchall()

    def neighbors(self, node_id: str, rel: str | None = None) -> list[tuple[str, str, float]]:
        """``(other_id, rel, weight)`` for every edge touching ``node_id`` (either direction)."""
        return [(dst if src == node_id else src, r, w) for src, r, dst, w in self.edges(node_id, rel)]

    def rewire(self, old_id: str, new_id: str) -> None:
        """Move every edge of ``old_id`` onto ``new_id`` (used when merging lessons)."""
        for src, rel, dst, weight in self.edges(old_id):
            a, b = (new_id if src == old_id else src), (new_id if dst == old_id else dst)
            if a != b:
                self.link(a, rel, b, weight)
        self.delete(old_id)

    # -- queries ------------------------------------------------------------ #
    def spread(self, seeds: dict[str, float], hops: int = 2, decay: float = 0.5) -> dict[str, float]:
        """Spreading activation: push energy from ``seeds`` along weighted edges.

        Each hop, a node passes ``decay * min(1, edge_weight)`` of its energy
        to every neighbour. Returns the total activation per node.
        """
        total = dict(seeds)
        frontier = dict(seeds)
        for _ in range(hops):
            nxt: dict[str, float] = {}
            for node_id, energy in frontier.items():
                for other, _, weight in self.neighbors(node_id):
                    nxt[other] = nxt.get(other, 0.0) + energy * decay * min(1.0, weight)
            for node_id, energy in nxt.items():
                total[node_id] = total.get(node_id, 0.0) + energy
            frontier = nxt
        return total

    def stats(self) -> dict[str, Any]:
        kinds = dict(self.db.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind").fetchall())
        return {"nodes": kinds, "edges": self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]}

    def to_dot(self, kinds: tuple = ("lesson", "concept")) -> str:
        """Graphviz rendering of the knowledge graph, for inspection."""
        keep = {n["id"]: n for kind in kinds for n in self.nodes(kind)}
        shape = {"lesson": "box", "concept": "ellipse", "episode": "note"}
        lines = ["graph memory {", "  rankdir=LR;"]
        for node in keep.values():
            label = node["label"].replace('"', "'")[:60]
            lines.append(f'  "{node["id"]}" [label="{label}", shape={shape.get(node["kind"], "box")}];')
        for src, rel, dst, weight in self.edges():
            if src in keep and dst in keep:
                lines.append(f'  "{src}" -- "{dst}" [label="{rel}", penwidth={min(4.0, weight):.1f}];')
        return "\n".join(lines + ["}"])

    def close(self) -> None:
        self.db.close()
