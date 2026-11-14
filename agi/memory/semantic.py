"""Semantic memory: an entity-relationship knowledge graph with time and provenance.

Facts are ``(subject, predicate, object)`` triples between typed entities. Each
fact carries a confidence, a source, and the period during which it held, so
the graph can answer "what is true now", "what was true then" and "who says
so". Predicates declared *functional* (``capital_of``, ``version``) hold one
value at a time: asserting a new one closes the old one instead of
contradicting it.

On top of the facts: dependency queries (what does X need, what breaks if X
changes, in what order to build), shortest paths, cycle detection, a small
rule engine for inferred facts (transitivity, inverses), and - when
``networkx`` is installed - export to a ``MultiDiGraph`` and PageRank centrality.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable

from pydantic import Field

from ..core.types import Model

try:
    import networkx as _nx
except Exception:                                           # pragma: no cover - optional
    _nx = None


def canonical(name: str) -> str:
    """The key an entity is stored under: case- and whitespace-insensitive."""
    return re.sub(r"\s+", " ", name.strip()).lower()


class Entity(Model):
    id: str
    name: str
    type: str = "thing"
    attributes: dict[str, Any] = Field(default_factory=dict)


class Fact(Model):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = ""
    valid_from: float = 0.0
    valid_to: float | None = None        # None = still true
    inferred: bool = False

    def __str__(self) -> str:
        return f"{self.subject} --{self.predicate}--> {self.object}"


class CycleError(ValueError):
    """A dependency cycle makes an ordering impossible."""


# "X depends on Y", "X is a Y", "X requires Y" ... -> (predicate, swap subject/object?)
_PATTERNS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"^(.+?) (?:depends on|requires|needs|relies on) (.+)$", re.I), "depends_on", False),
    (re.compile(r"^(.+?) is required by (.+)$", re.I), "depends_on", True),
    (re.compile(r"^(.+?) is (?:a|an) (?:kind|type|sort) of (.+)$", re.I), "is_a", False),
    (re.compile(r"^(.+?) is part of (.+)$", re.I), "part_of", False),
    (re.compile(r"^(.+?) (?:contains|includes|consists of) (.+)$", re.I), "part_of", True),
    (re.compile(r"^(.+?) (?:causes|leads to|results in) (.+)$", re.I), "causes", False),
    (re.compile(r"^(.+?) is caused by (.+)$", re.I), "causes", True),
    (re.compile(r"^(.+?) is the (\w[\w ]*?) of (.+)$", re.I), "", False),          # "Delhi is the capital of India"
    (re.compile(r"^(.+?) is located in (.+)$", re.I), "located_in", False),
    (re.compile(r"^(.+?) (?:is|are) (?:a|an) (.+)$", re.I), "is_a", False),
    (re.compile(r"^(.+?) (?:uses|calls|imports) (.+)$", re.I), "uses", False),
]


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Pull simple facts out of plain sentences. Deliberately conservative: unmatched text yields nothing."""
    triples = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        sentence = sentence.strip().rstrip(".!?;").strip()
        if not sentence or len(sentence) > 200:
            continue
        for pattern, predicate, swap in _PATTERNS:
            match = pattern.match(sentence)
            if not match:
                continue
            groups = [g.strip() for g in match.groups()]
            if len(groups) == 3:                            # "<s> is the <relation> of <o>"
                subject, predicate, obj = groups[0], groups[1].lower().replace(" ", "_") + "_of", groups[2]
            else:
                subject, obj = (groups[1], groups[0]) if swap else groups
            article = re.compile(r"^(?:the|a|an) ", re.I)
            subject = article.sub("", subject)
            for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", obj):
                part = article.sub("", part.strip())
                if part:
                    triples.append((subject, predicate, part))
            break
    return triples


class SemanticGraph:
    def __init__(self, path: str = ":memory:", functional: Iterable[str] = (),
                 clock: Callable[[], float] = time.time):
        self.clock = clock
        self.functional = set(functional)
        self._rules: list[tuple[str, str, str]] = []        # (kind, predicate, other)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript("""
            create table if not exists entities (id text primary key, name text, type text, attributes text);
            create table if not exists facts (subject text, predicate text, object text, confidence real,
                source text, valid_from real, valid_to real);
            create index if not exists facts_s on facts(subject, predicate);
            create index if not exists facts_o on facts(object, predicate);""")
        self._db.commit()

    # -- entities ---------------------------------------------------------- #
    def add_entity(self, name: str, type: str | None = None, **attributes: Any) -> Entity:
        key = canonical(name)
        if not key:
            raise ValueError("an entity needs a name")
        with self._lock:
            existing = self.entity(name)
            if existing:
                merged = Entity(id=key, name=existing.name, type=type or existing.type,
                                attributes={**existing.attributes, **attributes})
            else:
                merged = Entity(id=key, name=name.strip(), type=type or "thing", attributes=attributes)
            self._db.execute("insert or replace into entities values (?,?,?,?)",
                             (key, merged.name, merged.type, json.dumps(merged.attributes, default=str)))
            self._db.commit()
        return merged

    def entity(self, name: str) -> Entity | None:
        with self._lock:
            row = self._db.execute("select id, name, type, attributes from entities where id=?",
                                   (canonical(name),)).fetchone()
        return Entity(id=row[0], name=row[1], type=row[2], attributes=json.loads(row[3])) if row else None

    def entities(self, type: str | None = None) -> list[Entity]:
        with self._lock:
            rows = self._db.execute("select id, name, type, attributes from entities "
                                    "where (? is null or type=?) order by id", (type, type)).fetchall()
        return [Entity(id=r[0], name=r[1], type=r[2], attributes=json.loads(r[3])) for r in rows]

    # -- facts ------------------------------------------------------------- #
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0, source: str = "",
                 at: float | None = None) -> Fact:
        """Assert a fact. Re-asserting raises its confidence; a functional predicate replaces its old value."""
        s, o = canonical(subject), canonical(obj)
        predicate = predicate.strip().lower().replace(" ", "_")
        now = self.clock() if at is None else at
        fact = Fact(subject=s, predicate=predicate, object=o, confidence=confidence, source=source, valid_from=now)
        with self._lock:
            for name, key in ((subject, s), (obj, o)):
                if self.entity(key) is None:
                    self.add_entity(name)
            same = self._db.execute(
                "select rowid, confidence, source from facts where subject=? and predicate=? and object=? "
                "and valid_to is null", (s, predicate, o)).fetchone()
            if same:
                # Independent confirmations compound: 1 - (1-a)(1-b).
                known = [part for part in same[2].split("; ") if part]
                independent = bool(source) and source not in known
                combined = 1 - (1 - same[1]) * (1 - confidence) if independent else max(same[1], confidence)
                sources = "; ".join(known + [source]) if independent else same[2]
                self._db.execute("update facts set confidence=?, source=? where rowid=?", (combined, sources, same[0]))
                self._db.commit()
                return fact.model_copy(update={"confidence": combined, "source": sources})
            if predicate in self.functional:
                self._db.execute("update facts set valid_to=? where subject=? and predicate=? and valid_to is null",
                                 (now, s, predicate))
            self._db.execute("insert into facts values (?,?,?,?,?,?,?)", (s, predicate, o, confidence, source, now, None))
            self._db.commit()
        return fact

    def add_text(self, text: str, source: str = "", confidence: float = 0.8) -> list[Fact]:
        """Read simple declarative sentences into facts (see :func:`extract_triples`)."""
        return [self.add_fact(s, p, o, confidence, source) for s, p, o in extract_triples(text)]

    def retract(self, subject: str, predicate: str, obj: str | None = None) -> int:
        """Mark matching facts as no longer true (they stay in history)."""
        with self._lock:
            cursor = self._db.execute(
                "update facts set valid_to=? where subject=? and predicate=? and (? is null or object=?) "
                "and valid_to is null",
                (self.clock(), canonical(subject), predicate, obj and canonical(obj), obj and canonical(obj)))
            self._db.commit()
            return cursor.rowcount

    def facts(self, subject: str | None = None, predicate: str | None = None, obj: str | None = None,
              at: float | None = None, include_inferred: bool = True, min_confidence: float = 0.0) -> list[Fact]:
        """Facts matching the pattern (``None`` is a wildcard) that held at time ``at`` (default: now)."""
        s, o = (canonical(subject) if subject else None), (canonical(obj) if obj else None)
        with self._lock:
            if at is None:
                clause, params = "valid_to is null", ()
            else:
                clause, params = "valid_from<=? and (valid_to is null or valid_to>?)", (at, at)
            rows = self._db.execute(
                f"select subject, predicate, object, confidence, source, valid_from, valid_to from facts "
                f"where {clause} and confidence>=? order by rowid", (*params, min_confidence)).fetchall()
        stored = [Fact(subject=r[0], predicate=r[1], object=r[2], confidence=r[3], source=r[4],
                       valid_from=r[5], valid_to=r[6]) for r in rows]
        everything = stored + (self._infer(stored) if include_inferred and self._rules else [])
        return [f for f in everything if (s is None or f.subject == s)
                and (predicate is None or f.predicate == predicate) and (o is None or f.object == o)]

    def ask(self, subject: str, predicate: str, at: float | None = None) -> list[str]:
        """The object(s) of ``subject --predicate-->``, most confident first, by display name."""
        found = sorted(self.facts(subject, predicate, at=at), key=lambda f: -f.confidence)
        return [(self.entity(f.object) or Entity(id=f.object, name=f.object)).name for f in found]

    def holds(self, subject: str, predicate: str, obj: str, at: float | None = None) -> bool:
        return bool(self.facts(subject, predicate, obj, at=at))

    def history(self, subject: str, predicate: str) -> list[Fact]:
        with self._lock:
            rows = self._db.execute(
                "select subject, predicate, object, confidence, source, valid_from, valid_to from facts "
                "where subject=? and predicate=? order by valid_from, rowid", (canonical(subject), predicate)).fetchall()
        return [Fact(subject=r[0], predicate=r[1], object=r[2], confidence=r[3], source=r[4],
                     valid_from=r[5], valid_to=r[6]) for r in rows]

    # -- inference --------------------------------------------------------- #
    def add_rule(self, kind: str, predicate: str, other: str = "") -> None:
        """``transitive`` (a>b, b>c => a>c), ``symmetric`` (a~b => b~a), ``inverse`` (a p b => b other a),
        or ``inherits`` (a is_a b, b p c => a p c; pass the inherited predicate as ``other``)."""
        if kind not in ("transitive", "symmetric", "inverse", "inherits"):
            raise ValueError(f"unknown rule kind {kind!r}")
        if kind in ("inverse", "inherits") and not other:
            raise ValueError(f"a {kind} rule needs a second predicate")
        self._rules.append((kind, predicate, other))

    def _infer(self, stored: list[Fact]) -> list[Fact]:
        known = {(f.subject, f.predicate, f.object): f.confidence for f in stored}
        derived: dict[tuple, float] = {}

        def propose(triple: tuple, confidence: float) -> bool:
            if triple[0] == triple[2] or triple in known or derived.get(triple, -1.0) >= confidence:
                return False
            derived[triple] = confidence
            return True

        for _ in range(12):                                 # to a fixed point, bounded
            changed = False
            everything = {**known, **derived}
            by_subject: dict[tuple, list] = {}
            for (s, p, o), c in everything.items():
                by_subject.setdefault((s, p), []).append((o, c))
            for kind, predicate, other in self._rules:
                for (s, p, o), c in list(everything.items()):
                    if kind == "symmetric" and p == predicate:
                        changed |= propose((o, p, s), c)
                    elif kind == "inverse" and p == predicate:
                        changed |= propose((o, other, s), c)
                    elif kind == "transitive" and p == predicate:
                        for o2, c2 in by_subject.get((o, p), []):
                            changed |= propose((s, p, o2), c * c2)
                    elif kind == "inherits" and p == predicate:
                        for o2, c2 in by_subject.get((o, other), []):
                            changed |= propose((s, other, o2), c * c2)
            if not changed:
                break
        return [Fact(subject=s, predicate=p, object=o, confidence=c, source="inferred", inferred=True)
                for (s, p, o), c in derived.items()]

    # -- graph queries ----------------------------------------------------- #
    def _adjacency(self, predicate: str | None, reverse: bool = False) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for fact in self.facts(predicate=predicate, include_inferred=False):
            a, b = (fact.object, fact.subject) if reverse else (fact.subject, fact.object)
            graph.setdefault(a, []).append(b)
        return graph

    def _reach(self, start: str, graph: dict[str, list[str]]) -> list[str]:
        seen, order, queue = {start}, [], deque([start])
        while queue:
            for nxt in graph.get(queue.popleft(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    order.append(nxt)
                    queue.append(nxt)
        return order

    def dependencies(self, name: str, predicate: str = "depends_on") -> list[str]:
        """Everything ``name`` needs, directly or indirectly (nearest first)."""
        return self._reach(canonical(name), self._adjacency(predicate))

    def dependents(self, name: str, predicate: str = "depends_on") -> list[str]:
        """Everything that would be affected if ``name`` changed."""
        return self._reach(canonical(name), self._adjacency(predicate, reverse=True))

    def find_cycle(self, predicate: str = "depends_on") -> list[str] | None:
        graph = self._adjacency(predicate)
        state: dict[str, int] = {}

        def visit(node: str, trail: list[str]) -> list[str] | None:
            state[node] = 1
            for nxt in graph.get(node, []):
                if state.get(nxt) == 1:
                    return trail[trail.index(nxt):] + [nxt] if nxt in trail else [node, nxt]
                if nxt not in state:
                    found = visit(nxt, trail + [nxt])
                    if found:
                        return found
            state[node] = 2
            return None

        for node in list(graph):
            if node not in state:
                cycle = visit(node, [node])
                if cycle:
                    return cycle
        return None

    def build_order(self, targets: Iterable[str] | None = None, predicate: str = "depends_on") -> list[str]:
        """An order in which every item comes after what it depends on."""
        cycle = self.find_cycle(predicate)
        if cycle:
            raise CycleError("dependency cycle: " + " -> ".join(cycle))
        graph = self._adjacency(predicate)
        if targets is None:
            nodes = sorted(set(graph) | {n for deps in graph.values() for n in deps})
        else:
            nodes = []
            for target in map(canonical, targets):
                nodes += [n for n in [target, *self._reach(target, graph)] if n not in nodes]
        order: list[str] = []
        done: set[str] = set()

        def place(node: str) -> None:
            if node in done:
                return
            done.add(node)
            for dep in sorted(graph.get(node, [])):
                place(dep)
            order.append(node)

        for node in nodes:
            place(node)
        return order

    def path(self, start: str, goal: str, max_hops: int = 6) -> list[Fact] | None:
        """The shortest chain of facts connecting two entities, in either direction."""
        start, goal = canonical(start), canonical(goal)
        if start == goal:
            return []
        links: dict[str, list[tuple[str, Fact]]] = {}
        for fact in self.facts(include_inferred=False):
            links.setdefault(fact.subject, []).append((fact.object, fact))
            links.setdefault(fact.object, []).append((fact.subject, fact))
        queue, seen = deque([(start, [])]), {start}
        while queue:
            node, trail = queue.popleft()
            if len(trail) >= max_hops:
                continue
            for nxt, fact in links.get(node, []):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return trail + [fact]
                seen.add(nxt)
                queue.append((nxt, trail + [fact]))
        return None

    def neighbourhood(self, name: str, hops: int = 1) -> list[Fact]:
        frontier, seen, found = {canonical(name)}, {canonical(name)}, []
        facts = self.facts(include_inferred=False)
        for _ in range(hops):
            nxt = set()
            for fact in facts:
                if (fact.subject in frontier or fact.object in frontier) and fact not in found:
                    found.append(fact)
                    nxt |= {fact.subject, fact.object} - seen
            seen |= nxt
            frontier = nxt
        return found

    def describe(self, name: str, hops: int = 1, limit: int = 20) -> str:
        """What is known about an entity, as text for a prompt."""
        entity = self.entity(name)
        if entity is None:
            return f"Nothing is known about {name!r}."
        lines = [f"{entity.name} ({entity.type})" + (f" {json.dumps(entity.attributes)}" if entity.attributes else "")]
        for fact in self.neighbourhood(name, hops)[:limit]:
            hedge = "" if fact.confidence >= 0.9 else f" (confidence {fact.confidence:.2f})"
            lines.append(f"- {fact.subject} {fact.predicate.replace('_', ' ')} {fact.object}{hedge}")
        return "\n".join(lines)

    def relevant(self, text: str, limit: int = 12) -> list[Fact]:
        """Facts about entities that are mentioned in ``text``."""
        lowered = f" {canonical(text)} "
        found: list[Fact] = []
        for entity in self.entities():
            if len(entity.id) > 2 and re.search(rf"(?<![\w]){re.escape(entity.id)}(?![\w])", lowered):
                found += [f for f in self.neighbourhood(entity.id) if f not in found]
        return sorted(found, key=lambda f: -f.confidence)[:limit]

    # -- networkx ---------------------------------------------------------- #
    def to_networkx(self):
        if _nx is None:
            raise ImportError("networkx is not installed (pip install networkx)")
        graph = _nx.MultiDiGraph()
        for entity in self.entities():
            graph.add_node(entity.id, name=entity.name, type=entity.type, **entity.attributes)
        for fact in self.facts(include_inferred=False):
            graph.add_edge(fact.subject, fact.object, key=fact.predicate, predicate=fact.predicate,
                           confidence=fact.confidence, source=fact.source)
        return graph

    def central(self, k: int = 5) -> list[tuple[str, float]]:
        """The most connected entities: PageRank with networkx, plain degree without."""
        if _nx is not None:
            graph = _nx.DiGraph(self.to_networkx())
            if graph.number_of_nodes() == 0:
                return []
            try:
                ranks = _nx.pagerank(graph)
            except Exception:                               # pagerank needs numpy/scipy in some versions
                ranks = dict(_nx.degree_centrality(graph))
        else:
            ranks = {}
            for fact in self.facts(include_inferred=False):
                for node in (fact.subject, fact.object):
                    ranks[node] = ranks.get(node, 0.0) + 1.0
        return sorted(ranks.items(), key=lambda pair: (-pair[1], pair[0]))[:k]

    def stats(self) -> dict:
        with self._lock:
            entities = self._db.execute("select count(*) from entities").fetchone()[0]
            current = self._db.execute("select count(*) from facts where valid_to is null").fetchone()[0]
            total = self._db.execute("select count(*) from facts").fetchone()[0]
        return {"entities": entities, "facts": current, "historical_facts": total - current,
                "rules": len(self._rules), "networkx": _nx is not None}

    def close(self) -> None:
        with self._lock:
            self._db.close()
