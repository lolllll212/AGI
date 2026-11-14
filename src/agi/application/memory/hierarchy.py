"""The four memory tiers behind one object.

==========  ==================================  =====================================
tier        holds                               backed by
==========  ==================================  =====================================
working     the current context window          :class:`~agi.application.memory.working.WorkingMemory`
episodic    traces, preferences, snapshots      ``episodic_store`` adapter (SQLite + vector index)
semantic    entities, facts, dependencies       ``semantic_graph`` adapter (SQLite + graph)
procedural  verified code, tool trajectories    ``procedural_memory`` adapter (SQLite)
==========  ==================================  =====================================

plus the lesson engine from earlier versions (``lessons``), which turns
finished episodes into advice.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Sequence

from agi.application.memory.engine import MemoryEngine
from agi.application.memory.working import WorkingMemory
from agi.domain.ports import make


class HierarchicalMemory:
    def __init__(self, directory: str | None = None, sandbox=None, embed: Callable | None = None,
                 context_budget: int = 8000, summarizer: Callable[[str], str] | None = None,
                 vector_backend: str = "auto", functional: Iterable[str] = ("version", "status", "owner"),
                 lessons: MemoryEngine | None = None):
        """``directory=None`` keeps everything in RAM; otherwise each tier gets a file in it."""
        def path(name: str) -> str:
            if directory is None:
                return ":memory:"
            os.makedirs(directory, exist_ok=True)
            return os.path.join(directory, name)

        self.directory = directory
        self.working = WorkingMemory(budget=context_budget, summarizer=summarizer)
        self.episodic = make("episodic_store", path("episodic.db"), embed=embed, backend=vector_backend)
        self.semantic = make("semantic_graph", path("semantic.db"), functional=functional)
        self.procedural = make("procedural_memory", path("procedural.db"), sandbox=sandbox)
        self.lessons = lessons if lessons is not None else MemoryEngine(path("lessons.db"))
        self._owns_lessons = lessons is None

    # -- reading ----------------------------------------------------------- #
    def context_for(self, task: str, k: int = 3, max_chars: int = 3000) -> str:
        """What the tiers know that bears on ``task``, as a block of text for a prompt."""
        sections: list[str] = []
        preferences = self.episodic.preferences()
        if preferences:
            sections.append("USER PREFERENCES\n" + "\n".join(f"- {k_}: {v}" for k_, v in preferences.items()))
        runs = self.episodic.search(task, k=k, kind="trace", min_score=0.3)
        if runs:
            lines = []
            for hit in runs:
                data = hit.record.data
                actions = ", ".join(str(s.get("action")) for s in data.get("steps", []) if s.get("action"))
                lines.append(f"- [{data.get('outcome')}] {data.get('task', '')[:140]}"
                             + (f" (used: {actions})" if actions else "")
                             + (f" -> {str(data['answer'])[:120]}" if data.get("answer") else ""))
            sections.append("SIMILAR PAST RUNS\n" + "\n".join(lines))
        facts = self.semantic.relevant(task, limit=8)
        if facts:
            sections.append("KNOWN FACTS\n" + "\n".join(
                f"- {f.subject} {f.predicate.replace('_', ' ')} {f.object}" for f in facts))
        snippets = self.procedural.search(task, kind="snippet", k=2, min_score=0.3)
        if snippets:
            sections.append("VERIFIED CODE YOU CAN REUSE\n" + "\n\n".join(
                f"# {m.procedure.name}: {m.procedure.description}\n{m.procedure.code}" for m in snippets))
        text = "\n\n".join(sections)
        return text if len(text) <= max_chars else text[:max_chars] + "\n[memory context truncated]"

    # -- writing ----------------------------------------------------------- #
    def remember_run(self, task: str, success: bool, steps: Sequence[dict], answer: str | None,
                     tags: Iterable[str] = (), store_trajectory: bool = True, **data: Any) -> None:
        """File a finished run: a trace always, a reusable trajectory only for a clean success."""
        self.episodic.add_trace(task, "success" if success else "failure", steps, answer, tags, **data)
        calls = [{"tool": s["action"], "args": s.get("args", {})} for s in steps if s.get("action") and s.get("ok")]
        clean = success and calls and all(s.get("ok") for s in steps if s.get("action"))
        if clean and store_trajectory:
            last = next((s.get("observation", "") for s in reversed(steps) if s.get("action")), "")
            self.procedural.add_trajectory(task, calls, answer, last, tags)

    def stats(self) -> dict:
        return {"working": self.working.stats().model_dump(), "episodic": self.episodic.stats(),
                "semantic": self.semantic.stats(), "procedural": self.procedural.stats(),
                "lessons": self.lessons.stats()}

    def close(self) -> None:
        self.episodic.close()
        self.semantic.close()
        self.procedural.close()
        if self._owns_lessons:
            self.lessons.close()

    def __enter__(self) -> "HierarchicalMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
