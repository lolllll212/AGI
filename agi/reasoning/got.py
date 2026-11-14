"""Graph of Thoughts: reasoning as a graph, where thoughts can be merged as well as extended.

A tree of thoughts can only branch. A graph can also **aggregate** - combine
several partial results into one (merge sorted runs, reconcile two drafts) -
and **refine** a thought in place. The controller here runs a plan of such
operations over any thought type; the operations themselves are plain
callables, so they can be deterministic code or calls to a model
(:class:`LLMOperations`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.llm import extract_json


@dataclass
class Thought:
    id: int
    content: Any
    op: str                              # the operation that produced it
    parents: tuple = ()
    score: float | None = None
    alive: bool = True


@dataclass
class GraphResult:
    best: Thought | None
    thoughts: list[Thought]
    operations: int = 0
    solved: bool = False
    log: list[str] = field(default_factory=list)

    def lineage(self, thought: Thought | None = None) -> list[Thought]:
        """Every thought the given one was built from, oldest first."""
        thought = thought or self.best
        if thought is None:
            return []
        by_id = {t.id: t for t in self.thoughts}
        seen: dict[int, Thought] = {}

        def visit(node: Thought) -> None:
            for parent in node.parents:
                if parent not in seen:
                    visit(by_id[parent])
            seen[node.id] = node

        visit(thought)
        return list(seen.values())

    def to_dot(self) -> str:
        lines = ["digraph thoughts {"]
        for t in self.thoughts:
            label = f"{t.op}\\n{str(t.content)[:30]}" + (f"\\n{t.score:.2f}" if t.score is not None else "")
            lines.append(f'  t{t.id} [label="{label}"{"" if t.alive else ", style=dashed"}];')
            lines += [f"  t{p} -> t{t.id};" for p in t.parents]
        return "\n".join(lines + ["}"])


class GraphOfThoughts:
    """Executes a plan of operations over a growing graph of thoughts.

    Operations (all optional except ``score`` when ``keep`` is used):

    * ``generate(content, k) -> list``   - k new thoughts from one
    * ``aggregate(contents) -> content`` - one thought from several
    * ``refine(content) -> content``     - an improved version of one
    * ``score(content) -> float``        - higher is better
    * ``solved(content) -> bool``        - stop as soon as a thought passes

    A plan is a list of steps: ``("generate", k)``, ``("score",)``,
    ``("keep", n)``, ``("aggregate",)`` or ``("aggregate", group_size)``, and
    ``("refine", rounds)``.
    """

    def __init__(self, generate: Callable | None = None, aggregate: Callable | None = None,
                 refine: Callable | None = None, score: Callable | None = None,
                 solved: Callable | None = None, max_operations: int = 200):
        self.ops = {"generate": generate, "aggregate": aggregate, "refine": refine, "score": score}
        self.solved_check, self.max_operations = solved, max_operations

    def run(self, initial: Any, plan: Sequence[tuple]) -> GraphResult:
        thoughts = [Thought(0, initial, "input")]
        result = GraphResult(None, thoughts)
        frontier = [thoughts[0]]

        def add(content: Any, op: str, parents: Sequence[Thought]) -> Thought:
            thought = Thought(len(thoughts), content, op, tuple(p.id for p in parents))
            thoughts.append(thought)
            return thought

        def need(name: str) -> Callable:
            if self.ops[name] is None:
                raise ValueError(f"the plan uses '{name}' but no {name} operation was given")
            return self.ops[name]

        def spend(what: str) -> bool:
            if result.operations >= self.max_operations:
                result.log.append(f"stopped: operation limit of {self.max_operations} reached at {what}")
                return False
            result.operations += 1
            return True

        def rate(batch: Sequence[Thought]) -> None:
            for thought in batch:
                if thought.score is None and self.ops["score"] is not None:
                    thought.score = float(self.ops["score"](thought.content))

        for step in plan:
            name, args = step[0], step[1:]
            if name == "generate":
                k, fresh = (args[0] if args else 1), []
                for thought in frontier:
                    if not spend("generate"):
                        break
                    fresh += [add(content, "generate", [thought]) for content in need("generate")(thought.content, k)]
                frontier = fresh
            elif name == "score":
                need("score")
                rate(frontier)
            elif name == "keep":
                need("score")
                rate(frontier)
                ranked = sorted(frontier, key=lambda t: -(t.score or 0.0))
                for dropped in ranked[args[0] if args else 1:]:
                    dropped.alive = False
                frontier = ranked[: args[0] if args else 1]
            elif name == "aggregate":
                size = args[0] if args else len(frontier)
                merged = []
                for start in range(0, len(frontier), max(1, size)):
                    group = frontier[start:start + size]
                    if len(group) == 1:
                        merged.append(group[0])
                    elif spend("aggregate"):
                        merged.append(add(need("aggregate")([t.content for t in group]), "aggregate", group))
                frontier = merged
            elif name == "refine":
                for _ in range(args[0] if args else 1):
                    improved = []
                    for thought in frontier:
                        if not spend("refine"):
                            improved.append(thought)
                            continue
                        candidate = add(need("refine")(thought.content), "refine", [thought])
                        rate([thought, candidate])
                        # Keep a refinement only if it is not worse.
                        better = candidate.score is None or (candidate.score >= (thought.score or 0.0))
                        (thought if better else candidate).alive = False
                        improved.append(candidate if better else thought)
                    frontier = improved
            else:
                raise ValueError(f"unknown operation {name!r} in the plan")
            result.log.append(f"{name}{args or ''}: {len(frontier)} thought(s)")
            if self.solved_check is not None:
                done = next((t for t in frontier if self.solved_check(t.content)), None)
                if done is not None:
                    result.best, result.solved = done, True
                    return result
            if not frontier:
                break
        rate(frontier)
        result.best = max(frontier, key=lambda t: t.score if t.score is not None else 0.0) if frontier else None
        if result.best is not None and self.solved_check is not None:
            result.solved = bool(self.solved_check(result.best.content))
        return result


class LLMOperations:
    """Graph-of-Thoughts operations backed by a model, for free-text problems."""

    def __init__(self, llm: Callable[[str], str], task: str):
        self.llm, self.task = llm, task

    def _list(self, prompt: str) -> list[str]:
        try:
            value = extract_json(self.llm(prompt))
        except Exception:
            return []
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, list)), [])
        return [str(item) for item in value] if isinstance(value, list) else []

    def generate(self, content: str, k: int) -> list[str]:
        return self._list(f"TASK: {self.task}\n\nCURRENT THINKING:\n{content}\n\nPropose {k} different next "
                          f"steps or approaches. Reply with a JSON list of {k} strings.")[:k]

    def aggregate(self, contents: Sequence[str]) -> str:
        joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contents))
        return self.llm(f"TASK: {self.task}\n\nCombine these partial results into one that keeps what is "
                        f"right in each and resolves any conflict:\n\n{joined}\n\nCombined result:").strip()

    def refine(self, content: str) -> str:
        return self.llm(f"TASK: {self.task}\n\nDRAFT:\n{content}\n\nFind the weakest point of this draft and "
                        f"fix it. Reply with the improved draft only.").strip()

    def score(self, content: str) -> float:
        try:
            value = extract_json(self.llm(f"TASK: {self.task}\n\nCANDIDATE:\n{content}\n\nHow well does this "
                                          f'solve the task? Reply with JSON: {{"score": <0 to 10>}}'))
            return max(0.0, min(1.0, float(value["score"]) / 10))
        except Exception:
            return 0.0
