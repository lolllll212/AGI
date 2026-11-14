"""Procedural memory: things the agent knows how to *do*.

Three kinds of procedure, all retrieved by similarity to the task at hand:

* **snippet** - a piece of code that passed its tests in the sandbox before it was admitted;
* **trajectory** - a sequence of tool calls that solved a task, stored with the
  task it solved so it can be *adapted*: when a new task differs from the old
  one only in values that also appear in the calls ("North" -> "South"), the
  calls are rewritten and can be replayed without asking a model anything;
* **workflow** - a named, parameterised sequence of tool calls written on purpose.

Every use is scored. A procedure that keeps failing loses its reliability and
stops being offered - that is what makes it safe to act on without deliberation.
"""
from __future__ import annotations

import difflib
import json
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Literal, Sequence

from pydantic import Field

from ..core.types import Model, ToolCall
from .embed import HashingEmbedder, cosine

Kind = Literal["snippet", "trajectory", "workflow"]

_TOKEN = re.compile(r"\d[\d,]*\.\d+|\d[\d,]*|[A-Za-z_][\w'-]*|\S")


class Procedure(Model):
    id: str
    kind: Kind
    name: str
    description: str                     # for a trajectory: the task it solved
    code: str | None = None              # snippet
    tests: list[str] = Field(default_factory=list)
    calls: list[ToolCall] = Field(default_factory=list)       # trajectory / workflow
    params: list[str] = Field(default_factory=list)           # workflow: {placeholders} to fill
    answer_template: str | None = None   # trajectory: the answer with placeholders for the final output
    output_figures: int | None = None    # how many numbers that output held (a replay must match)
    tags: list[str] = Field(default_factory=list)
    uses: int = 0
    successes: int = 0
    created: float = 0.0
    last_used: float | None = None

    @property
    def reliability(self) -> float:
        """Laplace-smoothed success rate; admission counts as one success."""
        return (self.successes + 1) / (self.uses + 2) if self.uses else 0.75


class Adaptation(Model):
    procedure: Procedure
    calls: list[ToolCall]
    answer_template: str | None = None
    output_figures: int | None = None
    similarity: float
    substitutions: dict[str, str] = Field(default_factory=dict)
    exact: bool = False                  # the task is the same as the stored one

    @property
    def confidence(self) -> float:
        return self.similarity * self.procedure.reliability


class Match(Model):
    procedure: Procedure
    score: float


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _replace(value: Any, mapping: dict[str, str]) -> tuple[Any, int]:
    """Substitute whole-word occurrences of each old value inside strings, recursively."""
    if isinstance(value, str):
        count = 0
        for old, new in mapping.items():
            forms = {old, old.replace(",", "")} if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", old) else {old}
            pattern = re.compile(rf"(?<![\w.])(?:{'|'.join(map(re.escape, sorted(forms, key=len, reverse=True)))})(?![\w]|\.\d)")
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", new):
                new = new.replace(",", "")                  # "1,200" is the number 1200 inside a call
            value, n = pattern.subn(lambda _, new=new: new, value)
            count += n
        return value, count
    if isinstance(value, bool) or value is None:
        return value, 0
    if isinstance(value, (int, float)):
        for old, new in mapping.items():
            try:
                if float(old.replace(",", "")) == float(value):
                    replacement = float(new.replace(",", ""))
                    return (int(replacement) if isinstance(value, int) and replacement.is_integer()
                            else replacement), 1
            except ValueError:
                continue
        return value, 0
    if isinstance(value, list):
        pairs = [_replace(v, mapping) for v in value]
        return [p[0] for p in pairs], sum(p[1] for p in pairs)
    if isinstance(value, dict):
        pairs = {k: _replace(v, mapping) for k, v in value.items()}
        return {k: p[0] for k, p in pairs.items()}, sum(p[1] for p in pairs.values())
    return value, 0


def _figures(text: str) -> list[float]:
    from ..core.react import numbers_in
    return [value for value, _ in numbers_in(text)]


def render_answer(template: str, output: str, expected_figures: int | None = None) -> str | None:
    """Fill an answer template from a fresh tool output; ``None`` if the output has a different shape."""
    output = output.strip()
    figures = _figures(output)
    if expected_figures is not None and len(figures) != expected_figures:
        return None

    def fill(match: re.Match) -> str:
        index, spec = int(match.group(1)), match.group(2)
        if index >= len(figures):
            raise IndexError(index)
        return format(figures[index], spec)

    try:
        text = re.sub(r"\{#(\d+):([^}]*)\}", fill, template)
    except (IndexError, ValueError):
        return None
    return text.replace("{result}", output).replace("{{", "{").replace("}}", "}")


def align(old_task: str, new_task: str) -> dict[str, str] | None:
    """Which values changed between two tasks with the same wording.

    Returns ``{old value: new value}``, or ``None`` if the tasks differ by more
    than swapped values (words added or removed, or an inconsistent swap).
    """
    old, new = _tokens(old_task), _tokens(new_task)
    lowered_old, lowered_new = [t.lower() for t in old], [t.lower() for t in new]
    mapping: dict[str, str] = {}
    matcher = difflib.SequenceMatcher(a=lowered_old, b=lowered_new, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op != "replace":
            return None                                     # the wording itself changed
        if a1 - a0 == b1 - b0:                              # token-for-token swap
            pairs = [(old[a0 + i], new[b0 + i]) for i in range(a1 - a0)]
        else:
            pairs = [(" ".join(old[a0:a1]), " ".join(new[b0:b1]))]
        for before, after in pairs:
            if mapping.setdefault(before, after) != after:
                return None                                 # the same value maps to two things
    return mapping


class ProceduralMemory:
    def __init__(self, path: str = ":memory:", sandbox=None, clock: Callable[[], float] = time.time,
                 min_reliability: float = 0.5):
        """``sandbox`` (an :class:`~agi.core.sandbox.Sandbox`) is needed to admit snippets."""
        self.sandbox, self.clock, self.min_reliability = sandbox, clock, min_reliability
        self._embedder = HashingEmbedder()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("create table if not exists procedures (id text primary key, kind text, name text, body text)")
        self._db.commit()

    # -- storage ----------------------------------------------------------- #
    def _save(self, procedure: Procedure) -> Procedure:
        with self._lock:
            self._db.execute("insert or replace into procedures values (?,?,?,?)",
                             (procedure.id, procedure.kind, procedure.name, procedure.model_dump_json()))
            self._db.commit()
        return procedure

    def all(self, kind: Kind | None = None) -> list[Procedure]:
        with self._lock:
            rows = self._db.execute("select body from procedures where (? is null or kind=?) order by rowid",
                                    (kind, kind)).fetchall()
        return [Procedure.model_validate_json(row[0]) for row in rows]

    def get(self, name_or_id: str) -> Procedure | None:
        with self._lock:
            row = self._db.execute("select body from procedures where id=? or name=? order by rowid desc limit 1",
                                   (name_or_id, name_or_id)).fetchone()
        return Procedure.model_validate_json(row[0]) if row else None

    def forget(self, name_or_id: str) -> None:
        with self._lock:
            self._db.execute("delete from procedures where id=? or name=?", (name_or_id, name_or_id))
            self._db.commit()

    # -- admission --------------------------------------------------------- #
    def add_snippet(self, name: str, code: str, description: str, tests: Sequence[str],
                    tags: Iterable[str] = ()) -> Procedure:
        """Admit code only if every test passes in the sandbox. ``tests`` are assert statements."""
        if not tests:
            raise ValueError("a snippet needs at least one test; unverified code is not stored")
        if self.sandbox is None:
            raise RuntimeError("a sandbox is required to verify snippets")
        result = self.sandbox.run(code + "\n\n" + "\n".join(tests) + "\n")
        if not result.ok:
            raise ValueError(f"snippet {name!r} failed verification: {result.error_type}: {result.error}")
        return self._save(Procedure(id=uuid.uuid4().hex[:12], kind="snippet", name=name, description=description,
                                    code=code, tests=list(tests), tags=sorted(set(tags)), created=self.clock()))

    def add_trajectory(self, task: str, calls: Sequence[ToolCall | dict], answer: str | None = None,
                       last_output: str | None = None, tags: Iterable[str] = ()) -> Procedure | None:
        """Store the tool calls that solved ``task``. Call this only for verified successes.

        If the answer's figures all come from the final tool output, an answer
        template is kept as well, so a replay can phrase its own answer.
        """
        calls = [c if isinstance(c, ToolCall) else ToolCall(**c) for c in calls]
        if not calls:
            return None
        template = self._template(answer, last_output, task) if answer and last_output else None
        figures = len(_figures(last_output)) if template else None
        for existing in self.all("trajectory"):             # same task again: refresh, do not duplicate
            if _tokens(existing.description.lower()) == _tokens(task.lower()):
                existing.calls, existing.answer_template, existing.output_figures = calls, template, figures
                return self._save(existing)
        return self._save(Procedure(id=uuid.uuid4().hex[:12], kind="trajectory", name=f"trajectory-{uuid.uuid4().hex[:6]}",
                                    description=task, calls=calls, answer_template=template,
                                    output_figures=figures, tags=sorted(set(tags)), created=self.clock()))

    @staticmethod
    def _template(answer: str, last_output: str, task: str = "") -> str | None:
        """The answer with the final output's values replaced by placeholders, if that is unambiguous.

        ``{#i:spec}`` stands for the i-th number in the output (formatted the way
        the answer wrote it); ``{result}`` stands for a short non-numeric output.
        Returns ``None`` when the answer holds a figure that comes from neither
        the output nor the task - such an answer cannot be replayed mechanically.
        """
        from ..core.react import numbers_in                  # local: avoids an import cycle
        output = last_output.strip()
        if not output or len(output) > 2000:
            return None
        figures = [value for value, _ in numbers_in(output)]
        if not figures:
            if len(output) <= 80 and "\n" not in output and len(output) >= 2 and output in answer:
                return answer.replace("{", "{{").replace("}", "}}").replace(output, "{result}")
            return None
        task_figures = [value for value, _ in numbers_in(task)]
        pattern = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)")
        pieces, cursor, used = [], 0, 0
        for match in pattern.finditer(answer):
            token = match.group()
            value = float(token.replace(",", ""))
            decimals = len(token.partition(".")[2])
            tolerance = 0.5 * 10 ** -decimals + 1e-9
            where = [i for i, figure in enumerate(figures) if abs(figure - value) <= tolerance]
            if len({figures[i] for i in where}) > 1:
                return None                                 # two different output figures round to this one
            in_task = any(abs(figure - value) <= tolerance for figure in task_figures)
            if where and in_task:
                return None                                 # from the task or from the output? cannot tell
            if where:
                spec = f"{',' if ',' in token else ''}.{decimals}f"
                replacement = f"{{#{where[0]}:{spec}}}"
                used += 1
            elif in_task or (decimals == 0 and abs(value) < 13):
                replacement = token                         # quoted from the task, or a small count
            else:
                return None
            pieces.append(answer[cursor:match.start()].replace("{", "{{").replace("}", "}}") + replacement)
            cursor = match.end()
        if not used:
            return None
        return "".join(pieces) + answer[cursor:].replace("{", "{{").replace("}", "}}")

    def add_workflow(self, name: str, description: str, calls: Sequence[ToolCall | dict],
                     tags: Iterable[str] = ()) -> Procedure:
        """A named sequence of calls with ``{placeholders}`` in its arguments."""
        calls = [c if isinstance(c, ToolCall) else ToolCall(**c) for c in calls]
        params = sorted(set(re.findall(r"\{(\w+)\}", json.dumps([c.args for c in calls]))))
        return self._save(Procedure(id=uuid.uuid4().hex[:12], kind="workflow", name=name, description=description,
                                    calls=calls, params=params, tags=sorted(set(tags)), created=self.clock()))

    # -- retrieval --------------------------------------------------------- #
    def search(self, query: str, kind: Kind | None = None, k: int = 3, min_score: float = 0.2) -> list[Match]:
        target = self._embedder.embed(query)
        matches = []
        for procedure in self.all(kind):
            if procedure.reliability < self.min_reliability:
                continue
            score = cosine(target, self._embedder.embed(f"{procedure.name} {procedure.description} "
                                                        f"{' '.join(procedure.tags)}"))
            if score >= min_score:
                matches.append(Match(procedure=procedure, score=score))
        matches.sort(key=lambda m: -(m.score * m.procedure.reliability))
        return matches[:k]

    def adapt(self, task: str, min_similarity: float = 0.5) -> Adaptation | None:
        """A stored trajectory rewritten for ``task``, if one fits without guesswork.

        A trajectory fits when the new task has the same wording as the one it
        solved and every value that changed also appears in the stored calls
        (so the change is actually carried through). Anything else returns
        ``None``: that is the signal to stop and think instead.
        """
        best: Adaptation | None = None
        for procedure in self.all("trajectory"):
            if procedure.reliability < self.min_reliability:
                continue
            mapping = align(procedure.description, task)
            if mapping is None:
                continue
            old, new = _tokens(procedure.description), _tokens(task)
            similarity = difflib.SequenceMatcher(a=[t.lower() for t in old], b=[t.lower() for t in new],
                                                 autojunk=False).ratio()
            if similarity < min_similarity:
                continue
            calls, carried = [], set()
            for call in procedure.calls:
                args = call.args
                for before, after in mapping.items():
                    args, count = _replace(args, {before: after})
                    if count:
                        carried.add(before)
                calls.append(ToolCall(tool=call.tool, args=args))
            if carried != set(mapping):                     # a changed value the calls never used
                continue
            template = procedure.answer_template
            if template:                                    # rewrite the wording, never the placeholders
                parts = re.split(r"(\{#\d+:[^}]*\}|\{result\})", template)
                template = "".join(part if index % 2 else _replace(part, mapping)[0]
                                   for index, part in enumerate(parts))
            candidate = Adaptation(procedure=procedure, calls=calls, answer_template=template,
                                   output_figures=procedure.output_figures, similarity=similarity, substitutions=mapping, exact=not mapping)
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best

    def instantiate(self, name: str, **values: Any) -> list[ToolCall]:
        """A workflow's calls with its placeholders filled in."""
        workflow = self.get(name)
        if workflow is None or workflow.kind != "workflow":
            raise KeyError(f"no workflow named {name!r}")
        missing = [p for p in workflow.params if p not in values]
        if missing:
            raise ValueError(f"workflow {name!r} needs values for: {', '.join(missing)}")

        def fill(value: Any) -> Any:
            if isinstance(value, str):
                whole = re.fullmatch(r"\{(\w+)\}", value)
                if whole and whole.group(1) in values:      # a lone placeholder keeps the value's type
                    return values[whole.group(1)]
                return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), value)
            if isinstance(value, list):
                return [fill(v) for v in value]
            if isinstance(value, dict):
                return {k: fill(v) for k, v in value.items()}
            return value

        return [ToolCall(tool=c.tool, args=fill(c.args)) for c in workflow.calls]

    # -- feedback ---------------------------------------------------------- #
    def record_use(self, name_or_id: str, success: bool) -> Procedure | None:
        with self._lock:
            procedure = self.get(name_or_id)
            if procedure is None:
                return None
            procedure.uses += 1
            procedure.successes += int(success)
            procedure.last_used = self.clock()
            return self._save(procedure)

    def stats(self) -> dict:
        procedures = self.all()
        return {"snippets": sum(p.kind == "snippet" for p in procedures),
                "trajectories": sum(p.kind == "trajectory" for p in procedures),
                "workflows": sum(p.kind == "workflow" for p in procedures),
                "retired": sum(p.reliability < self.min_reliability for p in procedures)}

    def close(self) -> None:
        with self._lock:
            self._db.close()
