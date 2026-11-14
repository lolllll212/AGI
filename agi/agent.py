"""All four ideas in one loop: recall -> reason -> verify -> remember.

:class:`LearningAgent` solves programming-by-example tasks. Before searching
it asks memory what it already knows; remembered solutions are *re-verified*
in the sandbox rather than trusted; whatever the search learns - the answer,
and the dead ends on the way to it - is written back. The model weights (there
are none) never change. The agent still gets faster.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .memory import FAILURE, SUCCESS, Episode, Lesson, MemoryEngine, Step
from .reasoning import MCTS, PipelineSynthesis
from .sandbox import Sandbox


@dataclass
class Task:
    description: str
    examples: Sequence[tuple]
    tags: tuple = ()


@dataclass
class Attempt:
    task: Task
    solved: bool
    program: tuple | None
    source: str | None
    verifications: int            # hypotheses actually executed
    refuted: int                  # hypotheses refuted along the way
    how: str                      # "recalled" | "guided search" | "search"
    recalled: list = field(default_factory=list)
    learned: list = field(default_factory=list)
    seconds: float = 0.0

    def __str__(self) -> str:
        answer = " | ".join(self.program) if self.program else "no solution"
        return (f"{'solved' if self.solved else 'FAILED'} via {self.how}: {answer}  "
                f"[{self.verifications} programs executed, {self.refuted} refuted, "
                f"{len(self.recalled)} lessons recalled, {self.seconds:.1f}s]")


class LearningAgent:
    def __init__(self, memory: MemoryEngine, sandbox: Sandbox | None = None,
                 budget: int = 3000, seed: int = 0):
        self.memory, self.sandbox, self.budget, self.seed = memory, sandbox, budget, seed

    def solve(self, task: Task) -> Attempt:
        started = time.perf_counter()
        hits = self.memory.recall(task.description, task.tags, k=5)
        lessons = [hit.lesson for hit in hits]

        # 1. Turn what memory returned into guidance for the search.
        priors: dict[str, float] = {}
        avoid: set[tuple] = set()
        remembered: list[tuple[Lesson, tuple]] = []
        for hit in hits:
            payload = hit.lesson.payload
            solution = payload.get("solution")
            if hit.lesson.kind == "procedure" and isinstance(solution, list):
                remembered.append((hit.lesson, tuple(solution)))
                for op in solution:
                    priors[op] = max(priors.get(op, 1.0), 1.0 + 4.0 * min(1.0, hit.score))
            for failed in payload.get("failed", []):
                ops = [op.strip() for op in str(failed).split("|")]
                if payload.get("error_type") != "TypeMismatch" and ops and ops[0]:
                    avoid.add((ops[-2] if len(ops) > 1 else None, ops[-1]))

        problem = PipelineSynthesis(task.examples, self.sandbox, priors=priors, avoid=avoid)
        known = [(lesson, program) for lesson, program in remembered
                 if all(op in problem.ops for op in program)]
        verifications = refuted = 0
        program, how = None, "search"

        # 2. Try remembered solutions first - but verify them; memory can be wrong or stale.
        for lesson, candidate in known:
            verifications += 1
            if problem.verify(candidate).solved:
                program, how = candidate, "recalled"
                break
            refuted += 1

        # 3. Otherwise search, steered by the priors.
        if program is None:
            result = MCTS(problem, seed=self.seed,
                          max_verifications=self.budget).search(self.budget * 4)
            verifications += result.verifications
            refuted += result.backtracks
            if result.solved:
                program = result.best_state
                how = "guided search" if priors else "search"

        # 4. Write the experience back, then tell memory which lessons earned their keep.
        steps = [Step(problem.describe(p), ok=False, error=e) for p, e in problem.failures[:40]]
        if program is not None:
            steps.append(Step(problem.describe(program), ok=True))
        episode = Episode(task=task.description, outcome=SUCCESS if program is not None else FAILURE,
                          steps=steps, tags=task.tags,
                          solution=list(program) if program is not None else None,
                          error=None if program is not None else "Unsolved: search budget exhausted",
                          meta={"verifications": verifications, "how": how})
        learned = self.memory.record(episode)
        if program is not None:
            for lesson, candidate in known:
                overlap = len(set(candidate) & set(program)) / len(set(program))
                self.memory.feedback(lesson.id, helped=overlap >= 0.5)

        return Attempt(task, program is not None, program,
                       problem.source(program) if program is not None else None,
                       verifications, refuted, how, lessons, learned,
                       time.perf_counter() - started)
