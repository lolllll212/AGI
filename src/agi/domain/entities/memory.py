"""Records the memory engine stores: raw episodes and the lessons distilled from them."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

SUCCESS, FAILURE = "success", "failure"
#: Lesson kinds. ``avoid`` = something that failed, ``fix`` = failure plus what
#: worked instead, ``procedure`` = a known-good approach, ``fact`` = free-form.
KINDS = ("avoid", "fix", "procedure", "fact")


@dataclass
class Step:
    """One thing the agent did and what the world answered."""

    action: str
    ok: bool = True
    result: Any = None
    error: str | None = None


@dataclass
class Episode:
    """A complete attempt at a task: the unit of experience."""

    task: str
    outcome: str = SUCCESS
    steps: list = field(default_factory=list)
    tags: tuple = ()
    error: str | None = None
    solution: Any = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: "ep-" + uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        self.steps = [s if isinstance(s, Step) else Step(**s) for s in self.steps]
        self.tags = tuple(self.tags)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Lesson:
    """A reusable piece of knowledge, with evidence and a running utility estimate."""

    kind: str
    trigger: str            # the situation in which the lesson applies
    advice: str             # what to do (or not do) about it
    tags: tuple = ()
    payload: dict = field(default_factory=dict)   # machine-usable form of the advice
    support: int = 1        # how many episodes produced this lesson
    helped: int = 0         # times it was retrieved and the task then succeeded
    hurt: int = 0           # times it was retrieved and the task then failed
    episodes: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    last_used: float = 0.0
    id: str = ""

    def __post_init__(self):
        self.tags = tuple(self.tags)
        if not self.id:
            digest = hashlib.sha256(json.dumps(
                [self.kind, self.trigger, self.advice], sort_keys=True).encode()).hexdigest()
            self.id = "ls-" + digest[:12]

    @property
    def utility(self) -> float:
        """Posterior mean of 'retrieving this helps' under a uniform Beta prior."""
        return (self.helped + 1) / (self.helped + self.hurt + 2)

    @property
    def text(self) -> str:
        return f"{self.trigger} {self.advice} {' '.join(self.tags)}"

    def render(self) -> str:
        return f"[{self.kind}] When: {self.trigger}. Then: {self.advice}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Lesson":
        return cls(**data)
