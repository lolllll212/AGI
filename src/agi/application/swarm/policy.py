"""The rules agents cannot change.

Agents in this system may write new tools for themselves. What they may *not*
do is decide what "safe" means. The policy is an immutable value handed to the
forge at construction; it lives outside the replicated state, so no proposal,
vote or message can alter it, and the forge re-checks its fingerprint before
every decision.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agi.domain.value_objects.sandbox import SandboxPolicy

HARD_BOUNDARIES = (
    "Tool code only ever runs inside the sandbox - during review and after approval.",
    "The policy is fixed at construction; agents cannot propose, vote on or edit it.",
    "Only the forge can write the tool registry (enforced by signed, namespaced actions).",
    "Authors cannot approve their own tools; approvals must come from distinct, registered agents.",
    "Evidence beats votes: a reviewer whose own tests fail is counted as a rejection.",
    "Any safety concern from any reviewer is a veto.",
    "A new version must pass every test the previous version was approved with.",
    "Tools compose only through declared dependencies, to a bounded depth.",
    "Every decision is recorded in a hash-chained audit log.",
)


@dataclass(frozen=True)
class SafetyPolicy:
    sandbox: SandboxPolicy = field(default_factory=lambda: SandboxPolicy(
        wall_time_s=3.0, cpu_time_s=2, memory_mb=256, max_source_bytes=20_000))
    quorum: float = 2 / 3              # fraction of eligible reviewers that must approve
    min_reviewers: int = 2             # independent reviewers required (author excluded)
    max_tools: int = 64
    max_dependency_depth: int = 3

    def fingerprint(self) -> str:
        data = {"sandbox": self.sandbox.fingerprint(), "quorum": round(self.quorum, 6),
                "min_reviewers": self.min_reviewers, "max_tools": self.max_tools,
                "max_dependency_depth": self.max_dependency_depth}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
