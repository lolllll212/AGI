"""The tool forge: how agents extend themselves without escaping their limits.

A proposal passes five gates, in order, and stops at the first it fails:

1. **policy** - static analysis, naming, dependency and capacity rules;
2. **author-tests** - the author's own tests, run in the sandbox;
3. **regression** - if it replaces a tool, every test that tool was approved with;
4. **review** - each independent reviewer votes and may add tests; the forge
   *runs* those tests itself rather than trusting anyone's claim;
5. **consensus** - quorum of approvals, no safety veto.

Only then does the forge - the single writer of ``tools/`` - register it.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from agi.application.swarm.ledger import Ledger
from agi.application.swarm.policy import SafetyPolicy
from agi.domain.ports import make
from agi.domain.services.source_check import check_source
from agi.domain.value_objects.sandbox import SandboxResult


def _sha(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ToolProposal:
    name: str
    description: str
    code: str
    tests: str
    author: str
    depends_on: tuple = ()

    @property
    def id(self) -> str:
        return _sha(self.name, self.code, self.tests, list(self.depends_on), self.author)[:12]

    def to_dict(self) -> dict:
        return {**asdict(self), "depends_on": list(self.depends_on), "id": self.id}


@dataclass
class Review:
    reviewer: str
    approve: bool
    reasons: list = field(default_factory=list)
    tests: str = ""                    # extra tests the forge will run itself
    safety_concern: bool = False       # a veto

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    proposal_id: str
    name: str
    author: str
    accepted: bool
    stage: str                         # the gate that decided
    reasons: list = field(default_factory=list)
    approvals: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    version: int | None = None

    def __str__(self) -> str:
        if self.accepted:
            return (f"ACCEPTED {self.name} v{self.version} by {self.author} "
                    f"(approved by {', '.join(self.approvals)})")
        shown = "; ".join(self.reasons[:2]) + (f" (+{len(self.reasons) - 2} more)"
                                               if len(self.reasons) > 2 else "")
        return f"REJECTED {self.name} by {self.author} at gate '{self.stage}': {shown}"


class AuditLog:
    """Append-only log in which each entry commits to the one before it."""

    def __init__(self):
        self.entries: list[dict] = []

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else "genesis"

    def append(self, event: str, **data: Any) -> dict:
        entry = {"index": len(self.entries), "event": event, "data": data, "prev": self.head}
        entry["hash"] = _sha(entry["index"], event, data, entry["prev"])
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev = "genesis"
        for i, entry in enumerate(self.entries):
            expected = _sha(i, entry["event"], entry["data"], prev)
            if entry["index"] != i or entry["prev"] != prev or entry["hash"] != expected:
                return False
            prev = entry["hash"]
        return True


class PolicyTampered(RuntimeError):
    pass


class ToolForge:
    def __init__(self, ledger: Ledger, policy: SafetyPolicy | None = None):
        self.ledger = ledger
        self.policy = policy or SafetyPolicy()
        self._pinned = self.policy.fingerprint()
        self.sandbox = make("sandbox", self.policy.sandbox)
        self.audit = AuditLog()
        self.audit.append("forge-started", policy=self._pinned, isolation=self.sandbox.isolation)

    # -- registry views ----------------------------------------------------- #
    def _check_policy(self) -> None:
        if self.policy.fingerprint() != self._pinned or self.sandbox.policy is not self.policy.sandbox:
            self.audit.append("policy-tampered")
            raise PolicyTampered("safety policy changed after construction; refusing to continue")

    def tools(self, at: str | None = None, include_revoked: bool = False) -> dict[str, dict]:
        """Registered tools as seen from replica ``at`` (default: the forge's own)."""
        records = (self.ledger.read(key, at) for key in self.ledger.keys("tools/", at))
        return {r["name"]: r for r in records
                if isinstance(r, dict) and (include_revoked or r.get("status") == "active")}

    def _closure(self, names: Sequence[str], registry: dict) -> list[str]:
        """Dependencies of ``names`` in load order (dependencies first)."""
        ordered: list[str] = []

        def visit(name: str, trail: tuple) -> None:
            if name in trail:
                raise ValueError(f"dependency cycle through '{name}'")
            if name not in registry:
                raise ValueError(f"unknown or revoked dependency '{name}'")
            for dep in registry[name]["depends_on"]:
                visit(dep, (*trail, name))
            if name not in ordered:
                ordered.append(name)

        for name in names:
            visit(name, ())
        return ordered

    def _depth(self, names: Sequence[str], registry: dict) -> int:
        return 1 + max((self._depth(registry[n]["depends_on"], registry) for n in names), default=0)

    def source(self, name: str, at: str | None = None) -> str:
        """Full source for a registered tool: its dependencies, then itself."""
        registry = self.tools(at)
        chunks = []
        for dep in self._closure([name], registry):
            record = registry[dep]
            if _sha(record["code"]) != record["sha256"]:
                raise ValueError(f"tool '{dep}' failed its integrity check")
            chunks.append(record["code"])
        return "\n\n".join(chunks)

    def call(self, name: str, *args: Any, at: str | None = None) -> SandboxResult:
        """Run a registered tool - in the sandbox, like everything else."""
        self._check_policy()
        registry = self.tools(at)
        if name not in registry:
            return SandboxResult(ok=False, error_type="UnknownTool",
                                 error=f"'{name}' is not a registered, active tool")
        if registry[name]["policy"] != self._pinned:
            return SandboxResult(ok=False, error_type="PolicyMismatch",
                                 error=f"'{name}' was approved under a different policy")
        try:
            source = self.source(name, at)
        except ValueError as exc:
            return SandboxResult(ok=False, error_type="IntegrityError", error=str(exc))
        return self.sandbox.call(source, name, args)

    def revoke(self, name: str, reason: str) -> None:
        """Quarantine a tool. Anything that depends on it stops loading too."""
        record = self.tools().get(name)
        if record is None:
            raise KeyError(name)
        self.ledger.put(self.ledger.kernel, f"tools/{name}", {**record, "status": "revoked",
                                                             "revoked_because": reason})
        self.ledger.sync()
        self.audit.append("revoked", name=name, reason=reason)

    # -- the gates ---------------------------------------------------------- #
    def _admissible(self, proposal: ToolProposal, registry: dict) -> list[str]:
        """Gate 1. Everything that can be decided without running the code."""
        name, problems = proposal.name, []
        if (not name.isidentifier() or keyword.iskeyword(name) or name.startswith(("_", "test_"))
                or hasattr(builtins, name) or len(name) > 40):
            return [f"'{name}' is not an acceptable tool name"]
        problems += check_source(proposal.code, self.policy.sandbox)
        problems += [f"tests: {v}" for v in check_source(proposal.tests, self.policy.sandbox)]
        if problems:
            return problems
        defined = {n.name for n in ast.parse(proposal.code).body if isinstance(n, ast.FunctionDef)}
        if name not in defined:
            problems.append(f"code must define a top-level function '{name}'")
        tests = [n.name for n in ast.parse(proposal.tests).body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        if not tests:
            problems.append("at least one test_* function is required")
        if name not in registry and len(registry) >= self.policy.max_tools:
            problems.append(f"registry is full ({self.policy.max_tools} tools)")
        others = {k: v for k, v in registry.items() if k != name}
        try:
            closure = self._closure(list(proposal.depends_on), others)
            clash = defined & set(closure)
            if clash:
                problems.append(f"code redefines its dependency {sorted(clash)}")
            if self._depth(list(proposal.depends_on), others) > self.policy.max_dependency_depth:
                problems.append(f"dependency depth exceeds {self.policy.max_dependency_depth}")
        except ValueError as exc:
            circular = str(exc).endswith(f"dependency '{name}'")
            problems.append(f"'{name}' cannot depend on itself, directly or indirectly"
                            if circular else str(exc))
        return problems

    def submit(self, proposal: ToolProposal, reviewers: Sequence, need: Any = None) -> Verdict:
        """Run a proposal through every gate and record the outcome."""
        self._check_policy()
        ledger, kernel = self.ledger, self.ledger.kernel
        ledger.put(proposal.author, f"proposals/{proposal.author}/{proposal.id}", proposal.to_dict())
        ledger.sync()
        self.audit.append("proposed", id=proposal.id, name=proposal.name, author=proposal.author)
        registry = self.tools()

        def decide(accepted: bool, stage: str, reasons=(), approvals=(), rejections=(), version=None):
            verdict = Verdict(proposal.id, proposal.name, proposal.author, accepted, stage,
                              list(reasons), list(approvals), list(rejections), version)
            ledger.put(kernel, f"verdicts/{proposal.id}", asdict(verdict))
            ledger.count(kernel, "stats/accepted" if accepted else "stats/rejected")
            ledger.sync()
            self.audit.append("accepted" if accepted else "rejected", id=proposal.id,
                              name=proposal.name, stage=stage, reasons=list(reasons),
                              approvals=list(approvals))
            return verdict

        problems = self._admissible(proposal, registry)                         # gate 1
        if problems:
            return decide(False, "policy", problems)

        others = {k: v for k, v in registry.items() if k != proposal.name}
        prelude = "\n\n".join(others[d]["code"] for d in self._closure(list(proposal.depends_on), others))
        program = f"{prelude}\n\n{proposal.code}" if prelude else proposal.code

        outcome = self.sandbox.run_tests(program, proposal.tests)                # gate 2
        if not outcome.ok:
            return decide(False, "author-tests", [outcome.failure])

        previous = registry.get(proposal.name)                                  # gate 3
        for suite in (previous["tests"] if previous else []):
            outcome = self.sandbox.run_tests(program, suite)
            if not outcome.ok:
                return decide(False, "regression",
                              [f"breaks a test of v{previous['version']}: {outcome.failure}"])

        eligible = list({r.id: r for r in reviewers                             # gate 4
                         if r.id != proposal.author and r.id in ledger.replicas
                         and r.id != kernel}.values())
        context = ReviewContext(need=need, prelude=prelude, policy=self.policy)

        def ask(reviewer) -> Review:
            try:
                return reviewer.review(proposal, context)
            except Exception as exc:        # a crashing reviewer is a missing review
                return Review(reviewer.id, False, [f"reviewer crashed: {type(exc).__name__}"])

        with ThreadPoolExecutor(max_workers=8) as pool:      # reviewers work independently
            reviews = list(pool.map(ask, eligible))
        for reviewer, review in zip(eligible, reviews):
            ledger.put(reviewer.id, f"reviews/{reviewer.id}/{proposal.id}",
                       {**review.to_dict(), "reviewer": reviewer.id})
        ledger.sync()

        # Read every review back from the forge's own replica: only what was validly
        # signed by that reviewer, in that reviewer's namespace, counts.
        records = {r.id: ledger.read(f"reviews/{r.id}/{proposal.id}") for r in eligible}
        suites = list(dict.fromkeys(rec["tests"] for rec in records.values()
                                    if isinstance(rec, dict) and rec.get("tests")))
        with ThreadPoolExecutor(max_workers=8) as pool:      # ...and the forge trusts none of them
            checked = dict(zip(suites, pool.map(lambda t: self.sandbox.run_tests(program, t), suites)))

        approvals, rejections, vetoed = [], [], False
        vetoes: dict[str, list[str]] = {}
        objections: dict[str, list[str]] = {}
        for reviewer in eligible:
            record = records[reviewer.id]
            if not isinstance(record, dict):
                rejections.append(reviewer.id)
                objections.setdefault("no valid review received", []).append(reviewer.id)
                continue
            approve = bool(record.get("approve"))
            outcome = checked.get(record.get("tests") or "")
            if outcome is not None and not outcome.ok:
                approve = False
                why = "review tests violate policy" if outcome.violations else outcome.failure
                objections.setdefault(why, []).append(reviewer.id)
            concern = bool(record.get("safety_concern"))
            if concern:
                vetoed, approve = True, False
            if not approve:
                for reason in record.get("reasons", []) or (["safety concern"] if concern else []):
                    bucket = (vetoes if concern else objections).setdefault(str(reason), [])
                    if reviewer.id not in bucket:
                        bucket.append(reviewer.id)
            (approvals if approve else rejections).append(reviewer.id)
        reasons = [f"VETO by {', '.join(who)}: {why}" for why, who in vetoes.items()]
        reasons += [f"{', '.join(who)}: {why}" for why, who in objections.items()]
        passing = [proposal.tests] + [t for t in suites if checked[t].ok and t != proposal.tests]

        if vetoed:                                                              # gate 5
            return decide(False, "safety-veto", reasons, approvals, rejections)
        if len(eligible) < self.policy.min_reviewers:
            return decide(False, "consensus", [f"needs {self.policy.min_reviewers} independent "
                                               f"reviewers, has {len(eligible)}"], approvals, rejections)
        needed = math.ceil(self.policy.quorum * len(eligible) - 1e-9)
        if len(approvals) < needed:
            return decide(False, "consensus", reasons or [f"{len(approvals)}/{len(eligible)} "
                                                           f"approvals, {needed} required"],
                          approvals, rejections)

        version = previous["version"] + 1 if previous else 1
        ledger.put(kernel, f"tools/{proposal.name}", {
            "name": proposal.name, "version": version, "status": "active",
            "description": proposal.description, "code": proposal.code,
            "sha256": _sha(proposal.code), "tests": passing, "author": proposal.author,
            "approved_by": approvals, "depends_on": list(proposal.depends_on),
            "proposal": proposal.id, "policy": self._pinned})
        return decide(True, "accepted", (), approvals, rejections, version)


@dataclass(frozen=True)
class ReviewContext:
    """What a reviewer is given besides the proposal itself."""

    need: Any            # the ToolNeed that prompted the proposal, if any
    prelude: str         # source of the proposal's dependencies
    policy: SafetyPolicy
