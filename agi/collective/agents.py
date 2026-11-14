"""Agents that write and review tools.

An agent is anything with an ``id``, ``propose(need, context)`` and
``review(proposal, context)``. The reference agents below are deterministic so
the whole pipeline can be exercised offline; :class:`LLMAgent` plugs a
language model into the same two roles.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..llm import LLM, extract_json
from ..sandbox import Sandbox, check_source
from .forge import Review, ReviewContext, ToolProposal


@dataclass(frozen=True)
class ToolNeed:
    """A capability the group lacks, specified by examples: ``(args, expected)``."""

    name: str
    description: str
    examples: tuple = ()

    def spec_tests(self, prefix: str = "spec") -> str:
        """The examples rendered as test functions."""
        lines = []
        for i, (args, expected) in enumerate(self.examples):
            call = f"{self.name}({', '.join(repr(a) for a in args)})"
            lines.append(f"def test_{prefix}_{i}():\n    got = {call}\n"
                         f"    assert got == {expected!r}, '{call} should be {expected!r}"
                         f" but is ' + repr(got)\n")
        return "\n".join(lines)


class Agent(Protocol):
    id: str

    def propose(self, need: ToolNeed, context: ReviewContext) -> ToolProposal | None: ...

    def review(self, proposal: ToolProposal, context: ReviewContext) -> Review: ...


class SpecReviewer:
    """Checks a proposal against the *specification*, not against the author's tests.

    It turns the need's examples into tests and runs them in its own sandbox.
    The tests are attached to the review, so the forge re-runs them too.
    """

    def __init__(self, agent_id: str):
        self.id = agent_id

    def propose(self, need: ToolNeed, context: ReviewContext) -> ToolProposal | None:
        return None

    def review(self, proposal: ToolProposal, context: ReviewContext) -> Review:
        need = context.need
        if need is None or not need.examples or need.name != proposal.name:
            return Review(self.id, False, ["no specification to check the tool against"])
        tests = need.spec_tests()
        program = f"{context.prelude}\n\n{proposal.code}" if context.prelude else proposal.code
        outcome = Sandbox(context.policy.sandbox).run_tests(program, tests)
        if outcome.ok:
            return Review(self.id, True, [f"all {len(need.examples)} specification examples pass"], tests)
        return Review(self.id, False, [outcome.failure or "specification tests failed"], tests)


class SafetyReviewer:
    """Looks for ways a tool could misbehave rather than whether it is correct.

    Static: policy violations and loops with no way out. Dynamic: probes the
    tool on the specification's inputs under the sandbox limits, twice, and
    objects to non-termination, resource exhaustion and non-determinism.
    """

    def __init__(self, agent_id: str):
        self.id = agent_id

    def propose(self, need: ToolNeed, context: ReviewContext) -> ToolProposal | None:
        return None

    @staticmethod
    def _endless_loops(code: str) -> list[int]:
        lines = []
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
                exits = (ast.Break, ast.Return, ast.Raise)
                if not any(isinstance(n, exits) for n in ast.walk(node)):
                    lines.append(node.lineno)
        return lines

    def review(self, proposal: ToolProposal, context: ReviewContext) -> Review:
        violations = check_source(proposal.code, context.policy.sandbox)
        if violations:
            return Review(self.id, False, violations, safety_concern=True)
        loops = self._endless_loops(proposal.code)
        if loops:
            return Review(self.id, False, [f"loop at line {n} can never exit" for n in loops],
                          safety_concern=True)
        need = context.need
        if need is not None and need.examples and need.name == proposal.name:
            program = f"{context.prelude}\n\n{proposal.code}" if context.prelude else proposal.code
            sandbox = Sandbox(context.policy.sandbox)
            inputs = [list(args) for args, _ in need.examples]
            first = sandbox.call_many(program, proposal.name, inputs)
            if first.timed_out or first.error_type in ("ResourceLimit", "SandboxCrash", "MemoryError"):
                return Review(self.id, False, [f"probe run: {first.failure}"], safety_concern=True)
            second = sandbox.call_many(program, proposal.name, inputs)
            if first.calls != second.calls:
                return Review(self.id, False, ["non-deterministic: two identical runs disagree"])
        return Review(self.id, True, ["no safety concerns found"])


class Builder(SpecReviewer):
    """Proposes tools from a library of drafts and reviews others against the spec.

    ``drafts`` maps a tool name to ``{"code": ..., "tests": ..., "depends_on": (...)}``;
    ``tests`` defaults to the need's own examples.
    """

    def __init__(self, agent_id: str, drafts: Mapping[str, Mapping] | None = None):
        super().__init__(agent_id)
        self.drafts = dict(drafts or {})

    def propose(self, need: ToolNeed, context: ReviewContext) -> ToolProposal | None:
        draft = self.drafts.get(need.name)
        if draft is None:
            return None
        return ToolProposal(need.name, need.description, draft["code"],
                            draft.get("tests") or need.spec_tests("author"), self.id,
                            tuple(draft.get("depends_on", ())))


_PROPOSE = """You are agent '{agent}' in a group of AI agents that write tools for each other.
Write a Python tool. Rules: define a top-level function named `{name}`; pure computation only;
you may import only: {imports}; no file, network or process access.
These approved tools are already defined and may be called: {available}.

NEED: {description}
EXAMPLES (args -> expected): {examples}

Reply with JSON only: {{"code": "<python source>", "tests": "<python source defining test_* functions using assert>", "depends_on": ["<tool name>", ...]}}
"""
_REVIEW = """You are agent '{agent}', reviewing a tool written by another agent. Be adversarial:
look for wrong results on edge cases, non-termination, and anything unsafe.

NEED: {description}
EXAMPLES (args -> expected): {examples}
CODE:
{code}

Reply with JSON only: {{"approve": true|false, "reasons": ["..."], "safety_concern": true|false,
"tests": "<optional python source defining extra test_* functions that the tool must pass>"}}
"""


class LLMAgent:
    """A language model in the author and reviewer roles.

    Fails closed: an unparseable reply is no proposal, or a rejecting review.
    """

    def __init__(self, agent_id: str, llm: LLM):
        self.id = agent_id
        self.llm = llm

    @staticmethod
    def _describe(need: ToolNeed | None) -> dict:
        if need is None:
            return {"description": "(not provided)", "examples": "(none)"}
        return {"description": need.description, "examples": json.dumps(need.examples, default=str)}

    def propose(self, need: ToolNeed, context: ReviewContext) -> ToolProposal | None:
        prompt = _PROPOSE.format(agent=self.id, name=need.name, available=context.prelude or "(none)",
                                 imports=", ".join(sorted(context.policy.sandbox.allowed_imports)),
                                 **self._describe(need))
        try:
            reply = extract_json(self.llm(prompt))
            return ToolProposal(need.name, need.description, str(reply["code"]), str(reply["tests"]),
                                self.id, tuple(map(str, reply.get("depends_on", ()))))
        except Exception:
            return None

    def review(self, proposal: ToolProposal, context: ReviewContext) -> Review:
        prompt = _REVIEW.format(agent=self.id, code=proposal.code, **self._describe(context.need))
        try:
            reply = extract_json(self.llm(prompt))
            return Review(self.id, reply["approve"] is True, [str(r) for r in reply.get("reasons", [])],
                          str(reply.get("tests") or ""), reply.get("safety_concern") is True)
        except Exception as exc:
            return Review(self.id, False, [f"unusable review ({type(exc).__name__}); failing closed"])
