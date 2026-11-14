"""Typed interfaces shared across the framework (pydantic models).

Everything that crosses a package boundary - routing decisions, guardrail
verdicts, trace events, engine results, API payloads - is one of these models,
so malformed data is rejected at the boundary instead of deep inside a loop.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    """Base class: unknown fields are an error, and assignments are validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolCall(Model):
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class Observation(Model):
    tool: str
    ok: bool
    output: str
    error_type: str | None = None
    seconds: float = 0.0
    repaired: str | None = None          # what the self-healing loop changed, if anything


Mode = Literal["reflex", "react", "deliberate"]


class Route(Model):
    """A routing decision: which process handles the task, and why."""

    mode: Mode
    reason: str
    signals: list[str] = Field(default_factory=list)      # what pushed towards deliberation
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reply: str | None = None             # reflex: a direct conversational answer
    calls: list[ToolCall] = Field(default_factory=list)   # reflex: the action(s) to take
    answer_template: str | None = None
    expected_figures: int | None = None  # numbers the final tool output must hold for the template to apply
    policy_id: str | None = None

    @property
    def system(self) -> int:
        return 1 if self.mode == "reflex" else 2


class GuardrailVerdict(Model):
    allowed: bool
    boundary: Literal["input", "output", "observation"]
    text: str                            # the text to use (possibly redacted)
    rule: str | None = None              # the rule that blocked
    reason: str = ""
    redactions: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class TraceEvent(Model):
    trace_id: str
    span_id: str
    parent_id: str | None = None
    kind: Literal["run", "route", "thought", "action", "observation", "reward", "guardrail",
                  "heal", "plan", "memory", "error", "eval", "llm"]
    name: str
    start: float
    end: float
    status: Literal["ok", "error"] = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)


class StepRecord(Model):
    thought: str = ""
    action: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    ok: bool = True


class EngineResult(Model):
    """The outcome of one task handled by the cognitive engine."""

    task: str
    answer: str | None = None
    mode: Mode = "react"
    system: Literal[1, 2] = 2
    route: Route | None = None
    success: bool | None = None
    grounded: bool | None = None
    blocked: bool = False                # stopped by a guardrail
    stop_reason: str = ""
    steps: list[StepRecord] = Field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    repairs: int = 0
    plans_tried: int = 0
    compactions: int = 0                 # times the context window had to condense or drop earlier steps
    prompt_reuse: float | None = None    # share of each prompt that repeated the previous one's start
    escalated: bool = False              # the reflex path failed and deliberation took over
    trace_id: str | None = None
    seconds: float = 0.0
    error: str | None = None
    flags: list[str] = Field(default_factory=list)


class AgentConfig(Model):
    """Everything the optimiser is allowed to vary about an agent."""

    system_prompt: str | None = None     # None = the built-in prompt
    max_steps: int = Field(default=12, ge=1, le=100)
    heal: bool = True
    verify: bool = True
    max_challenges: int = Field(default=1, ge=0, le=5)
    notes: list[str] = Field(default_factory=list)        # why each change was made
