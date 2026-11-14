"""The cognitive engine: a fast path, a slow path, and the discipline around both.

One task goes through these stages::

    input guardrail -> router (System 1) -+-> reflex: canned reply / direct tool / cached policy
                                          +-> react: one reason-act-observe loop
                                          +-> deliberate: search over plans, each judged by executing it
    -> verification -> output guardrail -> memory (trace, trajectory, lessons) -> trace + reward

A path that fails **escalates** to the next slower one (reflex -> react ->
deliberate), so being fast is never bought with being wrong. Every model call
and tool call is counted against one budget for the task, a model that keeps
failing trips a circuit breaker, and nothing in here raises: the caller always
gets an :class:`~agi.core.types.EngineResult`.
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

from pydantic import Field

from ..memory.procedural import render_answer
from ..reasoning.planner import Execution, Plan, PlanProblem, search_plans
from ..tools.base import Tool, ToolRegistry, ToolResult
from ..tools.managed import ManagedRegistry
from ..tools.router import SemanticToolRouter
from .guardrails import Guardrails
from .react import _FAILURES, GroundingVerifier, ReActAgent, ReActStep
from .resilience import BudgetExceeded, CircuitBreaker, CircuitOpen, LoopDetector, RateLimiter, StepBudget
from .router import Router, RouterConfig
from .tracing import Tracer
from .types import EngineResult, Mode, Model, Route, StepRecord

_depth: contextvars.ContextVar[int] = contextvars.ContextVar("agi_engine_depth", default=0)

COMPOSE = """Answer the task using only the tool results below. Be brief. Do not add figures that are not there.

TASK: {task}

TOOL RESULTS
{results}

Answer:"""


class EngineConfig(Model):
    max_steps: int = Field(default=12, ge=1)             # model calls per reason-act loop
    max_total_steps: int = Field(default=80, ge=1)       # model calls + tool calls for the whole task
    max_seconds: float | None = None
    max_depth: int = Field(default=3, ge=1)              # how deep engines may delegate to engines
    max_plans: int = Field(default=3, ge=1)              # plans System 2 may execute
    plan_candidates: int = Field(default=3, ge=1)
    plan_strategy: Literal["mcts", "tot"] = "mcts"
    tool_k: int = Field(default=0, ge=0)                 # show only the k most relevant tools (0 = all)
    escalate: bool = True
    learn: bool = True
    verify: bool = True
    max_challenges: int = Field(default=1, ge=0)
    loop_repeats: int = Field(default=3, ge=2)           # identical results from one tool before it is a loop
    system_prompt: str | None = None
    router: RouterConfig = Field(default_factory=RouterConfig)


@dataclass
class _Outcome:
    """What one path produced."""

    answer: str | None = None
    success: bool = False
    grounded: bool | None = None
    steps: list = field(default_factory=list)
    stop_reason: str = ""
    error: str | None = None
    plans_tried: int = 0
    plan: Plan | None = None


class _RunState:
    """Per-task counters shared by every path the task goes through."""

    def __init__(self, engine: "CognitiveEngine"):
        config = engine.config
        self.budget = StepBudget(config.max_total_steps, config.max_depth, config.max_seconds)
        self.detector = LoopDetector(window=12, max_repeats=config.loop_repeats)
        self.llm_calls = self.tool_calls = self.tool_errors = 0
        self.halted: str | None = None
        self.flags: list[str] = []


class _RunTools:
    """The registry as one task sees it: every call is budgeted and watched for loops."""

    def __init__(self, registry: ToolRegistry, state: _RunState):
        self._registry, self._state = registry, state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def _refuse(self, name: str, args: Any, kind: str, message: str) -> ToolResult:
        self._state.tool_errors += 1
        return ToolResult(name, args if isinstance(args, dict) else {}, False, f"{kind}: {message}", None, kind)

    def call(self, name: str, args: Any = None) -> ToolResult:
        state = self._state
        if state.halted:
            return self._refuse(name, args, "Halted", state.halted)
        try:
            state.budget.tick(f"tool call '{name}'")
        except BudgetExceeded as exc:
            state.halted = str(exc)
            return self._refuse(name, args, "BudgetExceeded", str(exc))
        result = self._registry.call(name, args)
        state.tool_calls += 1
        state.tool_errors += 0 if result.ok else 1
        seen = state.detector.observe((name, result.ok, result.output[:300]))
        if seen == state.detector.max_repeats:
            result.output += (f"\n[loop warning: '{name}' has now returned exactly this {seen} times. "
                              f"Repeating it will not help; change the approach.]")
        elif seen > state.detector.max_repeats + 1:
            state.halted = f"loop detected: '{name}' returned the same result {seen} times"
            state.flags.append("loop_detected")
        return result


class CognitiveEngine:
    def __init__(self, llm: Callable[[str], str] | None = None, tools: ToolRegistry | None = None, memory=None,
                 guardrails: Guardrails | None = None, tracer: Tracer | None = None,
                 config: EngineConfig | None = None, breaker: CircuitBreaker | None = None,
                 rate_limiter: RateLimiter | None = None, healer=None, tags: Iterable[str] = ()):
        """``memory`` is a :class:`~agi.memory.HierarchicalMemory` (optional); ``llm`` is any
        ``callable(prompt) -> str`` (optional: without one only the reflex path can answer)."""
        self.llm, self.memory, self.healer = llm, memory, healer
        self.config = config or EngineConfig()
        self.tracer = tracer or Tracer()
        self.guardrails = guardrails or Guardrails()
        self.tags = tuple(tags)
        if isinstance(tools, ManagedRegistry):
            self.tools = tools
            self.tools.tracer = self.tools.tracer or self.tracer
            self.tools.guardrails = self.tools.guardrails or self.guardrails
        else:
            source = tools if tools is not None else ToolRegistry()
            self.tools = ManagedRegistry(list(source), tracer=self.tracer, guardrails=self.guardrails,
                                         max_output_chars=source.max_output_chars, max_calls=source.max_calls,
                                         allow=source.allow)
        try:
            from ..tools.builtin import evaluate
        except Exception:                                   # pragma: no cover
            evaluate = None
        self.router = Router(self.tools, getattr(memory, "procedural", None), getattr(memory, "episodic", None),
                             self.config.router, evaluate=evaluate)
        self.tool_router = SemanticToolRouter(
            self.tools, always=[n for n in ("list_tools", "find_tools") if n in self.tools])
        self.breaker = breaker or CircuitBreaker(failure_threshold=4, reset_after=30.0, name="model")
        self.rate_limiter = rate_limiter
        self.stats = {"runs": 0, "reflex": 0, "react": 0, "deliberate": 0, "escalations": 0, "blocked": 0,
                      "successes": 0, "llm_calls": 0, "tool_calls": 0}

    # -- the model, wrapped ------------------------------------------------ #
    def _model(self, state: _RunState) -> Callable[[str], str]:
        def call(prompt: str) -> str:
            if state.halted:
                raise BudgetExceeded(state.halted)
            try:
                state.budget.tick("model call")
            except BudgetExceeded as exc:
                state.halted = str(exc)
                raise
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            with self.tracer.span("llm", "completion", prompt_chars=len(prompt)) as span:
                reply = self.breaker.call(self.llm, prompt)
                span["reply_chars"] = len(reply)
            state.llm_calls += 1
            return reply
        return call

    # -- paths ------------------------------------------------------------- #
    def _reflex(self, task: str, route: Route, state: _RunState, tools: _RunTools,
                check: Callable[[str], bool] | None) -> _Outcome:
        if route.reply is not None:
            return _Outcome(answer=route.reply, success=True, stop_reason="answer")
        steps: list[StepRecord] = []
        for call in route.calls:
            result = tools.call(call.tool, call.args)
            steps.append(StepRecord(action=call.tool, args=call.args, observation=result.output, ok=result.ok))
            if not result.ok:
                return _Outcome(steps=steps, stop_reason="tool_error",
                                error=f"the cached action failed: {result.output[:200]}")
        last = steps[-1].observation.strip() if steps else ""
        answer = render_answer(route.answer_template, last, route.expected_figures) if route.answer_template else None
        if answer is not None:
            pass
        elif self.llm is not None:
            results = "\n".join(f"[{s.action}] {s.observation[:1500]}" for s in steps)
            try:
                answer = self._model(state)(COMPOSE.format(task=task, results=results)).strip()
            except Exception as exc:
                return _Outcome(steps=steps, stop_reason="llm_error", error=f"{type(exc).__name__}: {exc}")
        else:
            answer = last
        grounded = not GroundingVerifier().unsupported(task, answer, steps) if self.config.verify else None
        success = bool(answer) and grounded is not False
        if success and check is not None:
            success = self._checked(check, answer)
        return _Outcome(answer=answer, success=success, grounded=grounded, steps=steps, stop_reason="answer",
                        error=None if success else "the cached action's answer did not pass verification")

    @staticmethod
    def _checked(check: Callable[[str], bool], answer: str | None) -> bool:
        try:
            return bool(answer is not None and check(answer))
        except Exception:
            return False

    def _briefing(self, task: str, plan: Plan | None) -> str:
        parts = []
        if self.memory is not None:
            known = self.memory.context_for(task)
            if known:
                parts.append("WHAT MEMORY SAYS\n" + known)
        if plan is not None:
            parts.append("PLAN TO FOLLOW\n" + plan.render() + "\nCarry this plan out. If a step proves "
                         "impossible, say exactly what blocked it rather than quietly doing something else.")
        return "\n\n".join(parts)

    def _react(self, task: str, context: str, state: _RunState, tools: _RunTools,
               check: Callable[[str], bool] | None, plan: Plan | None = None) -> _Outcome:
        def on_step(step: ReActStep) -> None:
            if step.thought:
                self.tracer.event("thought", step.action or "reflection", text=step.thought)

        lessons = getattr(self.memory, "lessons", None) if self.config.learn else None
        agent = ReActAgent(self._model(state), tools, memory=lessons, max_steps=self.config.max_steps,
                           verifier="grounding" if self.config.verify else None,
                           max_challenges=self.config.max_challenges, tags=self.tags,
                           system=self.config.system_prompt, on_step=on_step)
        if self.config.tool_k:
            agent.visible = self.tool_router.select(task, self.config.tool_k)
        agent.briefing = self._briefing(task, plan)
        run = agent.run(task, check, context)
        steps = [StepRecord(thought=s.thought, action=s.action, args=s.args if isinstance(s.args, dict) else {},
                            observation=s.observation, ok=s.ok) for s in run.steps]
        stop = "budget" if state.halted and run.answer is None else run.stop_reason
        return _Outcome(answer=run.answer, success=bool(run.success), grounded=run.grounded, steps=steps,
                        stop_reason=stop, error=run.error or (state.halted if run.answer is None else None), plan=plan)

    @staticmethod
    def _why_failed(outcome: _Outcome) -> str:
        if outcome.answer is None:
            failing = next((s.observation for s in reversed(outcome.steps) if s.action and not s.ok), "")
            return (f"no answer was produced ({_FAILURES.get(outcome.stop_reason, outcome.stop_reason)}"
                    + (f": {outcome.error}" if outcome.error else "") + ")"
                    + (f"; the last failing step said: {failing[:200]}" if failing else ""))
        if outcome.grounded is False:
            return f"the answer {outcome.answer[:120]!r} contains figures that no tool result supports"
        return f"the answer {outcome.answer[:120]!r} was checked and is wrong"

    def _deliberate(self, task: str, context: str, state: _RunState, tools: _RunTools,
                    check: Callable[[str], bool] | None) -> _Outcome:
        executed: dict[tuple, _Outcome] = {}
        order: list[_Outcome] = []

        def execute(plan: Plan, earlier: Sequence[Plan]) -> Execution:
            if state.halted:
                return Execution(success=False, detail=state.halted)
            with self.tracer.span("plan", "execute", steps=plan.steps, revision_of=len(earlier)) as span:
                outcome = self._react(task, context, state, tools, check, plan)
                span.update(success=outcome.success, status="ok" if outcome.success else "error")
            executed[plan.key()] = outcome
            order.append(outcome)
            return Execution(success=outcome.success, score=0.5 if outcome.answer else 0.2,
                             detail="" if outcome.success else self._why_failed(outcome))

        visible = self.tool_router.select(task, self.config.tool_k) if self.config.tool_k else None
        problem = PlanProblem(task, self._model(state), execute, tools=self.tools.describe(visible),
                              tool_names=self.tools.names, candidates=self.config.plan_candidates,
                              context=self.memory.context_for(task) if self.memory is not None else "")
        found = search_plans(problem, self.config.max_plans, self.config.plan_strategy)
        for line in found.trace:
            self.tracer.event("plan", "search", detail=line)
        if not order:                                       # no usable plan was proposed: just try directly
            if state.halted:
                return _Outcome(stop_reason="budget", error=state.halted)
            outcome = self._react(task, context, state, tools, check)
            outcome.plans_tried = 0
            return outcome
        best = executed.get(found.plan.key()) if found.plan is not None else None
        if best is None or not best.success:
            best = next((o for o in order if o.success), None) or max(
                order, key=lambda o: (o.answer is not None, o.grounded is not False))
        best.plans_tried = len(order)
        return best

    # -- the whole task ---------------------------------------------------- #
    def run(self, task: str, check: Callable[[str], bool] | None = None, mode: Mode | None = None,
            context: str = "") -> EngineResult:
        """Handle one task. Never raises.

        ``check(answer) -> bool`` supplies ground truth when the caller has it;
        ``mode`` forces a path instead of asking the router.
        """
        started = time.perf_counter()
        depth = _depth.get()
        token = _depth.set(depth + 1)
        try:
            # Traces outlive the run: secrets are scrubbed before anything is written to them.
            shown = self.guardrails.redact(task)[0] if isinstance(task, str) else repr(task)
            with self.tracer.span("run", "task", task=shown, depth=depth) as span:
                if depth >= self.config.max_depth:
                    result = EngineResult(task=task, success=False, stop_reason="depth",
                                          error=f"recursion depth limit of {self.config.max_depth} reached")
                else:
                    try:
                        result = self._run(task, check, mode, context)
                    except Exception as exc:                # the engine itself must not take the caller down
                        result = EngineResult(task=task, success=False, stop_reason="error",
                                              error=f"{type(exc).__name__}: {exc}")
                result.trace_id = span["trace_id"]
                result.seconds = time.perf_counter() - started
                span.update(mode=result.mode, success=result.success, stop_reason=result.stop_reason,
                            answer=result.answer, llm_calls=result.llm_calls, tool_calls=result.tool_calls,
                            status="ok" if result.success or result.success is None else "error")
                self.tracer.reward(1.0 if result.success else 0.0, result.stop_reason, mode=result.mode)
        finally:
            _depth.reset(token)
        self.stats["runs"] += 1
        self.stats["successes"] += int(bool(result.success))
        self.stats["llm_calls"] += result.llm_calls
        self.stats["tool_calls"] += result.tool_calls
        return result

    def _run(self, task: str, check: Callable[[str], bool] | None, mode: Mode | None, context: str) -> EngineResult:
        verdict = self.guardrails.check_input(task)
        if not verdict.allowed:
            self.stats["blocked"] += 1
            self.tracer.event("guardrail", "input", status="error", rule=verdict.rule, reason=verdict.reason)
            return EngineResult(task=task, answer=f"I can't act on that request: {verdict.reason}.", blocked=True,
                                success=False, stop_reason="input_guardrail", mode="reflex", system=1,
                                flags=[f"blocked:{verdict.rule}"])
        flags = list(verdict.flags) + [f"redacted:{r}" for r in verdict.redactions]
        if flags:
            self.tracer.event("guardrail", "input", flags=flags)
        task = verdict.text

        state = _RunState(self)
        tools = _RunTools(self.tools, state)
        repairs_before = len(self.tools.repaired) + (self.healer.repairs if self.healer else 0)
        route = Route(mode=mode, reason="forced by the caller", confidence=1.0) if mode else self.router.route(task)
        if route.mode != "reflex" and self.llm is None:
            return EngineResult(task=task, success=False, route=route, mode=route.mode, stop_reason="no_model",
                                error="this task needs a model and none is configured", flags=flags)
        self.tracer.event("route", route.mode, reason=route.reason, signals=route.signals,
                          confidence=route.confidence)

        ladder: list[Mode] = [route.mode]
        if self.config.escalate and self.llm is not None:
            ladder += [m for m in ("react", "deliberate") if ("reflex", "react", "deliberate").index(m)
                       > ("reflex", "react", "deliberate").index(route.mode)]
        outcome, used, escalated = _Outcome(), route.mode, False
        for position, current in enumerate(ladder):
            used = current
            self.stats[current] += 1
            if current == "reflex":
                outcome = self._reflex(task, route, state, tools, check)
                if route.policy_id and self.memory is not None:
                    self.memory.procedural.record_use(route.policy_id, outcome.success)
            elif current == "react":
                outcome = self._react(task, context, state, tools, check)
            else:
                outcome = self._deliberate(task, context, state, tools, check)
            if outcome.success or state.halted or position == len(ladder) - 1:
                break
            if isinstance(getattr(self.breaker, "state", None), str) and self.breaker.state == "open":
                break                                       # the model is down: slower paths cannot help
            escalated = True
            self.stats["escalations"] += 1
            self.tracer.event("route", "escalate", status="error", from_mode=current, to_mode=ladder[position + 1],
                              reason=outcome.error or self._why_failed(outcome))

        answer = outcome.answer
        if answer is not None:
            out = self.guardrails.check_output(answer)
            if not out.allowed:
                self.stats["blocked"] += 1
                self.tracer.event("guardrail", "output", status="error", rule=out.rule, reason=out.reason)
                return EngineResult(task=task, answer="The answer was withheld by an output safety rule.",
                                    blocked=True, success=False, stop_reason="output_guardrail", mode=used,
                                    system=1 if used == "reflex" else 2, route=route, steps=outcome.steps,
                                    llm_calls=state.llm_calls, tool_calls=state.tool_calls,
                                    tool_errors=state.tool_errors, flags=flags + [f"blocked:{out.rule}"])
            if out.redactions:
                flags += [f"redacted_output:{r}" for r in out.redactions]
            answer = out.text

        if state.halted and not outcome.success:
            outcome.stop_reason = "loop" if "loop_detected" in state.flags else "budget"
        result = EngineResult(
            task=task, answer=answer, mode=used, system=1 if used == "reflex" else 2, route=route,
            success=outcome.success, grounded=outcome.grounded, stop_reason=outcome.stop_reason or "answer",
            steps=outcome.steps, tool_calls=state.tool_calls, tool_errors=state.tool_errors,
            llm_calls=state.llm_calls, plans_tried=outcome.plans_tried, escalated=escalated,
            repairs=len(self.tools.repaired) + (self.healer.repairs if self.healer else 0) - repairs_before,
            error=None if outcome.success else (outcome.error or state.halted), flags=flags + state.flags)
        if self.config.learn and self.memory is not None and (route.reply is None or used != "reflex"):
            self._remember(result, learned_by_thinking=used != "reflex")
        return result

    def _remember(self, result: EngineResult, learned_by_thinking: bool) -> None:
        steps = [{"action": s.action, "args": s.args, "ok": s.ok, "observation": s.observation[:600]}
                 for s in result.steps if s.action]
        try:
            self.memory.remember_run(result.task, bool(result.success), steps, result.answer, self.tags,
                                     store_trajectory=learned_by_thinking and result.grounded is not False,
                                     mode=result.mode, stop_reason=result.stop_reason, trace_id=result.trace_id)
            self.tracer.event("memory", "remember", success=bool(result.success), steps=len(steps))
        except Exception as exc:                            # memory trouble must not fail the task
            self.tracer.event("error", "memory", status="error", detail=f"{type(exc).__name__}: {exc}")

    # -- async ------------------------------------------------------------- #
    async def arun(self, task: str, check: Callable[[str], bool] | None = None, mode: Mode | None = None,
                   context: str = "") -> EngineResult:
        """:meth:`run` without blocking the event loop."""
        return await asyncio.to_thread(self.run, task, check, mode, context)

    async def run_many(self, tasks: Sequence[str], concurrency: int = 4,
                       timeout: float | None = None) -> list[EngineResult]:
        """Run independent tasks in parallel. One task failing or timing out does not affect the others."""
        gate = asyncio.Semaphore(max(1, concurrency))

        async def one(task: str) -> EngineResult:
            async with gate:
                try:
                    return await asyncio.wait_for(self.arun(task), timeout)
                except asyncio.TimeoutError:
                    return EngineResult(task=task, success=False, stop_reason="timeout",
                                        error=f"did not finish within {timeout:g}s")
                except Exception as exc:                    # pragma: no cover - run() does not raise
                    return EngineResult(task=task, success=False, stop_reason="error",
                                        error=f"{type(exc).__name__}: {exc}")

        return list(await asyncio.gather(*(one(task) for task in tasks)))

    # -- delegation -------------------------------------------------------- #
    def as_tool(self, name: str = "delegate", description: str | None = None) -> Tool:
        """This engine as a tool another agent can call. Nesting is limited by ``max_depth``."""
        from ..tools.base import Param, ToolError

        def delegate(task: str) -> str:
            result = self.run(task)
            if result.stop_reason == "depth":
                raise ToolError(f"DepthLimit: {result.error}. Solve this part yourself instead of delegating.")
            if not result.success:
                raise ToolError(f"SubtaskFailed: {result.error or result.stop_reason}"
                                + (f" (partial answer: {result.answer[:300]})" if result.answer else ""))
            return result.answer or ""

        return Tool(name, description or "Hand a self-contained sub-task to another agent and get its answer.",
                    delegate, {"task": Param("string", "the sub-task, stated completely")}, timeout_s=600.0)
