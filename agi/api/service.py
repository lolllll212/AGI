"""One object that wires the pieces together, with an asynchronous interface.

    service = AgentService.create(llm, workspace="./data", memory_dir="./state")
    result = await service.ask("What is the total for the North region in sales.csv?")

:func:`build_engine` assembles a :class:`~agi.core.engine.CognitiveEngine` with
the standard tools, a self-healing ``python`` tool, hierarchical memory,
guardrails and tracing. :class:`AgentService` adds what a long-running process
needs: bounded concurrency, sessions, parallel decomposition, drift monitoring.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.engine import CognitiveEngine, EngineConfig
from ..core.guardrails import Guardrails
from ..core.isolation import best_sandbox
from ..core.orchestrator import OrchestrationResult, Orchestrator
from ..core.sandbox import Sandbox, SandboxPolicy
from ..core.tracing import Tracer
from ..core.types import EngineResult
from ..evaluation.evaluator import DriftMonitor
from ..memory import HierarchicalMemory
from ..tools import SelfHealer, ToolRegistry, healing_python_tool, standard_tools, toolsmith_tools


def build_engine(llm: Callable[[str], str] | None = None, workspace: str | Path | None = None,
                 memory_dir: str | None = None, memory: Any = "auto", sandbox: Sandbox | str = "process",
                 heal: bool = True, toolsmith: bool = False, network: bool = False,
                 allow_hosts: Sequence[str] | None = None, read_only: bool = False,
                 extra_tools: Sequence[Any] = (), config: EngineConfig | None = None,
                 guardrails: Guardrails | None = None, trace_path: str | None = None,
                 policy: SandboxPolicy | None = None) -> CognitiveEngine:
    """Assemble an engine with sensible defaults.

    ``sandbox``: ``"process"`` (default: a locked-down child process), ``"wasm"``
    or ``"docker"`` (refuse to start if that backend is missing), ``"auto"`` (the
    strongest one available), or a ready :class:`Sandbox`.
    ``toolsmith=True`` lets the agent write and register new tools at runtime
    (each must pass its own tests inside the sandbox first).
    ``memory``: ``"auto"`` builds a :class:`HierarchicalMemory` (in RAM unless
    ``memory_dir`` is given); ``None`` runs without memory.
    """
    if isinstance(sandbox, str):
        if sandbox not in ("auto", "process", "wasm", "docker"):
            raise ValueError(f"unknown sandbox {sandbox!r} (use 'auto', 'process', 'wasm', 'docker' or a Sandbox)")
        sandbox = best_sandbox(policy) if sandbox == "auto" else best_sandbox(policy, require=sandbox)
    tracer = Tracer(path=trace_path)
    if memory == "auto":
        memory = HierarchicalMemory(memory_dir, sandbox=sandbox)
    tools: ToolRegistry = standard_tools(workspace=workspace, sandbox=sandbox, network=network,
                                         allow_hosts=allow_hosts, read_only=read_only,
                                         memory=getattr(memory, "lessons", None))
    healer = None
    if heal:
        healer = SelfHealer(sandbox, llm=llm, tracer=tracer)
        tools.register(healing_python_tool(healer), replace=True)
    if toolsmith:
        tools.extend(toolsmith_tools(tools, sandbox))
    for tool in extra_tools:
        tools.register(tool, replace=True)
    return CognitiveEngine(llm, tools, memory=memory, guardrails=guardrails, tracer=tracer,
                           config=config, healer=healer)


class AgentService:
    """An engine behind an async interface, safe to share between many callers."""

    def __init__(self, engine: CognitiveEngine, concurrency: int = 4, timeout: float | None = 300.0,
                 max_sessions: int = 1000, session_turns: int = 6, monitor: DriftMonitor | None = None):
        self.engine, self.concurrency, self.timeout = engine, max(1, concurrency), timeout
        self.monitor = monitor
        self.started = time.time()
        self._sessions: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        self._max_sessions, self._session_turns = max_sessions, session_turns
        self._gates: dict[int, asyncio.Semaphore] = {}       # one per event loop
        self.in_flight = 0
        self.alarms: list = []

    @classmethod
    def create(cls, llm: Callable[[str], str] | None = None, *, concurrency: int = 4,
               timeout: float | None = 300.0, monitor: DriftMonitor | None = None, **engine_options) -> "AgentService":
        return cls(build_engine(llm, **engine_options), concurrency=concurrency, timeout=timeout, monitor=monitor)

    def _gate(self) -> asyncio.Semaphore:
        loop = id(asyncio.get_running_loop())
        if loop not in self._gates:
            self._gates = {loop: asyncio.Semaphore(self.concurrency)}
        return self._gates[loop]

    def _context(self, session: str | None) -> str:
        turns = self._sessions.get(session or "", [])
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)

    def _remember_turn(self, session: str | None, task: str, result: EngineResult) -> None:
        if not session or result.blocked or not result.answer:
            return
        turns = self._sessions.setdefault(session, [])
        self._sessions.move_to_end(session)
        turns.append((self.engine.guardrails.redact(task)[0], result.answer))
        del turns[: max(0, len(turns) - self._session_turns)]
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    async def ask(self, task: str, mode: str | None = None, session: str | None = None,
                  check: Callable[[str], bool] | None = None, timeout: float | None = None) -> EngineResult:
        """Handle one task. Never raises: a timeout or failure comes back as a result."""
        limit = self.timeout if timeout is None else timeout
        async with self._gate():
            self.in_flight += 1
            try:
                result = await asyncio.wait_for(self.engine.arun(task, check, mode, self._context(session)), limit)
            except asyncio.TimeoutError:
                result = EngineResult(task=task, success=False, stop_reason="timeout",
                                      error=f"did not finish within {limit:g}s")
            except Exception as exc:
                result = EngineResult(task=task, success=False, stop_reason="error",
                                      error=f"{type(exc).__name__}: {exc}")
            finally:
                self.in_flight -= 1
        self._remember_turn(session, task, result)
        if self.monitor is not None and not result.blocked:
            self.alarms.extend(self.monitor.observe(result))
        return result

    async def ask_many(self, tasks: Sequence[str], mode: str | None = None) -> list[EngineResult]:
        """Independent tasks in parallel (bounded by the service's concurrency), results in order."""
        return list(await asyncio.gather(*(self.ask(task, mode) for task in tasks)))

    async def solve_parallel(self, task: str, subtasks: Sequence[str] | None = None) -> OrchestrationResult:
        """Split a task into independent parts, solve them at the same time, and combine the answers."""
        orchestrator = Orchestrator(lambda part: self.ask(part), llm=self.engine.llm,
                                    concurrency=self.concurrency, timeout=None)
        return await orchestrator.solve(task, subtasks)

    def end_session(self, session: str) -> bool:
        return self._sessions.pop(session, None) is not None

    # -- introspection ------------------------------------------------------ #
    def tools(self) -> list[dict]:
        return [{"name": tool.name, "description": tool.description, "risk": getattr(tool, "risk", "read"),
                 "parameters": {name: param.schema() for name, param in tool.params.items()}}
                for tool in self.engine.tools]

    def trace(self, trace_id: str) -> list[dict]:
        return [event.model_dump() for event in self.engine.tracer.trace(trace_id)]

    def health(self) -> dict:
        breaker = self.engine.breaker
        return {"status": "degraded" if breaker.state == "open" else "ok", "model": self.engine.llm is not None,
                "model_circuit": breaker.state, "uptime_s": round(time.time() - self.started, 1),
                "in_flight": self.in_flight, "tools": len(self.engine.tools)}

    def stats(self) -> dict:
        memory = self.engine.memory
        return {"engine": dict(self.engine.stats), "sessions": len(self._sessions),
                "memory": memory.stats() if hasattr(memory, "stats") else None,
                "drift": self.monitor.status() if self.monitor is not None else None,
                "drift_alarms": [alarm.model_dump() for alarm in self.alarms]}

    def close(self) -> None:
        memory = self.engine.memory
        if hasattr(memory, "close"):
            memory.close()
