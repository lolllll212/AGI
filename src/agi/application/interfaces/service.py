"""One object that wires the pieces together, with an asynchronous interface.

    service = AgentService.create(llm, workspace="./data", memory_dir="./state")
    result = await service.ask("What is the total for the North region in sales.csv?")

:class:`AgentService` wraps a :class:`~agi.application.cortex.engine.CognitiveEngine`
and adds what a long-running process needs: bounded concurrency, sessions,
parallel decomposition, drift monitoring. ``create`` asks the composition root
(:func:`agi.infrastructure.di.container.build_engine`) for an engine with the
standard tools, a self-healing ``python`` tool, hierarchical memory, guardrails
and tracing.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Sequence

from agi.application.cortex.engine import CognitiveEngine
from agi.application.memory.working import WorkingMemory
from agi.application.swarm.orchestrator import OrchestrationResult, Orchestrator
from agi.domain.ports import make
from agi.domain.value_objects.records import EngineResult


class AgentService:
    """An engine behind an async interface, safe to share between many callers."""

    def __init__(self, engine: CognitiveEngine, concurrency: int = 4, timeout: float | None = 300.0,
                 max_sessions: int = 1000, session_tokens: int = 2000, monitor: Any = None,
                 summarize_sessions: bool = False, session_store: Any = None):
        """A session's history is kept under ``session_tokens``: older turns are condensed into a
        summary (written by the model if ``summarize_sessions``, extractive otherwise), never just cut.

        ``session_store`` (a :class:`SessionStore` or a file path) keeps conversations on disk; then
        ``max_sessions`` only bounds how many are held in RAM at once. Without it they live in RAM."""
        self.engine, self.concurrency, self.timeout = engine, max(1, concurrency), timeout
        self.monitor = monitor
        self.started = time.time()
        self._sessions: OrderedDict[str, WorkingMemory] = OrderedDict()
        self._max_sessions, self._session_tokens = max_sessions, session_tokens
        self._summarize_sessions = summarize_sessions
        self.session_store = make("session_store", session_store) if isinstance(session_store, str) else session_store
        self._session_lock = threading.Lock()
        self._gates: dict[int, asyncio.Semaphore] = {}       # one per event loop
        self.in_flight = 0
        self.alarms: list = []
        self.observers: list[Callable[[EngineResult], None]] = []   # called with every finished result
        self.label: str | None = None                       # a name for the model, shown to callers
        self.demo = False                                   # True when the model is a scripted stand-in
        self.examples: list[str] = []                       # tasks worth trying, shown to callers

    @classmethod
    def create(cls, llm: Callable[[str], str] | None = None, *, concurrency: int = 4,
               timeout: float | None = 300.0, monitor: Any = None,
               session_store: Any = None, **engine_options) -> "AgentService":
        """Build the engine and wrap it. With ``memory_dir``, conversations are kept there too
        (``sessions.db``) unless ``session_store`` says otherwise."""
        directory = engine_options.get("memory_dir")
        if session_store is None and directory:
            os.makedirs(directory, exist_ok=True)
            session_store = os.path.join(directory, "sessions.db")
        return cls(make("engine", llm, **engine_options), concurrency=concurrency, timeout=timeout, monitor=monitor,
                   session_store=session_store)

    def _gate(self) -> asyncio.Semaphore:
        loop = id(asyncio.get_running_loop())
        if loop not in self._gates:
            self._gates = {loop: asyncio.Semaphore(self.concurrency)}
        return self._gates[loop]

    def _window(self, session: str, create: bool) -> WorkingMemory | None:
        """The session's history: from RAM, else from the store, else (if ``create``) a new one."""
        window = self._sessions.get(session)
        if window is None:
            saved = self.session_store.load(session) if self.session_store is not None else None
            if saved is None and not create:
                return None
            summarizer = self.engine.llm if self._summarize_sessions else None
            window = WorkingMemory(budget=self._session_tokens, keep_recent=4, summarizer=summarizer)
            if saved:
                window.load(saved)
            self._sessions[session] = window
        self._sessions.move_to_end(session)
        while len(self._sessions) > self._max_sessions:     # (still on disk, if there is a store)
            self._sessions.popitem(last=False)
        return window

    def _context(self, session: str | None) -> str:
        if not session:
            return ""
        with self._session_lock:
            window = self._window(session, create=False)
            if window is None:
                return ""
            label = {"user": "User", "assistant": "Assistant", "summary": "Earlier"}
            return "\n".join(f"{label.get(m.role, m.role)}: {m.content}" for m in window.messages)

    def _remember_turn(self, session: str | None, task: str, result: EngineResult) -> None:
        if not session or result.blocked or not result.answer:
            return
        with self._session_lock:
            window = self._window(session, create=True)
            window.add("user", self.engine.guardrails.redact(task)[0])
            window.add("assistant", self.engine.guardrails.redact(result.answer)[0])
            if self.session_store is not None:
                self.session_store.save(session, window.dump())

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
        for observer in list(self.observers):
            try:
                observer(result)
            except Exception:                               # measuring must never break answering
                pass
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
        """Forget a conversation, in RAM and on disk."""
        with self._session_lock:
            in_memory = self._sessions.pop(session, None) is not None
            stored = self.session_store.delete(session) if self.session_store is not None else False
        return in_memory or stored

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
        return {"engine": dict(self.engine.stats),
                "sessions": len(self.session_store) if self.session_store is not None else len(self._sessions),
                "sessions_in_memory": len(self._sessions),
                "routing": self.engine.router.learner.explain()[:10] if self.engine.router.learner else [],
                "routing_model": self.engine.router.learner.report() if self.engine.router.learner else None,
                "memory": memory.stats() if hasattr(memory, "stats") else None,
                "drift": self.monitor.status() if self.monitor is not None else None,
                "trace_export": [e.stats() for e in self.engine.tracer.exporters if hasattr(e, "stats")],
                "drift_alarms": [alarm.model_dump() for alarm in self.alarms]}

    def close(self) -> None:
        for exporter in self.engine.tracer.exporters:       # send the spans still queued
            if hasattr(exporter, "close"):
                exporter.close()
        if self.session_store is not None:
            self.session_store.close()
        memory = self.engine.memory
        if hasattr(memory, "close"):
            memory.close()
