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
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.engine import CognitiveEngine, EngineConfig
from ..core.guardrails import Guardrails
from ..core.isolation import best_sandbox
from ..core.orchestrator import OrchestrationResult, Orchestrator
from ..core.sandbox import Sandbox, SandboxPolicy
from ..core.tracing import OTLPExporter, Tracer
from ..core.types import EngineResult
from ..evaluation.evaluator import DriftMonitor
from ..memory import HierarchicalMemory
from ..memory.working import WorkingMemory
from ..tools import SelfHealer, ToolRegistry, healing_python_tool, standard_tools, toolsmith_tools


def _dense(embed: Any) -> Callable[[str], list[float]]:
    """Adapt an embedder (a function or an object with ``embed``) to the plain list episodic memory wants."""
    function = getattr(embed, "embed", embed)

    def dense(text: str) -> list[float]:
        vector = function(text)
        return [value for _, value in sorted(vector.items())] if isinstance(vector, dict) else list(vector)
    return dense


def build_engine(llm: Callable[[str], str] | None = None, workspace: str | Path | None = None,
                 memory_dir: str | None = None, memory: Any = "auto", sandbox: Sandbox | str = "process",
                 heal: bool = True, toolsmith: bool = False, network: bool = False,
                 allow_hosts: Sequence[str] | None = None, read_only: bool = False,
                 extra_tools: Sequence[Any] = (), config: EngineConfig | None = None,
                 guardrails: Guardrails | None = None, trace_path: str | None = None,
                 policy: SandboxPolicy | None = None, embed: Any = None,
                 tracer: Tracer | None = None, otlp_endpoint: str | None = None,
                 otlp_headers: dict[str, str] | None = None) -> CognitiveEngine:
    """Assemble an engine with sensible defaults.

    ``sandbox``: ``"process"`` (default: a locked-down child process), ``"wasm"``
    or ``"docker"`` (refuse to start if that backend is missing), ``"auto"`` (the
    strongest one available), or a ready :class:`Sandbox`.
    ``toolsmith=True`` lets the agent write and register new tools at runtime
    (each must pass its own tests inside the sandbox first).
    ``memory``: ``"auto"`` builds a :class:`HierarchicalMemory` (in RAM unless
    ``memory_dir`` is given); ``None`` runs without memory.
    ``embed``: an embedding model (an object with ``embed(text)``, such as
    :class:`~agi.core.llm.OpenAICompatEmbedder`) used for both episodic memory
    and tool matching. Without one, matching is by words and word groups.
    ``otlp_endpoint``: send every span to an OpenTelemetry collector, for
    example ``http://localhost:4318/v1/traces``.
    """
    if isinstance(sandbox, str):
        if sandbox not in ("auto", "process", "wasm", "docker"):
            raise ValueError(f"unknown sandbox {sandbox!r} (use 'auto', 'process', 'wasm', 'docker' or a Sandbox)")
        sandbox = best_sandbox(policy) if sandbox == "auto" else best_sandbox(policy, require=sandbox)
    tracer = tracer or Tracer(path=trace_path)
    if otlp_endpoint:
        tracer.add_exporter(OTLPExporter(otlp_endpoint, otlp_headers, service=tracer.service))
    if memory == "auto":
        memory = HierarchicalMemory(memory_dir, sandbox=sandbox, embed=_dense(embed) if embed is not None else None)
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
    engine = CognitiveEngine(llm, tools, memory=memory, guardrails=guardrails, tracer=tracer,
                             config=config, healer=healer, embed=embed)
    engine.sandbox = sandbox                               # so callers can warm it up or inspect its isolation
    return engine


class SessionStore:
    """Conversations on disk (SQLite), so they survive a restart and can outnumber what fits in RAM."""

    def __init__(self, path: str, ttl_s: float | None = 30 * 86400, clock: Callable[[], float] = time.time):
        self.path, self.ttl_s, self.clock = path, ttl_s, clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("create table if not exists sessions "
                         "(id text primary key, updated real not null, messages text not null)")
        self._db.commit()
        self.prune()

    def load(self, session: str) -> list[dict] | None:
        with self._lock:
            row = self._db.execute("select updated, messages from sessions where id = ?", (session,)).fetchone()
        if row is None or (self.ttl_s is not None and self.clock() - row[0] > self.ttl_s):
            return None
        try:
            return json.loads(row[1])
        except ValueError:
            return None

    def save(self, session: str, messages: list[dict]) -> None:
        with self._lock:
            self._db.execute("insert or replace into sessions values (?, ?, ?)",
                             (session, self.clock(), json.dumps(messages)))
            self._db.commit()

    def delete(self, session: str) -> bool:
        with self._lock:
            found = self._db.execute("delete from sessions where id = ?", (session,)).rowcount > 0
            self._db.commit()
        return found

    def prune(self) -> int:
        """Delete conversations idle for longer than the time-to-live."""
        if self.ttl_s is None:
            return 0
        with self._lock:
            removed = self._db.execute("delete from sessions where updated < ?",
                                       (self.clock() - self.ttl_s,)).rowcount
            self._db.commit()
        return removed

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("select count(*) from sessions").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class AgentService:
    """An engine behind an async interface, safe to share between many callers."""

    def __init__(self, engine: CognitiveEngine, concurrency: int = 4, timeout: float | None = 300.0,
                 max_sessions: int = 1000, session_tokens: int = 2000, monitor: DriftMonitor | None = None,
                 summarize_sessions: bool = False, session_store: "SessionStore | str | None" = None):
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
        self.session_store = SessionStore(session_store) if isinstance(session_store, str) else session_store
        self._session_lock = threading.Lock()
        self._gates: dict[int, asyncio.Semaphore] = {}       # one per event loop
        self.in_flight = 0
        self.alarms: list = []

    @classmethod
    def create(cls, llm: Callable[[str], str] | None = None, *, concurrency: int = 4,
               timeout: float | None = 300.0, monitor: DriftMonitor | None = None,
               session_store: "SessionStore | str | None" = None, **engine_options) -> "AgentService":
        """Build the engine and wrap it. With ``memory_dir``, conversations are kept there too
        (``sessions.db``) unless ``session_store`` says otherwise."""
        directory = engine_options.get("memory_dir")
        if session_store is None and directory:
            os.makedirs(directory, exist_ok=True)
            session_store = os.path.join(directory, "sessions.db")
        return cls(build_engine(llm, **engine_options), concurrency=concurrency, timeout=timeout, monitor=monitor,
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
