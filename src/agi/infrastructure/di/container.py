"""The composition root: assemble an engine from concrete adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from agi.application.cortex.engine import CognitiveEngine, EngineConfig
from agi.application.cortex.guardrails import Guardrails
from agi.application.memory.hierarchy import HierarchicalMemory
from agi.application.tools.healing import SelfHealer, healing_python_tool
from agi.application.tools.router import toolsmith_tools
from agi.domain.entities.tool import ToolRegistry
from agi.domain.value_objects.sandbox import SandboxPolicy
from agi.infrastructure.adapters.execution.builtin import standard_tools
from agi.infrastructure.adapters.observability.otlp import OTLPExporter
from agi.infrastructure.adapters.observability.tracing import Tracer
from agi.infrastructure.adapters.sandbox.process import Sandbox
from agi.infrastructure.adapters.sandbox.selection import best_sandbox


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
