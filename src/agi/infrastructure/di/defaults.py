"""Fill the adapter registry with the standard choices.

Each entry is looked up only when something asks for it, so importing ``agi``
does not import SQLite stores, sandboxes or HTTP clients that a program never uses.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from agi.domain.ports import register

_ADAPTERS = "agi.infrastructure.adapters."

DEFAULTS: dict[str, tuple[str, str]] = {
    "sandbox": (_ADAPTERS + "sandbox.process", "Sandbox"),
    "tracer": (_ADAPTERS + "observability.tracing", "Tracer"),
    "knowledge_graph": (_ADAPTERS + "persistence.knowledge_graph", "KnowledgeGraph"),
    "episodic_store": (_ADAPTERS + "persistence.episodic", "EpisodicStore"),
    "semantic_graph": (_ADAPTERS + "persistence.semantic", "SemanticGraph"),
    "procedural_memory": (_ADAPTERS + "persistence.procedural", "ProceduralMemory"),
    "session_store": (_ADAPTERS + "persistence.sessions", "SessionStore"),
    "engine": ("agi.infrastructure.di.container", "build_engine"),
}


def _lazy(module: str, name: str) -> Callable[..., Any]:
    def factory(*args: Any, **kwargs: Any) -> Any:
        return getattr(importlib.import_module(module), name)(*args, **kwargs)
    factory.__name__ = name
    factory.__qualname__ = f"{module}.{name}"
    return factory


def wire_defaults() -> None:
    """Register the standard adapter for every port. Safe to call again: it restores the defaults."""
    for port, (module, name) in DEFAULTS.items():
        register(port, _lazy(module, name))
