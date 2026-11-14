"""Import paths from 0.4 and earlier keep working.

The package was rearranged into layers in 0.5 (``agi.domain``, ``agi.application``,
``agi.infrastructure``, ``agi.eval``, ``agi.training``). Rather than leave a stub
file at every old location, one import hook answers for all of them:

* an old module that moved whole *is* the new module (the same object, so
  ``isinstance`` and monkeypatching behave as before);
* an old module or package whose contents were divided between layers becomes a
  read-only *facade* holding the names from each of its new homes. Assigning to
  an attribute of a facade does not change the real module.

Each old name warns once (``DeprecationWarning``) and says where to import from
instead. ``docs/migration-0.5.md`` has the full table.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

# old module -> the module it became
MOVED: dict[str, str] = {
    "agi.agent": "agi.application.cortex.agent",
    "agi.api.http": "agi.infrastructure.api.server",
    "agi.collective": "agi.application.swarm",
    "agi.collective.agents": "agi.application.swarm.agents",
    "agi.collective.collective": "agi.application.swarm.collective",
    "agi.collective.forge": "agi.application.swarm.forge",
    "agi.collective.ledger": "agi.application.swarm.ledger",
    "agi.collective.policy": "agi.application.swarm.policy",
    "agi.core.agent": "agi.application.cortex.agent",
    "agi.core.engine": "agi.application.cortex.engine",
    "agi.core.guardrails": "agi.application.cortex.guardrails",
    "agi.core.orchestrator": "agi.application.swarm.orchestrator",
    "agi.core.router": "agi.application.cortex.router",
    "agi.core.types": "agi.domain.value_objects.records",
    "agi.evaluation.dataset": "agi.training.dataset",
    "agi.evaluation.evaluator": "agi.eval.evaluator",
    "agi.evaluation.extended": "agi.eval.extended",
    "agi.evaluation.optimizer": "agi.eval.optimizer",
    "agi.evaluation.suite": "agi.eval.suite",
    "agi.memory.embed": "agi.domain.services.embedding",
    "agi.memory.engine": "agi.application.memory.engine",
    "agi.memory.extract": "agi.application.memory.extract",
    "agi.memory.graph": "agi.infrastructure.adapters.persistence.knowledge_graph",
    "agi.memory.hierarchy": "agi.application.memory.hierarchy",
    "agi.memory.models": "agi.domain.entities.memory",
    "agi.memory.semantic": "agi.infrastructure.adapters.persistence.semantic",
    "agi.memory.vector": "agi.infrastructure.adapters.persistence.episodic",
    "agi.memory.working": "agi.application.memory.working",
    "agi.reasoning.got": "agi.application.reasoning.got",
    "agi.reasoning.logic": "agi.application.reasoning.logic",
    "agi.reasoning.mcts": "agi.application.reasoning.mcts",
    "agi.reasoning.planner": "agi.application.reasoning.planner",
    "agi.reasoning.policy": "agi.training.policy",
    "agi.reasoning.problem": "agi.domain.entities.problem",
    "agi.reasoning.problems": "agi.application.reasoning.problems",
    "agi.reasoning.tot": "agi.application.reasoning.tot",
    "agi.tools.base": "agi.domain.entities.tool",
    "agi.tools.healing": "agi.application.tools.healing",
    "agi.tools.introspect": "agi.domain.entities.tool_schema",
    "agi.tools.managed": "agi.application.tools.managed",
    "agi.tools.router": "agi.application.tools.router",
    "agi.world": "agi.application.world",
    "agi.world.causal": "agi.application.world.causal",
    "agi.world.env": "agi.application.world.env",
    "agi.world.model": "agi.application.world.model",
    "agi.world.physics": "agi.application.world.physics",
}

# old module or package -> the new modules that now hold its names
DIVIDED: dict[str, list[str]] = {
    "agi.api": [
        "agi.infrastructure.api",
        "agi.application.interfaces.service",
        "agi.infrastructure.di.container",
        "agi.infrastructure.adapters.persistence.sessions",
    ],
    "agi.api.service": [
        "agi.infrastructure.di.container",
        "agi.infrastructure.adapters.persistence.sessions",
        "agi.application.interfaces.service",
    ],
    "agi.core": [
        "agi.application.cortex",
        "agi.application.swarm.orchestrator",
        "agi.infrastructure.adapters.sandbox",
        "agi.infrastructure.adapters.llm",
        "agi.infrastructure.adapters.embedding",
        "agi.infrastructure.adapters.observability",
        "agi.domain.value_objects",
        "agi.domain.exceptions",
        "agi.domain.services.text",
        "agi.domain.services.source_check",
        "agi.domain.ports.llm",
    ],
    "agi.core.isolation": [
        "agi.domain.exceptions",
        "agi.infrastructure.adapters.sandbox.process",
        "agi.infrastructure.adapters.sandbox.docker",
        "agi.infrastructure.adapters.sandbox.wasm",
        "agi.infrastructure.adapters.sandbox.selection",
    ],
    "agi.core.llm": [
        "agi.domain.ports.llm",
        "agi.domain.exceptions",
        "agi.domain.services.text",
        "agi.infrastructure.adapters.llm.scripted",
        "agi.infrastructure.adapters.llm.openai_compat",
        "agi.infrastructure.adapters.embedding.openai_compat",
    ],
    "agi.core.react": [
        "agi.domain.services.text",
        "agi.application.cortex.react",
    ],
    "agi.core.resilience": [
        "agi.domain.exceptions",
        "agi.application.cortex.resilience",
    ],
    "agi.core.sandbox": [
        "agi.domain.value_objects.sandbox",
        "agi.domain.services.source_check",
        "agi.infrastructure.adapters.sandbox.process",
    ],
    "agi.core.tracing": [
        "agi.infrastructure.adapters.observability.otlp",
        "agi.infrastructure.adapters.observability.tracing",
    ],
    "agi.evaluation": [
        "agi.eval",
        "agi.training.dataset",
    ],
    "agi.memory": [
        "agi.application.memory",
        "agi.infrastructure.adapters.persistence",
        "agi.domain.entities.memory",
        "agi.domain.services.embedding",
        "agi.domain.services.text",
    ],
    "agi.memory.procedural": [
        "agi.domain.services.text",
        "agi.infrastructure.adapters.persistence.procedural",
    ],
    "agi.reasoning": [
        "agi.application.reasoning",
        "agi.domain.entities.problem",
        "agi.training.policy",
    ],
    "agi.tools": [
        "agi.domain.entities.tool",
        "agi.domain.entities.tool_schema",
        "agi.application.tools",
        "agi.infrastructure.adapters.execution",
        "agi.domain.services.arithmetic",
    ],
    "agi.tools.builtin": [
        "agi.domain.services.arithmetic",
        "agi.infrastructure.adapters.execution.builtin",
    ],
}

# old names that were already aliases before 0.5
SAME: dict[str, str] = {
    "agi.evals": "agi.evaluation.suite",
    "agi.llm": "agi.core.llm",
    "agi.react": "agi.core.react",
    "agi.sandbox": "agi.core.sandbox",
}


def new_home(name: str) -> str | list[str] | None:
    """Where an old import path points now: a module, a list of modules, or ``None`` if it is not an old path."""
    name = SAME.get(name, name)
    return MOVED.get(name) or DIVIDED.get(name)


def _facade(name: str, sources: list[str]) -> types.ModuleType:
    module = types.ModuleType(name, f"Compatibility facade for the 0.4 module {name!r}; see {', '.join(sources)}.")
    module.__path__ = []                                    # lets "import {name}.submodule" reach the hook
    for source in sources:
        for key, value in vars(importlib.import_module(source)).items():
            if not (key.startswith("__") and key.endswith("__")):
                module.__dict__.setdefault(key, value)
    module.__all__ = sorted(key for key in module.__dict__ if not key.startswith("_"))
    return module


class _LegacyImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self) -> None:
        self._specs: dict[int, object] = {}

    def find_spec(self, name, path=None, target=None):
        if new_home(name) is None:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        home = new_home(spec.name)
        target = ", ".join(home) if isinstance(home, list) else home
        warnings.warn(f"{spec.name} is an import path from before 0.5; import from {target} instead",
                      DeprecationWarning, stacklevel=2)
        if isinstance(home, list):
            return _facade(spec.name, home)
        module = importlib.import_module(home)
        self._specs[id(module)] = module.__spec__           # the import system is about to overwrite it
        return module

    def exec_module(self, module) -> None:
        original = self._specs.pop(id(module), None)
        if original is not None:
            module.__spec__ = original


_hook = _LegacyImports()


def install() -> None:
    if _hook not in sys.meta_path:
        sys.meta_path.insert(0, _hook)
