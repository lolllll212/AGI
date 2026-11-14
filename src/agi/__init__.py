"""Building blocks for more capable, more trustworthy AI agents.

The package is arranged in layers; each may import only from the ones above it in this list:

* :mod:`agi.domain`         - plain data and rules: records, tools, memories, ports, pure text and arithmetic helpers
* :mod:`agi.application`    - the engine and what it orchestrates: ``cortex`` (route, act, verify), ``memory``,
  ``reasoning``, ``tools``, ``swarm``, ``world``, and ``interfaces`` (the service object callers use)
* :mod:`agi.eval`           - the benchmark suite, baselines, drift detection, A/B-tested optimisation
* :mod:`agi.training`       - a trained search policy, and fine-tuning data from agent runs
* :mod:`agi.infrastructure` - adapters (model clients, sandboxes, SQLite stores, tracing and metrics, the
  standard tools), the HTTP API, the command line, and ``di``, which wires adapters to ports

The names most programs need are importable from here. Import paths from 0.4
(``agi.core``, ``agi.memory``, ``agi.tools``, ``agi.api``, ...) still work and warn once.

Run ``python -m agi`` for a guided demonstration.
"""
# isort: skip_file  (the order below matters: the legacy hook and the adapters come first)
from . import _legacy
from .infrastructure.di.defaults import wire_defaults

_legacy.install()
wire_defaults()

from .application.cortex.engine import CognitiveEngine, EngineConfig                      # noqa: E402
from .application.cortex.agent import Attempt, LearningAgent, Task                        # noqa: E402
from .application.cortex.guardrails import Guardrails                                     # noqa: E402
from .application.cortex.react import Conversation, GroundingVerifier, ReActAgent, Run    # noqa: E402
from .application.interfaces.service import AgentService                                  # noqa: E402
from .application.memory.hierarchy import HierarchicalMemory                              # noqa: E402
from .domain.entities.tool import Tool, ToolRegistry                                      # noqa: E402
from .domain.exceptions import LLMError                                                   # noqa: E402
from .domain.ports.llm import LLM                                                         # noqa: E402
from .domain.services.source_check import check_source                                    # noqa: E402
from .domain.value_objects.records import EngineResult                                    # noqa: E402
from .domain.value_objects.sandbox import SandboxPolicy, SandboxResult                    # noqa: E402
from .infrastructure.adapters.embedding.openai_compat import OpenAICompatEmbedder         # noqa: E402
from .infrastructure.adapters.execution.builtin import standard_tools                     # noqa: E402
from .infrastructure.adapters.llm.openai_compat import OpenAICompatLLM                    # noqa: E402
from .infrastructure.adapters.llm.scripted import ScriptedLLM                             # noqa: E402
from .infrastructure.adapters.observability.tracing import Tracer                         # noqa: E402
from .infrastructure.adapters.sandbox.process import Sandbox                              # noqa: E402
from .infrastructure.api.server import AgentServer, serve                                 # noqa: E402
from .infrastructure.di.container import build_engine                                     # noqa: E402

__version__ = "0.6.0"

__all__ = ["AgentServer", "AgentService", "Attempt", "CognitiveEngine", "Conversation", "EngineConfig",
           "EngineResult", "GroundingVerifier", "Guardrails", "HierarchicalMemory", "LLM", "LLMError",
           "LearningAgent", "OpenAICompatEmbedder", "OpenAICompatLLM", "ReActAgent", "Run", "Sandbox",
           "SandboxPolicy", "SandboxResult", "ScriptedLLM", "Task", "Tool", "ToolRegistry", "Tracer",
           "__version__", "build_engine", "check_source", "serve", "standard_tools"]
