"""Building blocks for more capable, more trustworthy AI agents.

* :mod:`agi.core`       - the dual-process engine, routing, sandboxes, guardrails, tracing, resilience
* :mod:`agi.memory`     - working, episodic (vector), semantic (graph) and procedural memory
* :mod:`agi.reasoning`  - plan search with real verification (MCTS, Tree and Graph of Thoughts)
* :mod:`agi.tools`      - typed tools, semantic tool routing, runtime tool creation, self-healing
* :mod:`agi.evaluation` - benchmark suite, baselines, drift detection, A/B-tested optimisation
* :mod:`agi.api`        - an async service object and a small HTTP server
* :mod:`agi.world`      - world models and causal graphs learned by acting
* :mod:`agi.collective` - multi-agent consensus and sandboxed, bounded self-improvement
* :mod:`agi.training`   - a trained search policy, and fine-tuning data from agent runs

``agi.llm``, ``agi.sandbox``, ``agi.react``, ``agi.agent`` and ``agi.evals`` still import as before.

Run ``python -m agi`` for a guided demonstration.
"""
from .core.agent import Attempt, LearningAgent, Task
from .core.engine import CognitiveEngine, EngineConfig
from .core.guardrails import Guardrails
from .core.tracing import Tracer
from .core.types import EngineResult
from .core.llm import LLM, LLMError, OpenAICompatEmbedder, OpenAICompatLLM, ScriptedLLM
from .core.react import Conversation, GroundingVerifier, ReActAgent, Run
from .core.sandbox import Sandbox, SandboxPolicy, SandboxResult, check_source
from .tools import Tool, ToolRegistry, standard_tools

__version__ = "0.3.0"

__all__ = ["CognitiveEngine", "EngineConfig", "EngineResult", "Guardrails", "Tracer", "LearningAgent", "Task", "Attempt", "Sandbox", "SandboxPolicy", "SandboxResult",
           "check_source", "LLM", "LLMError", "OpenAICompatLLM", "OpenAICompatEmbedder",
           "ScriptedLLM", "ReActAgent", "Run", "Conversation", "GroundingVerifier", "Tool",
           "ToolRegistry", "standard_tools", "__version__"]
