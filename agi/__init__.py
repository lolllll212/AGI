"""Four building blocks for more capable, more trustworthy AI agents.

* :mod:`agi.memory`     - continual learning: episodic memory over a knowledge graph
* :mod:`agi.reasoning`  - System-2 reasoning: search + real verification + backtracking
* :mod:`agi.world`      - grounding: world models and causal graphs learned by acting
* :mod:`agi.collective` - multi-agent consensus and sandboxed, bounded self-improvement
* :mod:`agi.sandbox`    - the execution boundary all of the above rely on
* :mod:`agi.agent`      - the pieces wired into one learning loop

Run ``python -m agi`` for a guided demonstration.
"""
from .agent import Attempt, LearningAgent, Task
from .llm import LLM, OpenAICompatLLM, ScriptedLLM
from .sandbox import Sandbox, SandboxPolicy, SandboxResult, check_source

__version__ = "0.1.0"

__all__ = ["LearningAgent", "Task", "Attempt", "Sandbox", "SandboxPolicy", "SandboxResult",
           "check_source", "LLM", "OpenAICompatLLM", "ScriptedLLM", "__version__"]
