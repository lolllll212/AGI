"""Building blocks for more capable, more trustworthy AI agents.

* :mod:`agi.memory`     - continual learning: episodic memory over a knowledge graph
* :mod:`agi.reasoning`  - System-2 reasoning: search + real verification + backtracking
* :mod:`agi.world`      - grounding: world models and causal graphs learned by acting
* :mod:`agi.collective` - multi-agent consensus and sandboxed, bounded self-improvement
* :mod:`agi.tools`      - typed, logged tools: files, data, HTTP, sandboxed Python, memory
* :mod:`agi.react`      - the reason-act-observe loop, with answer grounding and memory
* :mod:`agi.training`   - a trained search policy, and fine-tuning data from agent runs
* :mod:`agi.evals`      - a benchmark of real-world tasks for tool-using agents
* :mod:`agi.sandbox`    - the execution boundary the rest relies on
* :mod:`agi.agent`      - memory, verified search and the trained policy in one loop

Run ``python -m agi`` for a guided demonstration.
"""
from .agent import Attempt, LearningAgent, Task
from .llm import LLM, LLMError, OpenAICompatEmbedder, OpenAICompatLLM, ScriptedLLM
from .react import Conversation, GroundingVerifier, ReActAgent, Run
from .sandbox import Sandbox, SandboxPolicy, SandboxResult, check_source
from .tools import Tool, ToolRegistry, standard_tools

__version__ = "0.2.0"

__all__ = ["LearningAgent", "Task", "Attempt", "Sandbox", "SandboxPolicy", "SandboxResult",
           "check_source", "LLM", "LLMError", "OpenAICompatLLM", "OpenAICompatEmbedder",
           "ScriptedLLM", "ReActAgent", "Run", "Conversation", "GroundingVerifier", "Tool",
           "ToolRegistry", "standard_tools", "__version__"]
