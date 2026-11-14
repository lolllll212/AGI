"""Core runtime: the dual-process engine and what it stands on.

* :mod:`.engine`       - :class:`CognitiveEngine`: reflex -> reason-act -> deliberate, with escalation
* :mod:`.router`       - decides which of those a task needs
* :mod:`.orchestrator` - async multi-agent messaging and parallel sub-tasks
* :mod:`.sandbox`, :mod:`.isolation` - where untrusted code runs (process, WebAssembly, Docker)
* :mod:`.guardrails`   - input, observation and output filters
* :mod:`.resilience`   - circuit breaker, rate limiter, step budget, loop detector
* :mod:`.tracing`      - every thought, action, observation and reward, as spans
* :mod:`.types`        - the validated data that crosses component boundaries
"""
from .engine import CognitiveEngine, EngineConfig
from .guardrails import Guardrails, Rule
from .isolation import DockerSandbox, SandboxUnavailable, WasmSandbox, available_backends, best_sandbox
from .llm import LLM, LLMError, OpenAICompatEmbedder, OpenAICompatLLM, ScriptedLLM, extract_code, extract_json
from .orchestrator import Actor, Message, MessageBus, OrchestrationResult, Orchestrator, SubResult
from .react import Conversation, GroundingVerifier, ReActAgent, Run
from .resilience import (BudgetExceeded, CircuitBreaker, CircuitOpen, LoopDetected, LoopDetector, RateLimiter,
                         StepBudget)
from .router import RouteLearner, Router, RouterConfig
from .sandbox import Sandbox, SandboxPolicy, SandboxResult, check_source
from .tracing import OTLPExporter, Tracer, load_jsonl, otlp_payload
from .types import (AgentConfig, EngineResult, GuardrailVerdict, Observation, Route, StepRecord, ToolCall,
                    TraceEvent)

__all__ = ["LLM", "Actor", "AgentConfig", "BudgetExceeded", "CircuitBreaker", "CircuitOpen", "CognitiveEngine",
           "Conversation", "DockerSandbox", "EngineConfig", "EngineResult", "GroundingVerifier", "GuardrailVerdict",
           "Guardrails", "LLMError", "LoopDetected", "LoopDetector", "Message", "MessageBus", "Observation",
           "OTLPExporter", "OpenAICompatEmbedder", "OpenAICompatLLM", "OrchestrationResult", "Orchestrator", "RateLimiter",
           "ReActAgent", "Route", "RouteLearner", "Router", "RouterConfig", "Rule", "Run", "Sandbox", "SandboxPolicy",
           "SandboxResult", "SandboxUnavailable", "ScriptedLLM", "StepBudget", "StepRecord", "SubResult",
           "ToolCall", "TraceEvent", "Tracer", "WasmSandbox", "available_backends", "best_sandbox", "check_source",
           "extract_code", "extract_json", "load_jsonl", "otlp_payload"]
