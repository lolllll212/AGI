# Moving from 0.4 to 0.5

0.5 rearranged the package into layers and moved it under `src/`. Nothing was
removed. Every import path from 0.4 (and the older aliases such as `agi.llm`)
still works and raises a `DeprecationWarning` that names the new location:

```
agi.core.engine is an import path from before 0.5; import from agi.application.cortex.engine instead
```

To find what to change, run your tests with `-W error::DeprecationWarning`.
The names re-exported from the package root (`from agi import ...`) are unchanged.

## Modules that moved whole

| Before | Now |
| --- | --- |
| `agi.core.types` | `agi.domain.value_objects.records` |
| `agi.tools.base` | `agi.domain.entities.tool` |
| `agi.tools.introspect` | `agi.domain.entities.tool_schema` |
| `agi.memory.models` | `agi.domain.entities.memory` |
| `agi.reasoning.problem` | `agi.domain.entities.problem` |
| `agi.memory.embed` | `agi.domain.services.embedding` |
| `agi.core.engine`, `.router`, `.agent`, `.guardrails` | `agi.application.cortex.*` |
| `agi.core.orchestrator` | `agi.application.swarm.orchestrator` |
| `agi.collective.*` | `agi.application.swarm.*` |
| `agi.memory.working`, `.hierarchy`, `.engine`, `.extract` | `agi.application.memory.*` |
| `agi.tools.router`, `.healing`, `.managed` | `agi.application.tools.*` |
| `agi.reasoning.got`, `.logic`, `.mcts`, `.planner`, `.problems`, `.tot` | `agi.application.reasoning.*` |
| `agi.world.*` | `agi.application.world.*` |
| `agi.memory.graph` | `agi.infrastructure.adapters.persistence.knowledge_graph` |
| `agi.memory.vector` | `agi.infrastructure.adapters.persistence.episodic` |
| `agi.memory.semantic` | `agi.infrastructure.adapters.persistence.semantic` |
| `agi.reasoning.policy` | `agi.training.policy` |
| `agi.evaluation.dataset` | `agi.training.dataset` |
| `agi.evaluation.evaluator`, `.extended`, `.optimizer`, `.suite` | `agi.eval.*` |
| `agi.api.http` | `agi.infrastructure.api.server` |
| `agi.__main__` | `agi.infrastructure.cli.main` (`python -m agi` is unchanged) |

## Modules that were split

| Before | Name | Now |
| --- | --- | --- |
| `agi.core.llm` | `LLM` | `agi.domain.ports.llm` |
| | `LLMError` | `agi.domain.exceptions` |
| | `extract_json`, `extract_code` | `agi.domain.services.text` |
| | `ScriptedLLM` | `agi.infrastructure.adapters.llm.scripted` |
| | `OpenAICompatEmbedder` | `agi.infrastructure.adapters.embedding.openai_compat` |
| | everything else | `agi.infrastructure.adapters.llm.openai_compat` |
| `agi.core.sandbox` | `check_source` | `agi.domain.services.source_check` |
| | `SandboxPolicy`, `SandboxResult`, the allow and deny lists | `agi.domain.value_objects.sandbox` |
| | `Sandbox` | `agi.infrastructure.adapters.sandbox.process` |
| `agi.core.isolation` | `SandboxUnavailable` | `agi.domain.exceptions` |
| | `DockerSandbox` | `agi.infrastructure.adapters.sandbox.docker` |
| | `WasmSandbox` | `agi.infrastructure.adapters.sandbox.wasm` |
| | everything else | `agi.infrastructure.adapters.sandbox.selection` |
| `agi.core.tracing` | `OTLPExporter`, `otlp_payload` | `agi.infrastructure.adapters.observability.otlp` |
| | everything else | `agi.infrastructure.adapters.observability.tracing` |
| `agi.core.resilience` | the exceptions | `agi.domain.exceptions` |
| | `RateLimiter`, `CircuitBreaker` | `agi.application.cortex.resilience` |
| `agi.core.react` | `numbers_in` | `agi.domain.services.text` |
| | everything else | `agi.application.cortex.react` |
| `agi.memory.procedural` | `render_answer` | `agi.domain.services.text` |
| | everything else | `agi.infrastructure.adapters.persistence.procedural` |
| `agi.tools.builtin` | `evaluate`, `format_number` | `agi.domain.services.arithmetic` |
| | everything else | `agi.infrastructure.adapters.execution.builtin` |
| `agi.api.service` | `build_engine` | `agi.infrastructure.di.container` |
| | `SessionStore` | `agi.infrastructure.adapters.persistence.sessions` |
| | `AgentService` | `agi.application.interfaces.service` |

## Two things that behave differently

- **An old path to a split module is a read-only view.** `agi.core.sandbox`
  now gathers names from three modules. Reading from it works; replacing a name
  on it (for example `monkeypatch.setattr("agi.core.sandbox.check_source", ...)`)
  no longer changes what `Sandbox` calls. Patch the module that uses the name:
  `agi.infrastructure.adapters.sandbox.process`.
- **The project is installed from `src/`.** Running from a checkout without
  installing needs `PYTHONPATH=src`, or `pip install -e .` once.

## Server

`AgentServer` takes the same arguments as before and adds `metrics`, `ui`,
`ui_dir`, `allow_framing` and `access_log`. Two defaults are new: the console is
served at `/` and metrics at `/metrics` (behind the token). Pass `ui=False` and
`metrics=False`, or `--no-ui` and `--no-metrics`, to get exactly the 0.4 surface.
`GET /health` gained one field, `auth_required`.
