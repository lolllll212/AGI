# Architecture

The package has three layers and two side packages. An arrow means "may import".

```
infrastructure ──► training ──► eval ──► application ──► domain
       └──────────────┴───────────┴───────────┴────────────┘
```

| Layer | Holds | May import |
| --- | --- | --- |
| `agi.domain` | Data and rules with no input or output: tools and the registry, memory records, validated value objects, the ports, pure functions | nothing else in `agi` |
| `agi.application` | What the agent does: the engine and its router, the reason-act loop, memory logic, reasoning, tool routing and repair, the swarm, the service object | `domain` |
| `agi.eval` | The benchmark, the evaluator, the optimiser | `domain`, `application` |
| `agi.training` | The search policy and data export | those, and `eval` |
| `agi.infrastructure` | Everything that touches the outside: model clients, sandboxes, SQLite stores, metrics, the HTTP server, the CLI, the wiring | all of the above |

`tests/architecture/layers.py` reads every import statement in `src/agi` and
fails if one points the wrong way. It runs as a test and as a script:

```bash
python tests/architecture/layers.py     # prints "layers are clean", or each breach with file and line
```

## How application code gets an adapter

The engine needs a sandbox, a tracer and four stores, and those live in
`infrastructure`, which `application` may not import. So application code asks
a registry for them by name:

```python
from agi.domain.ports import make
sandbox = make("sandbox")
```

`agi.infrastructure.di.defaults` registers the standard adapter for each name
when the package is imported:

| Port | Default adapter |
| --- | --- |
| `sandbox` | the process sandbox |
| `tracer` | the in-memory tracer |
| `knowledge_graph`, `episodic_store`, `semantic_graph`, `procedural_memory`, `session_store` | the SQLite-backed stores in `adapters/persistence` |
| `engine` | `build_engine` in `infrastructure/di/container.py` |

To use something else, either pass it in (every constructor that uses a port
also accepts the object directly) or register a replacement:

```python
from agi.domain.ports import register
register("sandbox", lambda **options: MySandbox(**options))
```

This keeps the import rule honest but it is a service locator: the dependency
is not visible in a signature, and the layer test cannot see through it.

## A request, end to end

1. `infrastructure/api/server.py` accepts the connection, finds the route in
   `api/routes/`, checks the token (`adapters/security/auth.py`) and the rate
   limit, and validates the body against `api/schemas.py`.
2. The route calls `AgentService` (`application/interfaces/service.py`), which
   holds the concurrency limit, the timeout and the conversation for the session.
3. The service calls `CognitiveEngine.run` (`application/cortex/engine.py`).
   The router picks reflex, react or deliberate; guardrails screen the input,
   every tool result and the answer; tools run through the registry, code
   through the sandbox.
4. The finished result goes back up. On the way, the service calls its
   observers: the server's metrics and its list of recent runs. An observer
   that raises is ignored.
5. The server writes the response, counts it in the metrics and writes one
   access-log line.

## The console

`web/` is a separate TypeScript project. `npm run build` writes static files
into `src/agi/infrastructure/api/static/`, and `api/routes/ui.py` serves them.
The console only ever talks to the server it was loaded from, with relative
URLs, through the same JSON endpoints any other client uses.

## Where the layout came from, and where it differs

The folder names follow a larger project's hexagonal layout. Only folders with
something real in them exist here. Not present, because this project has no
such feature: `auth` as a domain concept (there are API keys, in
`adapters/security`), `crdt`, `speech`, `mcp`, `deployment`, `eventbus`,
background `workers`, `backup`, a separate `backend/`, and hardware code.
