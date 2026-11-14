# Running the server

> The container setup described under "With Docker Compose" was written without
> Docker available and **has never been started**. The single-process setup
> above it has been run. Treat the Compose file as a careful first draft: expect
> to fix things on first use, and run `scripts/smoke.py` once it is up.

## One process

```bash
pip install ".[graph]"
export AGI_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export OPENAI_API_KEY=...
agi serve --model gpt-4o-mini --workspace ./workspace --state ./state --access-log -
```

Then open http://127.0.0.1:8000 and enter the token. To try it without a model:
`agi serve --demo` (answers are replayed from the benchmark's reference
solutions; the console says so).

The same settings can live in a file: `agi serve --config config/agi.toml`.
Every key is the name of a flag with underscores; a flag on the command line
wins over the file. A misspelt key is an error.

| Setting | Meaning |
| --- | --- |
| `host`, `port` | Where to listen. Any host other than 127.0.0.1 requires a token or keys. |
| `token_env`, `token_file` | Where the bearer token comes from: an environment variable (default `AGI_API_TOKEN`) or a file, which wins. |
| `keys_file` | A JSON file `{"name": "secret"}` giving each caller their own key, rate limit and conversations. |
| `tls_cert`, `tls_key` | Serve HTTPS directly. Behind a proxy, leave these out and let the proxy do it. |
| `rate_limit`, `max_connections` | Requests per second per caller; connections at once. |
| `workspace`, `read_only`, `network` | The one directory the agent may read and write; forbid writing; allow `http_get`. |
| `sandbox` | `process`, `wasm`, `docker` or `auto`: where the agent's code runs. |
| `state` | A directory for long-term memory, learned routing and conversations. Without it everything is in RAM. |
| `plugins` | A directory of extra tools. Trusted code. |
| `traces`, `otlp_endpoint`, `access_log` | A JSONL trace file; an OpenTelemetry collector; one JSON line per request (`-` for standard output). |
| `ui`, `metrics`, `allow_framing` | Serve the console; serve `/metrics`; let other sites frame the console. |

## Checking a running server

```bash
python scripts/smoke.py http://127.0.0.1:8000                      # token from AGI_API_TOKEN
python scripts/smoke.py https://agent.example.com --token-file secrets/agi_api_token.txt
```

It checks health, that a request without the token is refused, one task and its
trace, the run list, the metrics page and the console, and exits non-zero if
any of them fails. Pass `--task` and `--expect` to use a task of your own; the
default is arithmetic, which needs no model.

## With Docker Compose (not yet run)

```bash
cp .env.example .env                 # set AGI_MODEL, OPENAI_API_KEY, GRAFANA_ADMIN_PASSWORD
python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/agi_api_token.txt
chmod 644 secrets/agi_api_token.txt  # the containers read it as other users
# put fullchain.pem and privkey.pem in monitoring/nginx/ssl/ (see the README there)
docker compose up -d --build
python scripts/smoke.py https://localhost --token-file secrets/agi_api_token.txt --insecure
```

| Service | Role | Reachable from outside |
| --- | --- | --- |
| `nginx` | TLS, redirects HTTP to HTTPS, limits requests per address, hides `/metrics` | ports 80 and 443 |
| `api` | The agent server, read-only file system apart from its `/data` volume, no added capabilities | through nginx |
| `prometheus` | Scrapes `api:8000/metrics` every 15 s with the token; evaluates the alert rules | no |
| `alertmanager` | Groups alerts. Ships with no receiver: add yours in `monitoring/alertmanager/alertmanager.yml` | no |
| `grafana` | The "Agent overview" dashboard, with Prometheus and Loki already connected | at `/grafana/` through nginx |
| `loki`, `promtail` | Collect the API's access log and the proxy's log; kept 14 days | no |

Things to know before trusting it:

- **Where the agent's code runs.** Inside the `api` container, in the process
  sandbox. The container is the outer wall. `sandbox = "docker"` would need the
  Docker socket inside the container, which hands it the host; do not do that.
- **Promtail reads the Docker socket** (read-only) to find containers. That is
  how it normally works, and it is still broad access; drop `promtail` and
  `loki` if you do not want it.
- **The token file is world-readable on the host** so that two containers
  running as different users can read it. Keep the `secrets/` directory itself
  closed to other users of the machine.
- **One API process.** The server keeps the run list, the rate limits and the
  metrics in memory, so run one replica. Conversations and memory are on the
  `agi-data` volume and survive a restart.
- **Back up the `agi-data` volume** if the agent's memory matters to you;
  nothing here does it.
- **Image versions are pinned** to releases that existed when this was written.
  Review and update them.

## Alerts

| Alert | Fires when | Severity |
| --- | --- | --- |
| `AgiDown` | Prometheus cannot scrape the API for 2 minutes | critical |
| `AgiModelCircuitOpen` | Calls to the model have been failing for 2 minutes | critical |
| `AgiHighFailureRate` | Over 30% of tasks fail, error or time out, for 10 minutes | warning |
| `AgiSlowRuns` | 95% of tasks take up to more than 60 s, for 10 minutes | warning |
| `AgiServerErrors` | More than 0.1 responses per second are 5xx, for 5 minutes | warning |
| `AgiConnectionsRefused` | The connection limit was hit in the last 5 minutes | warning |
| `AgiAuthFailures` | More than 20 requests with a wrong or missing token in 5 minutes | warning |
| `AgiUnsupportedAnswers` | Over 10% of answers contain figures no tool returned, for 30 minutes | warning |
| `AgiDriftAlarm` | The drift monitor reports a change from the baseline | warning |
| `AgiTraceExportDropping` | Trace spans are being discarded | info |

The thresholds are starting points chosen without real traffic. Adjust them
after watching the dashboard for a while.

The dashboard is generated: edit `scripts/build_dashboard.py` and run it, and a
test fails if the committed JSON and the script disagree, or if a panel names a
metric the server does not export.
