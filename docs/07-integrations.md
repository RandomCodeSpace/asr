# 07 — Integrations

External systems the framework talks to, plus their dev / local
alternatives.

---

## LLM providers

Source: `src/runtime/llm.py:get_llm`. Each provider kind maps to a
LangChain chat-model class.

| Provider kind | Production class | Auth | Local alternative |
|---|---|---|---|
| `ollama` | `langchain_ollama.ChatOllama` | `api_key` (Ollama Cloud) or none (local Ollama) | Run Ollama locally (`ollama serve`); set `base_url: http://localhost:11434` |
| `azure_openai` | `langchain_openai.AzureChatOpenAI` | `api_key`, `endpoint`, `deployment` | None — Azure is cloud-only. Use `stub` for tests. |
| `openai_compat` | `langchain_openai.ChatOpenAI` (with `base_url=`) | `api_key` | Any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, …) |
| `stub` | `runtime.llm.StubChatModel` | none | Built-in canned-response chat model for tests / smoke |

Switching providers: edit `llm.providers` + `llm.models` in
`config/config.yaml`; per-skill override via `skill.model` in the
skill's YAML.

429 retry: free / shared upstream tiers (e.g. OpenRouter `…:free`)
are protected by the rate-limit retry regime added in v1.5-D
(`_RATE_LIMIT_MARKERS` in `src/runtime/graph.py`).

Live verification: `tests/test_integration_driver_s1.py` parametrises
three legs (`local`, `workhorse`, `azure`); each independently skips
on missing keys. `tests/test_llm_providers_smoke.py` is the
single-call smoke gated on `OLLAMA_LIVE=1`.

---

## MCP servers

Source: `src/runtime/mcp_loader.py`,
`src/runtime/config.py:MCPServerConfig`.

Three transports:

| Transport | Connection | Use case |
|---|---|---|
| `in_process` | Loads a Python module that exports a `mcp = FastMCP(...)` instance | Default for example apps; zero network cost |
| `stdio` | Spawns a subprocess command, talks JSON-RPC over stdio | Wrapping a 3rd-party MCP CLI |
| `http` | Talks JSON-RPC over HTTP | Remote MCP server (often with bearer auth via `headers`) |
| `sse` | Server-sent events transport | Inference: present in `MCPServerConfig.transport` literal but not exercised in tests; status: scaffold |

Configuration:

```yaml
mcp:
  servers:
    - name: local_inc
      transport: in_process
      module: examples.incident_management.mcp_server
      category: incident_management
    - name: ext_metrics
      transport: http
      url: ${EXTERNAL_MCP_URL}
      headers:
        Authorization: "Bearer ${EXT_TOKEN}"
      category: observability
```

The example apps' MCP servers all use `in_process` — the bundle
ships with the MCP code in the same process. Tests fixture sample at
`tests/fixtures/sample_config.yaml` covers `http` + bearer auth.

---

## Auth providers

The framework does not integrate with external auth providers
(no SSO, OIDC, SAML, …). Air-gap deploys live behind corporate
network controls.

The only auth touched by the framework:

- **MCP server bearer auth** — `headers.Authorization: "Bearer
  ${EXT_TOKEN}"` per server config.
- **Webhook trigger bearer auth** — `auth: bearer` +
  `auth_token_env: <ENV_VAR>` per trigger config; constant-time
  comparison via `hmac.compare_digest`.

Both read tokens from env vars at process start; rotating a secret
requires a process restart.

---

## Queues / messaging

The framework has no built-in queue. The closest thing is the
**trigger registry** (`src/runtime/triggers/`), which can fire a
session start from:

- HTTP POST (webhook)
- APScheduler cron (in-process)
- Custom plugin transport (entry-point or explicit registration)

There is no SQS / Kafka / NATS / RabbitMQ integration shipped, but
the `TriggerTransport` ABC and `plugin_transports` kwarg on
`TriggerRegistry.create` exist for adding one. The
`src/runtime/triggers/transports/plugin.py` file is a stub —
Inference: scaffold for future SQS/Kafka work.

---

## Observability / external services (referenced by the
incident_management example)

Source: `examples/incident_management/mcp_servers/observability.py`,
`mcp_servers/remediation.py`, `mcp_servers/user_context.py`.

The example app's MCP servers expose **mock** versions of operational
tools:

| Tool | Purpose | Real backend (production) | Mock (this repo) |
|---|---|---|---|
| `get_logs(service, minutes)` | Recent logs | Datadog / Loki / Splunk | Returns canned WARN/ERROR/INFO lines |
| `get_metrics(service, minutes)` | CPU/latency/error-rate samples | Prometheus / Datadog | Returns canned numeric envelope |
| `get_service_health(env)` | Service-level health | Service registry / k8s health | Returns canned per-service health dict |
| `check_deployment_history(hours, env)` | Recent deploys | ArgoCD / Spinnaker / Octopus | Returns canned recent-release list |
| `notify_oncall(team, message)` | Page oncall | PagerDuty / Opsgenie | Returns synthesised page id |
| `apply_fix(proposal_id, env)` | Run a remediation script | Ansible / Salt / custom | Returns deterministic success/failure |
| `propose_fix(hypothesis, env)` | Generate a fix proposal | LLM-driven (this remains LLM-only in production) | Returns canned proposal_id |

To wire real backends: replace the `_impl` body in the corresponding
`mcp_servers/<name>.py` file with the real client call, keeping the
function signature stable (the LLM-visible tool surface comes from
the signature + docstring).

---

## Code review tools

`examples/code_review/mcp_server.py` ships **mocked**:

- `fetch_pr_diff(repo, number)` — reads from
  `tests/fixtures/code_review/<repo>/<number>.json` if present;
  otherwise returns a tiny synthetic diff.
- `add_review_finding(...)` and `set_recommendation(...)` —
  in-process state mutation only.

There is no real GitHub or GitLab integration. To wire one up,
replace `fetch_pr_diff` with a `gh` API call or PyGithub /
python-gitlab client.

---

## Memory layers (incident_management example)

Source: `examples/incident_management/asr/`.

| Layer | Backing files | Lifecycle |
|---|---|---|
| L2 Knowledge Graph | `incidents/kg/{components,edges}.json` (or seed bundle at `examples/incident_management/asr/seeds/kg/`) | Read-only; populated by ops, consumed by intake |
| L5 Release Context | `incidents/releases/recent.json` (or seed bundle) | Read-only; populated by deploy pipeline (out of scope), consumed by triage |
| L7 Playbook Store | `incidents/playbooks/*.yaml` (or seed bundle) | Read-only; authored by SREs, consumed by resolution |

Filesystem-backed by design — no Neo4j / Redis / pgvector dependency
keeps the framework air-gap-friendly. When the configured layer
directory is empty, each store falls back to the bundled seeds so a
fresh checkout has working data.

Mutation paths (write-back from agents, playbook authoring) are
deferred — Inference: planned for a later milestone.

---

## CI / external services for development

| Service | Purpose | Configuration |
|---|---|---|
| GitHub Actions | CI (lint / type-check / test / sonar / bundle freshness) | `.github/workflows/ci.yml` |
| SonarCloud | Code quality + coverage gate | `sonar-project.properties`, `SONAR_TOKEN` repo secret |
| CodeQL | Security analysis | Default GitHub setup; `.github/workflows/` (auto-generated) |
| Socket Security | Dependency security scan | Auto-detected on PRs |
| OpenRouter | Live LLM smoke (when keys present) | `OPENROUTER_API_KEY` repo secret (Inference: project owner controls) |

CI does not call live LLM providers — the test suite is
stub-mode-only. Live integration smokes (`tests/test_integration_driver_s1.py`,
`tests/test_llm_providers_smoke.py`) are gated on env vars and skipped
in CI.

---

## Where to override for local dev

| Want to | Override |
|---|---|
| Use local Ollama instead of Ollama Cloud | `llm.providers.ollama.base_url: http://localhost:11434` |
| Use SQLite in `/var/lib/asr/` instead of `/tmp` | `storage.metadata.url: sqlite:////var/lib/asr/asr.db`, `storage.vector.path: /var/lib/asr/faiss` |
| Use Postgres instead of SQLite | `pip install asr[postgres]`; `storage.metadata.url: postgresql://…` |
| Skip MCP entirely for an integration test | Use `LLMConfig.stub()` + an empty `MCPConfig` (see `tests/_envelope_helpers.py`) |
| Test webhook trigger locally | Set `triggers:` in a local `config.yaml`; `curl -H 'Authorization: Bearer …' -X POST http://localhost:8000/triggers/<name>` |
