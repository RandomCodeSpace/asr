# 05 — Configuration

## Layered config

Two layers, in order of precedence:

| Layer | File(s) | Owns |
|---|---|---|
| **Framework** | `config/config.yaml` (or `${APP_CONFIG}`) | LLM providers + models, MCP servers, storage URL, gateway policy, framework knobs (confidence threshold, escalation roster, dedup), trigger registry, runtime tunables |
| **App** | `examples/<app>/config.yaml`, `config/<app>.yaml` (composite) | Domain-specific knobs: severity aliases, escalation teams, environments, similarity thresholds |

Source: `src/runtime/config.py` (~1100 lines) holds every pydantic
schema. Framework reads + validates at orchestrator boot via
`load_config(path)`.

The framework's `AppConfig` does **not** contain incident-shaped
keys — they live on `IncidentAppConfig`. Adding a new domain
field is a one-line addition to `IncidentAppConfig`, never to
`runtime.config.AppConfig`.

---

## Environment variables

Used in `config.yaml` via `${VAR_NAME}` interpolation
(`src/runtime/config.py:_interpolate`). Strict-mode resolver
**fails at config-load** if a referenced var is missing — this
is by design, so missing keys can't silently fall through to
"use default model".

| Var | Used by | Default | Notes |
|---|---|---|---|
| `OLLAMA_API_KEY` | `ollama_cloud` provider | none | Required if any `llm.providers.*.kind: ollama` entry references it |
| `OPENROUTER_API_KEY` | `openai_compat` provider via OpenRouter | none | |
| `AZURE_OPENAI_KEY` | `azure_openai` provider | none | |
| `AZURE_ENDPOINT` | `azure_openai` provider | none | Full URL incl. trailing `/` |
| `AZURE_DEPLOYMENT` | `smart` model in default config | `gpt-4o` (test driver default) | Per-deployment Azure name |
| `EXTERNAL_MCP_URL` | external HTTP MCP server | none | See `tests/fixtures/sample_config.yaml` |
| `EXT_TOKEN` | external HTTP MCP server bearer auth | none | |
| `ASR_LOG_LEVEL` | `src/runtime/ui.py:46-65` | unset (silent) | `DEBUG` / `INFO` / `WARNING` / `ERROR`; takes effect via `force=True` `logging.basicConfig` |
| `APP_CONFIG` | `src/runtime/ui.py:68` | `config/config.yaml` | Path override |
| `OLLAMA_LIVE` | `tests/test_llm_providers_smoke.py` | unset (skip) | Set to `1` to opt into live Ollama smoke |
| `OLLAMA_BASE_URL` | `tests/test_integration_driver_s1.py` | unset | Required for the integration driver `local` arm |

CI config (`.github/workflows/ci.yml:71-83`) sets dummy values for
all the above so the strict `_interpolate` check passes — tests
don't call live providers.

---

## Config file: `config/config.yaml`

Top-level structure (see
`config/config.yaml.example` for an annotated template):

```yaml
storage:
  metadata:
    url: "sqlite:////tmp/asr.db"     # SQLAlchemy URL
    pool_size: 5                     # postgres only; sqlite uses NullPool
    echo: false                      # SQL echo to stdout
  vector:
    backend: faiss                   # faiss | pgvector | none
    path: "/tmp/asr-faiss"           # FAISS only
    collection_name: "incidents"
    distance_strategy: cosine        # cosine | euclidean | inner_product

llm:
  default: workhorse                 # name from llm.models below
  providers:
    ollama_cloud:
      kind: ollama
      base_url: https://ollama.com
      api_key: ${OLLAMA_API_KEY}
    azure:
      kind: azure_openai
      endpoint: ${AZURE_ENDPOINT}
      api_version: 2024-08-01-preview
      api_key: ${AZURE_OPENAI_KEY}
    openrouter:
      kind: openai_compat
      base_url: https://openrouter.ai/api/v1
      api_key: ${OPENROUTER_API_KEY}
    stub:
      kind: stub                     # in-memory canned responses for tests
  models:
    workhorse:
      provider: openrouter
      model: inclusionai/ring-2.6-1t:free
      temperature: 0.0
    gpt_oss:
      provider: ollama_cloud
      model: gpt-oss:20b
      temperature: 0.0
    gpt_oss_cheap:
      provider: ollama_cloud
      model: gpt-oss:20b
      temperature: 0.4
    smart:
      provider: azure
      model: gpt-4o
      deployment: gpt-4o
      temperature: 0.0
  embedding:
    provider: ollama_cloud
    model: nomic-embed-text          # single embedding model

mcp:
  servers:
    - name: local_inc
      transport: in_process          # in_process | stdio | http | sse
      module: examples.incident_management.mcp_server
      category: incident_management
    - name: local_observability
      transport: in_process
      module: examples.incident_management.mcp_servers.observability
      category: observability
    # ...

runtime:
  state_class: examples.incident_management.state.IncidentState
  gateway:
    policy:                          # tool_name -> low | medium | high
      apply_fix: high
      restart_service: medium
      get_logs: low
  max_concurrent_sessions: 8         # SessionCapExceeded → HTTP 429

orchestrator:
  entry_agent: intake                # name of the first skill in the graph
  default_terminal_status: needs_review
  signals: [success, failed, needs_input]
  injected_args:
    environment: state.environment   # session-derived args injected before LLM-visible signature
  terminal_tools:                    # tool_name -> status transition rules
    - tool_name: mark_resolved
      status: resolved
      kind: terminal
    - tool_name: mark_escalated
      status: escalated
      kind: escalation
      extract_fields: { team: args.team }
  patch_tools: [submit_hypothesis, update_incident]
  default_llm_request_timeout: 120.0

framework:
  confidence_threshold: 0.75
  escalation_teams: [payments-oncall, infra-oncall, ...]
  approval_timeout: 1800             # seconds; ApprovalWatchdog timeout
  intake_context: {}                 # generic intake bag
  session_id_prefix: INC             # apps override (CR for code-review)

dedup:
  enabled: true
  stage1_top_k: 5
  stage1_threshold: 0.82
  stage2_model: workhorse
  prompt_template: |                 # LLM judge prompt (defaultable)
    ...

triggers:                            # optional; trigger registry transports
  - name: pagerduty-incident
    transport: webhook
    target_app: incident_management
    payload_schema: examples.incident_management.triggers.PagerDutyPayload
    transform: examples.incident_management.triggers.transform_pagerduty
    auth: bearer
    auth_token_env: PAGERDUTY_WEBHOOK_TOKEN
    idempotency_ttl_hours: 24

learning:
  scheduler:
    enabled: true
    cron: "0 2 * * *"                # nightly 02:00 UTC
```

Inference: not every block above is required for a minimal boot;
omitting `triggers` / `dedup` / `learning` is supported (they're
optional).

---

## Per-skill config

Each skill is a `<skill_dir>/config.yaml` + `<skill_dir>/system.md`
pair under `examples/<app>/skills/`.

```yaml
# examples/incident_management/skills/triage/config.yaml
description: Hypothesis-loop triage agent
kind: responsive                     # responsive | supervisor | monitor
model: gpt_oss_cheap                 # optional per-agent override; falls back to llm.default
tools:
  local_inc:
    - submit_hypothesis
    - update_incident
  local_observability:
    - get_logs
    - get_metrics
    - get_service_health
    - check_deployment_history
routes:
  - when: success
    next: deep_investigator
  - when: needs_input
    next: __end__
    gate: confidence
  - when: default
    next: deep_investigator
```

The accompanying `system.md` is the system prompt template. It must
include the markdown turn-output contract block (see
`examples/incident_management/skills/_common/output.md`) — failure
to include it will trip the envelope parser unless gpt-oss
synthesises something Path 6 can salvage.

---

## Feature flags

There are no first-class feature flags. Toggles are config-driven:

| Toggle | Mechanism |
|---|---|
| Disable dedup | `dedup.enabled: false` |
| Disable auto-learning scheduler | `learning.scheduler.enabled: false` |
| Disable HITL gating per env | `gate_policy.gated_environments: []` |
| Disable a tool's risk tier | Remove from `runtime.gateway.policy` (defaults to `auto`) |
| Disable a trigger | Remove from `triggers:` block; restart |
| Switch checkpointer to postgres | Install `asr[postgres]`; change `storage.metadata.url` to a postgres URL |

---

## Secrets required (production)

For a typical incident-management deploy:

| Secret | Purpose |
|---|---|
| `OLLAMA_API_KEY` (or `OPENROUTER_API_KEY`, etc.) | LLM provider auth |
| `AZURE_OPENAI_KEY` + `AZURE_ENDPOINT` | If Azure provider used |
| Webhook bearer tokens (e.g. `PAGERDUTY_WEBHOOK_TOKEN`) | If webhook triggers configured |
| Postgres credentials in the SQLAlchemy URL | If `storage.metadata.url` points at postgres |

**Do NOT commit secrets.** The framework reads them from env vars
via `${VAR_NAME}` interpolation; bind them via your deploy's
secret manager (k8s secret / docker `--env-file` / etc.).

`.env` is gitignored at the repo root. CI uses dummy values.

---

## Safe defaults

The shipped `config/config.yaml.example` documents safe defaults:

- `llm.default: stub_default` — runs without any LLM
  provider keys (useful for first boot / smoke)
- `storage.metadata.url: sqlite:///incidents/incidents.db` — local
  SQLite, no external service
- `vector.backend: faiss` — local FAISS, no external service
- No `triggers:` block — trigger registry off; only `POST /sessions`
  works
- No `dedup:` block — dedup off
- No `learning.scheduler.enabled` block — scheduler off

These give a working framework boot with zero external dependencies.
Production deploys swap in a real LLM provider and (optionally)
real triggers / dedup / scheduler.

---

## Validators

`src/runtime/config.py` enforces:

- `LLMConfig.default` must exist in `llm.models`
- Every `llm.models[*].provider` must exist in `llm.providers`
- Every `${VAR}` placeholder must resolve at config-load (strict)
- Every `skill.model` must exist in `llm.models` (skill-level
  validator, separate from `LLMConfig`)

Errors raise typed exceptions (`LLMConfigError`, `ValueError`) at
boot — the framework refuses to start with a misconfigured registry.
