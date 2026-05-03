# Incident Management — Example Application

The flagship example app for the `runtime` framework. Demonstrates how to layer a domain-specific agent application on top of the generic orchestration runtime.

## Run

```bash
python -m examples.incident_management
```

This launches the Streamlit UI for incident triage and resolution.

## Architecture

This example extends the generic `Session` model with incident-specific state and provides a 4-agent investigation pipeline (intake → triage → deep_investigator → resolution). The framework owns session lifecycle, agent dispatch, and tool gateway; this example owns domain shape, skill prompts, and MCP tools.

```
examples/incident_management/
├── state.py             IncidentState(Session) + Reporter + IncidentStatus
├── config.py            IncidentAppConfig + load_incident_app_config
├── config.yaml          severity_aliases, escalation_teams, environments, thresholds
├── mcp_server.py        IncidentMCPServer with 3 tools
├── asr/                 ASR memory layers (Phase 9)
│   ├── memory_state.py    MemoryLayerState + L2/L5/L7 pydantic models
│   ├── kg_store.py        L2 Knowledge Graph (filesystem)
│   ├── release_store.py   L5 Release Context (filesystem)
│   ├── playbook_store.py  L7 Playbook Store (filesystem)
│   └── seeds/             bundled JSON / YAML seed data per layer
├── skills/              4 agent YAML configs + _common/ shared prompts
│   ├── _common/
│   ├── intake/
│   ├── triage/
│   ├── deep_investigator/
│   └── resolution/
├── ui.py                Streamlit accordion-per-incident UI
├── __main__.py          Entry point
└── README.md            this file
```

## Configuration

Two layers, in order of precedence:

| Layer | File | What it owns |
|---|---|---|
| Framework | `config/config.yaml` | LLM providers + models, MCP servers, storage URL, paths |
| App | `examples/incident_management/config.yaml` | severity_aliases, escalation_teams, environments, similarity_threshold, confidence_threshold |

The framework's `AppConfig` does **not** contain incident-flavored keys — they all live in `IncidentAppConfig`. Adding a new domain field is a one-line addition to `IncidentAppConfig`, never to `runtime.config.AppConfig`.

## State Model

`IncidentState(Session)` extends the framework's `Session` base with:

- `query: str` — initial user description
- `environment: str` — production/staging/dev/local
- `reporter: Reporter` — who filed the incident
- `summary: str` — agent-produced narrative
- `tags: list[str]`
- `severity: str | None` — high/medium/low after triage
- `category: str | None`
- `matched_prior_inc: str | None` — id of similar resolved incident, if any
- `resolution: Any` — final outcome
- `memory: MemoryLayerState` — ASR memory-layer slots (L2 KG / L5 Release / L7 Playbooks); see "ASR memory layers" below

The framework only reads/writes the inherited `Session` fields (id, status, created_at, agents_run, tool_calls, findings, pending_intervention, token_usage). Domain fields above are read/written exclusively by example-app code.

## MCP Tools

`IncidentMCPServer` exposes three tools to the agents:

- `lookup_similar_incidents(query, environment)` — embedding similarity over closed incidents
- `create_incident(query, environment, reporter_id, reporter_team)` — start a new investigation
- `update_incident(incident_id, patch)` — write to status, severity, category, summary, tags, findings, resolution

The MCP loader reads the registry from `config/config.yaml` (`mcp.servers[*].module`), which points at `examples.incident_management.mcp_server` for this app.

## Skills

Each agent (intake/triage/deep_investigator/resolution) is a `Skill` defined by a `config.yaml` + `system.md` pair under `skills/<agent>/`. The `_common/` directory holds shared snippets all skills inherit. The framework's skill loader (`runtime.skill.load_all_skills`) takes a directory; `paths.skills_dir` in the framework config points at this directory.

## Durable Memory

Sessions survive cold restart. The framework wires LangGraph's `AsyncSqliteSaver` (or `AsyncPostgresSaver` for production) to the same database URL declared in `config.yaml`'s `storage.metadata.url`, on a separate connection pool with WAL + `busy_timeout=30s` configured on both sides. Resume after a crash: load the session by id and call `Orchestrator.resume_session(incident_id, user_input)` — it dispatches `Command(resume=...)` against the persisted graph state. Pending interventions are dual-written to both the LangGraph checkpoint and `IncidentRow.pending_intervention` so dashboards reading the relational row stay accurate.

The state class is configurable. `config/config.yaml` sets `runtime.state_class: examples.incident_management.state.IncidentState` so row hydration produces `IncidentState` instances, not bare `Session` instances. A different app subclassing `Session` simply points this key at its own state class — no framework changes.

## Multi-Session

The orchestrator runs as a long-lived `OrchestratorService` (single asyncio loop on a background thread, single shared FastMCP client pool). Each session is an asyncio task on that loop, started via `service.start_session(query=..., environment=..., reporter_id=..., reporter_team=...)` which returns the session id immediately while the agent run continues in the background.

Concurrent sessions are isolated at the row level (each writes to its own `IncidentRow`) but share the MCP client pool. `service.list_active_sessions()` returns a thread-safe snapshot of in-flight sessions; `service.stop_session(session_id)` cancels a task and marks the row `status="stopped"`. Default cap is 8 concurrent sessions; raise `SessionCapExceeded` (HTTP 429) on overflow. Configure via `runtime.max_concurrent_sessions` in `config.yaml`.

Three new HTTP endpoints expose this surface: `POST /sessions` (start), `GET /sessions` (list active), `DELETE /sessions/{id}` (stop). Legacy `POST /investigate` is preserved as a deprecated alias delegating to the same code path.

The Streamlit UI now shows two sections in the sidebar: **In-flight** (live, polled from `list_active_sessions()`) and **History** (closed sessions). The detail pane auto-polls every 1.5s while a session's status is non-terminal; polling stops once status is `resolved` / `escalated` / `stopped`.

## Risk-rated tool gateway

Phase 4 adds a per-tool risk gateway that sits between every agent and every MCP tool call. Each tool is tagged in `runtime.gateway.policy` (`low` / `medium` / `high`) and the gateway dispatches on the resolved action: `low` runs without overhead, `medium` runs and persists `ToolCall(status="executed_with_notify")` for soft audit, and `high` raises `langgraph.types.interrupt(...)` to pause the graph for human approval — the wrap closure captures the live `Session` per agent invocation so audit lands on the right row.

A prod-environment override tightens the policy further: when a session's `environment` is in `prod_environments` and the tool name matches a `resolution_trigger_tools` glob, the gateway forces `approve` regardless of the tool's risk tier. This guarantees that "blast-radius" tools (apply_fix, deploy, mass_update_*) always get a human in the loop in production, even when the underlying tier is `low` or `medium`.

Operators resolve pending approvals via `POST /sessions/{sid}/approvals/{tool_call_id}` (decision=approve|reject + approver + optional rationale) or via the **Pending Approvals** cards in the Streamlit detail pane. Both paths drive `Command(resume={...})` against the same graph thread_id so HTTP and UI clients share the resume contract.

Legacy `tool_calls` rows from before Phase 4 are migrated lazily by `runtime.storage.migrate_tool_calls_audit` — idempotent JSON walk that fills the new audit fields with their defaults. Run once at orchestrator startup or as a one-off ops job.

## ASR memory layers (Phase 9)

Phase 9 lays the foundation for ASR's 7-layer memory architecture (see `ASR.md` §3 / §6). Three of those layers ship in this batch as filesystem-backed read-only stores under `examples/incident_management/asr/`. No Neo4j / Redis / pgvector dependency — air-gapped friendly per `rules/build.md`.

| Layer | Class | Backing files | Surface |
|---|---|---|---|
| **L2** Knowledge Graph | `KGStore` | `incidents/kg/{components,edges}.json` | `get_component` / `find_by_name` / `neighbors` / `subgraph` |
| **L5** Release Context | `ReleaseStore` | `incidents/releases/recent.json` | `recent_for_service` / `suspect_at` / `context` |
| **L7** Playbook Store | `PlaybookStore` | `incidents/playbooks/*.yaml` | `get` / `list_all` / `match` |

Each store accepts a `root: Path` for testability. When the configured layer directory is empty, the store falls back to the seed bundle at `examples/incident_management/asr/seeds/<layer>/` so a fresh checkout has working data without provisioning `incidents/`.

Investigations attach context fetched from each layer to `IncidentState.memory` — the `MemoryLayerState` container with `l2_kg: L2KGContext | None`, `l5_release: L5ReleaseContext | None`, `l7_playbooks: list[L7PlaybookSuggestion]`. The whole bundle round-trips through the P8-J `extra_fields` JSON column, so no row schema change is needed. Mutation paths (writes from agents, playbook authoring) are deferred to later sub-phases (9e–9g).

## ASR MVP investigation flow (Phase 9 — 9h/9i/9k/9m)

The MVP slice wires a deliberate, end-to-end investigation pipeline on top of the memory-layer foundation. Three new skills + helpers + UI panels:

```
intake → triage (hypothesis loop) → deep_investigator → resolution (L7 + gateway)
```

**1. Supervisor (`intake`, P9-9h + 2026-05-03 generalisation).** Default entry agent (framework default `entry_agent='intake'` matches; no override needed in `config/config.yaml`). The `intake` skill is `kind: supervisor` whose runner composes `runtime.intake.default_intake_runner` (framework — similarity retrieval + dedup gate) with `examples.incident_management.asr.supervisor_node:default_supervisor_runner`'s memory hydration. Hydrates `session.memory` with L2 KG / L5 Release / L7 Playbook context fetched from the affected service set (extracted heuristically from the query). Applies the **single-active-investigation gate**: if another in-flight session is already covering the same components, the new session is tagged `status="duplicate"` with `parent_session_id` pointing at the active one (reuses P7 dedup linkage), and routed to `__end__`. Helper module: `examples/incident_management/asr/supervisor_node.py`.

**2. Triage hypothesis loop (P9-9i).** The triage skill now runs a bounded inner loop: generate hypothesis → gather evidence (L1 current findings, L3-equivalent past similar incidents via `lookup_similar_incidents`, L5 recent suspect deploys from `session.memory.l5_release`) → score → refine or accept. Hard cap of 3 iterations. The deterministic scorer (`asr.hypothesis_loop.score_hypothesis` — token-overlap, no LLM) and the `should_refine` predicate are unit-tested separately so the loop's safety net isn't LLM-dependent. Each iteration writes `{iteration, hypothesis, score, rationale}` to `findings.findings_triage` for the UI's hypothesis trail panel.

**3. Resolution + prod-HITL (P9-9k).** The resolution skill consults `session.memory.l7_playbooks`, picks the top match, and translates `playbook.remediation` into tool calls via `asr.resolution_helpers.playbook_to_tool_calls`. Every call routes through the framework gateway. The `runtime.gateway` block in `config/config.yaml` locks the prod-environment override: `update_incident` (medium) and any `remediation:*` tool ALWAYS require approval in `production`, regardless of risk tier. The override only TIGHTENS — it can never relax a higher-risk tool to `auto`.

**4. UI panels (P9-9m-sliver).** Two read-only views on the incident detail page:

- **Approval Inbox** — already shipped in P4-H; surfaces every tool call with `status="pending_approval"` as an Approve / Reject card.
- **Hypothesis Trail** — collapsed accordion showing the triage agent's iterative `{iteration, hypothesis, score, rationale}` log, sourced from `session.findings`. No new persistent state.

## Agent kinds

Phase 6 introduces a `kind` discriminator on every `Skill`, allowing three execution models behind a single config schema:

| `kind`       | Where it runs                                | Writes `AgentRun`? |
|--------------|----------------------------------------------|--------------------|
| `responsive` | LangGraph node, on a session turn (today's path) | yes |
| `supervisor` | LangGraph node, dispatches to a subordinate via `Send()` | **no** (dispatch log only) |
| `monitor`    | Out-of-band, scheduled via `MonitorRunner`   | no (signals only) |

Each existing skill in this example carries `kind: responsive` explicitly; the loader still defaults the field to `responsive` when omitted, so legacy YAML keeps working unchanged. A `supervisor` skill declares `subordinates`, `dispatch_strategy: llm|rule`, and either a `dispatch_prompt` (for `llm`) or a `dispatch_rules` list (for `rule`); supervisor dispatches emit a structured `supervisor_dispatch` log entry instead of bloating `agents_run` with router rows. A `monitor` skill declares a 5-field `schedule:` cron expression, an `observe:` list of tool names, an `emit_signal_when:` safe-eval expression, and a `trigger_target:` naming a Phase-5 trigger to fire when the expression is true. Monitors run on a small bounded thread pool (`max_workers=4`); each tick has a per-monitor `tick_timeout_seconds` so one slow `observe` tool cannot stall the others. Dangerous expression constructs (calls, attribute access, comprehensions, lambda) are rejected by an AST allowlist at skill-load time — `eval()`/`exec()` are never used on user-supplied strings.

## Triggers

Phase 5 adds a declarative trigger registry that generalises session-start beyond the legacy `POST /investigate` route. After Phase 5 the framework can fire `Orchestrator.start_session` from four transport flavours: `api` (back-compat), `webhook` (third-party POST `/triggers/{name}`), `schedule` (in-process APScheduler cron), and `plugin` (custom transport registered via setuptools entry-points or explicit `plugin_transports={"kind": Class}` on `TriggerRegistry.create`). All four are wired off a single `triggers:` block in `config.yaml`.

```yaml
triggers:
  - name: pagerduty-incident
    transport: webhook
    target_app: incident_management
    payload_schema: examples.incident_management.triggers.PagerDutyPayload
    transform: examples.incident_management.triggers.transform_pagerduty
    auth: bearer
    auth_token_env: PAGERDUTY_WEBHOOK_TOKEN
    idempotency_ttl_hours: 24

  - name: nightly-prod-scan
    transport: schedule
    target_app: incident_management
    transform: examples.incident_management.triggers.transform_schedule_heartbeat
    schedule: "0 2 * * *"        # 5-field cron (UTC by default)
    timezone: UTC
    payload:
      query: "Nightly health check"
      environment: production
```

**Webhook routing:** the registry mounts one `POST /triggers/{name}` route per webhook trigger. Each trigger config declares a Pydantic `payload_schema` (validated on every request — bad body returns 422) and a `transform` callable that maps the parsed payload to `start_session(**kwargs)`. The transform error policy is fail-closed: any exception from `transform` returns `422 Unprocessable Entity` and is **not** cached for idempotency, so a retried request gets a fresh attempt.

**Bearer auth:** when `auth: bearer`, the route requires `Authorization: Bearer $auth_token_env`. The token is read from the named env var **at app startup** — rotating the secret requires a process restart. No raw secrets ever land in YAML. Constant-time comparison (`hmac.compare_digest`) guards against timing oracles. HMAC signature transports (PagerDuty `x-pagerduty-signature`, Slack `x-slack-signature`) are deferred to a later phase via the same `auth:` discriminator.

**Idempotency-Key:** webhook clients can include `Idempotency-Key: <token>` to dedupe retries. The registry stores `(trigger_name, key)` -> `session_id` in a per-process LRU and a SQLite-backed table `trigger_idempotency_keys` on the same DB used for session metadata (`storage.metadata.url`). Cold restart is survived: on LRU miss, the disk row is read; entries past `ttl_hours` are purged opportunistically. Content-based dedup (hash of body) is **out of scope until Phase 7**; only the explicit `Idempotency-Key` header is honoured in Phase 5.

**Schedule cron:** `schedule:` is a standard 5-field cron string interpreted via APScheduler's `CronTrigger.from_crontab`. The 6-field APScheduler-native form is rejected at config-load time. Drift: in-process APScheduler is good for ±1 minute under normal load — tighter SLOs need an external scheduler (Celery beat, k8s `CronJob`).

**Plugin transports:** to ship a transport for SQS / Kafka / NATS, subclass `runtime.triggers.base.TriggerTransport` and register the class either via the `runtime.triggers` setuptools entry-point group or by passing `plugin_transports={"kind": Class}` to `TriggerRegistry.create`. Explicit registrations win on key collision.

**Provenance:** every session started via a trigger receives a `TriggerInfo(name, transport, target_app, received_at)` stamped onto `inc.findings['trigger']` before the graph runs, so dashboards and audit logs can answer "where did this session come from?" without re-deriving from disjoint sources.

## Adding a new app

The framework is genuinely generic — Phase 8 lifted every domain-specific assumption out of `src/runtime/` and pinned it with the second example at `examples/code_review/`. To stand up your own app, mirror this structure under `examples/<your_app>/` (no framework changes required):

| File | What it owns | Hook into the framework |
|---|---|---|
| `state.py` | Your `Session` subclass with domain fields | `runtime.state_class` (dotted path) |
| `state.py` (`id_format` classmethod) | Your session id shape (e.g. `MYAPP-NNN`) | `Session.id_format(seq=...)` (P8-C) |
| `config.py` / `config.yaml` | Your `AppConfig` subclass for app-specific tunables | Loaded by your own loader; framework doesn't touch it |
| `mcp_server.py` | Your domain MCP tools | `mcp.servers[*].module` |
| `skills/<name>/{config,system}.{yaml,md}` | Per-skill prompt + tool wiring | `paths.skills_dir` |
| `__main__.py` + `ui.py` | Streamlit entry point | run via `python -m examples.<your_app>` |

**Round-trip:** any field you declare on your `Session` subclass that is *not* an incident-shaped typed column on `IncidentRow` (`query`, `environment`, `severity`, `tags`, ...) lands in the row's `extra_fields` JSON column on save and is hydrated back via `state_cls.model_fields` on load (P8-J). You don't need to touch the framework's row schema or converters.

**Bundle:** add a `<YOUR_APP>_APP_MODULE_ORDER` and a `build_<your_app>_app()` function in `scripts/build_single_file.py`, then call it from `main()`. The flattening pipeline + intra-import stripping pattern is the same for every app (see how `examples.code_review` does it).

The second example at [`examples/code_review/`](../code_review/README.md) is a deliberate non-incident-flavored app (PR review). It exists to *prove* the framework is generic by being a second concrete instance of the same pattern. If you're stuck on how a piece should land, check what code-review does first — the pattern is almost always already there.

## Testing

```bash
pytest tests/ -q --no-cov
```

Pin tests for this example live in `tests/test_incident_state.py` (state shape), `tests/test_mcp_incident_server.py` (MCP server), and the broader integration suite under `tests/test_*`.

## Genericity ratchet

`scripts/check_genericity.py` counts occurrences of incident-flavored tokens (`incident`, `severity`, `reporter`) inside `src/runtime/`. `tests/test_genericity_ratchet.py` enforces that the total stays at or below `BASELINE_TOTAL` — so new domain leaks into the framework layer fail CI.

```bash
python scripts/check_genericity.py            # print current counts
python scripts/check_genericity.py --baseline 140  # exit non-zero if exceeded
```

To lower the baseline: refactor a leak out of `src/runtime/`, then update `BASELINE_TOTAL` in `tests/test_genericity_ratchet.py` in the same commit. Raising the baseline requires an architecture rationale in the commit message and is a code-review red flag.
