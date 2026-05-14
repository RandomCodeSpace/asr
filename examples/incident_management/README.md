# Incident Management — Example Application

The flagship example app for the framework. A 4-skill investigation
pipeline (intake → triage → deep_investigator → resolution) with
ASR memory layers (L2 Knowledge Graph, L5 Release Context, L7
Playbook Store).

For framework-wide design + decisions, see
[`docs/DESIGN.md`](../../docs/DESIGN.md). This README only covers
the bits specific to this app.

## Run

```bash
uv run python -m runtime --config config/incident_management.yaml
ASR_LOG_LEVEL=INFO uv run streamlit run src/runtime/ui.py --server.port 37777
```

## Layout

```
examples/incident_management/
├── state.py                 IncidentState(Session) + Reporter + IncidentStatus
├── config.py                IncidentAppConfig + load_incident_app_config
├── config.yaml              severity_aliases, escalation_teams, environments, thresholds
├── mcp_server.py            IncidentMCPServer + 3 tools
├── mcp_servers/             observability + remediation + user_context tools
├── asr/                     ASR memory layers
│   ├── memory_state.py        MemoryLayerState + L2/L5/L7 pydantic models
│   ├── kg_store.py            L2 Knowledge Graph (filesystem)
│   ├── release_store.py       L5 Release Context (filesystem)
│   ├── playbook_store.py      L7 Playbook Store (filesystem)
│   └── seeds/                 seed data per layer
├── skills/                  4 agent YAML configs + _common/ shared prompts
│   ├── _common/
│   ├── intake/                kind: supervisor, runs similarity + memory hydration
│   ├── triage/                hypothesis-loop investigator
│   ├── deep_investigator/     evidence gathering
│   └── resolution/            propose / apply fix or escalate
├── ui.py                    Streamlit accordion-per-incident view
└── __main__.py              entry point
```

## Domain shape

`IncidentState(Session)` adds `query`, `environment`, `reporter`,
`summary`, `tags`, `severity`, `category`, `matched_prior_inc`,
`resolution`, `memory: MemoryLayerState`. Session ids look like
`INC-YYYYMMDD-NNN`.

## ASR memory layers

| Layer | Class | Backing |
|---|---|---|
| L2 Knowledge Graph | `KGStore` | `incidents/kg/{components,edges}.json` (or seeds) |
| L5 Release Context | `ReleaseStore` | `incidents/releases/recent.json` (or seeds) |
| L7 Playbook Store | `PlaybookStore` | `incidents/playbooks/*.yaml` (or seeds) |

The intake supervisor hydrates `IncidentState.memory` from these
stores using components extracted from the user's query. The
triage / DI / resolution agents read the bundle as additional
context. Mutation paths (write-back) are deferred.

## MCP tools

`IncidentMCPServer` exposes `lookup_similar_incidents`,
`create_incident`, `update_incident`, `submit_hypothesis`,
`mark_resolved`, `mark_escalated`. Sibling MCP servers under
`mcp_servers/` add observability (`get_logs`, `get_metrics`,
`get_service_health`, `check_deployment_history`) and remediation
(`propose_fix`, `apply_fix`, `notify_oncall`).

The risk-rated gateway (`runtime.gateway.policy`) tags `apply_fix`
as `high` so production runs pause for operator approval before
applying any fix. See [DESIGN § 7](../../docs/DESIGN.md#7-hitl-approve--reject)
for the HITL pause/resume mechanics.

## Skill model

Per-agent LLM override: intake declares `model: gpt_oss_cheap` (a
fast / cheap model on Ollama Cloud) so the supervisor pre-filter is
cheap; downstream agents follow `llm.default`. See
[DESIGN § 5.3](../../docs/DESIGN.md#5-llm-provider-story) for the
per-agent dispatch.
