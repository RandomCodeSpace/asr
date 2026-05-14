# 02 — Architecture

> Companion to [`docs/DESIGN.md`](DESIGN.md), which carries the
> long-form design narrative. This file is the quick-scan summary.

## Major components

```
+------------------------------------------------------------+
| App layer (examples/incident_management, examples/code_review)
| - state.py, config.py, skills/, mcp_server.py, ui.py       |
+------------------------------------------------------------+
| Framework — runtime/                                       |
| - Session, Skill, AgentRun, ToolCall, AgentTurnOutput      |
| - Orchestrator, OrchestratorService                        |
| - Gateway (wrap_tool), policies, ToolRegistry              |
| - SessionStore, HistoryStore, EventLog                     |
| - graph.py: build_graph + make_agent_node                  |
| - llm.py: provider abstraction                             |
| - ui.py: Streamlit shell                                   |
| - api.py: FastAPI surface                                  |
+------------------------------------------------------------+
| LangGraph 1.x  (orchestration / state / checkpointing)     |
| LangChain 1.x  (chat models, agents.create_agent, tools)   |
| FastMCP        (in-process / stdio / http MCP servers)     |
+------------------------------------------------------------+
| Providers: Ollama Cloud · OpenRouter · Azure OpenAI · …    |
+------------------------------------------------------------+
```

| Component | Source | Responsibility |
|---|---|---|
| `Session` (model) | `src/runtime/state.py:70-172` | Lifecycle + telemetry fields. Apps subclass. |
| `Skill` (config) | `src/runtime/skill.py` | YAML-driven agent declaration: kind, model, tools, routes, system_prompt |
| `Orchestrator` | `src/runtime/orchestrator.py` | Owns compiled langgraph + SessionStore + per-session lock |
| `OrchestratorService` | `src/runtime/service.py` | Long-lived asyncio loop wrapper. Thread-safe `submit_async` / `submit_and_wait` bridge |
| `make_agent_node` | `src/runtime/graph.py:539+` and `src/runtime/agents/responsive.py:49+` | Builds one langgraph node per skill |
| `_drive_agent_with_resume` | `src/runtime/graph.py:202+` | Drives `langchain.agents.create_agent` executor with HITL pause/resume |
| `wrap_tool` (Gateway) | `src/runtime/tools/gateway.py:224+` | Risk-rated tool wrapper; injects session-derived args; raises `interrupt()` on high-risk |
| `parse_envelope_from_result` | `src/runtime/agents/turn_output.py` | 6-path envelope parser (markdown-primary, with synthesis fallbacks) |
| `SessionStore` | `src/runtime/storage/session_store.py` | CRUD over `IncidentRow` + FAISS write-through |
| `HistoryStore` | `src/runtime/storage/history_store.py` | Read-only similarity search over the same engine |
| `EventLog` | `src/runtime/storage/event_log.py` | Append-only `session_events` table |
| `ApprovalWatchdog` | `src/runtime/tools/approval_watchdog.py` | Background task that times out stale `pending_approval` rows |

## Request / data flow (one session, end-to-end)

```
UI / API ──start_session(query, environment, …)──▶ OrchestratorService
                                                       │
                                                       ▼
                                  Orchestrator (per-session lock)
                                                       │
                              new IncidentRow inserted ▼
                                  langgraph compiled graph (Pregel)
                                                       │
                              ┌────────────────────────┴────────────────┐
                              │ for each skill in topological order:    │
                              │   make_agent_node:                      │
                              │     reload session from store           │
                              │     wrap tools (gateway)                │
                              │     create_agent (langgraph subgraph)   │
                              │     _drive_agent_with_resume:           │
                              │       inner.ainvoke(messages)           │
                              │       if __interrupt__:                 │
                              │         raise GraphInterrupt → outer    │
                              │         pauses; UI sees pending_approval│
                              │       else parse envelope, record       │
                              │         AgentRun, route on signal       │
                              │   gate node (low-confidence)?           │
                              │   route to next skill / __end__         │
                              └─────────────────────────────────────────┘
                                                       │
                                  finalize: terminal-tool match? ▼
                                  default_terminal_status?
                                  agent failure? → status='error'
                                  paused on HITL? → SKIP finalize
```

Detailed contract: `docs/DESIGN.md` § 4 + § 7.

## Storage choices

| Store | Backend | Default URL / path | Owner |
|---|---|---|---|
| Session metadata | SQLAlchemy (SQLite or Postgres) | `sqlite:////tmp/asr.db` | `SessionStore`, `HistoryStore` |
| Vector similarity | FAISS (filesystem) | `/tmp/asr-faiss/` | `SessionStore._add_vector` |
| LangGraph checkpoints | `langgraph-checkpoint-sqlite` (default) or `langgraph-checkpoint-postgres` (opt-in) | Same SQLite DB as session metadata | `make_checkpointer` (`src/runtime/checkpointer.py`) |
| Event log | SQLAlchemy `session_events` table | Same SQLite DB | `EventLog.append` |
| Memory layers (incident_management only) | Filesystem JSON | `incidents/{kg,releases,playbooks}/` (or seed bundle) | `examples/incident_management/asr/*_store.py` |
| Lesson store (auto-learning) | SQLAlchemy `session_lessons` table | Same SQLite DB | `LessonStore` |

The whole framework runs against ONE durable backend (SQLite or
Postgres) carrying four separate concerns. Apps don't get to choose
backends per-store — the storage URL is a single config knob.

## External systems

The runtime in production reaches out to:

- **LLM providers** (variable): Ollama Cloud, Azure OpenAI,
  OpenAI-compatible endpoints (OpenRouter, etc.). Configured per
  `llm.providers` in `config/config.yaml`. Stub provider for tests.
- **MCP servers**: in-process (Python module) by default; `stdio`
  and `http` transports also supported per `mcp.servers[*].transport`.
  Schema: `MCPServerConfig` in `src/runtime/config.py`.
- **APScheduler** (in-process): drives nightly `LessonRefresher`
  jobs and any `schedule:` triggers from the trigger registry.

The runtime does NOT reach out to:

- The public internet at boot or runtime in air-gapped deploys —
  every provider URL is configurable; the hardcoded
  `https://ollama.com` fallback was removed in Phase 13 (HARD-05).
- Any package mirror at deploy time — the deploy is copy-only;
  `uv sync` runs at *install* time inside CI / the dev box.

## Important tradeoffs

| Decision | Trade | Where decided |
|---|---|---|
| LangGraph as orchestration engine | Don't maintain a graph engine; pay for langgraph version churn | `docs/DESIGN.md` DEC-001 |
| `langchain.agents.create_agent` for the per-agent loop | Single tool-loop with native ToolStrategy fallback; we're tied to langchain v1.x's agent API | `docs/DESIGN.md` DEC-002, Phase 15 |
| Markdown contract over `response_format` JSON | Lenient parsing in our code; 7 parse paths instead of 1 schema | DEC-003, Phase 22 |
| Pure-policy HITL gating | One source of truth (`should_gate`); everywhere else just calls it | DEC-004, Phase 11 |
| Generic `Session` + `extra_fields` JSON | Apps can extend without schema migrations; loses some type safety on app fields | DEC-005, v1.1 |
| Per-agent `skill.model` override | Cheap models for cheap agents; one config to think about | DEC-006, v1.5-C |
| Single-file bundle | Air-gap deployable; large files for review (~600KB each) | DEC-007, BUNDLER-01 |
| Concept-leak ratchet | CI gate keeps framework generic; some legitimate `incident` references look like leaks until cleaned | DEC-008, v1.5-B |
| 429 separate retry regime (longer backoff) | Free-tier OpenRouter survives transient throttles; non-429 4xx still fail fast | DEC-009, v1.5-D |
| Inner agent checkpointer + reload-on-entry | HITL Approve/Reject actually drives the gated tool; more state per agent invocation | DEC-010, PR #6 |

## What this architecture is NOT

- **Not a workflow engine** — agents are LLM-driven, not declarative
  state machines. Routing is signal-based, not condition-tree.
- **Not multi-tenant by default** — one process, one orchestrator,
  one storage URL. Multi-tenant deployments need a separate
  process/DB per tenant.
- **Not horizontally scalable** — `OrchestratorService` is a
  single-process / single-loop model. The lock registry
  (`SessionLockRegistry`) prevents concurrent writes per session
  but assumes one orchestrator per DB.
- **Not authenticated** — there's no built-in user authentication on
  the FastAPI surface. Air-gap deploys live behind corporate
  network controls; trigger webhook auth is bearer-token only.
