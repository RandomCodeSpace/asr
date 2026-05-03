# Generic Framework + ASR Example App — Roadmap

> **For agentic workers:** This is a *roadmap* — phase-level intent. Each phase will get its own detailed implementation plan (TDD task breakdown) at execution time. Run those via `superpowers:subagent-driven-development`.

**Vision:** Evolve the current incident-flavored orchestrator into a **generic agent-orchestration framework** with the ASR.md platform-monitoring system as its **flagship example application**. Other use cases (code review, customer support, etc.) drop in as additional apps without framework changes.

**Selling point right now:** the platform-monitoring solution from `ASR.md`. That's what gets demonstrated, what's tuned for production. Genericness is the hidden lever — same framework can host unrelated agent systems.

**Tech invariants:** Python 3.14, LangGraph, Pydantic v2, MCP via FastMCP, SQLAlchemy 2.x, sync runtime, air-gapped-friendly deps.

---

## 1. Naming + structural shift

The framework is *not* "the incident orchestrator with extra features". It is "an agent runtime; one of its example apps is incident management".

| Current name (framework) | New name (framework) | Stays as | Lives in |
|---|---|---|---|
| `Incident` | `Session` | example app's extended state | `examples/incident-management/state.py` |
| `IncidentRepository` | `SessionStore` (active) + `HistoryStore` (closed) | repository facets, both pure-relational+vector | framework |
| `incident_management` MCP server | (removed from framework) | example app's MCP server | `examples/incident-management/mcp_server.py` |
| `cfg.incidents.*`, `cfg.orchestrator.severity_aliases`, `cfg.intervention.*`, `cfg.environments` | (removed from framework) | example app's config | `examples/incident-management/config.yaml` |
| Framework directory `src/orchestrator/` | `src/runtime/` (or kept) | — | rename optional but recommended for clarity |
| `INC-YYYYMMDD-NNN` id format | generic `Session.id` (UUID/ULID) | example app keeps its INC- format via custom id-minter | example app |

The framework after rename owns:
- Generic session lifecycle (start, pause, resume, close, archive)
- Generic state shape: `id`, `status`, `created_at`, `updated_at`, `deleted_at`, `agents_run`, `tool_calls`, `findings`, `pending_intervention`, `user_inputs`, `token_usage`. **No domain fields.**
- Tool gateway, trigger registry, agent kinds, dedup pipeline (Phase 3+ items)

---

## 2. Surface area summary (where we land)

```
src/runtime/                          ← the framework
├── config.py                           ← framework config only (LLM, MCP, storage, triggers, agents, gateway, dedup)
├── state.py                            ← Session base (generic only); app extends
├── storage/
│   ├── session_store.py                ← active session reads/writes; LangGraph checkpointer hookup
│   ├── history_store.py                ← closed-session search (similarity + filters)
│   ├── engine.py                       ← SQLAlchemy engine factory
│   ├── embeddings.py                   ← embedder factory
│   └── vector.py                       ← VectorStore factory (FAISS dev, PGVector prod)
├── triggers/
│   ├── base.py                         ← Trigger ABC
│   ├── api.py                          ← FastAPI mount
│   └── schedule.py                     ← cron loop
├── agents/
│   ├── base.py                         ← AgentKind ABC
│   ├── responsive.py                   ← LLM-driven (existing)
│   ├── supervisor_router.py            ← no-LLM dispatch
│   └── monitor_loop.py                 ← background poller
├── tools/
│   ├── registry.py                     ← MCP tool loader (existing mcp_loader)
│   └── gateway.py                      ← risk-rated middleware + HITL
├── dedup/
│   ├── pipeline.py                     ← pluggable steps
│   └── steps/                          ← embedding, llm-confirm, structured-key
├── graph.py                            ← LangGraph builder (state-schema agnostic)
├── orchestrator.py                     ← public Orchestrator facade
└── api.py                              ← FastAPI service surface (generic)

examples/
├── incident-management/                ← the ASR.md flagship app
│   ├── config.yaml                     ← app config (severity, escalation_teams, environments, etc.)
│   ├── state.py                        ← extends Session with incident-specific fields
│   ├── mcp_server.py                   ← create/update/lookup_similar — incident-flavored
│   ├── skills/                         ← intake, triage, deep_investigator, resolution YAMLs
│   ├── ui.py                           ← Streamlit (incident-flavored UI lives here, not framework)
│   └── README.md                       ← how to run this example
└── code-review/                        ← second example to prove genericness (Phase 8)

dist/
├── framework.py                        ← bundled runtime (was app.py)
└── apps/
    └── incident-management.py          ← bundled app (lib + main entrypoint)
```

The dist split is a key signal: bundling **two artifacts** (framework lib + app entrypoint) cements the separation. Apps depend on framework; framework doesn't depend on apps.

---

## 3. Phase plan

Nine phases. Sequencing reflects dependency order — earlier phases unblock later ones. Per-phase implementation plans (with TDD task breakdown) get written when we start each phase.

---

### Phase 1 — Domain extraction + rename + repository split

**Goal:** Lift every incident-flavored type, field, config key, MCP tool, and skill out of the framework. Establish `examples/incident-management/` as the flagship app. Rename `Incident → Session` framework-side. Split `IncidentRepository` into `SessionStore` (active) + `HistoryStore` (closed).

**Why first:** every later phase touches state shape, repository, or naming. Doing genericification first means each subsequent phase is built on the generic abstraction — no rework.

**Deliverables:**
- Framework `Session` model: only generic fields (id, status, timestamps, agents_run, tool_calls, findings, pending_intervention, user_inputs, token_usage).
- `examples/incident-management/state.py` defines `IncidentState(Session)` with severity, environment, reporter, summary, etc.
- `examples/incident-management/mcp_server.py` is the new `incident_management` MCP server with the three existing tools, now backed by the framework's history-store primitive.
- `examples/incident-management/config.yaml` carries severity_aliases, escalation_teams, environments.
- `examples/incident-management/skills/` contains intake/triage/deep_investigator/resolution YAMLs.
- Framework `config.py` strips all incident-specific keys. Framework runs without an example app and just exposes session lifecycle + tools (when wired up).
- Existing UI keeps working pointed at the incident-management example app.
- All 191 tests still pass (most need import path updates; behavior unchanged).

**Risks:**
- Naming churn touches almost every file. High diff volume, low semantic risk.
- The `dist/app.py` bundler needs to know about the new directory layout — bundler script update required.
- UI is currently coupled to `Orchestrator.list_recent_incidents()` etc. — those become `list_recent_sessions()`; UI fetches stay incident-flavored via the example app's repository view.

**Dependencies:** none (Phase 1 is the load-bearing first move).

**Estimated effort:** 1 week.

---

### Phase 2 — Extensible state schema + LangGraph PostgresSaver

**Goal:** Move from a fixed `GraphState` shape to one where each app declares its own state TypedDict. Replace the hand-rolled pause/resume with the official LangGraph checkpointer (`PostgresSaver` for prod, `SqliteSaver` for dev).

**Deliverables:**
- Framework's `graph.py` accepts an app-supplied state TypedDict; merges it with the framework's required base fields.
- `LANGGRAPH_CHECKPOINTER` config key chooses between `sqlite` and `postgres`.
- The bespoke `pending_intervention` + `resume_graph` get replaced by `interrupt()` / `Command(resume=...)` patterns that LangGraph natively supports.
- `examples/incident-management/state.py` extends Session with the example app's fields; the LangGraph thread_id == session_id.
- Tests for state-schema merging, checkpointer round-trip on both backends.

**Risks:**
- LangGraph's `interrupt()` API needs careful review — the contract is different from our hand-rolled approach. Some flow nuances (e.g., re-running the gate node post-resume) may need re-implementation.
- PostgresSaver requires Postgres access in prod. Air-gapped deploys need the package vendored and the schema pre-migrated.

**Dependencies:** Phase 1 (rename so checkpoint's thread_id maps to `session_id`, not `incident_id`).

**Estimated effort:** 1 week.

---

### Phase 3 — Multi-session capability

**Goal:** Drop the singleton-investigation assumption. Multiple sessions live concurrently, each with its own LangGraph thread, each addressable by id.

**Deliverables:**
- `Orchestrator.start_session(trigger, payload) → session_id` replaces `start_investigation`.
- `Orchestrator.resume_session(session_id, decision)` replaces `resume_investigation`.
- `list_active_sessions()` API (was: hardcoded "the current investigation").
- UI shows multiple sessions; user can pick which to view.
- Internal: `Orchestrator` is no longer per-session; it's a long-lived service.

**Risks:**
- UI rework. Today's "current investigation" mental model is baked in.
- Race conditions on shared state (e.g., active-session registry) need basic locking.
- Migration: existing API endpoints stay (singleton-shim) for back-compat during transition; new API is multi-session-native.

**Dependencies:** Phase 2 (checkpointer per thread_id is the multi-session enabler).

**Estimated effort:** 1 week.

---

### Phase 4 — Risk-rated tool gateway + generic HITL

**Goal:** Every tool call flows through a framework-owned middleware. Per-tool risk declared in config; per-env policy overrides. HITL approval is a generic state machine, not a per-tool bespoke flow.

**Deliverables:**
- `framework/tools/gateway.py` wraps every tool invocation with a policy lookup.
- Config block:
  ```yaml
  gateway:
    policy:
      apply_fix:
        risk:
          default: high
          per-env:
            staging: low
        action:
          high: require_approval
          medium: notify_on_execute
          low: auto
  ```
- Default policy: `auto` for unconfigured tools (preserves current behavior).
- HITL state machine: `pending → approved | rejected | timeout`. Approval API endpoint generic across tools.
- Notification dispatch: hook-based (apps wire via tools — Slack/PagerDuty as MCP).
- Today's `apply_fix` confidence-HITL becomes a policy entry, not bespoke code.

**Risks:**
- The interaction between gateway HITL and LangGraph's `interrupt()` (Phase 2) needs careful design — only one suspension mechanism should own state at a time.
- Existing UI must surface "pending approval" for any tool, not just apply_fix. Modest UI work.

**Dependencies:** Phase 2 (HITL state machine sits inside the LangGraph state).

**Estimated effort:** 1 week.

---

### Phase 5 — Trigger registry (api + cron + plugin interface)

**Goal:** Triggers fire sessions. Today only API. After Phase 5: api + scheduled (cron) + plugin interface for event-stream sources.

**Deliverables:**
- `framework/triggers/base.py` Trigger ABC.
- Built-ins: `api` (FastAPI mount, current behavior), `schedule` (cron loop in a background task).
- Plugin interface via `pyproject.toml` entry-points group `framework.triggers`.
- Trigger config:
  ```yaml
  triggers:
    - kind: api
      mount: /sessions
    - kind: schedule
      cron: "*/15 * * * *"
      payload_template: { "trigger": "health_check" }
      target_agent: monitor
  ```
- Each trigger fires `start_session(trigger_kind, payload)` through the orchestrator.
- Trigger lifecycle: started on framework boot, stopped on shutdown.

**Risks:**
- Daemon management (cron loop is a long-running task) introduces a new failure mode — needs supervised retry and graceful shutdown.
- Concurrent triggers can race on session creation; dedup pipeline (Phase 7) handles the user-visible behavior, but the registry must be atomic.

**Dependencies:** Phase 3 (multi-session — triggers can fire many sessions in parallel).

**Estimated effort:** 4 days.

---

### Phase 6 — Agent kinds (responsive + supervisor-router + monitor-loop)

**Goal:** Generalize beyond LLM-driven request-handlers. Three kinds with different lifecycle wiring.

**Deliverables:**
- `framework/agents/base.py` AgentKind ABC.
- `responsive.py` — current LLM-driven node behavior. Default kind in YAML.
- `supervisor_router.py` — no-LLM dispatch. Receives state, returns next-route as a Python function or YAML rule table.
- `monitor_loop.py` — background daemon. Polls a configured source (via tool) at an interval, applies a rule (statistical or threshold), emits events that fire triggers.
- YAML extension:
  ```yaml
  - name: supervisor
    kind: supervisor-router
    rules: [...]
  - name: monitor
    kind: monitor-loop
    interval_seconds: 30
    poll_tool: get_hot_buffer
    detect_rule: ...
  ```
- Lifecycle: responsive = node in graph; router = state → route function; loop = supervised daemon.

**Risks:**
- Monitor loop integrates with Phase 5's scheduled trigger — design tension: should monitor be its own kind, or a special schedule trigger that calls a no-LLM agent? Worth resolving in design.
- "No LLM in hot path" is a discipline; the framework's contract should make it impossible to accidentally invoke an LLM from a monitor loop.

**Dependencies:** Phase 5 (loop agents + scheduled triggers are tightly coupled).

**Estimated effort:** 1 week.

---

### Phase 7 — Dedup pipeline + investigation manager

**Goal:** When a trigger fires, decide *new session* vs *merge into existing*. Pluggable dedup steps; subscriber model.

**Deliverables:**
- `framework/dedup/pipeline.py` — ordered list of `DedupStep`s.
- Built-in steps:
  - `embedding-similarity` — vs active-session triggers, threshold-gated
  - `llm-confirm` — cheap LLM "are these the same issue?"
  - `structured-key` — exact match on configurable key fields
- Active-session registry (already built in Phase 3) is the dedup target.
- Config:
  ```yaml
  dedup:
    enabled: true
    pipeline:
      - kind: embedding-similarity
        threshold: 0.82
      - kind: llm-confirm
        model: cheap
  subscribers: track  # or "ignore"
  ```
- Subscriber list per session; close-time fanout via tool.
- Off-by-default. Apps that want it set `enabled: true`.

**Risks:**
- LLM confirmation costs money — must be gated behind embedding stage (already the design).
- Merging two triggers' payloads into one session changes payload semantics; needs an explicit `on_merge(existing, incoming)` hook.

**Dependencies:** Phase 3 (active-session registry), Phase 5 (triggers feed the pipeline).

**Estimated effort:** 4 days.

---

### Phase 8 — Second example application (proof of genericness)

**Goal:** Stand up a second example app — code-review, customer-support, or compliance audit. Anything genuinely unrelated to incident management. Goal isn't a polished product; it's *forcing* the framework to demonstrate it doesn't leak incident-flavored assumptions.

**Deliverables:**
- `examples/<second-app>/` directory.
- App-extended state schema with no overlap to `IncidentState`.
- Skills YAML for that domain's agents.
- App-specific MCP server with that domain's tools.
- App config.
- README with run instructions.
- Tests demonstrate the framework runs both example apps from one codebase.

**Risks:**
- Second app reveals every assumption we hardcoded. Each leak found = framework fix needed.
- Estimating effort is hard; depends on what's revealed.

**Dependencies:** Phases 1, 2, 3 minimum. Full validation needs Phase 4–7 done.

**Estimated effort:** 1–2 weeks (depending on how leaky the abstractions turn out to be).

---

### Phase 9 — Incident-management app build-out (the ASR.md flagship)

**Goal:** Once the framework is generic enough, build the full ASR.md vision *as an example app*. This is the "selling point" — what we demo, what runs in production. Most of this is app code, not framework code.

**Sub-phases (each is its own implementation plan):**

| Sub-phase | What it ships |
|---|---|
| 9a | Enrich `IncidentState` schema per ASR.md (symptoms, root_cause, resolution_steps, hypothesis array). Update embedding source to `symptoms + root_cause + resolution`. |
| 9b | L2 Knowledge Graph MCP server. Two implementations: file-system (markdown topology files) and Neo4j-backed. |
| 9c | L5 Release Context MCP server. Postgres + vector embedding of release notes. Tool: `recent_releases(components, hours)`. |
| 9d | L7 Playbook Store MCP server. Git-backed markdown index + risk-rated step model. Integrates with framework tool gateway. |
| 9e | L4 Domain Knowledge MCP server. Doc ingestion pipeline (chunking, L2-tagged embedding), filtered semantic search tool. |
| 9f | L1 Hot Buffer MCP server. Redis-backed pre-aggregated LGTM snapshots. |
| 9g | Monitor agent (`kind: monitor-loop`) wired to L1 + L2; emits anomaly-triggered sessions. |
| 9h | Supervisor agent (`kind: supervisor-router`) — pure no-LLM dispatch. |
| 9i | Triage agent enhancement: hypothesis schema, evidence collection from L1/L3/L4/L5. |
| 9j | Deep Investigator agent: low-confidence triage hands off to deeper LLM-driven analysis. |
| 9k | Resolution agent: integrates with L7 + framework risk-gate. |
| 9l | Two-stage dedup configured via Phase 7's pipeline (embedding + LLM-confirm). |
| 9m | Production-grade UI updates: hypothesis graph, subscriber list, multi-session view, evidence trail. |

**Risks:**
- Some sub-phases (especially 9b/Neo4j and 9f/Redis) introduce new infra that needs vendored install paths for air-gapped deploy.
- Each sub-phase is essentially "build a new MCP server" — well-scoped but adds up.

**Dependencies:** Framework (Phases 1–7) must be solid before building on top.

**Estimated effort:** 2–3 months for full ASR.md vision. Realistically incremental — ship the first agent slices that prove the architecture; layer on as the rest gets prioritized.

---

## 4. Phase dependency graph

```
P1 (rename + extract) ─┬─ P2 (schemas + checkpointer) ─┬─ P3 (multi-session) ─┬─ P4 (gateway)
                       │                                │                      │
                       │                                │                      ├─ P5 (triggers) ─┬─ P6 (agent kinds)
                       │                                │                      │                  │
                       │                                │                      │                  └─ P7 (dedup)
                       │                                │                      │
                       │                                └─ P8 (second example app)
                       │
                       └─ P9 (ASR app build-out — depends on P1–P7 incrementally per sub-phase)
```

P1 is the gate. P2 unblocks P3, P3 unblocks the rest. P9 sub-phases interleave with framework phases (e.g., 9b can land after P1; 9g needs P5 and P6).

## 5. Definition of done (framework as a whole)

- Framework directory contains zero incident-management terminology in code or config.
- `examples/incident-management/` contains the entire flagship app and is functionally equivalent to today's UI/CLI.
- `examples/<second-app>/` runs from the same codebase, in <100 LOC of app code beyond skills + MCP server.
- Both examples can run side-by-side in the same orchestrator process (different sessions, different state schemas).
- Test coverage: framework tests don't reference incident-management; example tests live alongside the example.
- Bundle: `dist/framework.py` (lib) + `dist/apps/<app>.py` (per-app entry).
- README at root repositions the project: "agent orchestration framework, with incident management as flagship example".

## 6. Risks across the whole roadmap

| Risk | Mitigation |
|---|---|
| Phase 1 churn breaks UI / dist bundle | Land behind a flag if needed; ship the example-incident-management app the same week so end-to-end stays green |
| Phase 2's LangGraph `interrupt()` semantics differ from hand-rolled flow | Spike + adversarial review before committing to the migration |
| Phase 3 multi-session race conditions | Single-writer per session via thread_id locking; don't share mutable state across sessions |
| Phase 4 gateway adds latency to every tool call | Measure overhead; default policy = `auto` is a single dict lookup, should be sub-ms |
| Phase 9 sub-phases pull infra deps (Neo4j, Redis) into the air-gapped story | Each sub-phase ships its own vendored install path; mark as "optional infra"; framework runs without them |
| Genericness might prove leaky (Phase 8 surfaces leaks) | Treat Phase 8 leaks as Phase 1 follow-ups; don't ship as "framework" until both examples run cleanly |
| Roadmap takes longer than planned | Each phase is independently shippable behind feature flags; can pause anywhere with framework still functional |

## 7. What this roadmap is NOT

- Not a commitment to build all 9 phases sequentially before any user value.
- Not a guarantee that the order is final — Phase 4 could move ahead of Phase 3 if HITL is more urgent than concurrency, etc.
- Not a complete spec — each phase still needs its own implementation plan with TDD task breakdown when we execute.
- Not a green light to start coding Phase 9 today; start with P1, validate, iterate.

## 8. Immediate next step

Spec out Phase 1 (domain extraction + rename + repository split) as a detailed implementation plan with tasks A–N (TDD breakdown). Run that plan via subagent-driven-development. Once P1 lands, the rest of the roadmap unfolds.

---

*End of roadmap.*
