# Phase 9 — ASR Build-Out Implementation Plan

> **Phase 9 is the actual product.** It evolves `examples/incident_management/` into the ASR flagship described in `ASR.md`. Most of this is *app code*, layered on top of the framework that Phases 1–7 deliver. Phase 9 itself is divided into 13 sub-phases (9a–9m), each effectively its own implementation plan.

**Goal:** Deliver ASR per `ASR.md` — a multi-agent platform-monitoring solution with a 7-layer memory architecture, supervisor-routed agent topology (Monitor / Intake / Triage / Deep Investigator / Resolution), risk-rated HITL on resolution actions, and two-stage dedup of user reports — running on top of the generic framework.

**Sequencing strategy:** ASR is too large to ship in a single push (the roadmap budgets 2–3 months for the full vision). This plan picks an **MVP slice** that proves the architecture end-to-end with the minimum new infra, then tracks the remaining sub-phases as progressive enhancements. Locked product decisions:

- **ASR EXTENDS / EVOLVES `examples/incident_management/`** — not a side-by-side rewrite. Every sub-phase touches files inside `examples/incident_management/` (or adds peer MCP servers under `examples/incident_management/mcp_servers/`). The framework (`src/runtime/`) gains *only* the abstractions ASR's agents need; app-shaped logic stays in the app.
- **Optional infra deferred for MVP** — Neo4j (L2), Redis (L1, L6), Postgres+pgvector (L3, L5) are *out of MVP*. MVP uses filesystem markdown / SQLite / FAISS substitutes that the roadmap explicitly authorises. Production deployments swap implementations later via config; the MCP-server boundary makes the swap safe.
- **Prod-HITL override is non-negotiable.** Per Phase 4's gateway policy and the user-mandated production rule: any tool that resolves an incident (mutates infrastructure, runs a runbook step) is forced to `require_approval` when `environment in {prod, prod-like}`, regardless of declared risk tier. This overrides the per-tool `risk` config for the prod environment.

**Architecture summary (from ASR.md, §3 + §7 + §10):**

```
Triggers          ─▶ Supervisor (no-LLM router + Investigation Manager + dedup)
(L1 anomaly,           │
 user input,           ├─▶ Monitor Agent       (always-on, rule/statistical anomaly detection on L1)
 alert manager)        ├─▶ Intake Agent        (dedup new triggers, scope investigation, write L6)
                       ├─▶ Triage Agent        (evidence collection L1/L3/L4/L5, hypothesis loop in L6)
                       ├─▶ Deep Investigator   (low-confidence handoff for deeper LLM analysis)
                       └─▶ Resolution Agent    (matches L7 playbook, runs risk-gated steps, HITL on high)

Memory: L1 Hot Buffer · L2 Knowledge Graph · L3 Incident Memory · L4 Domain Knowledge ·
        L5 Release Context · L6 Session/Investigation · L7 Playbook Store
```

---

## 0. Source-of-Truth Inventory

| Source | Status | Used for |
|---|---|---|
| `/home/dev/projects/asr/ASR.md` (957 lines) | **Read end-to-end** | Memory layers, agent topology, lifecycle, HITL policy, dedup, hypothesis schema, approval state machine |
| `docs/superpowers/plans/2026-05-02-framework-genericification-and-asr-roadmap.md` | Read | Phase 9 sub-phase table (9a–9m), dependency graph, surface area |
| `docs/superpowers/plans/2026-05-02-phase1-implementation-plan.md` | Read (headers) | Phase 1 contract: `Session` base, `IncidentState(Session)` extension |
| `docs/superpowers/plans/2026-05-02-phase2-extensible-state-and-checkpointer.md` | Read (headers) | `Generic[StateT]` + LangGraph `interrupt()` HITL — ASR consumes both |
| `examples/incident_management/{state.py,config.py,config.yaml,mcp_server.py,ui.py,skills/*}` | Read | Starting point. ASR builds on this; sub-phases reshape, not replace |
| Phase 3–8 plans | **Do not exist on disk** | Roadmap descriptions are the contract. ASR sub-phases assume what the roadmap says they ship |

> Where Phase 3–8 plans are not yet written, the corresponding ASR sub-phases reference the roadmap entry. If the future plan changes contract, the affected sub-phase replans on its dependency edge — not silently here.

---

## 1. ASR.md Summary (3 paragraphs)

**What ASR is.** ASR (Agent Service for Reliability — internal name: "Multi-Agent Platform Monitoring Solution") is an incident-management agent system for a 150+-service Java/React trading platform monitored by the LGTM stack (Loki/Grafana/Tempo/Mimir). It detects anomalies (Monitor agent on a 15–60s loop), accepts user reports and Alert Manager pages, dedups them against active investigations, runs structured triage with a hypothesis-driven evidence-collection loop, escalates low-confidence cases to a Deep Investigator, and proposes/executes resolution steps from a versioned playbook library — with risk-rated human approval gating destructive actions. The product accuracy bar is "highest priority"; performance and safety follow.

**Seven-layer memory architecture (§3 of ASR.md).** L1 Hot Operational Buffer (Redis-backed pre-aggregated LGTM snapshots; rolling 15-min/1-hr/24-hr windows; sub-second reads for the Monitor); L2 Component Knowledge Graph (Neo4j; components, dependencies, business functions, ownership; resolves "margin calc slow" → specific Java services); L3 Incident Memory (Postgres + pgvector; structured records embedded from `symptoms + root_cause + resolution`; symptom→component→recency retrieval during triage); L4 Domain Knowledge Store (vector DB; chunked specs/runbooks tagged with L2 components for filtered semantic search); L5 Release Context (Postgres + vector; correlates incidents with recent deploys, one of the highest-signal heuristics); L6 Session/Investigation Memory (working scratchpad: hypotheses, evidence, reasoning chain — ephemeral, archived to L3 on close); L7 Playbook Store (git-backed markdown + Postgres index; risk-rated step model with environment-aware permissions).

**Agent topology and lifecycle (§7 + §3 lifecycle view).** A no-LLM Supervisor routes triggers to specialists; an Investigation Manager inside the Supervisor handles concurrent-investigation dedup (two-stage: embedding similarity ≥0.82 → cheap LLM confirmation; matched users join as subscribers, not duplicate investigations). The lifecycle is Phase 1 Intake & Dedup → Phase 2 Triage (gather L1/L3/L4/L5 evidence, write hypotheses to L6, loop until top hypothesis confidence ≥0.8 or escalate to Deep Investigator) → Phase 3 Resolution (match L7 playbook, execute steps where risk policy allows; high-risk steps await human approval via a state machine with `pending → approved | rejected | timeout`) → Phase 4 Close & Notify (archive L6 → L3, notify subscribers via Slack/PagerDuty/email). LangGraph is the recommended orchestrator with `PostgresSaver` checkpointing — Phase 2 of the framework already commits to this.

---

## 2. Sub-Phase Breakdown (9a–9m)

> Format: each sub-phase gets goal, what-it-ships, framework dependencies, and an effort sketch. The MVP slice (§3) gets full TDD detail in §4; the rest get high-level task lists.

### 9a — IncidentState schema enrichment
- **Goal:** Bring `IncidentState` in line with ASR.md §3 L6 schema and §7 hypothesis schema. Update embedding source from current text to `symptoms + root_cause + resolution` (ASR.md §3 L3 retrieval rule).
- **Ships:** Extended `examples/incident_management/state.py` with `symptoms: list[str]`, `root_cause: str | None`, `resolution_steps: list[str]`, `hypotheses: list[Hypothesis]`, `mapped_components: list[str]`, `mapped_business_function: str | None`, `subscribers: list[str]`, `recent_releases: list[ReleaseRef]`, `similar_incidents: list[IncidentRef]`, `current_phase: Literal["intake","triage","deep","resolution","closed"]`, `triage_iterations: int`. New `Hypothesis` pydantic model (id, description, confidence, category, affected_component, causal_chain, evidence, status). MCP tool `update_investigation` writes hypotheses into L6.
- **Depends on:** Framework P1 (Session/IncidentState split) + P2 (`Generic[StateT]` + checkpointer).
- **Estimated tasks:** 6–8 TDD pairs.
- **MVP?** **Yes — gate sub-phase.** Every other 9x reads/writes this shape.

### 9b — L2 Knowledge Graph MCP server
- **Goal:** Map "margin calc slow" → list of Java services. Two implementations behind one MCP interface.
- **Ships:** `examples/incident_management/mcp_servers/knowledge_graph/` with `fs_backend.py` (markdown topology files: one `.md` per component with YAML frontmatter declaring `depends_on`, `business_function`, `owner`) and a stub `neo4j_backend.py` (interface only; real Cypher post-MVP). Tools: `lookup_components_for_business_function(name) -> list[component]`, `get_dependencies(component, depth) -> list[component]`, `find_components_for_symptom(text) -> list[component]` (substring + tag match in MVP, semantic post-MVP).
- **Depends on:** Framework P4 (gateway — read tools are `risk: low`, auto), P1 (MCP loader path).
- **Estimated tasks:** 5–7 TDD pairs (markdown loader, indexer, three tool handlers, ingestion of seed topology fixture).
- **MVP?** **Yes** — without L2 the Triage agent can't scope. FS backend only; Neo4j deferred.

### 9c — L5 Release Context MCP server
- **Goal:** Recent-change correlation — "did anything ship for `margin-calc-service` in the last 48h?"
- **Ships:** `examples/incident_management/mcp_servers/release_context/` with SQLite-backed store (post-MVP swap: Postgres + pgvector). Schema per ASR.md §3 L5 (`release_id`, `deployed_at`, `components_changed`, `change_summary`, `jira_tickets`, `risk_tags`, `embedding`). Tool: `recent_releases(components: list[str], hours: int) -> list[Release]`. Ingestion CLI: `python -m examples.incident_management.mcp_servers.release_context.ingest <yaml-file>`. Embeddings via the framework's `storage/embeddings.py` factory.
- **Depends on:** Framework P4 + framework `storage/embeddings.py` (delivered in framework P2 follow-on).
- **Estimated tasks:** 6–8 TDD pairs.
- **MVP?** **Yes** — release correlation is the single highest-signal triage heuristic per ASR.md.

### 9d — L7 Playbook Store MCP server
- **Goal:** Procedural memory: risk-rated runbook steps the Resolution agent executes through the gateway.
- **Ships:** `examples/incident_management/mcp_servers/playbooks/` with git-backed markdown loader (one `.md` per playbook; YAML frontmatter declares `triggers`, `applicable_components`, `steps[]` where each step has `name`, `tool`, `args`, `risk: low|medium|high`, `env_overrides: {prod: high}`). SQLite index for vector search over playbook descriptions. Tools: `match_playbook(symptoms, components, env) -> Playbook`, `next_step(playbook_id, completed_steps) -> Step`. Each step's risk feeds the framework P4 gateway directly.
- **Depends on:** Framework P4 (gateway is the consumer of `risk` tags).
- **Estimated tasks:** 8–10 TDD pairs.
- **MVP?** **Yes** — without L7 there is no "Resolution" agent path; ASR is just triage.

### 9e — L4 Domain Knowledge MCP server
- **Goal:** Searchable specs/runbooks/architecture docs for the Triage agent's "what does this component actually do?" calls.
- **Ships:** Doc ingestion pipeline (chunk by section heading; tag chunks with L2 component IDs; embed; store in FAISS for MVP, vector DB later). Filtered semantic search tool: `search_domain_knowledge(query, component_filter=None, business_function_filter=None) -> list[Chunk]`.
- **Depends on:** 9b (for L2 tagging during ingestion).
- **Estimated tasks:** 7–9 TDD pairs.
- **MVP?** **No** — Triage can run without L4 in MVP using L1+L3+L5 only. Defer to v2.

### 9f — L1 Hot Buffer MCP server
- **Goal:** Pre-aggregated LGTM snapshots so Monitor doesn't hammer Mimir.
- **Ships:** SQLite ring-buffer keyed `component:{name}:health|alerts|log_errors` (post-MVP swap: Redis). Ingest CLI from a JSON fixture for tests; in production wire to LGTM webhooks. Tool: `read_health_snapshot(components, window) -> dict`.
- **Depends on:** None framework-wise (MCP P4 read-only path).
- **Estimated tasks:** 5–6 TDD pairs.
- **MVP?** **No** — defer with the Monitor agent (9g). Without 9g, no consumer.

### 9g — Monitor agent (`kind: monitor-loop`)
- **Goal:** Always-on background poller that emits anomaly-triggered sessions.
- **Ships:** `examples/incident_management/agents/monitor.py` registering a `monitor-loop` agent kind; rule-set + 3-sigma statistical detection over L1 snapshots; per-anomaly 5-min dedup window; on detection, calls Supervisor's intake trigger.
- **Depends on:** Framework P5 (trigger registry, cron) + P6 (`monitor-loop` agent kind) + 9f (L1 source).
- **Estimated tasks:** 6–8 TDD pairs.
- **MVP?** **No** — MVP relies on user-initiated investigations (chat trigger). Background detection is post-MVP.

### 9h — Supervisor agent (`kind: supervisor-router`)
- **Goal:** No-LLM dispatch + Investigation Manager.
- **Ships:** `examples/incident_management/agents/supervisor.py`. Pure Python routing: trigger → dedup pipeline → either join existing investigation as subscriber or create new → assign to Triage / Deep / Resolution by `current_phase` field. No LLM in the hot path (per ASR.md §7).
- **Depends on:** Framework P6 (`supervisor-router` agent kind) + P7 (dedup pipeline).
- **Estimated tasks:** 7–9 TDD pairs.
- **MVP?** **Yes** — without a Supervisor the LangGraph topology in ASR.md §8 doesn't exist. We can stub the dedup until P7 lands (single-investigation MVP) but the router itself is mandatory.

### 9i — Triage agent enhancement
- **Goal:** Hypothesis-loop triage. Today's Triage skill upgrades to the loop in ASR.md §7 ("gather evidence → form hypotheses → if top confidence ≥0.8 confirm, else expand or escalate").
- **Ships:** Reworked `examples/incident_management/skills/triage/system.md`; new tool calls `query_logs` / `query_metrics` / `query_traces` (LogQL/PromQL/TraceQL stubs in MVP, real LGTM in prod); `update_investigation` writes hypotheses to L6 each iteration. Triage agent module `agents/triage.py` orchestrates the loop, capped at `max_iterations` per config.
- **Depends on:** 9a (state shape), 9b (component scoping), 9c (release correlation), framework P2 (state graph).
- **Estimated tasks:** 8–10 TDD pairs.
- **MVP?** **Yes** — Triage is the agent that proves L1+L3+L5 evidence collection actually works.

### 9j — Deep Investigator agent
- **Goal:** Low-confidence Triage hands off to a heavier LLM-driven analyst.
- **Ships:** `agents/deep_investigator.py` + `skills/deep_investigator/` overhaul. Receives the Triage state, runs broader queries (multi-component traces, log pattern mining), returns enriched evidence to Triage which re-iterates.
- **Depends on:** 9i.
- **Estimated tasks:** 5–7 TDD pairs.
- **MVP?** **No** — MVP assumes Triage either confirms or returns "needs human"; deep dive is v2.

### 9k — Resolution agent + risk-gate integration
- **Goal:** End-to-end resolution flow with the §7 approval state machine.
- **Ships:** `agents/resolution.py`. Consumes a matched playbook from L7, walks the step list, every step goes through framework P4 gateway. **Prod-HITL override** is enforced: if `state.environment ∈ {prod, prod-like}` and the tool is in the resolution-trigger set, the gateway forces `require_approval` regardless of the step's declared risk. State machine: `Start → PlaybookMatched → ExecutingStep → CheckRisk → {AutoExecute | AutoExecuteNotify | AwaitApproval} → StepComplete → (MoreSteps | VerifyFix) → (ArchiveIncident | RetriageSupervisor)`.
- **Depends on:** 9d (L7), framework P4 (gateway), framework P2 (LangGraph `interrupt()` for HITL pause).
- **Estimated tasks:** 9–11 TDD pairs.
- **MVP?** **Yes** — without Resolution we have a nice triage demo, not "ASR works end-to-end".

### 9l — Two-stage dedup wiring
- **Goal:** Wire the Supervisor's Investigation Manager into the framework's dedup pipeline (Phase 7) using the §10 algorithm.
- **Ships:** Pipeline config: `step 1 = embedding (cosine ≥ 0.82)`, `step 2 = llm-confirm (cheap model, single prompt)`. Subscriber-add path on confirmed match. Three test cases per ASR.md §9 ("margin calc slow" / "margin calc broken" → match; "K8s pod restart loop" → new).
- **Depends on:** Framework P7 (dedup pipeline) + 9h (Supervisor consumes the verdict).
- **Estimated tasks:** 4–5 TDD pairs.
- **MVP?** **No** — MVP can ship with single-investigation-at-a-time. Dedup adds when concurrency does.

### 9m — Production-grade UI
- **Goal:** Streamlit UI shows hypothesis graph, subscriber list, multi-session view, evidence trail, approval inbox.
- **Ships:** Reworked `examples/incident_management/ui.py`. Per-incident accordion (per global UI memory) gains: hypothesis tree with confidence bars; subscriber chip list; evidence timeline grouped by L1/L3/L4/L5 source; pending-approval inbox with approve/reject buttons that POST to the gateway HITL endpoint.
- **Depends on:** All preceding sub-phases for data; honours `prefers-reduced-motion` and `prefers-color-scheme`.
- **Estimated tasks:** 8–12 TDD pairs (mostly view-state tests; UI screenshots verified manually).
- **MVP?** **Partial** — MVP needs the approval inbox and a minimal hypothesis view. The full multi-session view is post-MVP.

---

## 3. Recommended MVP Slice

The MVP question is "what's the smallest set of sub-phases that makes ASR demonstrably work end-to-end against ASR.md's lifecycle (Intake → Triage → Resolution → Close)?"

**Recommended MVP: 9a + 9b + 9c + 9d + 9h + 9i + 9k + UI sliver of 9m.**

| Sub-phase | Why it's in MVP |
|---|---|
| **9a** | Schema gate. Every other sub-phase reads this shape. Cheapest of the bunch. |
| **9b** (FS backend only) | Triage cannot scope a symptom to components without L2. Markdown FS backend keeps infra cost zero. |
| **9c** (SQLite backend) | Release correlation is the single highest-signal triage heuristic per ASR.md. Without it the demo is unconvincing. |
| **9d** | Without L7 there is no Resolution path; the system is just a triage research tool. |
| **9h** | The LangGraph topology *is* a Supervisor-routed graph. Stub dedup to single-active-investigation; that's enough for MVP. |
| **9i** | Triage is the agent that exercises L1+L3+L5 evidence collection — the load-bearing demo of the memory architecture. |
| **9k** | Resolution closes the loop. Prod-HITL override lives here. Without 9k, ASR doesn't *resolve* anything. |
| **9m sliver** | Approval inbox + minimal hypothesis view. UI must surface what 9k produces. |

**Out of MVP (post-MVP, in priority order):** 9f+9g (Monitor agent path — adds anomaly-triggered investigations); 9j (Deep Investigator — Triage handles MVP); 9e (L4 — Triage works on L1+L3+L5 alone for MVP); 9l (multi-session dedup — single-active is fine); 9m full (multi-session view, evidence timeline grouping); 9b Neo4j backend; production swaps for L1/L3/L5/L6/L7 stores.

> **Sanity check vs roadmap "2–3 months for full vision":** MVP slice (8 sub-phases, ~50–65 TDD pairs) is roughly 4–6 weeks. Post-MVP (5 sub-phases + 4 store-swaps + Neo4j backend) is the remaining 6–8 weeks. Total tracks the roadmap estimate.

**Done = "ASR works end-to-end" criteria for MVP** are in §7 below.

---

## 4. Detailed Plan for MVP Slice

> TDD discipline (per `~/.claude/rules/testing.md`): every task is a **failing test → minimum implementation → green → refactor** pair. Tasks are labelled `P9{X}-{N}` where X is the sub-phase letter and N is sequence within. Each task has: rationale · failing test · implementation hint · acceptance.

### 9a — IncidentState schema enrichment

**File targets:** `examples/incident_management/state.py`, `examples/incident_management/mcp_server.py`, `tests/incident_management/test_state.py`, `tests/incident_management/test_mcp_update_investigation.py`.

#### P9a-A — `Hypothesis` pydantic model
- Failing test: `test_hypothesis_validates_required_fields()` asserts `id`, `description`, `confidence∈[0,1]`, `status∈{investigating,confirmed,refuted}` are required; extra fields rejected.
- Implementation: `class Hypothesis(BaseModel)` with `id: str`, `description: str`, `confidence: confloat(ge=0, le=1)`, `category: str | None = None`, `affected_component: str | None = None`, `causal_chain: list[str] = []`, `evidence: list[str] = []`, `status: Literal["investigating","confirmed","refuted"] = "investigating"`. `model_config = ConfigDict(extra="forbid")`.
- Acceptance: confidence=1.5 raises ValidationError; status="closed" raises ValidationError.

#### P9a-B — Extend `IncidentState`
- Failing test: `test_incident_state_has_asr_fields()` asserts new fields exist with correct defaults.
- Implementation: add `symptoms: list[str] = []`, `root_cause: str | None = None`, `resolution_steps: list[str] = []`, `hypotheses: list[Hypothesis] = []`, `mapped_components: list[str] = []`, `mapped_business_function: str | None = None`, `subscribers: list[str] = []`, `recent_releases: list[dict] = []`, `similar_incidents: list[dict] = []`, `current_phase: Literal["intake","triage","deep","resolution","closed"] = "intake"`, `triage_iterations: int = 0`.
- Acceptance: schema serialises to JSON deterministically; defaulted fields don't leak across instances (no shared mutable defaults).

#### P9a-C — Embedding source change
- Failing test: `test_embedding_text_includes_symptoms_root_cause_resolution()` constructs an `IncidentState` with all three populated and asserts `state.embedding_text()` is `f"{symptoms_joined}\n{root_cause}\n{resolution_joined}"` (matches ASR.md §3 L3).
- Implementation: add `embedding_text(self) -> str` method on `IncidentState`; replace any current `text_for_embedding` usage in `mcp_server.lookup_similar_incidents` to call this.
- Acceptance: existing `lookup_similar_incidents` test updated to seed `symptoms`/`root_cause` instead of legacy `summary` and still passes (after fixture update).

#### P9a-D — `update_investigation` MCP tool
- Failing test: `test_update_investigation_appends_hypothesis_and_increments_iterations()` calls the tool with a hypothesis dict, asserts the state in the session store has the hypothesis appended and `triage_iterations` incremented by 1.
- Implementation: new tool handler in `mcp_server.py`: `_tool_update_investigation(state_id, hypothesis: dict | None = None, evidence: dict | None = None, current_phase: str | None = None)` writes via the framework's `SessionStore`.
- Acceptance: hypothesis dict that fails `Hypothesis` validation returns a structured error and does not mutate state.

#### P9a-E — Migration of existing fixtures
- Failing test: `test_legacy_fixtures_migrate_cleanly()` — load the existing test fixture, run a one-shot migration helper that maps `summary → symptoms[0]`, asserts no field is lost.
- Implementation: small `migrations/0001_asr_state_shape.py` script + invocation from the test fixture loader.
- Acceptance: all 191 existing tests still pass.

#### P9a-F — `Hypothesis` rendering in UI
- Failing test: `test_ui_renders_hypothesis_with_confidence_badge()` (UI snapshot test using Streamlit's testing harness).
- Implementation: small render helper in `ui.py`; defer the full hypothesis tree to 9m.
- Acceptance: confidence ≥0.8 renders a "confirmed" badge; <0.5 renders "low".

---

### 9b — L2 Knowledge Graph MCP server (FS backend)

**File targets:** `examples/incident_management/mcp_servers/knowledge_graph/{__init__.py,fs_backend.py,server.py,fixtures/}`, `tests/incident_management/test_kg_fs_backend.py`.

#### P9b-A — Markdown topology loader
- Failing test: `test_load_component_from_markdown()` — given a fixture file with frontmatter `name: margin-calc-service`, `business_function: Margin Calculation`, `depends_on: [margin-db, calc-queue]`, `owner: trade-platform-team`, the loader returns a `Component` dataclass with those fields populated.
- Implementation: parse YAML frontmatter via `python-frontmatter` (already vendored or add to lockfile). Component is a dataclass; loader caches.
- Acceptance: malformed frontmatter raises `KGLoadError` with file path; empty file is skipped with a warning.

#### P9b-B — Component index
- Failing test: `test_kg_index_builds_lookup_tables()` builds an index over a 10-component fixture, asserts `by_business_function`, `by_owner`, and `dependency_graph` are all populated.
- Implementation: `KGIndex` dataclass with three dicts; built in O(n).
- Acceptance: cyclic dependency raises a structured warning but doesn't crash.

#### P9b-C — `lookup_components_for_business_function` tool
- Failing test: returns `["margin-calc-service", "pricing-engine"]` for input `"Margin Calculation"`.
- Implementation: simple lookup against the index.
- Acceptance: case-insensitive match; unknown function returns `[]`.

#### P9b-D — `get_dependencies` tool
- Failing test: `get_dependencies("margin-calc-service", depth=2)` returns the transitive set within depth 2.
- Implementation: BFS up to `depth`; cap at 1000 nodes for safety.
- Acceptance: depth=0 returns just the input.

#### P9b-E — `find_components_for_symptom` tool (MVP impl)
- Failing test: `find_components_for_symptom("margin calculation slow")` returns components whose business_function tokens overlap the symptom.
- Implementation: tokenise + exact substring match (semantic search post-MVP).
- Acceptance: unknown symptom returns `[]`, never raises.

#### P9b-F — MCP server wiring
- Failing test: spin up the MCP server in a fixture; call each tool through the framework's `mcp_loader`; assert round-trip.
- Implementation: `server.py` uses the same `_default_server` pattern as `examples/incident_management/mcp_server.py`.
- Acceptance: server starts, all three tools register, gateway sees them as `risk: low`.

#### P9b-G — Seed fixture
- Failing test: `test_seed_topology_loads()` — load the bundled `fixtures/asr_seed/*.md` (10 components covering the ASR.md "margin calculation" walkthrough), assert KG has the right edges.
- Implementation: hand-author 10 fixture files mirroring ASR.md §3 L2 graph model.
- Acceptance: matches every component named in the ASR.md §4 cross-layer walkthrough.

---

### 9c — L5 Release Context MCP server

**File targets:** `examples/incident_management/mcp_servers/release_context/{__init__.py,store.py,ingest.py,server.py}`, `tests/incident_management/test_release_context.py`.

#### P9c-A — SQLite schema
- Failing test: `test_release_table_schema()` — fresh DB, asserts table `releases` with columns matching ASR.md §3 L5 schema.
- Implementation: SQLAlchemy model `Release` with `release_id PK`, `deployed_at`, `components_changed JSON`, `change_summary`, `detailed_changes JSON`, `jira_tickets JSON`, `risk_tags JSON`, `embedding BLOB` (numpy bytes for MVP; pgvector swap later).

#### P9c-B — Ingest from YAML
- Failing test: `test_ingest_yaml_creates_release_with_embedding()` — supply a YAML fixture, assert the row exists and `embedding` is non-null.
- Implementation: `ingest.py` CLI; embedding via framework `storage/embeddings.py` factory.

#### P9c-C — `recent_releases(components, hours)` tool
- Failing test: with three releases (one matching component+window, one outside window, one wrong component), tool returns exactly the matching one.
- Implementation: SQL `WHERE components_changed @> ? AND deployed_at > now() - ?` (SQLite uses `json_each`).

#### P9c-D — Triage integration smoke test
- Failing test: `test_triage_calls_recent_releases_for_each_mapped_component()` — mock Triage agent, assert tool invoked with the right args during evidence gathering.

#### P9c-E — Risk-tag filter
- Failing test: `recent_releases(components, hours, risk_tag="orm-upgrade")` filters correctly.

#### P9c-F — MCP server wiring
- Same pattern as 9b-F. `risk: low`, auto.

---

### 9d — L7 Playbook Store MCP server

**File targets:** `examples/incident_management/mcp_servers/playbooks/{__init__.py,loader.py,store.py,server.py,fixtures/}`, `tests/incident_management/test_playbooks.py`.

#### P9d-A — Playbook schema
- Failing test: validates a playbook YAML frontmatter matches the schema (`triggers`, `applicable_components`, `steps[]` each with `name`, `tool`, `args`, `risk`, optional `env_overrides`).
- Implementation: `Playbook` and `Step` pydantic models.

#### P9d-B — Markdown loader
- Failing test: load `fixtures/restart_consumer.md`, assert all steps parse with their risk tags.

#### P9d-C — SQLite vector index
- Failing test: build the index over 5 playbooks, assert k-NN search by description embedding returns the expected top match.
- Implementation: FAISS in-memory; persist via `numpy.save` for MVP.

#### P9d-D — `match_playbook(symptoms, components, env)` tool
- Failing test: given a symptom set + component list, returns the highest-scoring playbook whose `applicable_components` intersects.
- Implementation: combined component-overlap score + embedding similarity.

#### P9d-E — `next_step(playbook_id, completed_steps)` tool
- Failing test: returns the next step in declared order; returns `None` when complete.

#### P9d-F — Step risk projection through gateway
- Failing test: a step with `risk: medium` and `env_overrides: {prod: high}` is reported as `high` when called with `env=prod` — matches Phase 4 gateway expectation.
- Implementation: `Step.effective_risk(env)` method.

#### P9d-G — **Prod-HITL override** test (load-bearing)
- Failing test: `test_prod_resolution_tool_always_requires_approval()` — *any* tool whose name appears in `config.resolution_trigger_tools` (declared in `examples/incident_management/config.yaml`) returns `require_approval` from the gateway when `env=prod`, **even if the step's effective risk is `low`**.
- Implementation: gateway lookup is augmented in `framework/tools/gateway.py` (Phase 4) — but the *config* and the *test* live with ASR. ASR ships a config-only patch on the framework gateway: `gateway.policy.prod_overrides.resolution_trigger_tools: [restart_service, scale_replicas, modify_schema, kill_session, rollback_release, ...]`. **If Phase 4's gateway does not support `prod_overrides`, this is a contract gap that must be raised against P4 before 9d-G lands.**
- Acceptance: in `env=staging` the same step is `auto`; in `env=prod` it is `require_approval`.

#### P9d-H — Seed playbook fixtures
- Failing test: 5 fixtures covering ASR.md §3 L7 examples (rollback, schema modify, kill session, scale replicas, query slow log).
- Acceptance: fixtures exercise all three risk tiers across both envs.

---

### 9h — Supervisor agent

**File targets:** `examples/incident_management/agents/supervisor.py`, `examples/incident_management/agents/__init__.py`, `tests/incident_management/test_supervisor.py`.

#### P9h-A — `kind: supervisor-router` registration
- Failing test: registry contains `supervisor-router` after import.
- Implementation: subclass framework's `AgentKind` ABC (delivered by P6); register via the framework's agent registry.
- **Dependency note:** if P6 doesn't exist yet, this sub-phase is gated. ASR sub-phases must not stub framework primitives.

#### P9h-B — Trigger → phase routing
- Failing test: trigger of type `user_input` with no active investigation → routes to `intake`. Trigger with active investigation in phase `triage` → re-routes to `triage`.
- Implementation: pure dict-based dispatch table.

#### P9h-C — Single-active-investigation guard (MVP dedup stub)
- Failing test: when an investigation is open, a new `user_input` trigger calls `add_subscriber` instead of creating a new investigation. (Full two-stage dedup is 9l.)
- Implementation: simple lookup against `SessionStore.active()`.

#### P9h-D — Resolution → close transition
- Failing test: when Resolution returns `state.current_phase = "closed"`, Supervisor archives L6 → L3 and END's the LangGraph.
- Implementation: explicit edge in the LangGraph builder.

#### P9h-E — No-LLM contract test
- Failing test: assert no `LLMClient.generate()` is called during a 10-trigger run. (Spy fixture.)
- Acceptance: enforced contract — Supervisor stays rule-based.

#### P9h-F — Concurrent-trigger ordering
- Failing test: two simultaneous triggers serialise via the framework's per-thread-id lock (Phase 3 contract).

---

### 9i — Triage agent enhancement

**File targets:** `examples/incident_management/agents/triage.py`, `examples/incident_management/skills/triage/system.md`, `tests/incident_management/test_triage.py`.

#### P9i-A — Triage state machine
- Failing test: state graph has nodes `gather → analyse → decide → {confirm | expand | escalate}`.
- Implementation: small state machine (not LLM-driven for the routing — only the `analyse` step is LLM).

#### P9i-B — Evidence collection
- Failing test: `gather` step calls L1 (`read_health_snapshot` if 9f exists, else stubbed snapshot fixture), L3 (`lookup_similar_incidents`), L5 (`recent_releases`) for every mapped component. (L4 deferred per MVP.)
- Implementation: parallel tool dispatch via the framework's tool runner.

#### P9i-C — Hypothesis generation
- Failing test: given a fixture evidence packet, the LLM call (mocked via deterministic fixture) produces ≥1 hypothesis matching the schema; `update_investigation` writes it to L6.
- Implementation: prompt the model with evidence + the §7 hypothesis schema; require structured JSON output.

#### P9i-D — Confidence-gate routing
- Failing test: top hypothesis confidence ≥0.8 → routes to Resolution. <0.8 + iterations < `max` → expands component scope and loops. <0.8 + iterations == `max` → escalates to Deep Investigator (in MVP this short-circuits to "needs human").

#### P9i-E — `triage_iterations` cap
- Failing test: cap of 5 (configurable) terminates with status "needs human" rather than infinite-looping.

#### P9i-F — Skill prompt update
- Failing test: golden-prompt test that the triage system prompt contains the §7 hypothesis schema and references L1/L3/L5 retrieval rules.
- Implementation: rewrite `skills/triage/system.md` per ASR.md §7.

#### P9i-G — Component scope expansion
- Failing test: when `expand` is taken, L2's `get_dependencies` is queried with depth+1 and the new components are added to `mapped_components`.

#### P9i-H — End-to-end "margin calc slow" walkthrough
- Failing test: integration test that mirrors ASR.md §4 cross-layer walkthrough — user reports → Triage scopes to `margin-calc-service` + deps via L2 → finds `pg_stat_statements` evidence in L1 → finds matching v3.12.0 release in L5 → finds matching past incident in L3 → top hypothesis "ORM upgrade query regression" with confidence ≥0.8.
- Acceptance: this is *the* MVP demo test. If it doesn't pass, MVP isn't done.

---

### 9k — Resolution agent + risk-gate integration

**File targets:** `examples/incident_management/agents/resolution.py`, `examples/incident_management/skills/resolution/system.md`, `tests/incident_management/test_resolution.py`.

#### P9k-A — `kind: responsive` registration
- Failing test: Resolution agent registered as `responsive` kind (Phase 6).
- Acceptance: MVP uses the existing responsive kind; a custom kind is unnecessary.

#### P9k-B — Approval-flow state machine
- Failing test: enum/dataclass walk through ASR.md §7 state machine: `Start → PlaybookMatched → ExecutingStep → CheckRisk → AutoExecute|AutoExecuteNotify|AwaitApproval → StepComplete → MoreSteps → VerifyFix → ArchiveIncident|RetriageSupervisor`.
- Implementation: explicit `class ResolutionPhase(StrEnum)` and a transition table.

#### P9k-C — `match_playbook` call
- Failing test: at start of agent run, calls `match_playbook(symptoms, components, env)` from L7; if no match → posts a "no playbook" message and returns to Supervisor.

#### P9k-D — Step execution through gateway
- Failing test: each step is invoked via the framework gateway; gateway's verdict (`auto` / `notify_on_execute` / `require_approval`) drives the state-machine branch.
- Implementation: explicit call to `gateway.evaluate(step.tool, step.args, env)`.

#### P9k-E — **Prod-HITL override** end-to-end
- Failing test: with `env=prod` and a playbook step `tool=restart_service, risk=low` (a low-risk-in-staging step that's a resolution-trigger tool), the agent enters `AwaitApproval` and the LangGraph `interrupt()` fires.
- Acceptance: the override is verified at the *agent* level, not just the gateway level (defence in depth).

#### P9k-F — `notify_team` wiring
- Failing test: on `AutoExecuteNotify`, `notify_team` MCP tool is invoked with the incident summary.
- Implementation: `notify_team` is a stub in MVP (logs to stdout); Slack/PagerDuty wiring in production.

#### P9k-G — `VerifyFix` step
- Failing test: after the playbook completes, a `VerifyFix` LLM call decides `Fixed | NotFixed` against current L1 snapshot; `NotFixed` re-routes to Supervisor for re-triage.

#### P9k-H — Approval-timeout policy
- Failing test: an approval pending for >`approval_timeout_seconds` (default 1800) auto-rejects per the §7 state machine and re-routes.

#### P9k-I — Subscriber notification on close
- Failing test: on `ArchiveIncident`, every subscriber in `state.subscribers` is notified.

---

### 9m — UI sliver (MVP only)

**File targets:** `examples/incident_management/ui.py`.

#### P9m-A — Hypothesis list with confidence
- Failing test: snapshot test renders ≥1 hypothesis per accordion (per global UI memory: accordion-per-item, badge-rich).

#### P9m-B — Approval inbox
- Failing test: pending-approval rows render with Approve/Reject buttons; click POSTs to gateway HITL endpoint.

#### P9m-C — Subscribers chip list
- Failing test: subscribers render as chips; honour `prefers-reduced-motion`.

#### P9m-D — Evidence trail (minimal)
- Failing test: collapsed list of evidence entries with L1/L3/L5 source badges.

> Full multi-session view, hypothesis tree visualisation, evidence timeline grouping, and prefers-color-scheme polish are deferred to the post-MVP 9m completion.

---

## 5. Sequencing

### Inter-sub-phase dependencies (MVP slice)

```
9a (state) ──────────────────────────────────────────────┐
                                                          ├──▶ 9i (Triage)
9b (L2 KG, FS) ────────────────┐                          │
9c (L5 Release Context, SQLite) ┤                          │
                                ├──▶ 9h (Supervisor) ─────┘
9d (L7 Playbook, FS+SQLite) ───┤                            │
                                │                            ├──▶ 9k (Resolution) ──▶ 9m sliver
                                │                            │
                                └──── feeds Resolution ──────┘
```

### Framework ↔ ASR-sub-phase dependencies

| ASR sub-phase | Framework phase prerequisite | Status of framework phase |
|---|---|---|
| 9a | P1 (Session/IncidentState split) + P2 (`Generic[StateT]`) | **P1 plan exists**, P2 plan exists. P9a unblocked once both ship. |
| 9b, 9c, 9d | P4 (gateway) for risk policy at MCP-load time | **Roadmap-only**. Block 9b/9c/9d until P4 plan written and shipped. |
| 9h | P6 (`supervisor-router` agent kind) | **Roadmap-only**. Block 9h on P6. |
| 9i | P2 (LangGraph state graph) | Plan exists, blocking on P2 ship. |
| 9k | P4 (gateway HITL state machine) + P2 (`interrupt()` from LangGraph) | Block 9k on P2 + P4. |
| 9k prod-HITL override | P4 must accept `prod_overrides.resolution_trigger_tools` config | **Contract gap to raise.** If P4 ships without this config knob, ASR cannot enforce the prod-HITL override at the gateway and must enforce at the agent level only (acceptable but weaker). |

### Recommended landing order (MVP)

1. **9a** — schema gate; touches everything.
2. **9b**, **9c**, **9d** in parallel (independent MCP servers; three subagents per `superpowers:subagent-driven-development`).
3. **9h** — Supervisor stub with single-investigation guard.
4. **9i** — Triage with the §4 cross-layer walkthrough as the MVP integration test.
5. **9k** — Resolution + prod-HITL override.
6. **9m sliver** — UI for what 9k produces.

Each sub-phase is independently shippable behind a feature flag (`features.asr_l2: true`, etc.) so partial progress stays demoable.

---

## 6. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Framework P3–P8 plans don't yet exist on disk; ASR sub-phases assume contracts that may shift | High | Block each ASR sub-phase on its dependency framework-phase plan being *written*. Don't pre-implement against guessed contracts. Re-baseline this plan if any framework phase's contract changes. |
| Memory at scale: L3 (vector search over 10K+ closed incidents) and L4 (embedded specs) may exceed FAISS's in-memory bounds | High | MVP uses bounded fixtures. Document a hard cap (`features.l3_max_records: 10000`) and a clear "swap to pgvector" runbook. Production deploy must enable pgvector. |
| Telemetry-source integrations (LGTM, Slack, PagerDuty) don't exist in MVP — all stubbed | Medium | Stubs are clearly marked `examples/incident_management/integrations/_stubs/`. Integration tests use deterministic fixtures. Each stub has a `# PRODUCTION_REQUIRED` marker. Build an integration-readiness checklist for the prod-cutover doc (post-MVP). |
| HITL-in-prod policy: high-risk steps may stall investigations indefinitely if approvers are unavailable | Medium | Approval-timeout (`approval_timeout_seconds`, default 1800) auto-rejects; re-routes to Supervisor; human can manually re-issue. Escalation to a fallback approver list per `escalation_teams` config. |
| Air-gapped deploy: Phase 9 brings new MCP servers each with their own deps (frontmatter, FAISS, sentence-transformers) | Medium | Per `~/.claude/rules/build.md`: every new dep goes into the lockfile and `make vendor`. The bundle script (`dist/`) must include MCP servers in the air-gapped artifact. Document in `docs/build/asr-airgap.md` (post-MVP, but plan now). |
| Concurrent-investigation race conditions when 9l ships | Medium | Per Phase 3 contract: single-writer-per-`thread_id` lock. ASR must not share mutable state across investigations. Add a fuzz test as part of 9l. |
| Triage hypothesis-loop infinite-loops or hallucinates | Medium | `triage_iterations` cap (default 5). Each iteration must produce a *new* hypothesis (id-uniqueness check) or the loop terminates. |
| Prod-HITL override silently fails if `config.resolution_trigger_tools` is empty or out of sync with playbooks | High | Add a startup validator: every playbook step's `tool` whose `effective_risk(prod) ∈ {medium, high}` must appear in `resolution_trigger_tools`. Fail loud at boot. |
| Embedding model choice (`all-MiniLM-L6-v2`, 384-dim) drift across L3/L4/L5 stores | Medium | Single `EmbeddingFactory` from framework `storage/embeddings.py` — every memory layer uses the same instance. Versioned: `embeddings.model_version` stamped on every persisted embedding; mismatch on read raises a structured warning. |
| Streamlit perf on hypothesis tree + multi-session view | Low (MVP), Medium (post-MVP) | MVP UI sliver doesn't ship multi-session; hypothesis list is flat. When 9m completes, virtualise the multi-session list. |

### Conflicts found between ASR.md and locked Phase 2–8 decisions

**Zero hard conflicts found.** Three soft tensions worth flagging:

1. **L3 / L5 vector storage.** ASR.md mandates Postgres + pgvector. Phase 2 framework decision is "FAISS dev, PGVector prod" via the `storage/vector.py` factory. Resolution: ASR conforms to the framework factory; prod swap-in is a config flip, not a code change. No conflict.
2. **HITL mechanism.** ASR.md §7 describes a bespoke approval-flow state machine. Phase 2 commits to LangGraph `interrupt()` as the single suspension mechanism. Resolution: the ASR state machine is implemented *on top of* `interrupt()` — `AwaitApproval` is the LangGraph interrupt point. No double-implementation. This was already flagged in roadmap §6 risks.
3. **No-LLM Supervisor.** ASR.md §7 requires the Supervisor to be no-LLM; Phase 6 introduces `supervisor-router` as exactly that kind. Aligned by design. No conflict.

**One contract gap (not a conflict yet):** Phase 4's gateway plan must support `prod_overrides.resolution_trigger_tools`. If P4 ships without it, ASR enforces the prod-HITL override at the agent level (in 9k-E) only — weaker but workable. **Action:** raise this as a P4 input before P4 plan is written.

---

## 7. Done Criteria for MVP

ASR-MVP is "done" when **all** of the following are true. Any failure = not done.

### Functional acceptance

1. **End-to-end "margin calc slow" walkthrough (test P9i-H) passes.** User reports → Supervisor routes to Intake (existing skill) → Triage scopes to `margin-calc-service` + dependencies via L2 → gathers L1 health snapshot fixture, L3 similar incidents, L5 recent releases → produces top hypothesis with confidence ≥0.8 → routes to Resolution → matches L7 playbook → executes low-risk step auto, awaits approval on high-risk step → human approves via UI → step executes → `VerifyFix` confirms → `ArchiveIncident` writes L6 → L3 → notifies subscribers.
2. **Prod-HITL override (test P9k-E) passes.** A low-risk-in-staging resolution-trigger tool is gated when `env=prod`, regardless of the step's declared risk.
3. **Schema migration (test P9a-E) passes.** All 191 existing tests still green after the IncidentState change.
4. **Concurrent-investigation guard (test P9h-C) passes.** A second user reporting the same issue is added as a subscriber, not a new investigation.
5. **Approval-timeout (test P9k-H) passes.** An approval pending past timeout auto-rejects and re-routes.

### Quality bars (per `~/.claude/rules/`)

6. **Tests, lint, types, build all green.** No skipped tests except those explicitly marked post-MVP.
7. **Dependency audit clean.** Every new dep (frontmatter, faiss-cpu, sentence-transformers, etc.) passes `pip-audit` at moderate level. High/Critical CVEs = hard fail.
8. **Air-gapped build path documented.** `make vendor` produces a deterministic vendor tree; `dist/apps/incident-management.py` bundle includes every new MCP server. Doc lives at `docs/build/asr-airgap.md`.
9. **UI bar.** Approval inbox is keyboard-navigable; honours `prefers-reduced-motion`; contrast ≥4.5:1 on confidence badges; accordion-per-incident layout preserved per global UI memory.
10. **Performance bar.** Triage end-to-end (excluding LLM call) p95 <500ms over the demo fixture. Resolution gateway lookup <1ms. UI interaction response <100ms (per `rules/performance.md`).

### Verification

11. **`superpowers:verification-before-completion` skill invoked** with evidence: test output, audit output, build output, demo screencast of the end-to-end walkthrough.

### What MVP does NOT include (explicit non-goals)

- Background anomaly-triggered investigations (9f + 9g — defer).
- Deep Investigator handoff (9j — Triage either confirms or returns "needs human").
- L4 Domain Knowledge filtered semantic search (9e — defer).
- Two-stage dedup with LLM confirmation (9l — single-active investigation only).
- Multi-session UI view, hypothesis tree visualisation, full evidence timeline (9m full).
- Production swaps (Neo4j for L2, Postgres+pgvector for L3/L5, Redis for L1/L6).

These are **planned post-MVP** and tracked individually. Each ships behind its own feature flag once landed; partial-state ASR stays operational.

---

## 8. Out-of-Scope for This Plan

- Code for sub-phases 9e, 9f, 9g, 9j, 9l, full 9m, and the production store-swaps. These get their own implementation plans when prioritised.
- Operator runbooks for production deploy. Air-gapped build doc is in scope (above); operational runbooks are post-MVP.
- LGTM webhook receiver for L1 ingest. Stub fixture only in MVP.

---

## 9. Immediate Next Step (post-plan)

If approved, the first actionable PR is **9a — IncidentState schema enrichment.** It is the cheapest, lowest-risk, highest-leverage gate. Every other sub-phase reads this shape. P9a-A through P9a-F can ship in a single TDD branch, ~1 day of work, with all 191 existing tests still green.

Block until then on:
- User confirmation of the recommended MVP slice (§3).
- User confirmation of the prod-HITL override contract on P4 (§5 + §6).

*End of Phase 9 plan.*

---

## Post-Phase-9 generalisation (2026-05-03)

Phase 9-h's supervisor framework hook (`Skill.runner`) was generalised one
step further: the framework now ships `runtime.intake.default_intake_runner`
as the default runner for any `kind: supervisor` skill, doing similarity
retrieval (`HistoryStore.find_similar`) and dedup gating. ASR's
`default_supervisor_runner` is now a composition of that framework default
plus L2/L5/L7 memory hydration. The `asr_supervisor` skill was renamed to
`intake`, replacing the legacy LLM-driven intake skill. See
`docs/superpowers/plans/2026-05-03-framework-intake-extraction.md`.
