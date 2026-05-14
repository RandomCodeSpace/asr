# 03 — Code map

> Approximate line counts and per-folder purpose. Verify with
> `find <path> -name '*.py' -exec wc -l {} +`.

## Top-level directories

| Path | Purpose |
|---|---|
| `src/runtime/` | Framework code — the only thing the bundler reads to produce `dist/app.py` |
| `examples/incident_management/` | Flagship example app: SRE incident investigation pipeline |
| `examples/code_review/` | Second example app: PR review pipeline (proves framework genericity) |
| `tests/` | Pytest suite (149 test files; 1265 tests; 87% coverage on `src/runtime/`) |
| `scripts/` | Build + lint utilities |
| `config/` | Default framework config + per-app config + skill prompt directory |
| `docs/` | This documentation set + DESIGN narrative + dev / install how-tos |
| `dist/` | **Generated** by `scripts/build_single_file.py`; never hand-edit |
| `ui/` | Streamlit launcher shim (`streamlit_app.py`) |
| `.github/workflows/` | CI: lint + type-check + test + sonar (`ci.yml`) |
| `.planning/` | **Gitignored** local working state (GSD planning workflow); selected artifacts can be committed but rarely should be |

Top-level files:

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, deps, pytest/ruff/pyright/coverage config |
| `uv.lock` | Pinned dependency graph with hashes — reproducible installs |
| `pyrightconfig.json` | Pyright typing config (CI gate fails on errors per `ci.yml`) |
| `sonar-project.properties` | SonarCloud analysis config (sources, exclusions, CPD exclusions, coverage paths) |
| `README.md` | Repo intro pointing at `docs/DESIGN.md` |

---

## `src/runtime/` (~18 200 lines total)

### Top-level modules

| File | LOC | Purpose | Related |
|---|---|---|---|
| `__main__.py` | ~70 | argparse-only CLI entry: `python -m runtime --config <yaml>` | `orchestrator.py`, `service.py`, `api.py` |
| `__init__.py` | 0 | empty | |
| `state.py` | 173 | `Session`, `AgentRun`, `ToolCall`, `TokenUsage` pydantic models | All app `Session` subclasses extend the model here |
| `state_resolver.py` | ~70 | Loads the app's `state_class` from a dotted path (`runtime.state_class` config) | Wave-2 generic-runtime decoupling |
| `skill.py` | ~520 | `Skill`, `RouteRule`, `DispatchRule`, skill loader (reads YAML + system.md per skill folder) | `examples/*/skills/*/config.yaml` |
| `config.py` | ~1100 | All pydantic config schemas: `AppConfig`, `LLMConfig`, `MCPConfig`, `OrchestratorConfig`, `GatewayConfig`, `GatePolicy`, etc. | `config/config.yaml` |
| `errors.py` | ~50 | Typed exceptions: `LLMTimeoutError`, `LLMConfigError`, `EnvelopeMissingError` |
| `llm.py` | ~600 | `get_llm`, `get_embedding` — provider abstraction + `StubChatModel` for tests | `langchain-openai`, `langchain-ollama` |
| `mcp_loader.py` | ~270 | Loads MCP servers per `mcp.servers[*]`, builds `ToolRegistry` | `fastmcp`, `langchain-mcp-adapters` |
| `orchestrator.py` | ~1400 | `Orchestrator` class — `start_session`, `stream_session`, `resume_session`, `retry_session`, `_finalize_session_status_async`, `_is_graph_paused` | `graph.py`, `service.py` |
| `service.py` | ~830 | `OrchestratorService` — long-lived asyncio loop; thread-safe bridge | Used by both UI + API |
| `api.py` | ~880 | FastAPI surface — `/sessions/*` REST + SSE + WebSocket + approvals | `service.py` |
| `api_dedup.py` | ~110 | API endpoint for retracting dedup matches | `dedup.py` |
| `graph.py` | ~1430 | LangGraph build (`build_graph`, `_build_agent_nodes`), `make_agent_node`, `_drive_agent_with_resume`, `_ainvoke_with_retry`, `parse_envelope_from_result` callers | `langgraph`, `langchain.agents.create_agent` |
| `intake.py` | ~250 | Default intake supervisor runner — similarity retrieval + dedup gate | `dedup.py`, `LessonStore` |
| `dedup.py` | ~430 | Two-stage dedup pipeline (embedding similarity + LLM stage 2) | `HistoryStore` |
| `similarity.py` | ~50 | Cosine similarity helper |
| `policy.py` | ~270 | Pure functions — `should_gate`, `should_retry`, gate decision dataclass | `gateway.py`, `orchestrator.py` |
| `locks.py` | ~120 | `SessionLockRegistry` — per-session asyncio locks; `SessionBusy` exception; D-01 contract | `orchestrator.py`, `service.py` |
| `checkpointer.py` | ~120 | LangGraph checkpointer factory (sqlite default); `make_checkpointer` | `langgraph-checkpoint-sqlite` |
| `checkpointer_postgres.py` | ~80 | Postgres checkpointer (lazy-imported; `pip install asr[postgres]`) | `langgraph-checkpoint-postgres` |
| `dedup.py` | ~430 | Two-stage dedup (embedding + LLM) | (above) |
| `terminal_tools.py` | ~80 | Maps terminal tool names → status transitions per `cfg.orchestrator.terminal_tools` |
| `skill_validator.py` | ~110 | Validates `skill.model` references against `LLMConfig.models` at orchestrator boot |

### Subpackages

| Path | Purpose | Key files |
|---|---|---|
| `agents/` | Agent-kind factories | `responsive.py` (default LLM agent — mirrors `graph.py:make_agent_node`), `supervisor.py` (rule/llm dispatch), `monitor.py` (out-of-band runner), `turn_output.py` (envelope parser + `AgentTurnOutput` model) |
| `tools/` | Gateway + arg-injection + watchdog | `gateway.py` (~830 lines — risk-rated wrap + interrupt/resume), `arg_injection.py` (session-derived args), `approval_watchdog.py` (~320 lines — stale-approval timeout), `__init__.py` |
| `storage/` | Persistence | `models.py` (SQLAlchemy `IncidentRow` + `EventRow` + `LessonRow`), `engine.py` (engine factory), `embeddings.py` (FAISS-backed embedder), `vector.py` (vector store), `session_store.py` (~660 lines — CRUD), `history_store.py` (~230 lines — read-only similarity), `event_log.py` (~135 lines), `lesson_store.py` (~150 lines), `migrations.py` (~210 lines), `checkpoint_gc.py` (~50 lines) |
| `learning/` | Auto-learning (M5/M6) | `extractor.py` (lesson extraction at finalize), `scheduler.py` (~160 lines — APScheduler nightly refresher) |
| `memory/` | App-overridable memory hooks | `session_state.py`, `hypothesis.py` (triage hypothesis loop), `knowledge_graph.py`, `release_context.py`, `playbook_store.py`, `resolution.py` — these are runtime-agnostic helpers; the L2/L5/L7 stores in `examples/incident_management/asr/` use them |
| `triggers/` | Trigger registry | `base.py` (TriggerTransport ABC), `config.py`, `registry.py` (~320 lines), `idempotency.py` (~210 lines), `auth.py` (bearer), `resolve.py`, `transports/api.py`, `transports/webhook.py` (~140 lines), `transports/schedule.py` (~85 lines), `transports/__init__.py` |

---

## `examples/incident_management/`

| File | Purpose |
|---|---|
| `__init__.py` | empty |
| `state.py` | `IncidentState(Session)` subclass — `query`, `environment`, `reporter`, `summary`, `tags`, `severity`, `category`, `matched_prior_inc`, `resolution`, `memory: MemoryLayerState` |
| `mcp_server.py` | `IncidentMCPServer` — `lookup_similar_incidents`, `create_incident`, `update_incident`, `submit_hypothesis`, `mark_resolved`, `mark_escalated`, `hydrate_and_gate` (memory hydration + dedup gate) |
| `mcp_servers/observability.py` | Observability tools: `get_logs`, `get_metrics`, `get_service_health`, `check_deployment_history` |
| `mcp_servers/remediation.py` | Remediation tools: `propose_fix`, `apply_fix` (gated `high`), `notify_oncall` |
| `mcp_servers/user_context.py` | User-context tools |
| `asr/` | L2 Knowledge Graph + L5 Release Context + L7 Playbook stores (filesystem-backed) |
| `skills/intake/` | Supervisor skill: rule-dispatch to triage; runs similarity + memory hydration |
| `skills/triage/` | Hypothesis-loop investigator |
| `skills/deep_investigator/` | Evidence gathering |
| `skills/resolution/` | Propose / apply fix or escalate |
| `skills/_common/` | Shared prompt fragments (output contract, confidence calibration) |

Per-skill structure: `<skill>/config.yaml` + `<skill>/system.md`.

---

## `examples/code_review/`

| File | Purpose |
|---|---|
| `state.py` | `CodeReviewState(Session)` — `pr: PullRequest`, `review_findings: list[ReviewFinding]`, `overall_recommendation`, `review_summary`, `review_token_budget` |
| `mcp_server.py` | `CodeReviewMCPServer` — `fetch_pr_diff` (mock), `add_review_finding`, `set_recommendation` |
| `skills/intake/` `analyzer/` `recommender/` | 3-skill responsive pipeline |

Demonstration / mock; the diff fetch reads `tests/fixtures/code_review/<repo>/<number>.json` if present.

---

## `tests/` (149 files)

Test groups by topic (sample):

| Pattern | Topic |
|---|---|
| `test_agent_node*.py`, `test_real_llm_tool_loop_termination.py`, `test_integration_driver_s1.py` | Agent runner contract, live-LLM smoke |
| `test_interrupt_detection.py`, `test_gateway_persist_resolution.py`, `test_orchestrator_pause_detection.py`, `test_approval_*.py` | HITL approve/reject end-to-end |
| `test_markdown_turn_output.py` | Phase 22 envelope parser (36 tests) |
| `test_ainvoke_retry_429.py` | 429 retry backoff regime |
| `test_per_agent_model_dispatch.py` | v1.5-C per-agent dispatch contract |
| `test_genericity_ratchet.py`, `test_concept_leak_ratchet.py` | Framework-leak counters |
| `test_session_store.py`, `test_incident_store.py`, `test_history_store.py`, `test_dedup_*.py` | Storage layer |
| `test_telemetry_integration.py`, `test_event_log.py` | Per-step events |
| `test_api*.py`, `test_approval_api.py`, `test_session_lock.py` | FastAPI surface + lock contract |
| `test_bundle_*.py`, `test_build_*.py` | Bundler + bundle completeness |
| `test_triggers/` | Trigger registry transports |
| `test_ui_*.py`, `test_render_*.py` | Streamlit UI helpers |
| `test_skill*.py` | Skill loader, model override resolution |

Helpers: `tests/_envelope_helpers.py`, `tests/_policy_helpers.py`,
`tests/conftest.py` (if present), `tests/fixtures/` (sample
configs, mock PR diffs).

---

## `scripts/`

| Script | Purpose |
|---|---|
| `build_single_file.py` | The bundler. Reads `RUNTIME_MODULE_ORDER` + per-app order lists, flattens into `dist/`. **Must run after any change to `src/runtime/` or `examples/`** |
| `check_genericity.py` | Counts `incident` / `severity` / `reporter` tokens in `src/runtime/`. Powers the ratchet test |
| `lint_skill_prompts.py` | Phase 21 (SKILL-LINTER-01) — walks every `examples/*/skills/*/system.md` and asserts referenced tool names + arg fields exist in the inventory |
| `migrate_jsonl_to_sql.py` | One-off migration for legacy JSONL incident store → SQLAlchemy |
| `seed_demo_incidents.py` | Seeds the FAISS index + sqlite DB with demo data for UI walkthroughs |

---

## `config/`

| File | Purpose |
|---|---|
| `config.yaml` | Default framework config — LLM providers + models, MCP servers, storage URL, trigger registry, gateway policy |
| `config.yaml.example` | Annotated template for new deploys |
| `incident_management.yaml` | Incident-app composite config (framework + app keys) |
| `code_review.yaml`, `code_review.runtime.yaml` | Code-review composite config |
| `skills/` | Optional shared skill prompts (rare; usually skills live under `examples/<app>/skills/`) |

---

## `docs/`

| File | Purpose |
|---|---|
| `DESIGN.md` | Long-form architecture + decisions narrative |
| `DEVELOPMENT.md` | Day-to-day dev workflow |
| `AIRGAP_INSTALL.md` | Air-gap install procedure |
| `00-…` through `11-…` | This brownfield documentation set (you're reading it) |
| `adr/0001-…` | Architecture Decision Record |
