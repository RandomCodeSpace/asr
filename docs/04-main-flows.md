# 04 — Main flows

For each flow: **entry points**, **key files**, and **failure
modes**. Companion to `docs/DESIGN.md` § 2 (architecture overview)
and § 7 (HITL).

---

## Auth / login

**Status: not present in framework.** Air-gap deploys rely on
corporate network controls (the runtime never opens its own
auth surface).

The only auth surface in the framework is **bearer-token
auth on webhook trigger endpoints** (`auth: bearer` in
`triggers:` config; token read from env var at startup; constant-
time comparison via `hmac.compare_digest`).

Entry point: `src/runtime/triggers/auth.py`,
`src/runtime/triggers/transports/webhook.py`.

Failure modes:
- Missing/empty token env → trigger refuses to start
  (`LLMConfigError` analogue at config-load)
- Wrong bearer → `HTTP 401`
- Timing-safe comparison only — no rate limiting in the framework

---

## Session lifecycle (request → terminal)

Entry points (any of):

- **CLI** — `python -m runtime --config <yaml>` (boots the FastAPI surface)
- **API** — `POST /sessions` (`src/runtime/api.py`)
- **Streamlit UI** — "Start Investigation" button (`src/runtime/ui.py`)
- **Webhook trigger** — `POST /triggers/{name}` (configured per `triggers:` block in YAML)
- **Schedule trigger** — APScheduler cron (in-process)
- **Plugin trigger** — custom transport via setuptools entry-point

All entry points converge on
`OrchestratorService.start_session(query=…, environment=…, …)`,
which:

1. Allocates the session ID synchronously on the loop
2. Inserts the row (`status='new'`)
3. Spawns an `asyncio.Task` for `Orchestrator.graph.ainvoke(...)`
4. Returns the session ID immediately (caller polls or streams)

Key files:

- `src/runtime/service.py:start_session` (entry point)
- `src/runtime/orchestrator.py:start_session` (per-session lock + graph kick-off)
- `src/runtime/graph.py:make_agent_node` (per-skill agent step)
- `src/runtime/agents/turn_output.py:parse_envelope_from_result` (envelope contract enforcement)
- `src/runtime/orchestrator.py:_finalize_session_status_async` (terminal status assignment)

Per-step events emitted to `EventLog`:
`agent_started → tool_invoked* → confidence_emitted → route_decided
→ agent_finished` per agent; `gate_fired` at HITL boundaries;
`status_changed` on terminal transitions.

Failure modes:

| What | Symptom | Where caught |
|---|---|---|
| LLM 5xx / connection reset | Retried 3× with 1.5s/3s/4.5s backoff | `_ainvoke_with_retry` |
| LLM 429 rate-limit | Retried 3× with 7.5s/15s/22.5s backoff | `_ainvoke_with_retry` |
| LLM 4xx (non-429) | Fail immediately → `_handle_agent_failure` → `status='error'` | `make_agent_node` exception arm |
| LLM dropped markdown contract (no envelope) | Path 5 (terminal-tool args) → Path 6 (permissive synthesis) → 0.30-confidence placeholder | `parse_envelope_from_result` |
| LLM dropped contract AND no tool calls | Hard fail → `EnvelopeMissingError` → `status='error'` | Path 7 |
| HITL high-risk tool gate fires | `interrupt()` raised, session stays `in_progress`, pending_approval row written | `gateway.wrap_tool` |
| Operator times out an approval | `ApprovalWatchdog` resolves with `verdict=timeout` | `tools/approval_watchdog.py` |
| Stale-version save (concurrent writers) | `StaleVersionError` raised; caller reloads + retries | `SessionStore.save` |
| Recursion limit hit on inner agent | LangGraph `GraphRecursionError` propagates → `_handle_agent_failure` | langgraph default bound |

---

## HITL approve / reject (high-risk tool)

Trigger: an agent calls a tool tagged `high` in
`runtime.gateway.policy` (or matching `gate_policy.resolution_trigger_tools`
in production env).

Flow:

```
agent calls apply_fix
  └─ gateway _arun
       ├─ inject session-derived args (e.g. environment)
       ├─ should_gate → GateDecision(gate=True, reason=…)
       ├─ append ToolCall(status='pending_approval') + store.save
       └─ langgraph.types.interrupt(payload)  ← pauses inner agent
            ↓
inner.ainvoke returns with __interrupt__ in result dict
  └─ _drive_agent_with_resume detects, raises GraphInterrupt
       ↓
outer Pregel pauses (state checkpointed)
  └─ ainvoke returns with __interrupt__ on outer state
       ↓
finalize SKIPPED (Orchestrator._is_graph_paused → True)
       ↓
[UI / API: operator clicks Approve or POSTs to /approvals/{tcid}]
       ↓
graph.ainvoke(Command(resume={"decision": "approve", ...}))
  └─ outer node re-runs
       └─ _drive_agent_with_resume: aget_state(inner_cfg).next non-empty
            └─ outer interrupt() → returns the verdict dict
                 └─ inner.ainvoke(Command(resume=verdict), config=inner_cfg)
                      └─ gateway _arun re-enters
                           └─ verdict == "approve" → run apply_fix
                           └─ update pending row → status='approved' + save
                                ↓
inner agent finishes
  └─ envelope parsed, AgentRun recorded, route to next node / END
       ↓
outer ainvoke returns
  └─ finalize runs (no longer paused) → terminal status set
```

Key files:

- `src/runtime/tools/gateway.py:_arun` (and `_run` mirror) — the
  pause + resume entry points
- `src/runtime/graph.py:_drive_agent_with_resume` — the
  langgraph 1.x `__interrupt__` plumbing
- `src/runtime/orchestrator.py:_is_graph_paused` — finalize guard
- `src/runtime/api.py:submit_approval_decision` — HTTP approval handler
- `src/runtime/ui.py:_submit_approval_via_service` — UI approval handler
- `src/runtime/tools/approval_watchdog.py` — stale-approval timeout

Failure modes:

| What | Symptom | Where caught |
|---|---|---|
| `Command(resume=…)` raises `Cannot use Command(resume=...) without checkpointer` | Inner agent missing checkpointer | Inner `create_agent` always gets `checkpointer=` per PR #6 |
| Stale `state["session"]` on resume → gateway double-appends → `StaleVersionError` | Outer Pregel checkpoint at step boundaries doesn't reflect mid-step gateway saves | `make_agent_node` reloads from store at entry per PR #6 |
| Operator approves but DB row stays `pending_approval` | Gateway didn't save after status transition | `_record_pending_resolution` saves after every transition (approved/rejected/timeout) per PR #6 |
| Session goes to `error` instead of resuming | Pre-PR-#6 langgraph 1.x silently swallowed `interrupt()` and finalized the session | Fixed by `_drive_agent_with_resume` |

---

## Background jobs

### `LessonRefresher` (auto-learning, M5/M6)

Source: `src/runtime/learning/scheduler.py`.

Runs an APScheduler job (default: nightly 02:00 UTC; configurable
via `learning.scheduler` block in YAML). For each session resolved
since the last run, extracts a `Lesson` row capturing the winning
hypothesis + applied fix.

Entry: `LessonRefresher.start()` (called by lifespan hook in
`src/runtime/api.py`).

Failure modes:
- Job exception → APScheduler logs and continues to next tick
  (defensive `try/except` around the per-session extraction)
- Long-running extraction blocks subsequent ticks within the same
  scheduler — bounded by per-session timeout

### `ApprovalWatchdog`

Source: `src/runtime/tools/approval_watchdog.py`.

Polls the DB for `pending_approval` rows older than
`framework.approval_timeout`. Resolves them with
`verdict=timeout` so operators don't end up with permanently-paused
sessions.

Entry: `ApprovalWatchdog.start()` (called by lifespan hook).

Failure modes:
- DB unreachable → logged, retried next tick
- Resolution race with concurrent operator approval →
  `StaleVersionError`; watchdog reloads + retries

---

## Data ingestion / sync

### Trigger registry

Source: `src/runtime/triggers/`.

Three transport flavours configurable in `config.yaml`'s
`triggers:` block:

| Transport | Entry | Per-trigger config |
|---|---|---|
| `webhook` | `POST /triggers/{name}` (FastAPI route) | `payload_schema`, `transform`, `auth`, `idempotency_ttl_hours` |
| `schedule` | APScheduler in-process cron | `schedule:` 5-field cron, `payload:` static |
| `plugin` | custom (`TriggerTransport` ABC, setuptools entry-point) | per-plugin |
| `api` | back-compat for `POST /investigate` | (deprecated alias) |

Each trigger fires `OrchestratorService.start_session(...)` with
a synthetic payload. Provenance stamped on
`session.findings['trigger']` so dashboards can answer "where did
this come from?"

Failure modes:

| What | Symptom | Where caught |
|---|---|---|
| Transform raises | `HTTP 422 Unprocessable Entity`, NOT cached for idempotency | `transports/webhook.py` |
| Auth fails | `HTTP 401` | `triggers/auth.py` |
| Idempotency-Key replay | First request's `session_id` returned | `triggers/idempotency.py` |
| Schedule drift | ±1 minute under normal load (in-process APScheduler limit) | Inference: not measured; documented in legacy README |

### Two-stage dedup pipeline

Source: `src/runtime/dedup.py`.

Stage 1: embedding similarity over closed sessions
(`HistoryStore.find_similar`). Stage 2: LLM judge confirms (or
rejects) the match. Confirmed matches mark the new session
`status='duplicate'` with `parent_session_id` linkage.

Entry: `Orchestrator._run_dedup_check` called early in
`start_session`.

Failure modes:
- LLM stage 2 throws → degrades to "not a duplicate" so dedup
  never crashes intake (`Orchestrator._run_dedup_check` catches `Exception`)
- No similar sessions → returns False, normal flow proceeds

---

## Deployment

Source: `scripts/build_single_file.py`,
`docs/AIRGAP_INSTALL.md`, `docs/DEVELOPMENT.md`.

**Build (CI / dev box):**
```bash
uv sync --frozen --extra dev
uv run python scripts/build_single_file.py
git add dist/ && git commit
```

**Deploy (target host, copy-only):**

7-file payload:
```
app.py                    (renamed from dist/apps/<app>.py)
ui.py                     (dist/ui.py)
config/config.yaml        (framework: LLM, MCP, storage)
config/<app>.yaml         (app: severity aliases, escalation roster, …)
config/skills/            (optional skill prompt overrides)
.env                      (provider keys)
```

Boot:
```bash
python -m runtime --config config/<app>.yaml
streamlit run ui.py --server.port 37777
```

CI gate `Bundle staleness gate (HARD-08)` rebuilds the bundles
from source on every PR and refuses the merge if `dist/*` differs
from a fresh build. Means `dist/*` on `main` is always
deploy-ready.

Failure modes:

| What | Symptom | Where caught |
|---|---|---|
| New `src/runtime/` module not in `RUNTIME_MODULE_ORDER` | `tests/test_bundle_completeness.py` fails | Local pytest before push |
| Bundle drift (changed src without dist regen) | CI's "Bundle staleness gate" fails | CI |
| Bundle doesn't boot from a clean tmpdir | `tests/test_build_single_file.py` smoke check | Local |
| Lockfile drift | CI's "Lockfile freshness gate" fails | CI (`uv lock --check`) |

---

## Error handling (cross-cutting patterns)

| Pattern | Example | Source |
|---|---|---|
| Typed exception hierarchy | `LLMTimeoutError`, `LLMConfigError`, `EnvelopeMissingError`, `SessionBusy`, `StaleVersionError` | `src/runtime/errors.py`, `storage/session_store.py`, `locks.py`, `agents/turn_output.py` |
| Bounded retries on transient cloud errors | `_ainvoke_with_retry` (5xx + 429) | `src/runtime/graph.py` |
| Fail-fast on policy errors | `should_gate` raises before tool runs | `src/runtime/policy.py` |
| Defensive try/except around telemetry | EventLog failures NEVER break a tool call | `gateway.py` `_emit_invoked` |
| `_handle_agent_failure` for caught LLM exceptions | Marks session `error` + records failure agent_run | `src/runtime/graph.py` |
| Per-session async lock prevents concurrent writes | `SessionLockRegistry.acquire(session_id)` | `src/runtime/locks.py`, used by `service.py` + `api.py` |
| Optimistic concurrency on save | `version` column on `IncidentRow`; `StaleVersionError` on mismatch | `storage/session_store.py:save` |
| Silent-failure sweep (Phase 18 / HARD-04) | All `except Exception: pass` blocks replaced with logged re-raise or typed handler | `tests/test_silent_failure_sweep.py` (Inference: name based on phase) |
