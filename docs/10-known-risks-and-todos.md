# 10 — Known risks and TODOs

## Source-code TODO/FIXME/HACK markers

Verified via `grep -rnE "TODO|FIXME|XXX|HACK|DEPRECATED" src/ examples/`
on this branch (excluding `__pycache__` and the
`deprecated_kwargs` legitimate name).

| File | Marker | What |
|---|---|---|
| `src/runtime/locks.py:49` | `TODO(v2)` | Evict idle slots in `SessionLockRegistry` to cap memory in long-running servers |
| `src/runtime/locks.py:53` | `TODO(v2)` | Same — placement note on `_slots: dict[str, _Slot]` |

That's it. The codebase is otherwise free of TODO/FIXME debt — a
deliberate result of Phase 18 (HARD-04 silent-failure sweep) and the
overall "fix root cause, not workaround" project rule.

## Hardcoded values worth flagging

| Where | Value | Risk |
|---|---|---|
| `src/runtime/config.py` (default `MetadataConfig.url`) | `"sqlite:///incidents/incidents.db"` (relative path) | Default points at a relative path; CWD-dependent. The framework's actual default `config/config.yaml` overrides to `sqlite:////tmp/asr.db` (absolute). Operators who skip `config.yaml` get the relative-path default. |
| `src/runtime/config.py` (`storage.vector.path`) | `"incidents/faiss"` (relative) | Same as above |
| `src/runtime/llm.py` Phase 13 default request_timeout | `120.0` seconds | A 2-minute timeout is generous for LLM calls; some providers can hang longer on long-context responses. Per-provider override available |
| `runtime.locks.SessionLockRegistry` | unbounded dict | See `TODO(v2)` above |
| Bundle file sizes | ~660-700KB each | Large for code review. Inference: the flatten + intra-import-strip pattern is the only viable single-file deploy path. |
| `_RATE_LIMIT_MARKERS` in `src/runtime/graph.py` | string-match heuristic | If a provider invents a new 429 phrasing, retries fall back to fast-fail. Markers list comments out the variants observed in the wild. |

## Weak / incomplete features

### v1.5-D Azure leg of the integration driver
The `azure` parametrize arm in
`tests/test_integration_driver_s1.py` is wired but the dev
`.env` carries placeholder values for `AZURE_ENDPOINT`. Live
verification requires a real Azure deployment; framework code path
(`AzureChatOpenAI` construction) is intact.

### Duplicate ToolCall audit rows
The HITL fix in PR #6 left a known cosmetic duplication: when the
gateway records a high-risk tool, it stores the row under the
FastMCP composite name (`local_remediation:apply_fix`, colon form),
while the harvester later records the same tool call under the
LLM-visible name (`local_remediation__apply_fix`, double-underscore
form). Two rows for one logical event. Cosmetic in the UI; matters
if any consumer aggregates tool counts. Fix: align both on the `__`
form (~30 min). Out of scope for v1.5; deferred.

### `ApprovalWatchdog` regression test
PR #6 added gateway saves on resolution transitions. The watchdog
should observe a faster cleanup signal but no focused test verifies
that. Add a 1-test regression. ~15 min.

### `ASR_LOG_LEVEL` env var documentation
Added in PR #6, mentioned in `docs/01-local-setup.md` and
`docs/05-configuration.md` of this brownfield set, but not in the
main `README.md` or `docs/DEVELOPMENT.md`. One-line note worth
adding for operator visibility.

### Streamlit UI test coverage
`src/runtime/ui.py` is ~1700 lines, 0% coverage. Phase 20 (HARD-09)
scaffolded `tests/test_ui_*.py` with a few smoke tests but reaching
parity with backend coverage requires a dedicated UI-testing
milestone. Excluded from the coverage gate via
`pyproject.toml:[tool.coverage.run].omit`.

### Trigger registry plugin transport
`src/runtime/triggers/transports/plugin.py` is a stub —
Inference: scaffold for future SQS / Kafka / NATS work. The
`TriggerTransport` ABC + `plugin_transports` kwarg on
`TriggerRegistry.create` are usable today by external code, but no
in-repo transport beyond api / webhook / schedule.

### Postgres checkpointer
Optional via `pip install asr[postgres]`. CI is sqlite-only; the
postgres saver code (`src/runtime/checkpointer_postgres.py`) is
excluded from coverage. Production postgres deploys exist but
aren't exercised in the test suite. Risk: a postgres-specific bug
ships unnoticed.

### ASR memory layer write-back
The L2 / L5 / L7 stores in `examples/incident_management/asr/`
are read-only. Mutation paths (write-back from agents, playbook
authoring) are deferred. Inference: planned for a future
milestone; no roadmap entry confirms this.

### Dedup pipeline LLM error handling
`Orchestrator._run_dedup_check` catches all `Exception` from the
stage-2 LLM and degrades to "not a duplicate". Defensive but
silently masks a misconfigured stage-2 model. Inference: a typed
error path with logging would make ops triage faster.

## Security-sensitive areas

| Area | What to audit |
|---|---|
| `src/runtime/config.py:_interpolate` | Strict mode requires every `${VAR}` to exist; misses VAR-injection if `os.environ` itself is compromised. Standard env-var posture. |
| `src/runtime/triggers/auth.py` | Bearer token is read from env var at process start; rotation requires restart. `hmac.compare_digest` used. No HMAC-signature transport (PagerDuty / Slack) yet — `auth: bearer` only. |
| `src/runtime/tools/gateway.py` (HITL gate) | The risk policy is config-driven (`runtime.gateway.policy`) — operators MUST configure `apply_fix`-class tools as `high` for production environments to enforce HITL. The framework defaults to `auto` for unlisted tools. |
| `src/runtime/tools/gateway.py:_record_pending_resolution` | Verdict dict from operator → `Command(resume=verdict)` → tool args. Trust boundary: the operator is trusted; a malicious approver could pass arbitrary `rationale` text but cannot inject tool args (the gateway re-injects from session-derived state). |
| `src/runtime/dedup.py` (LLM stage 2) | Operator-supplied `query` text is interpolated into the LLM prompt. Standard prompt-injection surface — the LLM verdict can be steered by adversarial query content. Currently used only for soft routing (`status='duplicate'`); a misclassification doesn't escalate privileges. |
| `src/runtime/api.py` | NO authentication on `/sessions/*` endpoints. Air-gap deploys live behind corporate network controls. Webhook triggers have bearer auth via the trigger registry. |
| `src/runtime/intake.py` (similarity retrieval) | `query` text is embedded and matched against historical sessions. Low risk — the retrieved lessons are framing context, not authoritative. |
| Vector store (FAISS) | Local files. No encryption at rest; relies on filesystem permissions. Ops should chmod `/tmp/asr-faiss/` appropriately. |

## Migration risks

| Migration | Risk |
|---|---|
| Schema additive (new column, new table) | Low — `Base.metadata.create_all` at boot handles new tables; new columns get hand-rolled idempotent JSON-walk migrations under `migrations.py`. |
| Schema destructive (drop column, rename, change type) | High — there is no Alembic. A destructive change requires a one-shot script + a documented downtime window. None planned. |
| `extra_fields` JSON field reshape | Medium — apps store domain fields here. Renaming a field on the app's `Session` subclass without a `SessionStore` migration breaks load. Mitigation: app authors own their migrations. |
| FAISS index format change | Low — re-indexing is idempotent (delete the index file; the next save rebuilds). |
| Bundle format change | Low — `dist/*` is regenerated from source on every PR (HARD-08 gate). Bundle drift is mechanical. |
| `langgraph` major version bump | High — PR #6 caught a breaking semantic change in `interrupt()` between langgraph 0.x and 1.x. Future major bumps (2.x?) need similar smoke tests; the `_drive_agent_with_resume` helper is the most exposed surface. |
| `langchain` major version bump | High — `langchain.agents.create_agent` is the agent factory. A signature change there cascades through `make_agent_node`. |
| Provider model deprecation (e.g. OpenRouter free-tier model removed) | Low — config swap; no code change. The 429 retry helps with transient throttles, not deprecations. |

## Concurrency / race risks

| Risk | Mitigation |
|---|---|
| Concurrent session writes (UI + API approval simultaneously) | `SessionLockRegistry` enforces single writer per session; second writer gets `SessionBusy → HTTP 429`. |
| Concurrent retry on a session in `error` | `_retries_in_flight` set in `Orchestrator` rejects second retry. |
| Approval race with `ApprovalWatchdog` timeout | `StaleVersionError` → both reload, one wins. Watchdog re-checks before resolving. |
| LangGraph thread_id collision on retry | `retry_session` bumps `active_thread_id` to `<sid>:retry-N`; original thread stays at terminated checkpoint. |
| Stale state on HITL resume | PR #6 fix: `make_agent_node` reloads from store at entry. Past pain point — see `docs/DESIGN.md` DEC-010. |

## Operational risks

| Risk | Mitigation |
|---|---|
| `/tmp` filling up (SQLite + FAISS in `/tmp` per default config) | Operators should override `storage.metadata.url` and `storage.vector.path` in production to a persistent path. |
| Long-running orchestrator memory growth | `SessionLockRegistry` `TODO(v2)` — slots accumulate; add eviction. |
| Provider key rotation requires restart | Env vars read at process start. No SIGHUP reload. |
| Single-process limit | One `OrchestratorService` per host; `runtime.max_concurrent_sessions: 8` cap. Multi-host deploys need a separate orchestrator per host (and a separate metadata DB OR strict per-session locking via a shared lock service — not implemented). |
| Bundle drift on hand-edited `dist/` | CI catches via "Bundle staleness gate (HARD-08)". |
| Lockfile drift after `pip install` instead of `uv sync` | Operators MUST use `uv sync --frozen`; CI catches via `uv lock --check`. |

## Documentation drift risks

| Risk | Mitigation |
|---|---|
| Docs reference outdated test counts / coverage / ratchet baseline | `docs/00-project-overview.md` snapshots current values; refresh on milestone landings. |
| `.planning/` (gitignored) used as canonical state | Don't — the canonical state is `docs/DESIGN.md` § 13 and the git history. |
| `.env` placeholder vs real values mismatch | Operators must populate per-deploy; CI uses dummy values. |
| Skill prompts reference removed tool args | `scripts/lint_skill_prompts.py` (Phase 21 / SKILL-LINTER-01) catches as a CI gate. |
