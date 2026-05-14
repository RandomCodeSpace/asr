# 08 — Testing

## Framework

**pytest** with `pytest-asyncio` (asyncio_mode=auto), `pytest-cov`,
`pytest-repeat` (for D-13 stability gate). Config in
`pyproject.toml:53-58`.

```
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --cov=src/runtime --cov-report=term-missing --cov-report=xml"
pythonpath = ["src", "."]
```

Coverage gate **fails below 85%** when run with
`--cov-fail-under=85`. Current coverage: **87.04%** (post v1.5).

## How to run

```bash
# Full suite, fail-fast
uv run pytest -x

# Without coverage (faster iteration)
uv run pytest -x --no-cov

# Single file
uv run pytest tests/test_interrupt_detection.py -x -v

# Single test
uv run pytest tests/test_interrupt_detection.py::test_resume_forwards_verdict_to_inner_tool_and_completes -xvs

# With coverage gate
uv run pytest --cov=src/runtime --cov-fail-under=85 -x

# Stability check (50 iterations of one test — D-13 local gate)
uv run pytest tests/test_session_lock.py -x --count=50

# Live integration smoke (gated on env vars)
OLLAMA_API_KEY=... OLLAMA_BASE_URL=https://ollama.com \
  uv run pytest tests/test_integration_driver_s1.py -v
```

CI runs the full suite + coverage XML + JUnit XML for SonarCloud.

## Suite structure

149 test files; ~1265 tests; ~140s for the full suite.

### By topic

| Topic | Sample files |
|---|---|
| Agent runner contract + live-LLM smoke | `test_agent_node*.py`, `test_real_llm_tool_loop_termination.py`, `test_integration_driver_s1.py`, `test_per_agent_model_dispatch.py` |
| HITL approve/reject + gateway | `test_interrupt_detection.py`, `test_gateway_persist_resolution.py`, `test_orchestrator_pause_detection.py`, `test_approval_*.py`, `test_gateway_*.py`, `test_interrupt_status_handling.py` |
| Markdown turn-output parser | `test_markdown_turn_output.py` (36 tests) |
| Retry behaviour | `test_ainvoke_retry_429.py` (5 tests) |
| Storage layer | `test_session_store.py`, `test_incident_store.py`, `test_history_store.py`, `test_dedup_*.py`, `test_event_log.py` |
| FastAPI surface + locks | `test_api*.py`, `test_approval_api.py`, `test_session_lock.py`, `test_retry_concurrency.py` |
| Triggers | `test_triggers/test_*.py` (transport per file) |
| Bundler + bundle | `test_build_*.py`, `test_bundle_*.py` |
| Genericity ratchets | `test_genericity_ratchet.py`, `test_concept_leak_ratchet.py` |
| Skill loader | `test_skill*.py` |
| Telemetry + auto-learning | `test_telemetry_integration.py`, `test_lesson_*.py` |
| UI helpers | `test_ui_*.py`, `test_render_*.py` |
| Memory layers (incident_management) | `test_asr_*.py`, `test_kg_store.py`, `test_release_store.py`, `test_playbook_store.py` |
| Per-app tests | `test_code_review_*.py`, `test_two_apps_coexist.py`, `test_session_id_format.py`, `test_generic_round_trip.py` |

### Helpers + fixtures

| File | Purpose |
|---|---|
| `tests/_envelope_helpers.py` | `EnvelopeStubChatModel` — pydantic stub LLM that emits the markdown contract, used across HITL + agent tests |
| `tests/_policy_helpers.py` | Helpers for building synthetic gate decisions |
| `tests/fixtures/sample_config.yaml` | Reference config for config-loader tests |
| `tests/fixtures/code_review/<repo>/<number>.json` | Mock PR diffs for the code-review example app |

Conftest is implicit (no `tests/conftest.py` discovered;
fixtures defined per-file).

## What's covered well

- **Markdown envelope parser** — 36 tests covering 6 paths,
  Unicode dash variants, gpt-oss empty-closing pattern, terminal-tool
  args synthesis, permissive synthesis fallback.
- **HITL pause/resume on langgraph 1.x** — `test_interrupt_detection.py`
  proves the GraphInterrupt re-raise + Command(resume) forwarding;
  `test_gateway_persist_resolution.py` (10 tests) proves the DB row
  reflects the verdict for both sync + async paths.
- **Retry regimes** — `test_ainvoke_retry_429.py` pins both backoff
  windows (5xx and 429) plus fast-fail on non-transient errors.
- **Per-agent LLM dispatch** — `test_per_agent_model_dispatch.py`
  proves `_build_agent_nodes` calls `get_llm` with `model_name=skill.model`.
- **Storage round-trip** — `test_generic_round_trip.py` proves
  `extra_fields` JSON survives full save/load cycles for arbitrary
  `Session` subclasses.
- **Optimistic concurrency** — `test_session_lock.py` (over 1000
  lines) covers the D-01 / D-20 contracts: per-session lock holds
  across HITL pause; resume re-acquires cleanly; concurrent retry
  is rejected.
- **API surface** — `test_api_react_surface.py` covers `/sessions/*`
  + SSE + WebSocket + structured error envelope.
- **Two apps coexist** — `test_two_apps_coexist.py` proves an
  incident session and a code-review session can share the same
  metadata DB without collisions (per `Session.id_format`).

## What's covered weakly or not at all

| Gap | Why it matters | Where to start |
|---|---|---|
| `src/runtime/ui.py` (~1700 lines, 0% coverage) | Streamlit shell — exercised by manual smoke. Phase 20 (HARD-09) scaffolded `tests/test_ui_*.py` but UI parity coverage is a milestone. | `tests/test_ui_*.py` exists; extend with `streamlit.testing.v1.AppTest` |
| `src/runtime/__main__.py` | argparse-only CLI; covered by smoke only | Inference: low risk |
| `src/runtime/checkpointer_postgres.py` | Postgres saver; CI is sqlite-only | Run a postgres container in CI for a one-test postgres smoke |
| `src/runtime/triggers/transports/plugin.py` | Stub for future transports | n/a |
| `ApprovalWatchdog` × `gateway` saves on transition | I added gateway saves on transitions in PR #6; the watchdog should observe a faster cleanup signal but no focused test verifies that. ~15 min. | New test asserting the watchdog resolves a row faster after a gateway save |
| Live integration with all 3 providers green simultaneously | OpenRouter is out of credits and Azure has placeholder endpoint in this dev `.env` | Operator-side issue, not framework |
| `test_silent_failure_sweep.py` | Should assert no `except Exception: pass` survives | Inference: name based on Phase 18 / HARD-04; verify the test exists and passes |

## Risky areas needing more tests

1. **Multi-agent live runs against real providers** — only the
   single-agent S1 driver is live-gated. Multi-agent E2E (intake →
   triage → DI → resolution) only runs in stub mode. A live multi-
   agent driver would catch provider-quirk regressions earlier.
2. **`HistoryStore` filter dimensions** — apps build their own
   `filter_resolver`; the framework only tests the incident-shaped
   one. A code-review-shaped filter test would prove the seam holds.
3. **`OrchestratorService.stop_session` mid-pause** — what happens
   if the operator cancels a session that's currently `pending_approval`?
   `test_session_lock.py` covers locks; explicit cancellation
   semantics during HITL deserve a focused test.
4. **`migrations.py` rollback** — the migrations are forward-only
   and idempotent. A backward-compat regression test (run the new
   code against an old-shape DB) exists for `migrate_tool_calls_audit`;
   adding similar tests for future migrations would lock the
   contract.
5. **Trigger registry under concurrency** — `test_triggers/`
   covers each transport in isolation; a fan-in test (50 webhooks
   firing concurrently) would catch idempotency-key races.

## CI gates

`.github/workflows/ci.yml`:

| Gate | Tool | Failure behavior |
|---|---|---|
| Lockfile freshness (HARD-02) | `uv lock --check` | Fails if `pyproject.toml` drift from `uv.lock` |
| Bundle staleness (HARD-08) | `python scripts/build_single_file.py && git diff --exit-code dist/` | Fails if `dist/` would change |
| Lint | `ruff check src/ tests/` | Fails on any rule violation |
| Type check (HARD-03) | `pyright src/runtime` | Fail-on-error since Phase 19 |
| Test + coverage | `pytest --cov=src/runtime --cov-report=xml --junitxml=junit.xml` | Default fail on test failure; coverage gate via SonarCloud |
| Skill-prompt-vs-schema lint (SKILL-LINTER-01) | `python scripts/lint_skill_prompts.py` | Fails if any skill prompt references a tool name / arg field that doesn't exist |
| SonarCloud scan | `SonarSource/sonarqube-scan-action@v8.0.0` | Quality gate (coverage / hotspots / duplications) reported back to the PR |

## How to add a test

1. Pick the file matching the topic (or create a new one if cross-cutting).
2. If async, no decorator needed (`asyncio_mode=auto`).
3. If you need a stub LLM, use `EnvelopeStubChatModel` from
   `tests/_envelope_helpers.py` — it emits the markdown contract
   automatically.
4. If you need a `Session` instance with a particular state class,
   use `runtime.storage.session_store.SessionStore.create(...)`
   over a tmp_path engine (see `_make_repo` patterns in existing
   tests).
5. Run the new test with `-xvs` to iterate; then `-x` for the full
   suite to catch regressions.
