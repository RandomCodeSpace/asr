# 11 — Agent handoff

> Designed for AI coding agents picking this project up cold. If
> you're a human, this works for you too.

## Project summary in 20 lines

ASR is a generic Python multi-agent runtime framework. It wraps
**LangGraph** (orchestration / checkpointing) and **LangChain**
(`langchain.agents.create_agent` for the per-agent loop;
`Chat{OpenAI,Ollama}` and `AzureChatOpenAI` for provider abstraction).
Tools come from **FastMCP** servers (in-process / stdio / http).
A risk-rated **HITL gateway** wraps every tool — high-risk calls
raise `langgraph.types.interrupt(payload)` to pause the graph for
operator approval; resume via `Command(resume=verdict)`. Agent
output uses a **markdown contract block** (`## Response / ##
Confidence / ## Signal`) parsed by a 6-path lenient parser with
synthesis fallbacks for misbehaving models.

Two reference apps live in `examples/`: `incident_management` (4-skill
SRE investigation pipeline with ASR memory layers) and `code_review`
(3-skill PR review pipeline; mocked tools). Apps subclass `Session`
to add domain fields; the framework stays generic — a CI ratchet
(`tests/test_genericity_ratchet.py`) keeps it that way.

The deploy target is air-gapped corporate environments. The deploy
artifact is a single-file bundle under `dist/` (not a wheel) plus a
handful of YAML configs and `.env`. The bundler script
(`scripts/build_single_file.py`) flattens `src/runtime` + an example
app into one `.py` file; CI's "Bundle staleness gate" rebuilds on
every PR and refuses the merge if `dist/` would change.

`main` is at v1.5; 1265 tests passing; 87% coverage; ruff clean;
SonarCloud green; concept-leak ratchet at 39. v2.0 (React UI
replacing the Streamlit prototype) is the next big move.

## Top 20 files to read first

In order — each builds on the previous.

1. **`README.md`** — repo intro + quick start
2. **`docs/DESIGN.md`** — long-form architecture + decision log
   (12 numbered DEC-NNN entries) + milestone history (v1.0 → v1.5)
3. **`docs/02-architecture.md`** — quick-scan summary of the layers
4. **`pyproject.toml`** — deps, pytest/ruff/pyright/coverage config
5. **`config/config.yaml.example`** — annotated config template
6. **`src/runtime/state.py`** — `Session`, `AgentRun`, `ToolCall`,
   `TokenUsage` pydantic models
7. **`src/runtime/skill.py`** — `Skill` (YAML-driven agent declaration)
8. **`src/runtime/orchestrator.py`** — `Orchestrator` class + lifecycle
   methods (`start_session`, `stream_session`, `resume_session`,
   `_finalize_session_status_async`, `_is_graph_paused`)
9. **`src/runtime/service.py`** — `OrchestratorService` long-lived
   loop wrapper + thread-safe bridge
10. **`src/runtime/graph.py`** — `build_graph`, `make_agent_node`,
    `_drive_agent_with_resume`, `_ainvoke_with_retry`,
    `parse_envelope_from_result` callers
11. **`src/runtime/agents/turn_output.py`** — markdown envelope
    parser, 6-path fallback chain
12. **`src/runtime/tools/gateway.py`** — `wrap_tool` (~830 LOC) —
    risk-rated tool wrapper with HITL pause/resume
13. **`src/runtime/llm.py`** — `get_llm` provider abstraction
14. **`src/runtime/storage/session_store.py`** — CRUD + FAISS
    write-through + optimistic-version save
15. **`src/runtime/api.py`** — FastAPI `/sessions/*` REST + SSE +
    WebSocket + approvals
16. **`examples/incident_management/state.py`** — example
    `IncidentState(Session)` subclass
17. **`examples/incident_management/mcp_server.py`** — example MCP
    server pattern
18. **`tests/test_interrupt_detection.py`** — proves the HITL fix
    end-to-end (read this for the resume contract)
19. **`scripts/build_single_file.py`** — the bundler (the deploy
    pipeline)
20. **`.github/workflows/ci.yml`** — CI gates (lint / type / test /
    sonar / bundle / skill-lint)

## Commands future agents SHOULD use

| Goal | Command |
|---|---|
| Install / sync deps | `uv sync --frozen --extra dev` |
| Run full test suite | `uv run pytest -x` |
| Run single test fast | `uv run pytest tests/<file>.py::<test_name> -xvs --no-cov` |
| Lint | `uv run ruff check src/ tests/` |
| Type check | `uv run pyright src/runtime` |
| Coverage gate | `uv run pytest --cov=src/runtime --cov-fail-under=85 -x` |
| Regenerate single-file bundle | `uv run python scripts/build_single_file.py` |
| Concept-leak ratchet check | `python scripts/check_genericity.py` |
| Skill-prompt linter | `uv run python scripts/lint_skill_prompts.py` |
| Lockfile freshness | `uv lock --check` |
| Boot CLI | `uv run python -m runtime --config config/incident_management.yaml` |
| Boot Streamlit UI | `ASR_LOG_LEVEL=INFO uv run streamlit run src/runtime/ui.py --server.port 37777` |
| Reset local state | `rm /tmp/asr.db /tmp/asr.db-*; rm -rf /tmp/asr-faiss` |
| Inspect session events | `sqlite3 /tmp/asr.db "SELECT kind, datetime(ts), substr(payload,1,200) FROM session_events WHERE session_id='<sid>' ORDER BY ts;"` |
| Inspect a session row | `sqlite3 /tmp/asr.db "SELECT id, status, version FROM incidents WHERE id='<sid>';"` |
| Live integration smoke | `OLLAMA_API_KEY=… OLLAMA_BASE_URL=https://ollama.com uv run pytest tests/test_integration_driver_s1.py -v` |
| Open a PR | `gh pr create --base main --head <branch> --title "…" --body "…"` |
| Watch CI | `gh pr checks <pr_number> --watch` |
| Squash merge | `gh pr merge <pr_number> --squash --delete-branch --subject "…"` |

## Commands future agents SHOULD AVOID

| Avoid | Why | Use instead |
|---|---|---|
| `pip install …` | Bypasses uv lockfile; CI's "Lockfile freshness gate" will fail | `uv add <pkg>` then `uv sync` |
| `pytest …` (bare) | Doesn't pick up `pythonpath` from `pyproject.toml` | `uv run pytest …` |
| Editing `dist/*` directly | Bundles are generated; hand-edits get clobbered + CI's "Bundle staleness gate" fails | Edit `src/runtime/` or `examples/`, regenerate via `scripts/build_single_file.py` |
| `git commit` without bundle regen after touching `src/runtime/` or `examples/` | CI's bundle gate fails | Run `scripts/build_single_file.py`, `git add dist/` |
| `git push --force` to `main` (or any shared branch) | Rewrites history for everyone | Use a feature branch + PR |
| `git push origin --delete <branch>` for branches you didn't create | Destructive on shared state | Confirm with the owner |
| Adding a `TODO` to source | Project rule is "fix root cause, not workaround"; the only `TODO(v2)` in the repo is intentional | Open an issue or write the fix |
| Adding `except Exception: pass` | Phase 18 (HARD-04) explicitly removed all of these | Log + re-raise, or catch a typed exception |
| Touching schema columns on `IncidentRow` | Requires a migration; v1.5-B (DEC-008) explicitly left the incident-shaped columns alone | Use `extra_fields` JSON for app-specific data |
| Calling live LLM providers in tests | CI uses dummy keys; live tests are env-gated and skipped | Use `LLMConfig.stub()` + `EnvelopeStubChatModel` |
| Renaming `incident` → `session` in source code without bumping the ratchet test | `tests/test_genericity_ratchet.py` enforces the count downward only | Update `BASELINE_TOTAL` in the same commit with rationale comment (see history at `tests/test_genericity_ratchet.py:60-86`) |
| Writing agent-generated `*.md` outside `docs/` and committing | `docs/*` is gitignored except for explicit allowlist | Add to the allowlist in `.gitignore` if it's a real deliverable; otherwise keep it local |

## Architectural rules

These are **load-bearing** — if you're tempted to violate one, stop
and re-read `docs/DESIGN.md` § 12 (decision log).

1. **The framework stays domain-agnostic.** Apps subclass `Session`
   for domain data; framework code references `Session` and
   `extra_fields`, never app-specific fields. The concept-leak
   ratchet enforces this on `incident` / `severity` / `reporter`
   tokens.
2. **One source of truth per concern.** Gate decisions:
   `policy.should_gate`. Retry policy: `policy.should_retry`.
   Status finalization: `_finalize_session_status`. Don't reimplement.
3. **HITL pause is NOT an error.** `GraphInterrupt` and the
   `__interrupt__` field on the result dict signal a checkpointed
   pending_approval, not a failure. `_handle_agent_failure` must NOT
   fire; finalize must NOT run while paused. See PR #6.
4. **Append-only audit trails.** `agents_run`, `tool_calls`,
   `session_events` are never updated in place (the gateway's
   per-row pending→approved transition IS in-place but is the only
   exception, and it persists via `_record_pending_resolution`).
5. **The bundle is the deploy unit.** `dist/*` is regenerated, not
   hand-edited. Every PR touching `src/runtime/` or `examples/`
   commits a fresh bundle.
6. **Provider abstraction stays in `src/runtime/llm.py`.** Apps
   declare provider config; the framework owns the provider class
   selection (`langchain_openai.ChatOpenAI` vs
   `langchain_openai.AzureChatOpenAI` vs `langchain_ollama.ChatOllama`).
7. **Tests use stubs by default.** Live LLM tests are env-gated;
   the suite must run cleanly in CI without any provider keys.
8. **No public-internet calls at deploy time.** Air-gap is the
   target. The `https://ollama.com` hardcoded fallback was
   explicitly removed in Phase 13 (HARD-05); don't re-introduce.

## Coding conventions

| Convention | Example |
|---|---|
| Pydantic v2 BaseModel for every config / state | `src/runtime/state.py:Session` |
| Async first; sync wrappers as needed | `OrchestratorService.submit_async` is async; `submit_and_wait` wraps for sync callers |
| Type-hint everything; pyright fail-on-error gate | `src/runtime/graph.py` |
| Skill prompts as `system.md` not Python strings | `examples/*/skills/<name>/system.md` |
| Tools registered via `@mcp.tool()` decorator on FastMCP server | `examples/incident_management/mcp_server.py` |
| Per-line `# pyright: ignore[<rule>] -- <rationale>` for legitimate stub gaps | `src/runtime/orchestrator.py` (multiple) |
| String constants for envelope keys / status values | Avoid bare strings — use `runtime.state.ToolStatus` Literal or named constants |
| `_private_helper(*, kw=…)` for keyword-only args inside the framework | `src/runtime/graph.py:make_agent_node` |
| Test files mirror source: `src/runtime/X.py` → `tests/test_X.py` | Most do; some are topical (`test_interrupt_detection.py` ≠ one source file) |
| Conventional-commit subjects | `feat(retry): 429 rate-limit retry…`, `fix(hitl): …`, `refactor(v1.5-B): …`, `docs: …`, `build: …`, `chore(config): …` |
| Atomic commits per logical change; squash-merge into main | git history shows the pattern |

## Common traps

1. **`pytest` (bare) doesn't pick up the `pythonpath`** → `ModuleNotFoundError: runtime`. Use `uv run pytest …`.
2. **Touching `src/runtime/` or `examples/` without regenerating `dist/`** → CI bundle gate fails. Always run `uv run python scripts/build_single_file.py && git add dist/` before committing.
3. **Adding a kwarg to a framework function without checking callers** → `incident=` rename in v1.5-B caught the example app's `_record_success_run(incident=…)` call. Run `git grep -nE "<func>\\("` before any signature change.
4. **Approving a HITL session that was created on pre-PR-#6 code** → that session's checkpoint is poisoned (langgraph 1.x semantic mismatch). The Approve button silently no-ops. Tell the user to start a fresh session.
5. **Live OpenRouter `:free` model rate limits** → first call may 429. The v1.5-D 429 retry (7.5s/15s/22.5s) clears most short-window throttles; persistent 429 means quota exhaustion.
6. **Azure connection error** → check `.env` `AZURE_ENDPOINT` is a real URL, not a placeholder like `noop`.
7. **Pyright complains about langchain stubs** → use `# pyright: ignore[<rule>] -- <rationale>` per line; don't disable the gate.
8. **Streamlit `AssertionError: scope["type"] == "http"` storm under Python 3.14** → cosmetic Starlette compat bug; HTTP traffic still works. Filter logs.
9. **`StaleVersionError` on HITL resume** → was a real bug pre-PR-#6 (stale `state["session"]`); now mitigated by `make_agent_node` reload-on-entry. If you see it again, check whether you accidentally bypassed the reload.
10. **Two ToolCall rows for one apply_fix** → known cosmetic duplication (gateway colon-form vs harvester `__`-form). Documented as a small follow-up.

## Current unfinished work

From `docs/00-project-overview.md` § "What's next" and
`docs/10-known-risks-and-todos.md`:

| Item | Effort | Priority |
|---|---|---|
| **v2.0 — React UI** replacing Streamlit; parity-port against `/sessions/*` API | ~1–2 weeks | High |
| Duplicate ToolCall audit rows (gateway colon vs harvester `__`) | ~30 min | Low (cosmetic) |
| `ApprovalWatchdog` regression test (covers PR #6 saves) | ~15 min | Medium |
| `ASR_LOG_LEVEL` env var doc in main README | ~5 min | Low |
| `src/runtime/locks.py:49` — `TODO(v2)` slot eviction | ~1-2h | Low (relevant for long-running servers) |

**Environment-side (operator, not framework):**

- OpenRouter `workhorse` returns 402 on paid models — out of credits
- Azure live verification needs a real `AZURE_ENDPOINT` (`.env` placeholder)

## Recommended next tasks

In order of value × effort:

1. **Update `.planning/STATE.md` + `.planning/ROADMAP.md`** (gitignored,
   local) to reflect v1.5 fully shipped. ~5 min.
2. **Land the smaller cleanups together as a single "v1.5 polish" PR**:
   `ApprovalWatchdog` test + duplicate ToolCall fix + `ASR_LOG_LEVEL`
   doc. ~1h total. Closes the loop on v1.5.
3. **Brainstorm v2.0 React UI** — invoke `superpowers:brainstorming`.
   Stack pick (Next.js / Vite + React / Remix?), state management,
   API client codegen from `/sessions/*` OpenAPI?
4. **Scaffold v2.0 React UI** in a new top-level `web/` directory.
   Don't touch `src/runtime/` until the parity-port surfaces a real
   missing API.
5. **Build a multi-agent live driver** that runs intake → triage →
   resolution against a real provider end-to-end. Catch provider-quirk
   regressions earlier than the single-agent S1 driver.
6. **Postgres CI smoke** — one test against a postgres container so
   the optional checkpointer doesn't drift unnoticed.

## Where DESIGN.md and this handoff differ

`docs/DESIGN.md` is the **prose narrative** — read it once, top-to-
bottom, to build the mental model. This handoff is the **action card**
— skim it at the start of each new session to remember what to do
and what to avoid.

The 12 numbered files in this `docs/` directory (00 through 11) are
the **per-topic reference**: jump to whichever one matches your
current question.
