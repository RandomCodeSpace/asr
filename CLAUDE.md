# CLAUDE.md — project context for AI agents

> Loaded automatically by Claude Code (and equivalent agents) for
> every session in this repo. Companion to
> [`docs/11-agent-handoff.md`](docs/11-agent-handoff.md), which has
> the longer "action card" format with explanations.

## What this project is

Generic Python multi-agent runtime framework on **LangGraph**
(orchestration) + **LangChain** (provider + agent factory) +
**FastMCP** (tools). Single-file deploy bundle for air-gapped
corporate environments. Two reference apps in `examples/`:
`incident_management` (flagship) and `code_review` (proves the
framework is generic).

`main` is at v1.5 (see [`docs/DESIGN.md`](docs/DESIGN.md) § 13 for
milestone history).

## Read these first

In order:
1. [`docs/DESIGN.md`](docs/DESIGN.md) — long-form architecture +
   12 numbered DEC-NNN decisions + milestone history
2. [`docs/11-agent-handoff.md`](docs/11-agent-handoff.md) — top
   20 files to read, command allowlist / denylist, common traps
3. [`docs/02-architecture.md`](docs/02-architecture.md) —
   quick-scan layered diagram
4. [`docs/04-main-flows.md`](docs/04-main-flows.md) — entry points
   + failure modes per flow

## Always-on commands

```bash
# install / sync deps (uses uv.lock)
uv sync --frozen --extra dev

# tests (full)
uv run pytest -x

# tests (single file, fast)
uv run pytest tests/<file>.py -xvs --no-cov

# lint + type check + ratchets
uv run ruff check src/ tests/
uv run pyright src/runtime
python scripts/check_genericity.py
uv run python scripts/lint_skill_prompts.py

# regenerate single-file bundle (REQUIRED after touching src/runtime/ or examples/)
uv run python scripts/build_single_file.py

# coverage gate
uv run pytest --cov=src/runtime --cov-fail-under=85 -x
```

## DO

- Use `uv run pytest …` (NOT bare `pytest`) — pythonpath is in
  `pyproject.toml`.
- Regenerate `dist/*` after ANY change to `src/runtime/` or
  `examples/`. CI's "Bundle staleness gate (HARD-08)" fails
  otherwise.
- Run `uv lock` and commit `uv.lock` if you change `pyproject.toml`.
  CI's "Lockfile freshness gate (HARD-02)" fails otherwise.
- Work on a feature branch, open a PR, squash-merge.
  Conventional-commit subjects: `feat(area): …`, `fix(area): …`,
  `refactor(area): …`, `docs: …`, `build: …`, `chore(area): …`.
- Use `extra_fields` JSON for app-specific fields. Do NOT add
  app-specific columns to `IncidentRow`.
- Use stub LLMs (`LLMConfig.stub()` + `EnvelopeStubChatModel` from
  `tests/_envelope_helpers.py`) in tests. Live LLM tests are
  env-gated.
- Re-read [`docs/DESIGN.md`](docs/DESIGN.md) § 12 (decision log)
  before any architectural change.

## DO NOT

- Do NOT `pip install …` — bypasses uv lockfile. Use `uv add` +
  `uv sync`.
- Do NOT edit `dist/*` directly — they're generated.
- Do NOT add `TODO`/`FIXME`/`HACK` comments — fix root cause or
  open an issue. The only intentional `TODO(v2)` is in
  `src/runtime/locks.py:49` (slot eviction; documented).
- Do NOT add `except Exception: pass` — Phase 18 / HARD-04
  removed all of these. Log + re-raise or catch a typed exception.
- Do NOT touch SQLAlchemy column names on `IncidentRow` —
  destructive migration. Add to `extra_fields` instead.
- Do NOT commit anything in `.planning/` — gitignored;
  local-only working state for the GSD planning workflow.
- Do NOT commit agent-generated `*.md` outside `docs/` unless
  the user explicitly asks them to ship. `docs/*` is gitignored
  except for the explicit allowlist in `.gitignore`.
- Do NOT call live LLM providers in CI tests — keys are dummy in
  `.github/workflows/ci.yml`.
- Do NOT introduce a public-internet runtime dependency in
  `src/runtime/`. Air-gap is the deploy target. The hardcoded
  `https://ollama.com` fallback was explicitly removed in
  Phase 13 (HARD-05); don't re-introduce.
- Do NOT force-push or rewrite history on `main` (or any branch
  with collaborators). PRs only.
- Do NOT skip the bundle regeneration step ("I'll do it before
  PR" leads to CI fails and time wasted on rebases).
- Do NOT bypass the concept-leak ratchet by raising
  `BASELINE_TOTAL` without a rationale entry. Lowering it is
  encouraged; raising requires architectural justification in the
  commit message.

## Architectural rules (load-bearing)

See [`docs/11-agent-handoff.md`](docs/11-agent-handoff.md) §
"Architectural rules" for the 8 rules. Quick recap:

1. Framework stays domain-agnostic
2. One source of truth per concern (`should_gate`, `should_retry`,
   `_finalize_session_status`)
3. HITL pause is NOT an error
4. Append-only audit trails
5. The bundle is the deploy unit
6. Provider abstraction stays in `runtime.llm`
7. Tests use stubs by default
8. No public-internet runtime calls in air-gap path

## Common traps (skim before debugging)

- `pytest` (bare) → `ModuleNotFoundError: runtime`. Use `uv run pytest …`.
- Touching `src/` without regenerating `dist/` → CI bundle gate fails.
- Approving a HITL session created on pre-PR-#6 code → silent no-op.
  Tell the user to start a fresh session.
- Live OpenRouter `:free` model 429s on first call → retry usually
  works (v1.5-D 429 backoff is 7.5s/15s/22.5s).
- Streamlit `AssertionError: scope["type"] == "http"` storm under
  Python 3.14 → cosmetic Starlette compat bug; HTTP traffic still
  works.

## Repo conventions

- **Branches:** `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`,
  `build/`. Squash-merge into `main`.
- **Commits:** Conventional Commits style. Verbose body with the
  "why" + key file references when non-trivial.
- **PRs:** Use `gh pr create` with title + body; CI runs lint /
  type / test / sonar / bundle / skill-lint. Squash-merge with
  `gh pr merge <n> --squash --delete-branch --subject "…"`.
- **Tests:** `tests/test_*.py`. Async tests need no decorator
  (`asyncio_mode=auto`). Stub LLMs from `tests/_envelope_helpers.py`.
- **Coverage:** ≥ 85% on `src/runtime/`. UI / `__main__` /
  postgres saver / plugin transport are excluded
  (`pyproject.toml:[tool.coverage.run].omit`).
- **Type-checker:** pyright fail-on-error (Phase 19 / HARD-03);
  use `# pyright: ignore[<rule>] -- <rationale>` for legitimate
  stub gaps.
- **Skill prompts:** `examples/<app>/skills/<name>/{config.yaml, system.md}`.
  Must include the markdown turn-output contract block (see
  `_common/output.md`).

## Worktree workflow

This repo is set up for parallel-agent worktrees under
`.claude/worktrees/`. If you're given the EnterWorktree tool:

- Use it BEFORE making any code changes — keeps the user's main
  checkout clean.
- After CI passes and the PR merges, ExitWorktree with
  `action=remove, discard_changes=true` (the squashed commits are
  on `main`; the original SHAs are dropped, content is preserved).

If you're not given EnterWorktree, work in the main checkout but
let the user know.

## Current state snapshot (as of last update)

- Tests: 1265 passing, 8 skipped
- Coverage: 87.04%
- Concept-leak ratchet: 39 (down from 156 pre-v1.5-B)
- Ruff: clean
- SonarCloud quality gate: green
- Latest milestone: v1.5 (markdown turn output + HITL fix +
  generic-noun pass + per-agent LLM + 429 retry)
- Next big move: v2.0 React UI (Streamlit retirement)

## Where to find what

| You want to … | Read |
|---|---|
| Understand the architecture | [`docs/DESIGN.md`](docs/DESIGN.md), [`docs/02-architecture.md`](docs/02-architecture.md) |
| Local setup | [`docs/01-local-setup.md`](docs/01-local-setup.md) |
| Find a file by purpose | [`docs/03-code-map.md`](docs/03-code-map.md) |
| Understand a flow end-to-end | [`docs/04-main-flows.md`](docs/04-main-flows.md) |
| Configure deployment | [`docs/05-configuration.md`](docs/05-configuration.md) |
| Inspect storage / data | [`docs/06-data-model.md`](docs/06-data-model.md) |
| External integrations | [`docs/07-integrations.md`](docs/07-integrations.md) |
| Run / write tests | [`docs/08-testing.md`](docs/08-testing.md) |
| Build / deploy / release | [`docs/09-build-deploy-release.md`](docs/09-build-deploy-release.md) |
| Risk / debt inventory | [`docs/10-known-risks-and-todos.md`](docs/10-known-risks-and-todos.md) |
| Action card for AI agents | [`docs/11-agent-handoff.md`](docs/11-agent-handoff.md) |
| Architectural baseline | [`docs/adr/0001-current-architecture.md`](docs/adr/0001-current-architecture.md) |
| Dev workflow (regenerate dist, add module) | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Air-gap install | [`docs/AIRGAP_INSTALL.md`](docs/AIRGAP_INSTALL.md) |
