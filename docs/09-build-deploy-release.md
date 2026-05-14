# 09 — Build / deploy / release

## Build commands

| Step | Command | Source |
|---|---|---|
| Install dependencies (frozen, hash-verified) | `uv sync --frozen --extra dev` | `uv.lock`, `pyproject.toml:42-50` |
| Regenerate single-file bundle | `uv run python scripts/build_single_file.py` | `scripts/build_single_file.py` |
| Lint | `uv run ruff check src/ tests/` | |
| Type-check | `uv run pyright src/runtime` | `pyrightconfig.json` |
| Test + coverage | `uv run pytest --cov=src/runtime --cov-fail-under=85` | `pyproject.toml:53-58` |
| Skill-prompt linter | `uv run python scripts/lint_skill_prompts.py` | |
| Concept-leak ratchet | `uv run python scripts/check_genericity.py --baseline 39` | |
| Lockfile freshness | `uv lock --check` | |

The "build" of this project is **not a wheel** — wheels exist
(`pyproject.toml:[tool.hatch.build.targets.wheel]` declares
`packages = ["src/runtime", "examples"]`) but the deployed artifact
is the **single-file bundle** under `dist/`. Wheels are useful for
local `pip install -e .` development; the deployed shape is
copy-only.

## Packaging — the bundler

Source: `scripts/build_single_file.py`. Runs in three steps:

1. Read `RUNTIME_MODULE_ORDER` (a list of `(root, relpath)` tuples
   topologically ordered so each module's body sees its
   dependencies' symbols already in scope).
2. For each module: read source, strip intra-bundle imports
   (the bundle is one big namespace — `from runtime.config import X`
   becomes a no-op when `X` is already defined above).
3. Concatenate + emit four bundles:

| Output | Contents |
|---|---|
| `dist/app.py` (~660KB) | Framework only. Used to demonstrate the runtime stands on its own. |
| `dist/apps/incident-management.py` (~707KB) | Framework + `incident_management` example. The deployment ship target for the incident app. |
| `dist/apps/code-review.py` (~670KB) | Framework + `code_review` example. The second example, demonstrating genericity. |
| `dist/ui.py` (~68KB) | Streamlit shell. Sits next to whichever `app.py` you deployed and `from app import …` reaches into the deploy bundle's flattened namespace. |

The bundler also runs an `ast.parse` smoke on each output so a
broken bundle fails the script (rather than failing at deploy).

## CI/CD

Source: `.github/workflows/ci.yml`.

Single workflow `quality:` runs on every push to `main` and on every
PR. Steps:

```
checkout (fetch-depth: 0 for SonarCloud blame)
  ↓
setup-python @ 3.11
  ↓
setup-uv @ 0.11.7
  ↓
Lockfile freshness gate (uv lock --check)            # HARD-02
  ↓
Install deps (uv sync --frozen --extra dev)
  ↓
Bundle staleness gate (build + git diff --exit-code dist/)  # HARD-08
  ↓
Lint (ruff check src/ tests/)
  ↓
Type check (pyright src/runtime)                     # HARD-03 fail-on-error
  ↓
Test with coverage (pytest --cov= --cov-report=xml --junitxml=junit.xml)
  ↓
Skill-prompt-vs-schema lint (lint_skill_prompts.py)  # SKILL-LINTER-01
  ↓
SonarCloud Scan
```

Total CI time: ~2-3 minutes (most spent in test suite).

CI environment variables (dummy values for the
`_interpolate` strict check; tests don't call live providers):
- `OLLAMA_API_KEY=""`
- `OPENROUTER_API_KEY=""`
- `AZURE_OPENAI_KEY=""`
- `AZURE_DEPLOYMENT=""`
- `AZURE_ENDPOINT=https://ci-dummy.example/`
- `EXTERNAL_MCP_URL=https://ci-dummy.example/`
- `EXT_TOKEN=ci-dummy`

## Quality gates

Beyond CI's pass/fail, these soft gates guide PR review:

| Gate | Source | Threshold |
|---|---|---|
| Coverage | SonarCloud `new_coverage` | ≥ 80% on new code |
| Duplications | SonarCloud `new_duplicated_lines_density` | < 3% (with `sonar.cpd.exclusions` for intentional sync/async + responsive/graph mirrors) |
| Reliability | SonarCloud `new_reliability_rating` | A (=1) |
| Security | SonarCloud `new_security_rating` | A (=1) |
| Maintainability | SonarCloud `new_maintainability_rating` | A (=1) |
| Hotspots reviewed | SonarCloud `new_security_hotspots_reviewed` | 100% |
| Concept-leak ratchet | `tests/test_genericity_ratchet.py` | ≤ `BASELINE_TOTAL` (currently 39) |
| Bundle freshness | `tests/test_bundle_completeness.py` + CI gate | exit-code clean |
| Type errors | `pyright` fail-on-error | zero new errors |
| Lockfile drift | `uv lock --check` | clean |
| Skill prompts | `scripts/lint_skill_prompts.py` | binary pass |

## Containerisation

There is **no Dockerfile** in the repo (verified via
`find . -name Dockerfile`). Inference: the deploy target is bare-VM
or systemd, not container. A container deploy would need a
hand-rolled `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY dist/apps/incident-management.py app.py
COPY dist/ui.py ui.py
COPY config/ config/
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py", "--config", "config/incident_management.yaml"]
```

(Inference: above is illustrative; not tested in this repo.)

## Deployment model — air-gap copy

Source: `docs/AIRGAP_INSTALL.md`,
`docs/DEVELOPMENT.md`, `docs/DESIGN.md` § 10.

**The deploy target has NO public-internet access** at runtime. Two
phases:

### Phase A — install dependencies (one-time, on the dev/CI box or behind an internal mirror)

```bash
export UV_INDEX_URL="https://<internal-mirror>/simple/"
uv sync --frozen --extra dev          # populates ~/.cache/uv from the mirror
# or fully offline if the cache is pre-warmed:
uv sync --frozen --offline --extra dev
```

### Phase B — copy the 7-file payload onto the target host

```
app.py                    (renamed from dist/apps/<app>.py)
ui.py                     (dist/ui.py)
config/config.yaml        (framework: LLM, MCP, storage)
config/<app>.yaml         (app: severity aliases, escalation roster, …)
config/skills/            (optional skill prompt overrides)
.env                      (provider keys; secrets manager preferred)
```

### Phase C — boot

```bash
python -m runtime --config config/<app>.yaml &
streamlit run ui.py --server.port 37777 &
```

Or systemd units; or k8s `Pod`s. The framework doesn't care.

## Release flow

Source: git history + `docs/DESIGN.md` § 13.

The release pattern in this repo is **squash merge into `main`** via
GitHub PRs. Each milestone is a sequence of small PRs:

```
PR opened → CI runs (lint / type / test / sonar / bundle / skill-lint)
         → all green → squash merge with verbose subject
         → branch deleted
         → main moves to the squash SHA
```

There is **no separate release branch**, no semver tags, and no
release notes infrastructure. The "release" is `main` itself.

The milestone history (v1.0 → v1.5) is recorded in
`docs/DESIGN.md` § 13. New work goes on a feature branch (`feat/…`,
`fix/…`, `refactor/…`, `docs/…`); merge via PR.

## Rollback

Inference: not formally documented. Practical:

- **Code rollback** — `git revert <squash-sha>` and merge a revert
  PR. CI will re-run.
- **Bundle rollback** — copy the previous bundle from a known-good
  `main` commit; the deploy is copy-only so rolling back is just
  copying older files.
- **Schema rollback** — there's no Alembic. New columns / tables
  added via `Base.metadata.create_all` are forward-only;
  rolling back code that introduced a new column doesn't delete
  the column from the DB (harmless — old code ignores it). New
  rows in new tables are abandoned (also harmless).
- **Stuck session rollback** — operator can `DELETE /sessions/{sid}`
  (soft delete) or set `status='stopped'` via `stop_session(sid)`.

## Versioning

`pyproject.toml:8` declares `version = "0.1.0"`. The version has
not been bumped despite v1.0 → v1.5 of the **product** milestones —
Inference: the package version is independent of the milestone
labelling. There are no git tags pinning the milestones; the
squash SHAs in `docs/DESIGN.md` § 13 are the canonical reference.

## Operational concerns

- **Process lifecycle** — `OrchestratorService` runs a single
  asyncio loop on a background thread. SIGTERM cancels in-flight
  session tasks; the lifespan shutdown hook closes the FastMCP +
  SQLAlchemy + checkpointer transports.
- **Session capacity** — `runtime.max_concurrent_sessions: 8`
  (default); raises `SessionBusy → HTTP 429` on overflow.
- **Long-running approval** — `framework.approval_timeout` (default
  Inference: 1800 seconds) drives `ApprovalWatchdog`; sessions with
  pending approvals beyond that age get auto-resolved with
  `verdict=timeout`.
- **DB growth** — `EventLog` and `LessonStore` are append-only.
  No automatic pruning. Operators should periodically GC closed
  sessions via `delete_session(sid)` (soft delete) or run a
  manual VACUUM on SQLite. Inference: not documented; needs a
  runbook.
- **FAISS index growth** — vectors are written through on every
  save and removed on `delete_session`. The index size scales
  linearly with active sessions.
