# 01 — Local setup

## Prerequisites

- **Python 3.11** (`pyproject.toml:7` requires `>=3.11`; `pyrightconfig` /
  CI also pin 3.11). The dev environment in this repo runs Python
  3.13 / 3.14 successfully — Inference: 3.11 is the *floor*, newer
  3.x versions work in practice.
- **`uv`** package manager `>= 0.11.7` (CI pins this exact version
  in `.github/workflows/ci.yml`). Install via `pipx install uv` or
  the `uv` binary; do not `curl | sh`.
- **git** (for branch / PR workflow).
- **Optional, for live LLM smoke**: provider API keys —
  `OLLAMA_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_KEY` (+
  `AZURE_ENDPOINT`, `AZURE_DEPLOYMENT`). Stub-mode tests do NOT
  need any keys.
- **Optional, for postgres deployments**: install
  `pip install asr[postgres]` to pull `langgraph-checkpoint-postgres`
  and `psycopg-pool`. SQLite is the default and CI-tested path.

## Install

From a clean checkout:

```bash
git clone <repo-url>
cd asr
uv sync --frozen --extra dev
```

`--frozen` forbids re-resolving — installs the exact set pinned in
`uv.lock` with hash verification (HARD-02 reproducibility gate). For
fully air-gapped install with an internal mirror, see
[`docs/AIRGAP_INSTALL.md`](AIRGAP_INSTALL.md).

`--extra dev` pulls test runner, type checker, and linters per
`pyproject.toml:42-50`.

## Run

Two entry points share the same orchestrator service.

### CLI / API

```bash
uv run python -m runtime --config config/incident_management.yaml
```

Boots the long-lived `OrchestratorService` and FastAPI surface (`/sessions/*`
REST, SSE, WebSocket). Source: `src/runtime/__main__.py`.

### Streamlit UI

```bash
ASR_LOG_LEVEL=INFO uv run streamlit run src/runtime/ui.py --server.port 37777
```

`ASR_LOG_LEVEL` env var enables structured logs at the chosen level
(`DEBUG` / `INFO` / `WARNING` / `ERROR`). Source:
`src/runtime/ui.py:46-65` (`_maybe_configure_logging`).

The UI binds to the same `OrchestratorService` instance as the CLI;
both can run in the same process (Streamlit script imports the
service lazily on first session).

## Test

```bash
# Full suite
uv run pytest -x

# Without coverage (faster)
uv run pytest -x --no-cov

# A single file or test
uv run pytest tests/test_interrupt_detection.py -x -v

# With coverage gate (fails below 85%)
uv run pytest --cov=src/runtime --cov-fail-under=85 -x
```

Pytest config: `pyproject.toml:53-58` — `asyncio_mode = "auto"`,
`testpaths = ["tests"]`, `pythonpath = ["src", "."]`.

Coverage omits: `src/runtime/ui.py`,
`src/runtime/__main__.py`, `src/runtime/checkpointer_postgres.py`,
`src/runtime/triggers/transports/plugin.py`
(`pyproject.toml:71-76`).

## Lint + type check

```bash
uv run ruff check src/ tests/
uv run pyright src/runtime
```

CI runs both with `fail-on-error`.

## Concept-leak ratchet

```bash
python scripts/check_genericity.py            # current count
python scripts/check_genericity.py --baseline 39  # exit non-zero if exceeded
```

Enforced by `tests/test_genericity_ratchet.py` — the count must stay
at or below `BASELINE_TOTAL` (currently 39).

## Bundle regeneration

After ANY change to `src/runtime/` or `examples/*/`:

```bash
uv run python scripts/build_single_file.py
git add dist/
```

CI's "Bundle staleness gate (HARD-08)" rebuilds and fails the build
if `dist/*` doesn't match. See `docs/DEVELOPMENT.md` for the full
flow.

## Required services

Default config uses local-only services:

- **SQLite** at `/tmp/asr.db` (auto-created on first run)
- **FAISS** vector index at `/tmp/asr-faiss/` (auto-created)
- **Ollama Cloud** (when `llm.default` points there) — needs
  `OLLAMA_API_KEY`

To start fresh after testing:
```bash
rm /tmp/asr.db /tmp/asr.db-wal /tmp/asr.db-shm
rm -rf /tmp/asr-faiss
```

## Common setup issues

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: runtime` when running tests | `pythonpath` not picked up | Run via `uv run pytest …` (NOT bare `pytest`); pytest reads `[tool.pytest.ini_options].pythonpath` from `pyproject.toml` |
| CI fails "Lockfile freshness gate" | `pyproject.toml` changed without `uv lock` | Run `uv lock` and commit `uv.lock` |
| CI fails "Bundle staleness gate" | `src/runtime/` or `examples/*/` changed without `dist/` regen | Run `uv run python scripts/build_single_file.py` and commit `dist/*` |
| Live LLM tests fail with `Error code: 402` (OpenRouter) | Account out of credits | Switch `llm.default` to `gpt_oss` (Ollama) or another model in `config/config.yaml` |
| Live LLM tests fail with `Connection error` (Azure) | `.env` `AZURE_ENDPOINT` is a placeholder or unreachable | Set a real Azure endpoint, or skip Azure leg (test gates on `AZURE_OPENAI_KEY` + `AZURE_ENDPOINT` per `tests/test_integration_driver_s1.py`) |
| Streamlit dies on every Python 3.14 WebSocket request with `AssertionError: scope["type"] == "http"` | Streamlit ≤ x + Starlette static-files compat bug under Python 3.14 | Cosmetic — HTTP traffic still works. Filter logs with `grep -v "AssertionError\|scope.*type"`. Inference: fixed in newer Streamlit; not yet retested. |
| `gpt-oss:20b` returns errors / no envelope | Model dropped the markdown contract | Path 6 permissive synthesis (`turn_output.py`) emits a 0.30-confidence placeholder so the session still finalizes; retry the session for a real envelope |

## Environment variables

| Var | Required when | Notes |
|---|---|---|
| `OLLAMA_API_KEY` | `ollama_cloud` provider used | `config/config.yaml` references via `${OLLAMA_API_KEY}` |
| `OPENROUTER_API_KEY` | OpenRouter provider used | |
| `AZURE_OPENAI_KEY` | Azure provider used | |
| `AZURE_ENDPOINT` | Azure provider used | Full URL incl. trailing `/` |
| `AZURE_DEPLOYMENT` | Azure provider used | Defaults to `gpt-4o` in test driver |
| `EXTERNAL_MCP_URL` | external HTTP MCP server configured | (see `tests/fixtures/sample_config.yaml`) |
| `EXT_TOKEN` | external HTTP MCP server with bearer auth | |
| `ASR_LOG_LEVEL` | optional | `DEBUG` / `INFO` / `WARNING` / `ERROR`; UI uses it via `force=True` basicConfig |
| `APP_CONFIG` | optional | overrides default `config/config.yaml` path; read by `src/runtime/ui.py:68` |
| `OLLAMA_LIVE` | optional | gates live-LLM smoke tests in `tests/test_llm_providers_smoke.py` |
| `OLLAMA_BASE_URL` | required for `tests/test_integration_driver_s1.py` | Typically `https://ollama.com` for cloud or `http://localhost:11434` for local |

CI uses dummy values for the API keys (see `ci.yml` — they only need
to *exist* for the strict-mode `_interpolate` check; tests don't call
live providers).
