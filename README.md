# ASR — Multi-Agent Runtime Framework

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.x-purple?style=for-the-badge)](https://github.com/jlowin/fastmcp)
[![CI](https://img.shields.io/github/actions/workflow/status/RandomCodeSpace/asr/ci.yml?branch=main&style=for-the-badge&logo=github)](https://github.com/RandomCodeSpace/asr/actions/workflows/ci.yml)
[![Quality Gate](https://img.shields.io/sonar/quality_gate/RandomCodeSpace_asr?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud)](https://sonarcloud.io/project/overview?id=RandomCodeSpace_asr)
[![Coverage](https://img.shields.io/sonar/coverage/RandomCodeSpace_asr?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud)](https://sonarcloud.io/component_measures?id=RandomCodeSpace_asr&metric=coverage)
[![Tests](https://img.shields.io/badge/tests-1265%20passing-brightgreen?style=for-the-badge)](https://github.com/RandomCodeSpace/asr/actions)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?style=for-the-badge&logo=ruff)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/types-pyright-yellow?style=for-the-badge)](https://github.com/microsoft/pyright)

Python multi-agent runtime built on **LangGraph** (orchestration) +
**FastMCP** (tool dispatch), with HITL gate, markdown turn-output
contract, and a single-file deploy bundle for air-gapped corporate
targets.

Two reference apps live in the same repo to prove the runtime is
generic:

- **`examples/incident_management/`** — 4-skill investigation
  pipeline (intake → triage → deep_investigator → resolution) with
  ASR memory layers (Knowledge Graph, Release Context, Playbooks).
- **`examples/code_review/`** — 3-skill PR review pipeline (intake
  → analyzer → recommender).

## Quick start

```bash
uv sync --frozen --extra dev
uv run pytest tests/ -x

# Run the incident-management app via the CLI entrypoint
uv run python -m runtime --config config/incident_management.yaml

# Backend API (serves the React SPA at / when web/dist exists)
uv run uvicorn runtime.api:get_app --factory --port 8000

# React UI (Vite dev server; proxy /api/v1 -> :8000)
cd web && npm ci && npm run dev    # http://localhost:5173

# Production: build once, then the backend serves it at http://localhost:8000
cd web && npm run build && cd ..
uv run uvicorn runtime.api:get_app --factory --port 8000

# Legacy Streamlit UI (deprecated in v2; banner inside)
ASR_LOG_LEVEL=INFO uv run streamlit run src/runtime/ui.py --server.port 37777
```

Set provider keys in `.env` (`OLLAMA_API_KEY`, `OPENROUTER_API_KEY`,
`AZURE_OPENAI_KEY`, …) and switch `llm.default` /
`skill.model` overrides in `config/config.yaml`.

## Documentation

- **[`docs/DESIGN.md`](docs/DESIGN.md)** — architecture, core
  abstractions, runtime model, storage, deployment, decision log,
  milestone history. **Start here** if you're new to the codebase.
- **[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)** — day-to-day
  contributor loop: setup, regenerating `dist/`, adding a runtime
  module.
- **[`docs/AIRGAP_INSTALL.md`](docs/AIRGAP_INSTALL.md)** —
  air-gapped / internal-mirror install procedure (includes the v2
  React UI air-gap layout).
- **[`docs/RELEASE.md`](docs/RELEASE.md)** — cutting a release
  candidate; `npm run build` + `scripts/package_airgap.py` + git tag.

## Status

`main` carries v1.0 → v1.5. **v2.0.0-rc1** ships the React UI
(Vite + React 19 + TS + Tailwind v4 + Radix primitives) in `web/`
which replaces the Streamlit prototype. The legacy Streamlit UI
remains shippable behind a deprecation banner until parity is
verified (`docs/REACT_UI_PARITY.md`). See `docs/DESIGN.md` § 13
for the milestone history.
