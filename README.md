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

# Streamlit UI
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
  air-gapped / internal-mirror install procedure.

## Status

`main` carries v1.0 → v1.5. v2.0 (React UI replacing the Streamlit
prototype) is the next big move. See `docs/DESIGN.md` § 13 for the
milestone history and § 14 for the pending list.
