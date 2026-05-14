# ASR — Multi-Agent Runtime Framework

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
