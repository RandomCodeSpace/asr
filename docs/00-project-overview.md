# 00 — Project overview

## What it does

ASR is a generic Python multi-agent runtime framework that wraps
**LangGraph** (orchestration), **LangChain** (LLM provider
abstraction + agent factory), and **FastMCP** (tool dispatch). It
adds a risk-rated HITL gateway, a markdown turn-output contract,
per-step telemetry, an auto-learning lesson store, and a single-file
deploy bundle for air-gapped corporate targets.

Two reference apps live in the same repo to prove the runtime is
genuinely generic:

- **`examples/incident_management/`** — 4-skill investigation pipeline
  (intake → triage → deep_investigator → resolution) with ASR memory
  layers (L2 Knowledge Graph, L5 Release Context, L7 Playbook Store).
- **`examples/code_review/`** — 3-skill PR review pipeline (intake
  → analyzer → recommender). Built specifically to surface every
  framework leak that would have made the runtime
  incident-shaped — those leaks were lifted into the framework.

References: [`docs/DESIGN.md`](DESIGN.md), [`pyproject.toml`](../pyproject.toml).

## Target users

- **Operators** of internal SRE / on-call automation in regulated /
  air-gapped corporate environments. The deployment story is a
  copy-only 7-file payload (no `pip install` at deploy time, no
  runtime CDN/internet calls).
- **Application authors** building domain-specific agent apps on top
  of the framework. Add a folder under `examples/<your_app>/` with a
  `Session` subclass, MCP servers, and skill prompts.
- **Framework contributors** working on the `src/runtime/` layer.

## Core features

| Feature | Implemented in |
|---|---|
| LangGraph-driven multi-agent dispatch | `src/runtime/graph.py`, `src/runtime/agents/*.py` |
| LangChain-driven LLM provider abstraction (Ollama, Azure OpenAI, OpenAI-compat) | `src/runtime/llm.py` |
| FastMCP tool servers (in-process / stdio / http) | `src/runtime/mcp_loader.py` |
| Risk-rated HITL gateway with `interrupt()` / `Command(resume=…)` | `src/runtime/tools/gateway.py` |
| Markdown turn-output contract + 6-path parser + permissive fallback | `src/runtime/agents/turn_output.py` |
| Per-step telemetry events (agent_started, tool_invoked, gate_fired, etc.) | `src/runtime/storage/event_log.py` |
| Auto-learning lesson store + nightly refresher | `src/runtime/learning/extractor.py`, `src/runtime/learning/scheduler.py` |
| Two-stage dedup (embedding + LLM) | `src/runtime/dedup.py` |
| Optimistic-concurrency `SessionStore` over SQLAlchemy | `src/runtime/storage/session_store.py` |
| Read-only similarity store | `src/runtime/storage/history_store.py` |
| Trigger registry (api / webhook / schedule / plugin) | `src/runtime/triggers/` |
| Single-file deploy bundle (`dist/`) | `scripts/build_single_file.py` |
| Streamlit UI shell | `src/runtime/ui.py`, `ui/streamlit_app.py` |
| FastAPI surface (`/sessions/*`, SSE/WebSocket, approvals) | `src/runtime/api.py` |
| Concept-leak ratchet (CI-enforced framework genericity) | `tests/test_genericity_ratchet.py`, `scripts/check_genericity.py` |

## Current status

`main` is at v1.5 (last squash commit `b97ddb3`). All milestones
shipped:

| Milestone | Title | PR |
|---|---|---|
| v1.0 | Prompt-vs-Code Remediation | #1 |
| v1.1 | Framework De-coupling (generic runtime) | #2 |
| v1.2 + v1.3 + v1.4 | FOC + HARD + telemetry + auto-learning + React-ready API | bundled into #5 |
| v1.5-A | Markdown turn output + HITL fix on langgraph 1.x | #6 / #7 |
| v1.5-B | Generic-noun pass (concept-leak ratchet 156 → 39) | #8 |
| v1.5-C | Per-agent LLM proof point | #9 |
| v1.5-D | 429 rate-limit retry + multi-provider integration driver | #10 |

**1265 tests passing**, **87% coverage**, **ratchet at 39**, ruff
clean, SonarCloud quality gate green. See [`docs/DESIGN.md` § 13](DESIGN.md#13-milestone-history)
for the full history.

## Production-ready vs experimental

| Surface | Status | Notes |
|---|---|---|
| Framework runtime (`src/runtime/`) | **production** | Used in air-gapped corporate environments |
| `incident_management` example | **production** | Flagship use case |
| `code_review` example | **demo / proof-of-genericity** | Tools are mocks (no real GitHub/GitLab fetch) — `examples/code_review/README.md` |
| Streamlit UI | **prototype** | Stable but slated for replacement by React in v2.0 |
| FastAPI surface | **production-ready** | v1.4 added generic `/sessions/*` REST + SSE/WebSocket + CORS + structured error envelope |
| Postgres checkpointer | **optional / opt-in** | Default is SQLite; install `pip install asr[postgres]` (`pyproject.toml:39`) |
| Trigger registry — webhook / schedule | **functional, lightly exercised** | Used by the example apps; no large-scale fan-in tested |
| Trigger registry — plugin transport | **stub** (`src/runtime/triggers/transports/plugin.py`) — Inference: scaffold for future SQS/Kafka/NATS transports |
| ASR memory layers (incident_management) | **read-only** | Mutation paths (write-back) deferred per `examples/incident_management/README.md` |
| Auto-learning lesson refresher | **production** | Nightly APScheduler job, gated on config |

## What's next

- **v2.0 — React UI**, replacing the Streamlit prototype, parity-port
  against the v1.4 `/sessions/*` API surface. The long pole.
- Smaller cleanups: duplicate `ToolCall` audit rows
  (gateway colon-form vs harvester `__`-form), `ApprovalWatchdog`
  regression test, `ASR_LOG_LEVEL` doc, `src/runtime/locks.py:49`
  TODO. See [`docs/10-known-risks-and-todos.md`](10-known-risks-and-todos.md).
