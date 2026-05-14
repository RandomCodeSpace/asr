# ASR Multi-Agent Runtime Framework — Design & Decisions

> **Audience.** New contributors and operators who need one document
> covering what the framework is, how it composes, and *why* the
> non-obvious decisions are the way they are.
>
> **Scope.** Architecture, core abstractions, runtime model, storage,
> deployment, and a decision log. Operational how-tos live in
> `docs/DEVELOPMENT.md` (dev workflow) and `docs/AIRGAP_INSTALL.md`
> (corporate-mirror install).

---

## 1. What it is

ASR is a generic Python multi-agent runtime that wraps **LangGraph**
for orchestration and **FastMCP** for tool dispatch, adds a HITL
gateway and a markdown turn-output contract on top, and ships as a
single-file bundle into air-gapped corporate environments.

Two reference apps live in the same repo to prove the runtime is
genuinely generic:

- **`examples/incident_management/`** — 4-skill investigation
  pipeline (intake → triage → deep_investigator → resolution) with
  ASR memory layers (L2 Knowledge Graph, L5 Release Context, L7
  Playbook Store) and a remediation workflow that pauses on
  high-risk actions.
- **`examples/code_review/`** — 3-skill PR review pipeline (intake
  → analyzer → recommender). Built specifically to surface every
  framework leak that would have made the runtime
  incident-shaped — those leaks were lifted into the framework
  rather than worked around.

What the framework owns: session lifecycle, agent dispatch, tool
gateway, HITL pause/resume, telemetry, storage, deployment bundling.

What an app owns: domain `Session` subclass, MCP servers, skill
prompts + per-skill YAML, `App*Config` for cross-cutting domain
knobs (severity aliases, escalation roster, similarity thresholds).

---

## 2. Architecture at a glance

Layers from bottom to top:

```
+------------------------------------------------------------+
| App layer (examples/incident_management, examples/code_review)
| - state.py, config.py, skills/, mcp_server.py, ui.py       |
+------------------------------------------------------------+
| Framework — runtime/                                       |
| - Session, Skill, AgentRun, ToolCall, AgentTurnOutput      |
| - Orchestrator, OrchestratorService                        |
| - Gateway (wrap_tool), policies, ToolRegistry              |
| - SessionStore, HistoryStore, EventLog                     |
| - graph.py: build_graph + make_agent_node                  |
| - llm.py: provider abstraction                             |
| - ui.py: Streamlit shell                                   |
| - api.py: FastAPI surface                                  |
+------------------------------------------------------------+
| LangGraph 1.x  (orchestration / state / checkpointing)     |
| LangChain 1.x  (chat models, agents.create_agent, tools)   |
| FastMCP        (in-process / stdio / http MCP servers)     |
+------------------------------------------------------------+
| Providers: Ollama Cloud · OpenRouter · Azure OpenAI · …    |
+------------------------------------------------------------+
```

**Control flow for one session** (steady state):

```
UI / API  ──start_session──▶  OrchestratorService  ──▶  Orchestrator
                                                            │
                                                            ▼
                                  build_graph (langgraph StateGraph)
                                                            │
                                       per-agent step       ▼
                              ┌───────────────────────────────────┐
                              │ make_agent_node                   │
                              │ - reload session from store       │
                              │ - emit agent_started event        │
                              │ - wrap_tool(s) with gateway       │
                              │ - create_agent (langchain/langgraph)
                              │ - _drive_agent_with_resume        │
                              │     loop: ainvoke / handle pause  │
                              │ - parse_envelope_from_result      │
                              │ - record AgentRun                 │
                              │ - decide route from signal        │
                              └───────────────────────────────────┘
                                                            │
                              gate node? (low confidence) ──▼
                              terminal tool? ──▶ status set by tool
                              else ──▶ default_terminal_status
                                                            │
                                                            ▼
                                  finalize_session_status_async
```

---

## 3. Core abstractions

### 3.1 `Session` (`src/runtime/state.py`)

The framework's unit of work. All apps subclass it; the framework
itself only reads/writes the fields declared on the base class.

```python
class Session(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None
    agents_run: list[AgentRun]
    tool_calls: list[ToolCall]
    findings: dict[str, Any]
    token_usage: TokenUsage
    pending_intervention: dict | None
    user_inputs: list[str]
    parent_session_id: str | None       # dedup linkage
    dedup_rationale: str | None
    extra_fields: dict[str, Any]        # bag for app-specific domain data
    version: int                        # optimistic concurrency
    turn_confidence_hint: float | None  # transient (excluded from persistence)
```

Apps add domain fields on a subclass:

```python
class IncidentState(Session):
    query: str
    environment: str
    reporter: Reporter
    severity: str
    summary: str
    resolution: str | None
```

Fields the row schema doesn't have a column for round-trip via the
`extra_fields` JSON bag — see [§ 8 Storage](#8-storage).

### 3.2 `Skill` (`src/runtime/skill.py`)

YAML-driven configuration unit:

```yaml
name: triage
description: Hypothesis-loop triage agent
kind: responsive               # responsive | supervisor | monitor
model: gpt_oss_cheap           # optional per-agent override
tools:
  local_inc: [submit_hypothesis, update_incident]
  local_observability: [get_logs, get_metrics, ...]
routes:
  - when: success
    next: deep_investigator
  - when: needs_input
    next: __end__
    gate: confidence
  - when: default
    next: deep_investigator
system_prompt: |
  ...
```

Three `kind`s:

- `responsive` — ReAct LLM agent (the default; uses
  `langchain.agents.create_agent`).
- `supervisor` — non-LLM rule-based dispatcher (or LLM-dispatched
  via `dispatch_strategy: llm`); used by intake to pre-filter.
- `monitor` — out-of-band runner (e.g. `MonitorRunner`); not a graph
  node.

### 3.3 `AgentRun` + `ToolCall` (`src/runtime/state.py`)

Append-only audit rows:

```python
class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str                      # final_text, or "agent failed: <exc>"
    token_usage: TokenUsage
    confidence: float | None
    confidence_rationale: str | None
    signal: str | None

class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str
    risk: ToolRisk | None             # low | medium | high
    status: ToolStatus                # executed | executed_with_notify
                                      # | pending_approval | approved
                                      # | rejected | timeout
    approver: str | None
    approved_at: str | None
    approval_rationale: str | None
```

### 3.4 `Orchestrator` + `OrchestratorService`

- `Orchestrator` (`src/runtime/orchestrator.py`) — owns the compiled
  langgraph, the `SessionStore`, the per-session async lock
  registry, and the synchronous lifecycle methods (`start_session`,
  `stream_session`, `resume_session`, `retry_session`).
- `OrchestratorService` (`src/runtime/service.py`) — long-lived
  asyncio loop wrapper around `Orchestrator`. Owns the loop thread,
  registers in-flight sessions, exposes a thread-safe `submit_async`
  / `submit_and_wait` bridge so the Streamlit UI thread and the
  FastAPI request handlers can both schedule work without fighting
  over the same FastMCP / SQLAlchemy transports.

### 3.5 `wrap_tool` Gateway (`src/runtime/tools/gateway.py`)

Every `BaseTool` an agent sees is wrapped by the gateway. The
wrapper:

1. Injects session-derived args (e.g. `environment` from the
   session row) before the LLM-visible arg surface, so the LLM
   physically cannot fabricate them.
2. Consults the risk policy:
   - `low` → run, emit `tool_invoked` with `status=executed`.
   - `medium` → run, append a `executed_with_notify` audit row.
   - `high` → call `langgraph.types.interrupt(payload)`, append a
     `pending_approval` row, save to DB, pause the graph.
3. After resume:
   - On `approve` → run the inner tool, update the pending row to
     `approved`, save.
   - On `reject` / `timeout` → return a marker dict, update the
     pending row to the matching status, save.

### 3.6 `AgentTurnOutput` envelope
(`src/runtime/agents/turn_output.py`)

The structured output every agent must produce per turn:

```python
class AgentTurnOutput(BaseModel):
    content: str
    confidence: float                 # [0.0, 1.0], reconciled
    confidence_rationale: str
    signal: str | None                # success | failed | needs_input | None
```

How the envelope is sourced — see [§ 6 Markdown turn output](#6-markdown-turn-output-contract-phase-22).

---

## 4. Runtime model

### 4.1 Session lifecycle

States a session walks through:

```
new ─▶ in_progress ─▶ <terminal>
                       resolved | escalated | needs_review |
                       awaiting_input | error | stopped | duplicate
```

- `new` — row created, graph not yet entered.
- `in_progress` — at least one agent has run; non-terminal.
- Terminal states are set by:
  - **Terminal tool calls** (e.g. `mark_resolved` → `resolved`,
    `mark_escalated` → `escalated`); the tool registry maps tool
    names to status transitions.
  - **`default_terminal_status`** (`needs_review` for incident
    management) when the graph completes without a terminal tool.
  - **`_handle_agent_failure`** → `error` on agent exceptions.
  - **`stop_session()`** → `stopped` on explicit cancellation.
  - **`dedup_check`** → `duplicate` (with `parent_session_id`) when
    stage-2 LLM dedup confirms a match against a prior closed
    session.

### 4.2 Per-agent dispatch
(`src/runtime/graph.py:_build_agent_nodes`)

For every skill in `cfg.orchestrator.skills`:

```python
llm = get_llm(cfg.llm, skill.model, role=agent_name, ...)
node = make_agent_node(skill=skill, llm=llm, tools=run_tools, ...)
sg.add_node(agent_name, node)
```

`skill.model` is the per-agent override; falls through to
`cfg.llm.default` when `None`. This is what lets intake run on
Ollama while triage / DI / resolution run on OpenRouter — see the
v1.5-C decision below.

### 4.3 Routing

`skill.routes` is a list of `(when, next, gate?)` rules. The
runtime evaluates them after each agent step:

```yaml
routes:
  - when: success           # signal value
    next: deep_investigator
  - when: needs_input
    next: __end__
    gate: confidence        # route through gate node first
  - when: default           # fallback
    next: triage
```

The framework's gate node fires when the upstream agent's confidence
is below `framework.confidence_threshold` (default 0.75). The gate
emits a `pending_intervention` and the session moves to
`awaiting_input` until the operator supplies a `resume_with_input`
verdict. Agents emit signals via the `signal` arg of typed-terminal
or patch tools.

### 4.4 Termination

Three independent paths:

1. **Tool-driven** — an agent calls a tool the registry recognises
   as terminal (`local_inc:mark_resolved`, `…:mark_escalated`).
   The tool sets `inc.status` directly.
2. **Inferred** — `_finalize_session_status` walks `tool_calls`
   matching against `cfg.orchestrator.terminal_tools` rules.
3. **Default** — falls through to
   `cfg.orchestrator.default_terminal_status` when no rule fires
   AND the graph wasn't paused on a HITL gate.

The pause-aware guard
(`Orchestrator._is_graph_paused`) is what keeps a paused HITL
session from being coerced to `default_terminal_status` while the
operator is still deciding.

---

## 5. LLM provider story

### 5.1 Three layers

```
+----------------------------------------------------------+
| Skill (YAML)         model: gpt_oss_cheap                |
+----------------------------------------------------------+
| runtime.llm.get_llm  resolves name → cfg.models[name]    |
|                       → ProviderConfig → BaseChatModel    |
+----------------------------------------------------------+
| LangChain provider class                                 |
|   - ChatOpenAI         openai_compat (OpenRouter)        |
|   - ChatOllama         ollama (Ollama Cloud + local)     |
|   - AzureChatOpenAI    azure_openai                      |
+----------------------------------------------------------+
| Driven by langchain.agents.create_agent (langgraph subgraph) |
+----------------------------------------------------------+
```

### 5.2 Provider config

`config/config.yaml` declares providers + named models:

```yaml
llm:
  default: workhorse
  providers:
    ollama_cloud:
      kind: ollama
      base_url: https://ollama.com
      api_key: ${OLLAMA_API_KEY}
    azure:
      kind: azure_openai
      endpoint: ${AZURE_ENDPOINT}
      api_version: 2024-08-01-preview
      api_key: ${AZURE_OPENAI_KEY}
    openrouter:
      kind: openai_compat
      base_url: https://openrouter.ai/api/v1
      api_key: ${OPENROUTER_API_KEY}
  models:
    workhorse:
      provider: openrouter
      model: inclusionai/ring-2.6-1t:free
    gpt_oss:
      provider: ollama_cloud
      model: gpt-oss:20b
    smart:
      provider: azure
      model: gpt-4o
      deployment: gpt-4o
```

### 5.3 429 retry regime (v1.5-D)

`_ainvoke_with_retry` (`src/runtime/graph.py`) splits transient
errors into two classes:

| Class | Markers | Backoff | Total |
|---|---|---|---|
| 5xx + connection | `internal server error`, `status code: 5xx`, `connection reset`, `remoteprotocolerror`, `incomplete chunked read` | 1.5s × attempt | ~9s |
| 429 / rate-limit | `status code: 429`, `error code: 429`, ` 429`, `429 `, `ratelimiterror`, `rate limit`, `rate-limited`, `too many requests` | 7.5s × attempt | ~45s |

Non-429 4xx (auth, validation) propagates immediately so quota /
schema problems fail fast.

### 5.4 Live verification

`tests/test_integration_driver_s1.py` parametrises three legs
(`local`, `workhorse`, `azure`); each skips independently if its
keys are absent. Run with `OLLAMA_API_KEY + OLLAMA_BASE_URL`,
`OPENROUTER_API_KEY`, and/or `AZURE_OPENAI_KEY + AZURE_ENDPOINT`
exported.

---

## 6. Markdown turn-output contract (Phase 22)

### 6.1 Why

Pre-Phase-22 the framework forced agents through
`response_format=AgentTurnOutput` (a JSON schema). Multiple problems:

- gpt-oss / Ollama models drifted on JSON schema adherence.
- LangGraph's `with_structured_output` second pass interacted badly
  with the React END signal under `recursion_limit=25`.
- Adding tools to the schema confused some providers' tool dispatch.

Phase 22 dropped `response_format` and made the agent close its turn
with a markdown contract block. Markdown is the format every chat
model writes well; the parse step happens in the framework where
leniency is in our control.

### 6.2 The contract

Every skill prompt ends with:

```
## Output contract — REQUIRED

Every final reply MUST end with these three sections, in order, each
preceded by a level-2 markdown header:

  ## Response
  <body>

  ## Confidence
  <0.0-1.0 float> -- <one-line rationale>

  ## Signal
  <success|failed|needs_input|none>

**CRITICAL — final-reply rule:** the markdown envelope is mandatory;
the framework hard-fails if it is missing.
```

### 6.3 Parse paths

`parse_envelope_from_result` walks 6 paths and falls through to a
hard fail:

| Path | Source | When it fires |
|---|---|---|
| 1 | `result["structured_response"]` | Pre-Phase-22 stub fixtures and explicit-schema callers |
| 2 | JSON-decode last AIMessage content | Models that still emit valid JSON |
| 4 | `_parse_confidence_line` over the `## Confidence` body | Markdown-primary path; the production happy path |
| 5 | Typed-terminal-tool args (`confidence`, `confidence_rationale`, `resolution_summary`) | Models that treat a terminal tool call as completion |
| 6 | Permissive: any tool was called → synthesise a 0.30-confidence placeholder | Last-ditch fallback so the session reaches a terminal status instead of hard-failing |
| 7 | `raise EnvelopeMissingError` | Truly nothing parseable |

(Path 3 was the original location for what became Path 4; the
numbering is preserved in code comments to keep historical commits
diff-friendly.)

### 6.4 gpt-oss compatibility quirks

- gpt-oss prefers EN DASH (`–`, `–`) over EM DASH (`—`,
  `—`); the dash separator accepts the full Unicode Pd block.
- gpt-oss sometimes emits an empty closing AIMessage after a tool
  call; Path 5 / Path 6 cover that.
- The skill prompts carry an explicit
  `**CRITICAL — final-reply rule:**` paragraph because gpt-oss
  initially treated the first tool result as completion.

The procedural confidence-line parser
(`_parse_confidence_line`) replaces an earlier regex that Sonar's
S5852 (regex DoS) flagged; the procedural form has no backtracking
surface to attack.

---

## 7. HITL approve / reject

### 7.1 The risk-rated gateway

Tools are policy-gated per
`cfg.runtime.gateway.policy`:

```yaml
runtime:
  gateway:
    policy:
      apply_fix: high                 # gate
      restart_service: medium         # notify-only audit
      get_logs: low                   # default; no row written
```

Apps configure `cfg.orchestrator.gate_policy` for cross-cutting
behaviour:

```yaml
gate_policy:
  threshold: 0.75
  gated_environments: [production]
  gated_risk_actions: [approve]
  resolution_trigger_tools: ['local_remediation:apply_*']
```

### 7.2 Pause / resume on langgraph 1.x (PR #6)

langgraph 1.x changed the `interrupt()` contract: a tool calling
`interrupt()` no longer raises `GraphInterrupt` to the caller —
`agent.ainvoke()` returns a normal result with
`result["__interrupt__"]` populated. The framework's wrapper had to
catch up:

- `_drive_agent_with_resume` (`src/runtime/graph.py`) detects an
  inner pause via `agent_executor.aget_state(inner_cfg).next` being
  non-empty, calls outer `interrupt()` to fetch the verdict, and
  forwards via `agent_executor.ainvoke(Command(resume=verdict),
  config=inner_cfg)`.
- The inner `create_agent` now receives the orchestrator's
  checkpointer + a deterministic per-invocation thread id
  (`f"{inc_id}:agent:{skill.name}:turn{len(agents_run)}"`). Without
  these, `Command(resume=…)` raises and the gated tool gets silently
  skipped.
- `make_agent_node` reloads from `store.load(inc_id)` at entry —
  defends against stale `state["session"]` snapshots from outer
  Pregel checkpoints (which capture state at step boundaries, not
  mid-step).
- `gateway.wrap_tool` calls `store.save` after every status
  transition (rejected / timeout / approved) so the audit row in
  the DB matches the operator's actual decision.
- `Orchestrator._is_graph_paused` guards
  `_finalize_session_status_async` in `stream_session` /
  `retry_session` / the API approval handler — a HITL pause must
  not be coerced into `default_terminal_status`.

These five fixes shipped together as PR #6; before them, clicking
Approve would do nothing because the framework had already moved
past the pause point.

### 7.3 Approval surface

Two ways to resolve a `pending_approval`:

- **UI** — `_render_pending_approvals_block` shows the Approve /
  Reject buttons and rationale field; click drives
  `Command(resume={"decision": "approve", ...})` via
  `OrchestratorService.submit_and_wait`.
- **API** — `POST /sessions/{sid}/approvals/{tcid}` does the same
  resume, scoped under the per-session lock so two concurrent
  approvals on the same thread can't race.

### 7.4 Approval watchdog
(`src/runtime/tools/approval_watchdog.py`)

Background task that scans `pending_approval` rows older than
`framework.approval_timeout` and resolves them with `verdict=timeout`,
freeing operators from manual intervention on stale rows. Triggered
by the lifespan startup hook.

---

## 8. Storage

### 8.1 SessionStore (`src/runtime/storage/session_store.py`)

CRUD for the row schema. Owns:

- `_next_id` — monotonic per-day sequence; respects
  `state_cls.id_format(seq=…, prefix=…)` so each app picks its own
  ID namespace (`INC-…`, `CR-…`, etc.).
- `save` — optimistic-version update. Bumps `version`; raises
  `StaleVersionError` on mismatch so the caller can reload + retry.
- `_row_to_incident` / `_incident_to_row_dict` — round-trip
  between `IncidentRow` (SQLAlchemy) and the app's `Session`
  subclass. Fields the row schema has columns for go to typed
  fields; everything else lands in `extra_fields` JSON.
- Vector write-through — `_persist_vector` / `_add_vector` /
  `_refresh_vector` keep a FAISS index aligned with the row
  table.

### 8.2 IncidentRow (`src/runtime/storage/models.py`)

The persistent schema. Fields are deliberately broad enough to host
the example apps' typed fields (`severity`, `reporter_id`,
`reporter_team`, `summary`, `tags`, `parent_session_id`,
`dedup_rationale`, `extra_fields` JSON) without forcing every app
to declare them. An app's `Session` subclass declares whichever
typed fields it cares about; the rest stay in `extra_fields`.

The `severity` / `reporter_*` columns ARE incident-shaped — the
v1.5-B generic-noun pass left them in place because renaming would
require a schema migration. Apps that don't model severity or a
human submitter ignore those columns; the round-trip silently
omits them.

### 8.3 HistoryStore (`src/runtime/storage/history_store.py`)

Read-only similarity search over the same engine + vector store.
Used by intake's similarity retrieval (`lookup_similar_incidents`).
Filter dimensions are pluggable — apps construct a
`HistoryStore(filter_resolver=…)` matching their own row shape.

### 8.4 LangGraph checkpointer
(`src/runtime/checkpointer.py`)

Separate from the SessionStore. SQLite default (`sqlite:////tmp/asr.db`),
Postgres optional via `runtime.checkpointer_postgres`. Holds langgraph
Pregel state + pending interrupts. The HITL approve / reject path
relies on this checkpointer being durable.

### 8.5 EventLog (`src/runtime/storage/event_log.py`)

Append-only `session_events` table. Records:

- `agent_started`, `agent_finished`, `confidence_emitted`,
  `route_decided`
- `tool_invoked` (every wrapped tool call, with latency + result_kind)
- `gate_fired` (HITL gate decisions)
- `status_changed` (terminal-status transitions with cause)
- `lesson_extracted` (M5/M6 auto-learning)

Per-step events feed any external observability stack and the
auto-learning pipeline.

---

## 9. Memory layers (incident_management example)

The incident-management app ships an ASR (Automated Site Reliability)
memory bundle hydrated by the supervisor at intake:

| Layer | What | Backend |
|---|---|---|
| L2 | Knowledge Graph — services, owners, runbooks, dependencies | `examples/incident_management/asr/kg_store.py` (filesystem JSON) |
| L5 | Release Context — recent deploys per service | `release_store.py` (filesystem JSON) |
| L7 | Playbooks — known-good remediation steps per failure mode | `playbook_store.py` (filesystem JSON) |

`hydrate_and_gate` (in the example's MCP server) walks the user's
query, extracts mentioned components, and returns a `MemoryLayerState`
bundle that the triage agent reads as additional context.

This is **app-level**, not framework — the runtime stays memory-
agnostic. A different app can ship its own L1/L2/L3 memory layers
without touching `runtime/`.

---

## 10. Deployment

### 10.1 Air-gapped target

The deployment env is corporate / air-gapped: no public-internet
runtime calls, no CDN fetches, no `pip install` at deploy time.

### 10.2 Single-file bundle (BUNDLER-01)

`scripts/build_single_file.py` flattens the runtime + each app into
self-contained `.py` files under `dist/`:

| File | Contents |
|---|---|
| `dist/app.py` | framework only — no example code |
| `dist/apps/incident-management.py` | framework + incident_management example |
| `dist/apps/code-review.py` | framework + code_review example |
| `dist/ui.py` | Streamlit shell |

CI gate `Bundle staleness gate (HARD-08)` rebuilds the bundles
from `src/` and fails the build if they don't match the committed
`dist/*` — this keeps the deploy bundles "repaired by construction"
on every merge.

### 10.3 7-file deploy payload

Copy onto the target host:

```
app.py                    (renamed from dist/apps/<app>.py)
ui.py                     (dist/ui.py)
config/config.yaml        (framework: LLM, MCP, storage)
config/<app>.yaml         (app: severity aliases, escalation roster, …)
config/skills/            (skill prompts, optional override)
.env                      (provider keys)
```

Boot:

```bash
python -m runtime --config config/<app>.yaml
streamlit run ui.py --server.port 37777
```

### 10.4 Reproducible install (HARD-02)

`uv.lock` pins direct + transitive deps with sha256 hashes. CI
installs from the lock with `uv sync --frozen`; an internal
package mirror is sufficient for a fully offline build. See
`docs/AIRGAP_INSTALL.md`.

---

## 11. Telemetry + auto-learning (M1–M9)

### 11.1 Per-step events

Every meaningful boundary emits an `EventLog` row keyed by
`session_id`. The four agent-boundary events
(`agent_started → confidence_emitted → route_decided →
agent_finished`) fire in order; `tool_invoked` and `gate_fired`
fire at the gateway boundary.

### 11.2 Lesson store

`src/runtime/learning/extractor.py` runs at session finalize and
distills outcome + winning hypothesis + applied fix into a
`Lesson` row. The intake supervisor reads recent lessons via
`LessonStore.find_relevant(query, …)` to prime the next session.

### 11.3 Lesson refresher

`src/runtime/learning/scheduler.py` runs an APScheduler job
nightly (configurable) that walks recent sessions and extracts
lessons missed at finalize time (e.g. sessions resolved manually
in the UI long after the agent's run).

---

## 12. Decision log

Compact rationale for the non-obvious calls. Each entry is a single
"why".

### DEC-001. LangGraph as orchestration engine

**When.** From the start.
**Why.** Out-of-the-box Pregel-style step boundaries +
checkpointing + first-class HITL `interrupt()` semantics. We don't
maintain a graph engine ourselves; we just wrap it.

### DEC-002. `langchain.agents.create_agent` for the per-agent loop (Phase 15)

**When.** v1.3 hardening, after `langgraph.prebuilt.create_react_agent`
was deprecated.
**Why.** Single tool-loop with native ToolStrategy fallback, removes
the `recursion_limit=25` workaround we previously needed.

### DEC-003. Markdown contract over `response_format` JSON (Phase 22)

**When.** v1.5-A.
**Why.** JSON-schema-shaped output via `response_format` triggered a
class of brittleness across providers (model-specific JSON drift,
tool-strategy + React END interaction, recursion_limit ceilings).
Markdown is the native format every chat model writes well; the parse
step happens in the framework where leniency is in our control. Path
5 / Path 6 fallbacks cover models that occasionally drop the
contract.

### DEC-004. Pure-policy HITL gating (Phase 11)

**When.** v1.2.
**Why.** The gate decision (high-risk tool? gated env? low
confidence?) was previously scattered across the gateway, the
orchestrator, and the skill prompts. Phase 11 moved it into a single
pure function `should_gate(session, tool_call, confidence, cfg)` so
auditing what gates is a one-grep operation.

### DEC-005. Generic `Session` base + `extra_fields` JSON (v1.1 decoupling)

**When.** v1.1.
**Why.** Pre-v1.1 the framework had `IncidentState` baked in. Adding
a second app (code_review) was the forcing function — every
"incident-shaped" leak that surfaced moved into the framework as
`Session.extra_fields` (the JSON bag) or the row schema's existing
typed columns. Apps now subclass `Session` and write whatever fields
they need; the framework stays domain-agnostic.

### DEC-006. Per-agent `skill.model` override (v1.5-C / M8)

**When.** v1.5-C.
**Why.** The intake supervisor can run on a fast / cheap model
while the deep-investigator agent needs a smarter (more expensive)
one. `_build_agent_nodes` resolves `get_llm(cfg.llm, skill.model,
role=agent_name)` per skill; falls back to `cfg.llm.default` when
`model` is `None`.

### DEC-007. Single-file bundle for air-gap deploy (BUNDLER-01)

**When.** v1.3.
**Why.** Corporate deploy env is copy-only. A multi-file
`pip install` step is out of scope. The bundler turns the
multi-file source tree into the smallest possible deploy payload
(7 files total).

### DEC-008. Concept-leak ratchet for framework genericity (v1.5-B)

**When.** v1.5-B.
**Why.** The decoupling work (DEC-005) wasn't binary — `incident` /
`severity` / `reporter` tokens kept creeping into `src/runtime/` via
local variables, docstrings, and helper names. The ratchet test
counts those tokens and fails the build if the count grows. v1.5-B
took it from 156 down to 39 (the residual 39 are
schema-coupled / public-API / intentional example-app callouts).

### DEC-009. 429 separate retry regime with longer backoff (v1.5-D)

**When.** v1.5-D.
**Why.** Free / shared upstream tiers (e.g. OpenRouter `…:free`)
throttle on 30-60s windows; the 5xx default backoff (1.5s/3s/4.5s)
exhausted retries before the window cleared. Now 429 retries on
7.5s/15s/22.5s (~45s total).

### DEC-010. Inner agent checkpointer + reload-on-entry to fix HITL stale state (PR #6)

**When.** v1.5-A.
**Why.** Outer Pregel checkpoints at step boundaries, not mid-step.
On resume, `state["session"]` reflects the prior step's output, NOT
the gateway's pending_approval row + version bump that happened
mid-step. Without `make_agent_node` reloading from store at entry,
the gateway sees no pending row, double-appends, and `store.save`
raises `StaleVersionError`. The reload + the inner checkpointer
together are what make Approve / Reject actually drive the gated
tool to completion.

### DEC-011. Two example apps to prove genericity

**When.** v1.1 (incident_management lifted), Phase 8
(code_review added).
**Why.** Without a second app, "is the framework generic?" is
unanswerable. The code_review app was built specifically to surface
every incident-shaped assumption that hadn't been lifted yet — id
format, row schema, build pipeline, intra-bundle imports. Each
leak became a framework PR rather than an app workaround.

### DEC-012. Bundle staleness CI gate (HARD-08)

**When.** v1.3.
**Why.** dist/ files drift if a contributor updates `src/runtime/`
or `examples/` without re-running the bundler. The drift turns into
a deploy-time bug ("works in dev, broken in prod"). The CI gate
rebuilds the bundles from source on every PR and refuses the merge
if they differ from the committed `dist/*`.

---

## 13. Milestone history

| Milestone | Title | PR | Squash SHA | Headline change |
|---|---|---|---|---|
| v1.0 | Prompt-vs-Code Remediation | #1 | `02378dd` | Code becomes the authority — skill prompts no longer carry policy logic |
| v1.1 | Framework De-coupling | #2 | `0ff8914` | Generic runtime, ASR as use case |
| v1.2 | Framework Owns Flow Control | bundled into #5 | `9018371` | FOC-01..06 — gate / retry / signal / dedup all framework-owned |
| v1.3 | Hardening + Real-LLM Compatibility | bundled into #5 | `9018371` | HARD-01..09 + LLM-COMPAT-01 + BUNDLER-01 + SKILL-LINTER-01 |
| v1.4 | Per-step telemetry + auto-learning intake + React-ready API | #5 | `9018371` | M1..M9 telemetry + LessonStore + generic /sessions/* + SSE + WebSocket + CORS + structured error envelope |
| v1.5-A | Markdown turn output (Phase 22) + HITL approve/reject end-to-end on langgraph 1.x | #6 + #7 | `f0586a8`, `3f0eb5f` | DEC-003 + DEC-010 |
| v1.5-B | Generic-noun pass — concept-leak ratchet 156 → 39 | #8 | `25e363c` | DEC-008 |
| v1.5-C | Per-agent LLM proof point — intake on Ollama Cloud, downstream on `llm.default` | #9 | `54a830d` | DEC-006 |
| v1.5-D | 429 rate-limit retry + multi-provider integration driver | #10 | `adefae6` | DEC-009 |

Per-phase artefacts under `.planning/phases/<NN>-<slug>/` (gitignored
working state; selected artefacts are committed for historical record).

---

## 14. Pending / known gaps

### v2.0 — React UI (the long pole)

Stack pick + scaffold + parity-port against the v1.4
`/sessions/*` REST + SSE/WebSocket API. ~1–2 weeks. The Streamlit
shell stays as the prototype until React reaches parity.

### Smaller cleanups

- **Duplicate ToolCall audit rows.** The gateway records the gated
  tool under the FastMCP composite name (`local_remediation:apply_fix`,
  colon form), the harvester records the same tool call under the
  LLM-visible name (`local_remediation__apply_fix`, double-underscore
  form). Cosmetic in the UI; matters if any consumer aggregates tool
  counts. Fix: align both on the `__` form. ~30 min.
- **`ApprovalWatchdog` regression test.** PR #6 added gateway saves
  on resolution transitions; the watchdog should observe a faster
  cleanup signal but no focused test was added. ~15 min.
- **`ASR_LOG_LEVEL` env var documentation.** Added in PR #6, no
  README mention. One-line doc fix.
- **`src/runtime/locks.py:49` — `TODO(v2)`.** Evict idle slots to cap
  memory in long-running servers. Real concern for production; not
  urgent for HITL-paced workloads.

---

## 15. Where to find what

| You want to… | Look at |
|---|---|
| Add a new skill | `examples/<app>/skills/<name>/{config.yaml, system.md}` |
| Add a new app | New folder under `examples/`; subclass `Session` in `state.py`; declare `App*Config` in `config.py`; write MCP servers and skills |
| Add a tool | App's `mcp_server.py`; register in YAML; gateway picks up risk policy from `cfg.runtime.gateway.policy` |
| Change LLM provider | `config/config.yaml` `llm.providers` / `llm.models`; per-agent override on `skill.model` |
| Change HITL policy | `cfg.orchestrator.gate_policy` (cross-cutting), `cfg.runtime.gateway.policy` (per-tool) |
| Trace one session end-to-end | `EventLog` rows for that `session_id`; `agents_run` and `tool_calls` on the row; `session_events` table |
| Update the bundle | `uv run python scripts/build_single_file.py`; commit `dist/*` |
| Add a new framework module | `RUNTIME_MODULE_ORDER` in `scripts/build_single_file.py` (after deps); regen + commit |
| Run live LLM tests | Set `OLLAMA_API_KEY + OLLAMA_BASE_URL`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_KEY + AZURE_ENDPOINT`; `uv run pytest tests/test_integration_driver_s1.py -v` |
| Reset state for a fresh run | `rm /tmp/asr.db /tmp/asr.db-{wal,shm}; rm -rf /tmp/asr-faiss` then restart |

---

## 16. Document map

- **`docs/DESIGN.md`** (this file) — architecture, abstractions,
  decisions, milestone history.
- **`docs/DEVELOPMENT.md`** — day-to-day contributor loop (setup,
  bundle regeneration, adding modules).
- **`docs/AIRGAP_INSTALL.md`** — corporate-mirror install procedure.
- **`README.md`** (repo root) — one-screen overview pointing at the
  three docs above.
- **`examples/incident_management/README.md`** — incident-management
  app surface; per-skill prompts under `skills/`.
- **`examples/code_review/README.md`** — code-review app surface;
  per-skill prompts under `skills/`.
- **`.planning/`** (gitignored) — working state for the GSD planning
  workflow (`STATE.md`, `ROADMAP.md`, `phases/<NN>-<slug>/`). Not
  shipped; selected phase artefacts are committed for the historical
  record.
