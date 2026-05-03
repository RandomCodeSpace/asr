# Phase 6 — Agent Kinds (Responsive / Supervisor / Monitor)

**Status**: Plan only — no implementation in this document.
**Branch target**: feature branch off `main` (e.g., `feat/phase6-agent-kinds`).
**Author**: planning session 2026-05-02.
**Depends on**: Phase 5 (Trigger Registry + APScheduler) must be merged first; this plan
hooks `Monitor` skills into the Phase 5 trigger registry.

---

## 1. Goal

Extend the runtime so a `Skill` can declare one of three agent **kinds**, each with a
distinct execution model:

| kind         | When it runs                            | Where it lives                          | Writes `AgentRun`? |
|--------------|------------------------------------------|------------------------------------------|--------------------|
| `responsive` | Inside a session graph, on user turn     | LangGraph node (today's path)            | Yes                |
| `supervisor` | Inside a session graph, dispatches work  | LangGraph node using `Send()`            | **No** (log only)  |
| `monitor`    | Out-of-band, periodic                    | Orchestrator-level singleton (APScheduler)| No (signals only)  |

The schema is **single-model with discriminator**: one `Skill` Pydantic model with
`kind: Literal["responsive","supervisor","monitor"]`, validated per kind.

---

## 2. Locked decisions (NON-NEGOTIABLE)

These were decided in `docs/superpowers/plans/p6-research-decisions.md` and must not be
re-litigated in this phase:

- **P6.1 — Monitor lifetime**: orchestrator-level **singleton**. One `MonitorRunner`
  per `OrchestratorService` instance, shared across all sessions. Monitors fire P5
  triggers; they do **not** run inside session graphs.
- **P6.2 — Skill schema**: single Pydantic model with `kind` discriminator. Per-kind
  fields are optional at the model level and gated by validators.
- **P6.3 — Supervisor bookkeeping**: **skip** the `AgentRun` row for supervisor
  dispatches. The supervisor is bookkeeping/routing, not a token-burning agent run.
  Audit trail comes from a structured **dispatch log** (different stream).

---

## 3. Source material

- `docs/superpowers/plans/2026-05-02-framework-genericification-and-asr-roadmap.md`
  (Phase 6 section)
- Phase 5 plan — trigger registry + scheduler that monitors hook into
- `src/runtime/skill.py` — current responsive-only `Skill` model
- `src/runtime/graph.py` — `make_agent_node`, `route_from_skill`,
  `_build_agent_nodes`
- `src/runtime/orchestrator.py` — `OrchestratorService` lifecycle (where the
  `MonitorRunner` will be wired)
- `docs/superpowers/plans/p6-research-decisions.md` — locked decisions

---

## 4. Deliverables

1. **Skill schema** with `kind` discriminator + per-kind config blocks.
2. **Per-kind LangGraph wiring**:
   - `responsive` — unchanged.
   - `supervisor` — new `make_supervisor_node` using LangGraph `Send()` API.
   - `monitor` — new `make_monitor_node` factory that produces a callable for the
     out-of-graph runner (NOT a session-graph node).
3. **`MonitorRunner`** — orchestrator-level singleton, schedules monitor skills
   via APScheduler (reusing the Phase 5 scheduler).
4. **Supervisor dispatch logging** — structured log entry (no `AgentRun` row).
5. **Validation** — Pydantic per-kind validators that reject mis-typed config.
6. **Skill YAML migration** — existing skills get an explicit `kind: responsive`
   added one-shot; loader still defaults missing `kind` to `responsive` for
   forward safety.
7. **Tests** — three suites: skill loader, supervisor dispatch, monitor firing.
8. **Verification** — full diff re-read, lint/type/test pass, sample YAML for
   each kind committed under `examples/skills/`.

---

## 5. Non-goals

- **Recursive supervisors are bounded but not optimised** — depth limit only
  (R1). No multi-supervisor topology design in this phase.
- **Real-time monitor → session bridge** — monitors signal triggers, triggers
  start sessions; we do not introduce a streaming bus (R2).
- **Per-supervisor LLM routing strategies beyond `llm` and `rule`** — deferred.
- **No UI surface** for monitors/supervisors in this phase (status only via
  logs + existing audit endpoints).

---

## 6. Risks & mitigations

| ID | Risk                                                                                          | Mitigation                                                                                                      |
|----|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| R1 | Supervisor → supervisor recursion (infinite chains).                                          | Hard recursion-depth limit, default `3`, configurable via `Skill.max_dispatch_depth`. Validator at graph build. |
| R2 | Monitor → trigger → session that itself uses `interrupt()` produces async chains.             | Monitors emit signals into an in-memory queue; queue consumers start sessions. Not real-time, by design.        |
| R3 | Supervisor dispatches don't appear in `agent_runs` audit — incomplete audit trail.            | Emit a structured `supervisor_dispatch` log entry (separate stream) with `incident_id`, parent/child agent, ts. |
| R4 | Existing skill YAMLs have no `kind` field.                                                    | Loader defaults to `responsive`; one-shot migration adds `kind: responsive` explicitly to all existing YAMLs.   |
| R5 | APScheduler reuse: monitors and Phase 5 triggers might collide on job IDs.                    | Namespace job IDs: `monitor:<skill_name>` vs `trigger:<trigger_name>`. Validator rejects collisions.            |
| R6 | Monitor `observe` tools may block; one slow monitor stalls others.                            | Run each monitor on APScheduler's `ThreadPoolExecutor`; bound pool size (default 4); per-monitor timeout.       |
| R7 | `emit_signal_when` expression is user-supplied — code-injection risk.                         | Restrict to a safe expression evaluator (e.g. `simpleeval`), no `eval()`/`exec()`. Validator parses at load.    |

---

## 7. Tasks

Ten tasks, **P6-A through P6-J**. Each task lists scope, files touched, acceptance
criteria, and the test surface added.

---

### P6-A — Extend `Skill` model with `kind` and per-kind fields

**Scope**: Add the discriminator and the per-kind config blocks to the existing
`Skill` Pydantic model. No validators yet — those come in P6-B. No loader
changes yet — those come in P6-C.

**Files**:
- `src/runtime/skill.py`

**Shape** (illustrative, final naming TBD during implementation):

```python
from typing import Literal
from pydantic import BaseModel, Field

class RouteRule(BaseModel):
    ...  # existing

class DispatchRule(BaseModel):
    when: str            # condition expression
    target: str          # subordinate agent name

class Skill(BaseModel):
    name: str
    kind: Literal["responsive", "supervisor", "monitor"] = "responsive"

    # responsive (today's behaviour)
    system_prompt: str | None = None
    model: str | None = None
    tools: list[str] = []
    routes: list[RouteRule] = []
    stub_response: ... = None

    # supervisor
    subordinates: list[str] = []
    dispatch_strategy: Literal["llm", "rule"] = "llm"
    dispatch_prompt: str | None = None
    dispatch_rules: list[DispatchRule] = []
    max_dispatch_depth: int = 3   # R1 mitigation

    # monitor
    schedule: str | None = None              # cron expression
    observe: list[str] = []                  # tool names producing observations
    emit_signal_when: str | None = None      # safe-eval expression on observation
    trigger_target: str | None = None        # P5 trigger name to fire
```

**Acceptance**:
- Existing responsive skills parse unchanged (defaults preserve back-compat).
- New fields are visible to the model but not yet enforced.

**Tests**: `tests/runtime/test_skill_schema.py::test_responsive_back_compat`
(new file, populated incrementally across P6-A/B/C).

---

### P6-B — Per-kind validators

**Scope**: Pydantic validators that reject misconfigured skills at load time.

**Files**:
- `src/runtime/skill.py`

**Rules** (each enforced by a `model_validator(mode="after")`):

| kind         | Forbidden when set                                                                          | Required                                                       |
|--------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `responsive` | `subordinates`, `dispatch_strategy != "llm"`, `dispatch_prompt`, `dispatch_rules`, `schedule`, `observe`, `emit_signal_when`, `trigger_target` | `system_prompt` (or `stub_response`)                           |
| `supervisor` | `system_prompt`, `tools`, `routes`, `stub_response`, `schedule`, `observe`, `emit_signal_when`, `trigger_target` | `subordinates` (non-empty); `dispatch_prompt` if strategy=`llm`; `dispatch_rules` if strategy=`rule` |
| `monitor`    | `system_prompt`, `routes`, `stub_response`, `subordinates`, `dispatch_prompt`, `dispatch_rules` | `schedule`, `observe` (non-empty), `emit_signal_when`, `trigger_target` |

**Additional validations**:
- `max_dispatch_depth` between 1 and 10 (sanity bound).
- `schedule` parses as a valid cron expression (use `croniter` or APScheduler's
  parser; reject at load).
- `emit_signal_when` parses with the safe-eval AST (R7); reject on parse fail.
- `trigger_target` must exist in the registered Phase 5 triggers — checked at
  orchestrator startup, not at YAML load (lookup may not be available at parse
  time).

**Acceptance**: Each violation raises `ValidationError` with a clear message
naming the offending field and the kind.

**Tests**: `tests/runtime/test_skill_schema.py` with one parametrised case per
forbidden-field-per-kind combination, plus required-field-missing cases.

---

### P6-C — Skill YAML loader updates

**Scope**: Update the loader (currently in `src/runtime/skill.py` or wherever
`Skill.from_yaml` lives) to honour `kind` and run the validators.

**Files**:
- `src/runtime/skill.py`
- `src/runtime/config.py` (if loader entrypoint lives there — check during
  implementation)

**Behaviour**:
- `kind` missing → defaults to `"responsive"` (for forward compat with old
  YAML).
- `kind` present and known → validated per P6-B.
- `kind` present and unknown → `ValidationError`.

**One-shot migration**: under `examples/skills/` (and any in-repo skills
distributed with the framework), explicitly add `kind: responsive` to every
existing YAML. This is a content change, not a loader change, but it lives in
this task to keep the diff coherent.

**Acceptance**:
- Old YAML without `kind` still loads as responsive.
- New YAML for each of the three kinds loads and validates.
- One-shot migration commit adds `kind: responsive` to all in-repo example
  skill files.

**Tests**: `tests/runtime/test_skill_loader.py` — happy path per kind, plus
unknown-kind rejection.

---

### P6-D — `make_supervisor_node` using LangGraph `Send()`

**Scope**: New supervisor-node factory in `src/runtime/graph.py`. The node
inspects the supervisor's `dispatch_strategy`, picks subordinate(s), and uses
LangGraph's `Send()` API to dispatch. **No `AgentRun` row** is written.

**Files**:
- `src/runtime/graph.py`

**Behaviour**:
1. On entry, increment per-session `dispatch_depth` (carried in graph state);
   reject if `>= max_dispatch_depth` (R1).
2. If `dispatch_strategy == "llm"`: invoke a small LLM call with
   `dispatch_prompt` to choose subordinate(s) from `subordinates`. Strategy is
   not a full agent run; result is a list of names + payload.
3. If `dispatch_strategy == "rule"`: evaluate `dispatch_rules` in order; first
   match wins.
4. For each chosen subordinate, emit a LangGraph `Send(subordinate_name,
   payload)`. LangGraph fans out and joins.
5. Emit a `supervisor_dispatch` structured log entry (P6-H).

**`_build_agent_nodes`** updates: when iterating skills, route each to the
correct factory (`make_agent_node` for `responsive`,
`make_supervisor_node` for `supervisor`; `monitor` skills are skipped — they
do not become graph nodes).

**Acceptance**:
- Supervisor dispatches correctly to one or many subordinates via `Send()`.
- Recursion depth honoured.
- No `AgentRun` row inserted for the supervisor invocation.

**Tests**: in P6-I.

---

### P6-E — `MonitorRunner` (orchestrator-level singleton)

**Scope**: New module `src/runtime/monitor_runner.py`. Owns one APScheduler
`BackgroundScheduler` instance (or shares Phase 5's; decide at implementation
time based on whether P5 exposes a registry-level scheduler).

**Files**:
- `src/runtime/monitor_runner.py` (new)

**API**:

```python
class MonitorRunner:
    def __init__(self, scheduler, trigger_registry, tool_registry, logger): ...
    def register(self, skill: Skill) -> None: ...
        # validate kind == "monitor", schedule cron job, record handle
    def unregister(self, name: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
        # gracefully halt: scheduler.shutdown(wait=True)
```

**Job execution** (per monitor tick):
1. For each tool name in `observe`, call the tool via the existing tool
   registry/MCP client. Aggregate observations.
2. Evaluate `emit_signal_when` against the observation (safe-eval, R7).
3. If true, look up `trigger_target` in the trigger registry and fire it.
4. Per-tick timeout (R6). On timeout or exception: log + skip; do not crash
   the scheduler.

**Acceptance**:
- Cron schedule honoured.
- Failed tool call or expression eval does not stop the runner.
- `stop()` shuts down cleanly.

**Tests**: in P6-I.

---

### P6-F — `make_monitor_node` factory

**Scope**: A factory that produces the **callable** the `MonitorRunner` runs
per tick. This is intentionally **not** a LangGraph node; the name mirrors the
agent-node factories for symmetry but it returns a plain `Callable[[], None]`
(or `async`).

**Files**:
- `src/runtime/graph.py` (factory placement) **or**
  `src/runtime/monitor_runner.py` (co-located with runner).
  Decide at implementation time; co-location with the runner is preferred to
  avoid pulling LangGraph into a non-graph code path.

**Behaviour**: Returns the callable described in P6-E's "Job execution" so
that `MonitorRunner.register` can hand it to APScheduler.

**Acceptance**: Round-trip from skill YAML → `MonitorRunner.register` → tick
→ trigger fires.

**Tests**: in P6-I.

---

### P6-G — Wire monitor lifecycle to `OrchestratorService`

**Scope**: At service startup, build a `MonitorRunner`, register all skills
where `kind == "monitor"`, and start it. At shutdown, stop the runner.

**Files**:
- `src/runtime/orchestrator.py`

**Order of startup** (additive, do not reorder existing steps):
1. Existing: load config, init storage, init MCP clients.
2. Existing: build trigger registry (Phase 5).
3. **New**: instantiate `MonitorRunner(scheduler, trigger_registry, tool_registry, logger)`.
4. **New**: iterate loaded skills; for each `kind == "monitor"`, call
   `runner.register(skill)`.
5. **New**: call `runner.start()`.
6. Existing: bind API.

**Order of shutdown**:
1. **New**: `monitor_runner.stop()` (before tearing down MCP clients so monitor
   tool calls in-flight can finish or time out).
2. Existing: stop scheduler/registry, close MCP clients, close storage.

**Trigger-target validation** (deferred from P6-B): iterate monitor skills and
verify each `trigger_target` is registered; fail startup with a clear error if
not.

**Acceptance**:
- Service starts with monitor skills present; APScheduler shows monitor jobs.
- Service stops cleanly under SIGTERM and pytest fixture teardown.

**Tests**: in P6-I (lifecycle test that boots a minimal `OrchestratorService`).

---

### P6-H — Audit/log path for supervisor dispatch (no `AgentRun`)

**Scope**: A small structured-logging helper `log_supervisor_dispatch(...)`
emitted from `make_supervisor_node`. Lives in the existing logger module or a
new `src/runtime/audit.py` if cleaner.

**Files**:
- `src/runtime/audit.py` (new) **or** existing logging module.

**Schema** (one log line per dispatch):

```json
{
  "event": "supervisor_dispatch",
  "ts": "<iso8601>",
  "incident_id": "<uuid>",
  "session_id": "<uuid>",
  "supervisor": "<skill_name>",
  "strategy": "llm" | "rule",
  "depth": 1,
  "targets": ["agent_a", "agent_b"],
  "rule_matched": "<rule_when_string>",   // present only for strategy=rule
  "dispatch_payload_size": 1234            // bytes, not full payload
}
```

**Mitigation alignment** (R3): operators querying "what happened in this
incident" join `agent_runs` and the new log stream. Document this in a short
note in the existing audit/runbook docs (do not add a new doc file unless one
already exists; otherwise inline the note in the closest existing operator
doc).

**Acceptance**:
- Every supervisor dispatch emits exactly one log entry.
- Log entry survives JSON-line ingestion (no unserialisable fields).

**Tests**: in P6-I.

---

### P6-I — Test suites

Three suites, written in this order so failures bisect cleanly.

#### I.1 — Skill loader (`tests/runtime/test_skill_schema.py`, `tests/runtime/test_skill_loader.py`)

- Each kind parses correctly with minimal valid config.
- Each forbidden-field-per-kind combination from P6-B is rejected with a
  message naming the field.
- Required-field-missing cases (e.g. monitor without `schedule`) rejected.
- Cron expression rejected when malformed.
- `emit_signal_when` rejected when it fails the safe-eval AST parse.
- Old YAML without `kind` parses as responsive (back-compat).

#### I.2 — Supervisor dispatch (`tests/runtime/test_supervisor_dispatch.py`)

- Build a graph with one supervisor + two responsive subordinates.
- Run a session; assert:
  - Both subordinates were `Send()`-dispatched.
  - **Zero** `AgentRun` rows for the supervisor; subordinates each have one.
  - One `supervisor_dispatch` log entry per dispatch with correct `targets`.
- Recursion bound: build a supervisor whose subordinate is itself; assert the
  graph aborts at `max_dispatch_depth` with a clean error.
- `dispatch_strategy="rule"` path: rule matches the first target; assert the
  matched rule appears in the log entry.

#### I.3 — Monitor firing (`tests/runtime/test_monitor_runner.py`)

- Register a stub monitor whose `observe` tool returns a deterministic
  observation.
- Register a stub trigger that records calls.
- Use APScheduler's `BackgroundScheduler` test mode (or freeze time with
  `freezegun`) to advance the schedule and tick the monitor twice.
- Assert: stub trigger called twice, with expected payload.
- Failure path: stub `observe` tool raises; assert the runner logs and the
  next tick still runs.
- Lifecycle: `runner.stop()` halts further ticks.

**Acceptance**: All three suites pass; coverage on the new modules ≥ 85%.

---

### P6-J — Final verification

Per `~/.claude/rules/testing.md` and the global "Done = Done" checklist. This
task does **no new code**.

1. Re-read the full diff (P6-A → P6-I) end-to-end.
2. Run the project's test suite (unit + integration); confirm green.
3. Run linters, type checks, and build; confirm green.
4. Run dependency audit (`pip-audit` or repo equivalent); High/Critical = block.
5. Spot-check: bring up an `OrchestratorService` with one of each kind in
   `examples/skills/`; confirm:
   - Responsive skill answers a turn.
   - Supervisor dispatches and logs without an `AgentRun`.
   - Monitor fires its trigger on schedule.
6. Re-confirm air-gapped build (`make` from vendored deps; no public network
   calls at runtime).
7. Write a one-paragraph "Phase 6 complete" summary in the PR body covering:
   what changed, follow-ups, known limitations.

---

## 8. Out of scope follow-ups (track as issues, do not include in PR)

- Multi-supervisor topology design (graph of supervisors).
- Streaming bus for monitor → session real-time bridge (R2 alternative).
- Web UI surfaces for monitor status and supervisor dispatch logs.
- Per-supervisor LLM routing strategies beyond `llm` and `rule`.
- Hot-reload of monitor skills without restarting the service.

---

## 9. Constraints (from global rules)

- **No new `.md` files** beyond this plan and any explicitly-requested artifacts.
- **No commits** unless asked; this plan is purely planning output.
- **Air-gapped**: monitor `observe` tools must call only local/internal MCP
  endpoints, never the public internet.
- **Scope discipline**: do not refactor unrelated graph code while extending
  `_build_agent_nodes`. Surface adjacent issues as follow-up notes.
- **Testing**: never report tasks complete on "should work"; verify by running.
- **Performance**: the `MonitorRunner` thread pool is bounded (default 4); each
  tick has a per-monitor timeout. No unbounded queues, ever (R6).
- **Security**: `emit_signal_when` is safe-eval only; no `eval()`/`exec()` on
  user-supplied expressions (R7).
