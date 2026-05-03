# Phase 4 — Risk-rated Tool Gateway + Generic HITL Implementation Plan

**Status:** Draft (plan only, no implementation)
**Date:** 2026-05-02
**Depends on:** Phase 2 (`runtime/graph.py` already uses `interrupt()` and `Command(resume=...)`); Phase 1 (`Session.tool_calls` is the canonical audit log)
**Estimated effort:** 1 week
**Test count target:** +21 new tests (current passing baseline + 21)

---

## 1. Goal + Scope

Phase 1 produced a generic `Session` with a `tool_calls: list[ToolCall]` audit log; Phase 2 wired LangGraph's native `interrupt()` for the confidence gate and `Command(resume=...)` for resume.
Phase 4 generalises the human-in-the-loop pattern from "the confidence gate before resolution" to **every tool call** the framework executes, gated by a declared per-tool risk level and a per-environment override predicate.

The unit of intervention shifts from one bespoke `pending_intervention` per session to **one inline approval per high-risk `ToolCall`**. The audit lives directly on `Session.tool_calls` — no new audit table.

**Locked decisions (non-negotiable, see §10 for rationale):**

| # | Decision | Resolution |
|---|---|---|
| **P4.1** | HITL pause semantics | Hybrid by action: `auto` → no pause; `notify` → soft (no graph pause); `approve` → hard (`interrupt()` pauses the graph). |
| **P4.2** | Risk-policy locus | Gateway-only config. Skills cannot override the risk a tool has been declared with. Source of truth: `examples/incident_management/config.yaml` `gateway:` block (or per-app equivalent). Tool-side `meta.risk` is a *default suggestion* the app config may tighten. |
| **P4.3** | Approval scope | Per-call inline audit on `Session.tool_calls`. Each high-risk invocation has its own approval record; `ToolCall` extends with `risk`, `status`, `approver`, `approved_at`, `approval_rationale`. |
| **EXTRA** | Per-env force-approve | Configurable predicate. When the session's `environment` matches `gateway.prod_env_patterns` AND the tool name matches `gateway.prod_force_approval_tools`, the gateway forces `require_approval` *regardless* of the tool's declared risk. Wrapper runs *before* the risk-tier action lookup. |

**In scope:** gateway middleware (one async wrapper per `BaseTool`), policy resolution, prod-env override hook, `ToolCall` schema extension, generic approval API endpoint, Streamlit approval card, DB migration.

**Out of scope (defer):** approver authentication / RBAC (Phase 8 candidate), notification dispatch beyond a pluggable `Notifier` protocol with a `LoggingNotifier` default (Slack/PagerDuty MCP wiring is app-side and trivial once the protocol exists), approval timeouts as background timers (the API is stateless; if approval never arrives the session sits paused — operationally acceptable for this phase), multi-tool batched approval (one tool, one decision).

---

## 2. Target Architecture After Phase 4

```
src/runtime/
  config.py                     # +GatewayConfig (policy map, prod_env_patterns, prod_force_approval_tools)
  state.py                      # ToolCall extends: risk, status, approver, approved_at, approval_rationale
  graph.py                      # make_agent_node now wraps tools via gateway.wrap_tools(...)
  api.py                        # +POST /sessions/{id}/approvals/{idx}; +GET pending approvals helper
  orchestrator.py               # +resume_with_approval(session_id, idx, decision, rationale, approver)
  tools/                        # NEW PACKAGE
    __init__.py                 # public surface: wrap_tools, GatewayPolicy, resolve_action, Notifier
    gateway.py                  # NEW — wrap_tools(tools, *, policy, session_provider, notifier)
    policy.py                   # NEW — GatewayConfig parsing, resolve_risk, resolve_action, prod-env predicate
    notifier.py                 # NEW — Notifier protocol + LoggingNotifier default impl
  storage/
    models.py                   # ToolCall column unchanged (still JSON list); migration applied via JSON shape upgrade
    session_store.py            # adds back-compat hydration: missing status -> "auto"
    migrations/                 # NEW DIR (alembic-light: a single one-shot data migration script)
      0001_tool_call_status_default.py   # NEW — backfill status="auto" on existing rows

examples/incident_management/
  config.py                     # +GatewayConfig embedded in IncidentAppConfig (or surfaced via load)
  config.yaml                   # +gateway: block with policy entries for the 3 tools + prod_env settings
  mcp_server.py                 # tool docstrings get a [risk: low|medium|high] suffix (advisory only)

src/ui/
  streamlit_app.py              # detail pane: render approval card when an in-flight ToolCall has status="pending"
  api_client.py                 # +submit_approval(session_id, idx, decision, rationale, approver)

tests/
  test_tool_call_schema.py      # NEW — ToolCall back-compat default + new fields
  test_gateway_policy.py        # NEW — resolve_risk, resolve_action, default behaviours
  test_gateway_prod_env.py      # NEW — prod-env override forces approval (the EXTRA requirement)
  test_gateway_wrap_auto.py     # NEW — auto path runs tool, records approver="auto"
  test_gateway_wrap_notify.py   # NEW — notify path invokes notifier AND tool, records status="notified"
  test_gateway_wrap_approve.py  # NEW — approve path emits interrupt(), records status="pending"
  test_gateway_resume_accept.py # NEW — Command(resume=approve) → tool runs → status="approved"
  test_gateway_resume_reject.py # NEW — Command(resume=reject) → tool NOT run → result is error
  test_api_approvals.py         # NEW — POST /sessions/{id}/approvals/{idx} happy path + 404 + 409
  test_streamlit_approval_card.py # NEW — Streamlit renders card when pending ToolCall present
  test_migration_tool_call_status.py # NEW — pre-Phase-4 row hydrates with status="auto"
```

---

## 3. Naming Map / API Changes

| Before | After |
|---|---|
| `runtime.state.ToolCall(agent, tool, args, result, ts)` | + `risk: str \| None`, `status: Literal["auto","approved","notified","pending","rejected","timeout"] = "auto"`, `approver: str \| None = None`, `approved_at: str \| None = None`, `approval_rationale: str \| None = None` |
| (no gateway) | `runtime.tools.gateway.wrap_tools(tools, *, policy, session_provider, notifier)` |
| (no gateway config) | `runtime.config.GatewayConfig{ policy: dict[str, ToolPolicy], prod_env_patterns: list[str], prod_force_approval_tools: list[str] }` |
| (notification ad hoc) | `runtime.tools.notifier.Notifier` protocol; `LoggingNotifier` default |
| `Orchestrator.resume_session(session_id, decision)` (Phase 2) | unchanged for the *confidence* gate. **NEW** `Orchestrator.resume_with_approval(session_id, idx, decision, rationale, approver)` for the tool-gateway approval. Internally both call `graph.invoke(Command(resume=payload), config)` but with distinct payload shapes. |
| (no API) | `POST /sessions/{id}/approvals/{tool_call_index}` body: `{decision: "approve"\|"reject", rationale: str, approver: str}` → 200 `{status: "approved"\|"rejected", session: <session>}` |
| Streamlit detail pane (incident-only) | + render approval card when any in-flight ToolCall has `status="pending"` |

---

## 4. File Structure (decomposition)

Each file has one responsibility:

- `runtime/tools/policy.py` — pure functions, no I/O. `parse_gateway_config`, `resolve_risk`, `prod_env_matches`, `tool_matches_force_list`, `resolve_action`. Easy to unit-test exhaustively.
- `runtime/tools/notifier.py` — `Notifier` Protocol + `LoggingNotifier`. ~30 lines. Apps inject Slack/PagerDuty equivalents via DI.
- `runtime/tools/gateway.py` — the wrapper. Takes a `list[BaseTool]` and a `policy`, returns a list of wrapped tools where each `_arun`/`_run` call goes through the policy/notifier/interrupt path. Records to the `Session` via the supplied `session_provider` callable (avoids importing `SessionStore` directly — keeps the gateway storage-agnostic).
- `runtime/config.py` — extends with `GatewayConfig` (Pydantic). Parsed once at app startup; passed into `wrap_tools`.
- `runtime/state.py` — `ToolCall` schema extension only. No behaviour.
- `runtime/api.py` — one new endpoint + small helper for fetching pending approvals.
- `runtime/orchestrator.py` — `resume_with_approval` delegates to checkpointer-aware resume.
- `ui/streamlit_app.py` — pure presentation; pulls pending state from session API.
- `tests/test_gateway_*.py` — one file per behaviour cluster.

The gateway is intentionally **framework-side**, not example-app-side — the same wrapper handles incident-management, future apps, and the to-be-built `apply_fix` pattern.

---

## 5. Task Breakdown

Tasks are TDD, bite-sized (2–5 min/step), with explicit failing test → minimal code → passing test → commit.

---

### P4-A — Extend `ToolCall` schema (back-compat)

**Why first:** every later step writes to the new fields. Locking the schema first lets every other test rely on it.

**Files:**
- Modify: `src/runtime/state.py` (the `ToolCall` class)
- Test: `tests/test_tool_call_schema.py` (NEW)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_call_schema.py
from runtime.state import ToolCall


def test_toolcall_default_status_is_auto():
    tc = ToolCall(agent="x", tool="y", args={}, result=None, ts="2026-05-02T00:00:00Z")
    assert tc.status == "auto"
    assert tc.risk is None
    assert tc.approver is None
    assert tc.approved_at is None
    assert tc.approval_rationale is None


def test_toolcall_accepts_full_approval_record():
    tc = ToolCall(
        agent="resolution", tool="apply_fix", args={"target": "svc"}, result={"ok": True},
        ts="2026-05-02T00:00:00Z",
        risk="high", status="approved", approver="alice",
        approved_at="2026-05-02T00:01:00Z", approval_rationale="checked rollback path",
    )
    assert tc.status == "approved"
    assert tc.approver == "alice"


def test_toolcall_rejects_invalid_status():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ToolCall(agent="x", tool="y", args={}, result=None,
                 ts="2026-05-02T00:00:00Z", status="not_a_real_status")


def test_toolcall_legacy_payload_round_trips():
    """Pre-Phase-4 row in DB has no status key — must hydrate with default 'auto'."""
    legacy = {"agent": "x", "tool": "y", "args": {}, "result": None,
              "ts": "2026-05-02T00:00:00Z"}
    tc = ToolCall(**legacy)
    assert tc.status == "auto"
```

- [ ] **Step 2: Run test** → `pytest tests/test_tool_call_schema.py -v` → 4 failures (fields don't exist)

- [ ] **Step 3: Minimal implementation**

```python
# src/runtime/state.py — replace ToolCall class
from typing import Literal

ToolCallStatus = Literal["auto", "approved", "notified", "pending", "rejected", "timeout"]


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str
    risk: str | None = None
    status: ToolCallStatus = "auto"
    approver: str | None = None
    approved_at: str | None = None
    approval_rationale: str | None = None
```

- [ ] **Step 4: Run** → all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/state.py tests/test_tool_call_schema.py
git commit -m "feat(state): extend ToolCall with risk/status/approver fields (back-compat default)"
```

---

### P4-B — Define `GatewayConfig` in framework + per-app extension

**Files:**
- Modify: `src/runtime/config.py` (add `GatewayConfig`, attach to `AppConfig`)
- Modify: `examples/incident_management/config.py` (`IncidentAppConfig` exposes gateway block via the framework's `AppConfig.gateway`)
- Modify: `examples/incident_management/config.yaml` (sample gateway block)
- Test: `tests/test_gateway_policy.py` (config parsing portion only — resolve_* tested in P4-C)

- [ ] **Step 1: Failing test** for parsing

```python
# tests/test_gateway_policy.py (config-parsing section only)
from runtime.config import GatewayConfig, ToolPolicy


def test_gateway_config_defaults():
    cfg = GatewayConfig()
    assert cfg.policy == {}
    assert cfg.prod_env_patterns == ["prod*", "production"]
    assert cfg.prod_force_approval_tools == []


def test_gateway_config_parses_yaml_shape():
    raw = {
        "policy": {
            "incident_management:create_incident": {
                "risk": {"default": "medium", "per_env": {"staging": "low"}},
                "action": {"high": "require_approval", "medium": "notify_on_execute", "low": "auto"},
            },
        },
        "prod_env_patterns": ["prod*"],
        "prod_force_approval_tools": ["*update_incident*"],
    }
    cfg = GatewayConfig.model_validate(raw)
    assert "incident_management:create_incident" in cfg.policy
    assert cfg.policy["incident_management:create_incident"].risk.default == "medium"
    assert cfg.policy["incident_management:create_incident"].risk.per_env["staging"] == "low"
    assert cfg.policy["incident_management:create_incident"].action["high"] == "require_approval"
    assert cfg.prod_force_approval_tools == ["*update_incident*"]


def test_gateway_config_invalid_risk_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({
            "policy": {"x": {"risk": {"default": "MAYBE"}, "action": {}}},
        })
```

- [ ] **Step 2: Run** → import-time failures (`GatewayConfig` not defined).

- [ ] **Step 3: Implementation**

```python
# src/runtime/config.py (additions; place near other model classes)
from typing import Literal

RiskLevel = Literal["low", "medium", "high"]
GatewayAction = Literal["auto", "notify_on_execute", "require_approval"]


class RiskTiers(BaseModel):
    default: RiskLevel = "low"
    per_env: dict[str, RiskLevel] = Field(default_factory=dict)


class ToolPolicy(BaseModel):
    risk: RiskTiers = Field(default_factory=RiskTiers)
    action: dict[RiskLevel, GatewayAction] = Field(
        default_factory=lambda: {"low": "auto", "medium": "notify_on_execute", "high": "require_approval"}
    )


class GatewayConfig(BaseModel):
    policy: dict[str, ToolPolicy] = Field(default_factory=dict)
    prod_env_patterns: list[str] = Field(default_factory=lambda: ["prod*", "production"])
    prod_force_approval_tools: list[str] = Field(default_factory=list)


# AppConfig (existing) — add:
class AppConfig(BaseModel):
    # ... existing fields ...
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
```

- [ ] **Step 4: Run** → 3 pass.

- [ ] **Step 5: Add YAML example block** to `examples/incident_management/config.yaml`:

```yaml
gateway:
  policy:
    incident_management:create_incident:
      risk:
        default: medium
        per_env:
          dev: low
      action:
        high: require_approval
        medium: notify_on_execute
        low: auto
    incident_management:update_incident:
      risk: { default: low }
      action: { low: auto, medium: notify_on_execute, high: require_approval }
    incident_management:lookup_similar_incidents:
      risk: { default: low }
      action: { low: auto, medium: notify_on_execute, high: require_approval }
  prod_env_patterns: ["prod*", "production"]
  prod_force_approval_tools:
    - "*update_incident*"
    - "*apply_fix*"
    - "*_resolution_*"
```

- [ ] **Step 6: Commit**

```bash
git add src/runtime/config.py examples/incident_management/config.yaml tests/test_gateway_policy.py
git commit -m "feat(config): add GatewayConfig with policy + prod-env override settings"
```

---

### P4-C — Pure policy resolvers: `resolve_risk` + `resolve_action`

**Files:**
- Create: `src/runtime/tools/__init__.py`
- Create: `src/runtime/tools/policy.py`
- Test: extend `tests/test_gateway_policy.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_gateway_policy.py (additions)
from runtime.config import GatewayConfig, ToolPolicy, RiskTiers
from runtime.tools.policy import resolve_risk, resolve_action, prod_env_matches, tool_matches


def _cfg() -> GatewayConfig:
    return GatewayConfig.model_validate({
        "policy": {
            "incident_management:create_incident": {
                "risk": {"default": "medium", "per_env": {"staging": "low"}},
                "action": {"high": "require_approval", "medium": "notify_on_execute", "low": "auto"},
            },
        },
        "prod_env_patterns": ["prod*", "production"],
        "prod_force_approval_tools": ["*apply_fix*"],
    })


def test_resolve_risk_uses_per_env_when_set():
    assert resolve_risk("incident_management:create_incident", env="staging", cfg=_cfg()) == "low"


def test_resolve_risk_falls_back_to_default():
    assert resolve_risk("incident_management:create_incident", env="dev", cfg=_cfg()) == "medium"


def test_resolve_risk_unknown_tool_returns_low():
    """Default policy: any tool with no entry is 'low'/auto — preserves current behaviour."""
    assert resolve_risk("unknown:tool", env="prod", cfg=_cfg()) == "low"


def test_resolve_action_maps_risk_to_action():
    assert resolve_action(risk="high", tool="incident_management:create_incident", env="dev", cfg=_cfg()) == "require_approval"
    assert resolve_action(risk="medium", tool="incident_management:create_incident", env="dev", cfg=_cfg()) == "notify_on_execute"
    assert resolve_action(risk="low", tool="incident_management:create_incident", env="dev", cfg=_cfg()) == "auto"


def test_prod_env_predicate_matches_glob():
    assert prod_env_matches("production", ["prod*", "production"]) is True
    assert prod_env_matches("prod-eu-west-1", ["prod*", "production"]) is True
    assert prod_env_matches("staging", ["prod*", "production"]) is False
    assert prod_env_matches("preprod", ["prod*"]) is True  # documented: prod* matches preprod


def test_tool_matches_glob_list():
    assert tool_matches("incident_management:apply_fix", ["*apply_fix*"]) is True
    assert tool_matches("incident_management:lookup_similar_incidents", ["*apply_fix*"]) is False
```

- [ ] **Step 2: Run** → import errors.

- [ ] **Step 3: Implementation**

```python
# src/runtime/tools/__init__.py
"""Framework tool gateway: risk-tiered HITL middleware around BaseTool."""

# src/runtime/tools/policy.py
from __future__ import annotations
from fnmatch import fnmatchcase

from runtime.config import GatewayConfig, GatewayAction, RiskLevel


def resolve_risk(tool: str, *, env: str, cfg: GatewayConfig) -> RiskLevel:
    """Look up the configured risk for ``tool`` in the given ``env``.

    Default for unconfigured tools is "low" — preserves current behaviour
    where every tool ran without intervention.
    """
    policy = cfg.policy.get(tool)
    if policy is None:
        return "low"
    return policy.risk.per_env.get(env, policy.risk.default)


def resolve_action(*, risk: RiskLevel, tool: str, env: str, cfg: GatewayConfig) -> GatewayAction:
    policy = cfg.policy.get(tool)
    if policy is None:
        return "auto"
    return policy.action.get(risk, "auto")


def prod_env_matches(env: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(env, p) for p in patterns)


def tool_matches(tool: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(tool, p) for p in patterns)
```

- [ ] **Step 4: Run** → 6 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/tools/__init__.py src/runtime/tools/policy.py tests/test_gateway_policy.py
git commit -m "feat(tools): pure resolvers for risk + action + prod-env globs"
```

---

### P4-D — Notifier protocol + default `LoggingNotifier`

**Files:**
- Create: `src/runtime/tools/notifier.py`
- Test: inline assertion inside `test_gateway_wrap_notify.py` (next task)

Tiny task — kept separate so the gateway wrapper has a clean dependency.

- [ ] **Step 1: Implementation**

```python
# src/runtime/tools/notifier.py
from __future__ import annotations
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def notify(
        self,
        *,
        tool: str,
        risk: str,
        session_id: str,
        agent: str,
        args: dict,
    ) -> None: ...


class LoggingNotifier:
    """Default notifier — emits a structured log line. Apps inject Slack/PagerDuty equivalents."""
    async def notify(self, *, tool: str, risk: str, session_id: str, agent: str, args: dict) -> None:
        logger.info(
            "tool_notify session=%s agent=%s tool=%s risk=%s args=%s",
            session_id, agent, tool, risk, args,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/runtime/tools/notifier.py
git commit -m "feat(tools): Notifier protocol + LoggingNotifier default"
```

---

### P4-E — Prod-env force-approval predicate (composable)

**Files:**
- Modify: `src/runtime/tools/policy.py` (add the wrapper)
- Test: `tests/test_gateway_prod_env.py` (NEW)

This is the **EXTRA requirement**: the predicate runs *before* the risk-tier action lookup and forces `require_approval` when both predicates match.

- [ ] **Step 1: Failing test**

```python
# tests/test_gateway_prod_env.py
from runtime.config import GatewayConfig
from runtime.tools.policy import effective_action


def _cfg(force_tools=None, env_patterns=None):
    return GatewayConfig.model_validate({
        "policy": {
            "x:apply_fix": {
                "risk": {"default": "low"},
                "action": {"low": "auto", "medium": "notify_on_execute", "high": "require_approval"},
            },
        },
        "prod_env_patterns": env_patterns or ["prod*", "production"],
        "prod_force_approval_tools": force_tools or [],
    })


def test_low_risk_tool_in_prod_with_force_list_requires_approval():
    """The mandated extra: low-risk tool + prod env + force-list match → require_approval."""
    action = effective_action(tool="x:apply_fix", env="production", cfg=_cfg(force_tools=["*apply_fix*"]))
    assert action == "require_approval"


def test_low_risk_tool_in_prod_without_force_list_match_unchanged():
    action = effective_action(tool="x:apply_fix", env="production", cfg=_cfg(force_tools=["*remediation*"]))
    assert action == "auto"


def test_low_risk_tool_in_dev_with_force_list_match_unchanged():
    """Force list only fires when env predicate also matches."""
    action = effective_action(tool="x:apply_fix", env="dev", cfg=_cfg(force_tools=["*apply_fix*"]))
    assert action == "auto"


def test_high_risk_tool_in_prod_already_requires_approval():
    cfg = GatewayConfig.model_validate({
        "policy": {"x:y": {"risk": {"default": "high"}, "action": {"high": "require_approval"}}},
    })
    assert effective_action(tool="x:y", env="production", cfg=cfg) == "require_approval"


def test_force_list_glob_matches_resolution_pattern():
    """Spec example: '*_resolution_*' matches resolution-trigger tools."""
    cfg = _cfg(force_tools=["*_resolution_*"])
    assert effective_action(tool="x:trigger_resolution_apply", env="production", cfg=cfg) == "require_approval"
```

- [ ] **Step 2: Run** → `effective_action` not defined.

- [ ] **Step 3: Implementation**

```python
# src/runtime/tools/policy.py — append
from runtime.config import GatewayAction


def effective_action(*, tool: str, env: str, cfg: GatewayConfig) -> GatewayAction:
    """Resolve the final gateway action.

    Order:
    1. Prod-env override predicate — if env matches AND tool matches force list,
       returns "require_approval" unconditionally.
    2. Otherwise, risk-tier lookup: resolve_risk → resolve_action.

    The override layer composes with risk: it can only *tighten* the policy
    (force approval), never relax it.
    """
    if prod_env_matches(env, cfg.prod_env_patterns) and tool_matches(tool, cfg.prod_force_approval_tools):
        return "require_approval"
    risk = resolve_risk(tool, env=env, cfg=cfg)
    return resolve_action(risk=risk, tool=tool, env=env, cfg=cfg)
```

- [ ] **Step 4: Run** → 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/tools/policy.py tests/test_gateway_prod_env.py
git commit -m "feat(tools): prod-env override forces approval before risk lookup"
```

---

### P4-F — Gateway wrapper: `auto` action path

**Files:**
- Create: `src/runtime/tools/gateway.py`
- Test: `tests/test_gateway_wrap_auto.py` (NEW)

The wrapper takes a `list[BaseTool]` (LangChain) and a `session_provider: Callable[[], Session]` so it can append to `session.tool_calls`. Following the `_handle_agent_failure` pattern in `graph.py`, the gateway uses an injected callable rather than importing `SessionStore` — keeps it storage-agnostic and simple to mock.

- [ ] **Step 1: Failing test**

```python
# tests/test_gateway_wrap_auto.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import StructuredTool

from runtime.config import GatewayConfig
from runtime.tools.gateway import wrap_tools
from runtime.tools.notifier import LoggingNotifier
from runtime.state import Session, ToolCall


def _make_session(env: str) -> Session:
    return Session(
        id="sess-1", status="in_progress",
        created_at="2026-05-02T00:00:00Z", updated_at="2026-05-02T00:00:00Z",
    )


@pytest.mark.asyncio
async def test_auto_path_runs_tool_and_records_status_auto():
    session = _make_session(env="dev")
    inner = AsyncMock(return_value={"ok": True})
    base = StructuredTool.from_function(
        coroutine=inner, name="incident_management:lookup_similar_incidents",
        description="lookup", args_schema=None,
    )
    cfg = GatewayConfig()  # default = no policy → auto
    save = MagicMock()
    [wrapped] = wrap_tools(
        [base], policy=cfg, env="dev", agent="intake",
        session_provider=lambda: session, save_session=save,
        notifier=LoggingNotifier(),
    )

    result = await wrapped.ainvoke({"query": "x"})

    assert result == {"ok": True}
    assert len(session.tool_calls) == 1
    tc = session.tool_calls[0]
    assert tc.tool == "incident_management:lookup_similar_incidents"
    assert tc.status == "auto"
    assert tc.risk == "low"
    assert tc.approver == "auto"
    save.assert_called_once_with(session)
```

- [ ] **Step 2: Run** → `wrap_tools` not defined.

- [ ] **Step 3: Implementation (auto path only — notify + approve in P4-G/H)**

```python
# src/runtime/tools/gateway.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from runtime.config import GatewayConfig
from runtime.state import Session, ToolCall
from runtime.tools.notifier import Notifier
from runtime.tools.policy import effective_action, resolve_risk

_TS = "%Y-%m-%dT%H:%M:%SZ"


def wrap_tools(
    tools: list[BaseTool],
    *,
    policy: GatewayConfig,
    env: str,
    agent: str,
    session_provider: Callable[[], Session],
    save_session: Callable[[Session], None],
    notifier: Notifier,
) -> list[BaseTool]:
    """Wrap each tool with a risk-tier middleware.

    Returns a parallel list of StructuredTool instances. The wrapper preserves
    the inner tool's name + description (LangChain ReAct relies on these for
    LLM disambiguation) and delegates to it after policy resolution.
    """
    return [_wrap_one(t, policy=policy, env=env, agent=agent,
                       session_provider=session_provider,
                       save_session=save_session, notifier=notifier)
            for t in tools]


def _wrap_one(t: BaseTool, *, policy, env, agent, session_provider, save_session, notifier) -> BaseTool:
    name = t.name

    async def _gated(**kwargs):
        session = session_provider()
        risk = resolve_risk(name, env=env, cfg=policy)
        action = effective_action(tool=name, env=env, cfg=policy)
        ts = datetime.now(timezone.utc).strftime(_TS)

        if action == "auto":
            result = await t.ainvoke(kwargs)
            session.tool_calls.append(ToolCall(
                agent=agent, tool=name, args=kwargs, result=result, ts=ts,
                risk=risk, status="auto", approver="auto", approved_at=ts,
            ))
            save_session(session)
            return result
        # P4-G + P4-H wire notify and approve — for now, fall through to auto.
        result = await t.ainvoke(kwargs)
        session.tool_calls.append(ToolCall(
            agent=agent, tool=name, args=kwargs, result=result, ts=ts,
            risk=risk, status="auto", approver="auto", approved_at=ts,
        ))
        save_session(session)
        return result

    return StructuredTool.from_function(
        coroutine=_gated, name=name, description=t.description,
        args_schema=t.args_schema,
    )
```

- [ ] **Step 4: Run** → passes.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/tools/gateway.py tests/test_gateway_wrap_auto.py
git commit -m "feat(tools): gateway wrapper - auto action path"
```

---

### P4-G — Gateway wrapper: `notify` action path

**Files:**
- Modify: `src/runtime/tools/gateway.py`
- Test: `tests/test_gateway_wrap_notify.py` (NEW)

- [ ] **Step 1: Failing test**

```python
# tests/test_gateway_wrap_notify.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import StructuredTool

from runtime.config import GatewayConfig
from runtime.tools.gateway import wrap_tools
from runtime.state import Session


def _sess():
    return Session(id="s", status="in_progress",
                   created_at="2026-05-02T00:00:00Z", updated_at="2026-05-02T00:00:00Z")


@pytest.mark.asyncio
async def test_notify_path_invokes_notifier_and_runs_tool():
    cfg = GatewayConfig.model_validate({
        "policy": {"t": {"risk": {"default": "medium"}, "action": {"medium": "notify_on_execute"}}},
    })
    inner = AsyncMock(return_value="ok")
    notifier = MagicMock()
    notifier.notify = AsyncMock()
    base = StructuredTool.from_function(coroutine=inner, name="t", description="d", args_schema=None)
    session = _sess()
    [wrapped] = wrap_tools([base], policy=cfg, env="dev", agent="A",
                            session_provider=lambda: session, save_session=MagicMock(),
                            notifier=notifier)
    out = await wrapped.ainvoke({})
    assert out == "ok"
    notifier.notify.assert_awaited_once()
    assert session.tool_calls[-1].status == "notified"
    assert session.tool_calls[-1].risk == "medium"
    assert session.tool_calls[-1].approver == "auto"
```

- [ ] **Step 2: Run** → currently records `status="auto"`, fails.

- [ ] **Step 3: Implementation** — replace the `else` fallthrough in `_wrap_one`:

```python
        if action == "notify_on_execute":
            await notifier.notify(
                tool=name, risk=risk, session_id=session.id, agent=agent, args=kwargs,
            )
            result = await t.ainvoke(kwargs)
            session.tool_calls.append(ToolCall(
                agent=agent, tool=name, args=kwargs, result=result, ts=ts,
                risk=risk, status="notified", approver="auto", approved_at=ts,
            ))
            save_session(session)
            return result
```

- [ ] **Step 4: Run** → passes.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/tools/gateway.py tests/test_gateway_wrap_notify.py
git commit -m "feat(tools): gateway wrapper - notify action path"
```

---

### P4-H — Gateway wrapper: `require_approval` path via `interrupt()`

**Files:**
- Modify: `src/runtime/tools/gateway.py`
- Test: `tests/test_gateway_wrap_approve.py` (NEW)

The `require_approval` path:
1. Records the ToolCall with `status="pending"` first (so the UI can render).
2. Calls `interrupt({...})` — LangGraph pauses; on resume, the human's payload is in the resume value.
3. Inspects the resume payload; if `decision="approve"`, runs the tool and updates the same ToolCall to `status="approved"` with approver/rationale/approved_at; if `reject`, records `status="rejected"` and returns `{"error": "rejected: <rationale>"}` as the tool result.

The ToolCall index used for approval is `len(session.tool_calls) - 1` at the moment `interrupt()` is called (i.e., the just-appended pending entry).

- [ ] **Step 1: Failing test**

```python
# tests/test_gateway_wrap_approve.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt

from runtime.config import GatewayConfig
from runtime.tools.gateway import wrap_tools
from runtime.state import Session


def _sess():
    return Session(id="s", status="in_progress",
                   created_at="2026-05-02T00:00:00Z", updated_at="2026-05-02T00:00:00Z")


@pytest.mark.asyncio
async def test_approve_path_emits_interrupt_and_records_pending():
    cfg = GatewayConfig.model_validate({
        "policy": {"t": {"risk": {"default": "high"}, "action": {"high": "require_approval"}}},
    })
    inner = AsyncMock(return_value="ok")
    base = StructuredTool.from_function(coroutine=inner, name="t", description="d", args_schema=None)
    session = _sess()
    save = MagicMock()
    [wrapped] = wrap_tools([base], policy=cfg, env="prod", agent="A",
                            session_provider=lambda: session, save_session=save,
                            notifier=MagicMock())

    with pytest.raises(GraphInterrupt) as ei:
        await wrapped.ainvoke({"x": 1})

    # The pending ToolCall is recorded BEFORE the interrupt
    assert session.tool_calls[-1].status == "pending"
    assert session.tool_calls[-1].risk == "high"
    inner.assert_not_awaited()  # tool not executed yet
    # The interrupt payload carries enough for UI to show the card
    payload = ei.value.value if hasattr(ei.value, "value") else ei.value.args[0]
    assert payload["kind"] == "tool_approval"
    assert payload["tool"] == "t"
    assert payload["tool_call_index"] == len(session.tool_calls) - 1
```

- [ ] **Step 2: Run** → fails (no interrupt path yet).

- [ ] **Step 3: Implementation** — append to `_wrap_one`:

```python
        # action == "require_approval"
        session.tool_calls.append(ToolCall(
            agent=agent, tool=name, args=kwargs, result=None, ts=ts,
            risk=risk, status="pending",
        ))
        save_session(session)
        idx = len(session.tool_calls) - 1
        decision_payload = interrupt({
            "kind": "tool_approval",
            "session_id": session.id,
            "tool": name,
            "agent": agent,
            "args": kwargs,
            "risk": risk,
            "tool_call_index": idx,
        })
        # On resume, decision_payload is whatever the caller passed via Command(resume=...)
        return await _apply_decision(
            session=session, save_session=save_session,
            inner=t, kwargs=kwargs, idx=idx, decision_payload=decision_payload,
        )


async def _apply_decision(*, session: Session, save_session, inner: BaseTool, kwargs: dict,
                           idx: int, decision_payload: dict):
    pending = session.tool_calls[idx]
    decision = (decision_payload or {}).get("decision", "reject")
    rationale = (decision_payload or {}).get("rationale", "")
    approver = (decision_payload or {}).get("approver", "unknown")
    now = datetime.now(timezone.utc).strftime(_TS)

    if decision == "approve":
        result = await inner.ainvoke(kwargs)
        pending.result = result
        pending.status = "approved"
        pending.approver = approver
        pending.approved_at = now
        pending.approval_rationale = rationale
        save_session(session)
        return result

    # reject path
    err = {"error": f"rejected: {rationale}"}
    pending.result = err
    pending.status = "rejected"
    pending.approver = approver
    pending.approved_at = now
    pending.approval_rationale = rationale
    save_session(session)
    return err
```

- [ ] **Step 4: Run** → passes.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/tools/gateway.py tests/test_gateway_wrap_approve.py
git commit -m "feat(tools): gateway wrapper - approve path via interrupt() with pending audit"
```

---

### P4-I — Wire gateway into `make_agent_node`

**Files:**
- Modify: `src/runtime/graph.py` (specifically the `_build_agent_nodes` function around the `tools = registry.resolve(skill.tools, cfg.mcp)` line)
- Test: extend `tests/test_agent_node.py` with one wiring assertion

The hook is small: where `make_agent_node` receives `tools: list[BaseTool]`, the gateway wraps them before they reach `create_react_agent`. The session provider closure reads `state["incident"]` (Phase-1 alias) / `state["session"]`; the save callback reuses the existing `store.save(...)` path.

- [ ] **Step 1: Failing test**

```python
# tests/test_agent_node.py — append
def test_make_agent_node_wraps_tools_via_gateway(monkeypatch):
    """The tools list reaching create_react_agent must be wrapped, not raw."""
    from runtime import graph as g
    captured = {}

    def fake_wrap(tools, **kw):
        captured["count"] = len(tools)
        captured["env"] = kw["env"]
        return tools

    monkeypatch.setattr(g, "wrap_tools", fake_wrap)
    # ... build skill + tools + call _build_agent_nodes (existing fixture pattern)
    assert captured.get("count") == EXPECTED
    assert captured.get("env") in {"production", "staging", "dev"}  # whatever the fixture session uses
```

- [ ] **Step 2: Run** → fails.

- [ ] **Step 3: Implementation**

```python
# src/runtime/graph.py — at top
from runtime.tools.gateway import wrap_tools
from runtime.tools.notifier import LoggingNotifier

# inside _build_agent_nodes, after tools = registry.resolve(...):
        wrapped = wrap_tools(
            tools,
            policy=cfg.gateway,
            env=getattr(state_session_factory(), "environment", "dev"),
            agent=agent_name,
            session_provider=lambda s=state_session_factory: s(),
            save_session=store.save,
            notifier=LoggingNotifier(),
        )
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=wrapped,  # was: tools=tools
            decide_route=decide, store=store,
            valid_signals=valid_signals,
        )
```

Note: `state_session_factory` is a small helper that reads the current session out of the running graph state at *node entry time* — the lambda captures the closure correctly because each tool call happens inside the running node where `state["incident"]` is live. The implementor will route the session reference through `make_agent_node` (it already calls `incident = state["incident"]`) — concretely the cleanest approach is to construct the wrappers *inside* the node body instead of at factory time. That refactor lives in this task.

- [ ] **Step 4: Run** → passes; confirm full test suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/graph.py tests/test_agent_node.py
git commit -m "feat(graph): route tool calls through risk-tier gateway"
```

---

### P4-J — Resume payload handling: approve + reject end-to-end

**Files:**
- Modify: `src/runtime/orchestrator.py` (add `resume_with_approval`)
- Test: `tests/test_gateway_resume_accept.py` (NEW), `tests/test_gateway_resume_reject.py` (NEW)

This task validates the **full integration**: build a graph with a high-risk tool, invoke, hit `interrupt()`, then resume with `Command(resume={...})`, confirm the tool either runs (approve) or returns the error (reject) and the ToolCall is updated in place.

- [ ] **Step 1: Failing tests**

```python
# tests/test_gateway_resume_accept.py
@pytest.mark.asyncio
async def test_high_risk_tool_pause_resume_approve_runs_tool(test_orchestrator_factory):
    """Full flow: invoke → interrupt → resume(approve) → tool executes → status='approved'."""
    orch = await test_orchestrator_factory(gateway_high_risk_tool="incident_management:create_incident")
    sid = await orch.start_session(...)
    # First invoke pauses
    state = await orch.session(sid)
    assert any(tc.status == "pending" for tc in state.tool_calls)
    pending = next(tc for tc in state.tool_calls if tc.status == "pending")
    assert pending.result is None

    await orch.resume_with_approval(
        session_id=sid,
        tool_call_index=state.tool_calls.index(pending),
        decision="approve",
        rationale="ack rollback path",
        approver="alice",
    )
    state = await orch.session(sid)
    updated = state.tool_calls[state.tool_calls.index(pending)]
    assert updated.status == "approved"
    assert updated.approver == "alice"
    assert updated.result is not None
```

```python
# tests/test_gateway_resume_reject.py
@pytest.mark.asyncio
async def test_high_risk_tool_pause_resume_reject_returns_error(test_orchestrator_factory):
    orch = await test_orchestrator_factory(gateway_high_risk_tool="incident_management:create_incident")
    sid = await orch.start_session(...)
    state = await orch.session(sid)
    pending = next(tc for tc in state.tool_calls if tc.status == "pending")
    await orch.resume_with_approval(
        session_id=sid, tool_call_index=state.tool_calls.index(pending),
        decision="reject", rationale="not safe", approver="bob",
    )
    state = await orch.session(sid)
    rejected = state.tool_calls[state.tool_calls.index(pending)]
    assert rejected.status == "rejected"
    assert rejected.approver == "bob"
    assert isinstance(rejected.result, dict) and "error" in rejected.result
    assert "rejected: not safe" in rejected.result["error"]
```

- [ ] **Step 2: Run** → no `resume_with_approval`.

- [ ] **Step 3: Implementation**

```python
# src/runtime/orchestrator.py — append
from langgraph.types import Command


async def resume_with_approval(
    self,
    *,
    session_id: str,
    tool_call_index: int,
    decision: Literal["approve", "reject"],
    rationale: str,
    approver: str,
) -> Session:
    """Resume a graph paused at a tool-gateway interrupt with an approval decision.

    Distinct from ``resume_session`` (which handles the confidence gate). The
    payload shape is the contract the gateway's interrupt expects:
    ``{decision, rationale, approver}``.
    """
    payload = {"decision": decision, "rationale": rationale, "approver": approver,
               "tool_call_index": tool_call_index}
    cfg = {"configurable": {"thread_id": session_id}}
    await self._graph.ainvoke(Command(resume=payload), config=cfg)
    return await self.session(session_id)
```

- [ ] **Step 4: Run** → both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/orchestrator.py tests/test_gateway_resume_accept.py tests/test_gateway_resume_reject.py
git commit -m "feat(orchestrator): resume_with_approval for tool-gateway interrupts"
```

---

### P4-K — Approval API endpoint

**Files:**
- Modify: `src/runtime/api.py`
- Test: `tests/test_api_approvals.py` (NEW)

- [ ] **Step 1: Failing test**

```python
# tests/test_api_approvals.py
def test_post_approval_approve_decision(test_app):
    # Setup: paused session with one pending ToolCall at index 0
    sid = "sess-pending-1"
    seed_paused_session(test_app, sid, pending_idx=0)
    r = test_app.post(
        f"/sessions/{sid}/approvals/0",
        json={"decision": "approve", "rationale": "ok", "approver": "alice"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert body["session"]["tool_calls"][0]["status"] == "approved"


def test_post_approval_404_on_unknown_session(test_app):
    r = test_app.post("/sessions/nope/approvals/0",
                       json={"decision": "approve", "rationale": "", "approver": "x"})
    assert r.status_code == 404


def test_post_approval_409_when_index_not_pending(test_app):
    """Cannot approve a call that's already 'auto' or 'approved'."""
    sid = "sess-finished-1"
    seed_completed_session(test_app, sid)
    r = test_app.post(f"/sessions/{sid}/approvals/0",
                       json={"decision": "approve", "rationale": "", "approver": "x"})
    assert r.status_code == 409
```

- [ ] **Step 2: Run** → no endpoint.

- [ ] **Step 3: Implementation**

```python
# src/runtime/api.py — add
from typing import Literal

class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    rationale: str = ""
    approver: str


@app.post("/sessions/{session_id}/approvals/{tool_call_index}")
async def submit_approval(session_id: str, tool_call_index: int, body: ApprovalRequest):
    orch: Orchestrator = app.state.orchestrator
    try:
        session = await orch.session(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    if tool_call_index >= len(session.tool_calls):
        raise HTTPException(404, "tool_call_index out of range")
    if session.tool_calls[tool_call_index].status != "pending":
        raise HTTPException(409, "tool call is not pending approval")
    session = await orch.resume_with_approval(
        session_id=session_id, tool_call_index=tool_call_index,
        decision=body.decision, rationale=body.rationale, approver=body.approver,
    )
    return {"status": session.tool_calls[tool_call_index].status, "session": session.model_dump()}
```

- [ ] **Step 4: Run** → 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/api.py tests/test_api_approvals.py
git commit -m "feat(api): POST /sessions/{id}/approvals/{idx} for tool-gateway HITL"
```

---

### P4-L — Streamlit approval card

**Files:**
- Modify: `src/ui/streamlit_app.py` (the detail-pane render path)
- Modify: `src/ui/api_client.py` (add the call)
- Test: `tests/test_streamlit_approval_card.py` (NEW — uses Streamlit's `AppTest` runner)

The card renders inside the existing detail pane, gated on "session has at least one ToolCall with status='pending'":
- Headline: tool name + risk badge
- Body: agent, args (pretty JSON, collapsible), rationale textarea, approver name input, two buttons (Approve / Reject)
- On click: calls the API, refreshes the session view

UI rules from `~/.claude/rules/ui.md` apply: sharp corners (matches existing accordion shape), restrained palette, badge-rich header, keyboard accessible. The card respects `prefers-reduced-motion` (no expand animation on the args block).

- [ ] **Step 1: Failing test**

```python
# tests/test_streamlit_approval_card.py
from streamlit.testing.v1 import AppTest

def test_approval_card_renders_when_pending_present(monkeypatch, paused_session_fixture):
    monkeypatch.setattr("ui.api_client.get_session", lambda sid: paused_session_fixture)
    at = AppTest.from_file("src/ui/streamlit_app.py")
    at.session_state["selected_incident"] = paused_session_fixture["id"]
    at.run()
    cards = [w for w in at.text if "Tool approval required" in w.value]
    assert len(cards) == 1
    # Two buttons visible
    btn_labels = {b.label for b in at.button}
    assert {"Approve", "Reject"}.issubset(btn_labels)


def test_approval_card_absent_when_no_pending(monkeypatch, completed_session_fixture):
    monkeypatch.setattr("ui.api_client.get_session", lambda sid: completed_session_fixture)
    at = AppTest.from_file("src/ui/streamlit_app.py")
    at.session_state["selected_incident"] = completed_session_fixture["id"]
    at.run()
    assert not any("Tool approval required" in w.value for w in at.text)
```

- [ ] **Step 2: Run** → fails.

- [ ] **Step 3: Implementation** — in detail pane:

```python
# src/ui/streamlit_app.py — inside render_incident_detail
def _render_approval_cards(inc: dict) -> None:
    pending = [(i, tc) for i, tc in enumerate(inc.get("tool_calls", [])) if tc.get("status") == "pending"]
    for idx, tc in pending:
        with st.container(border=True):
            st.markdown(
                f"**Tool approval required** &nbsp;"
                f"`{tc['tool']}` &nbsp;"
                f"<span class='risk-badge risk-{tc.get('risk','low')}'>"
                f"{tc.get('risk','low').upper()}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"agent: {tc['agent']}")
            with st.expander("Args", expanded=False):
                st.json(tc.get("args", {}))
            rationale = st.text_area("Rationale", key=f"appr_rat_{inc['id']}_{idx}")
            approver = st.text_input("Approver", key=f"appr_who_{inc['id']}_{idx}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve", key=f"appr_yes_{inc['id']}_{idx}", type="primary",
                              disabled=not approver.strip()):
                    submit_approval(inc["id"], idx, "approve", rationale, approver)
                    st.rerun()
            with c2:
                if st.button("Reject", key=f"appr_no_{inc['id']}_{idx}",
                              disabled=not approver.strip()):
                    submit_approval(inc["id"], idx, "reject", rationale, approver)
                    st.rerun()
```

```python
# src/ui/api_client.py — add
def submit_approval(session_id: str, idx: int, decision: str, rationale: str, approver: str) -> dict:
    r = requests.post(
        f"{API_BASE}/sessions/{session_id}/approvals/{idx}",
        json={"decision": decision, "rationale": rationale, "approver": approver},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()
```

- [ ] **Step 4: Run** → passes.

- [ ] **Step 5: Commit**

```bash
git add src/ui/streamlit_app.py src/ui/api_client.py tests/test_streamlit_approval_card.py
git commit -m "feat(ui): approval card for pending tool calls in detail pane"
```

---

### P4-M — DB migration: backfill `status="auto"` on legacy `tool_calls` rows

**Files:**
- Create: `src/runtime/storage/migrations/__init__.py`
- Create: `src/runtime/storage/migrations/0001_tool_call_status_default.py`
- Modify: `src/runtime/storage/session_store.py` (hydration is already covered by the Pydantic default from P4-A — this migration brings *persisted JSON* up to date so audit views are clean)
- Test: `tests/test_migration_tool_call_status.py` (NEW)

Since `tool_calls` is a JSON column (per `storage/models.py`), no DDL is needed — the migration is a one-shot Python script that walks every row and rewrites missing keys. It is idempotent.

- [ ] **Step 1: Failing test**

```python
# tests/test_migration_tool_call_status.py
def test_migration_backfills_status_auto(in_memory_db):
    # Seed a legacy tool_calls JSON list with no status key
    legacy = [{"agent": "x", "tool": "y", "args": {}, "result": None, "ts": "2026-05-02T00:00:00Z"}]
    insert_raw_session(in_memory_db, sid="leg-1", tool_calls_json=legacy)

    from runtime.storage.migrations import zero_zero_zero_one_tool_call_status_default as m
    m.run(in_memory_db)

    row = fetch_raw_session(in_memory_db, "leg-1")
    assert row["tool_calls"][0]["status"] == "auto"
    # Idempotent re-run is a no-op
    m.run(in_memory_db)
    assert row["tool_calls"][0]["status"] == "auto"


def test_migration_does_not_clobber_explicit_status(in_memory_db):
    pre = [{"agent": "x", "tool": "y", "args": {}, "result": None,
            "ts": "2026-05-02T00:00:00Z", "status": "approved", "approver": "alice"}]
    insert_raw_session(in_memory_db, sid="ok-1", tool_calls_json=pre)
    from runtime.storage.migrations import zero_zero_zero_one_tool_call_status_default as m
    m.run(in_memory_db)
    row = fetch_raw_session(in_memory_db, "ok-1")
    assert row["tool_calls"][0]["status"] == "approved"
    assert row["tool_calls"][0]["approver"] == "alice"
```

- [ ] **Step 2: Run** → fails.

- [ ] **Step 3: Implementation**

```python
# src/runtime/storage/migrations/__init__.py — empty package init

# src/runtime/storage/migrations/0001_tool_call_status_default.py
"""One-shot migration: add status='auto' to tool_calls entries lacking it.

Idempotent. Safe to run on every startup if needed.
"""
from sqlalchemy import select, update

from runtime.storage.models import IncidentRow


def run(engine) -> int:
    """Returns the number of rows touched."""
    touched = 0
    with engine.begin() as conn:
        rows = conn.execute(select(IncidentRow.id, IncidentRow.tool_calls)).all()
        for sid, tcs in rows:
            if not tcs:
                continue
            changed = False
            new_tcs = []
            for tc in tcs:
                if "status" not in tc:
                    tc = {**tc, "status": "auto"}
                    changed = True
                new_tcs.append(tc)
            if changed:
                conn.execute(update(IncidentRow).where(IncidentRow.id == sid).values(tool_calls=new_tcs))
                touched += 1
    return touched
```

Wire-in: orchestrator startup calls `migrations.zero_zero_zero_one_tool_call_status_default.run(engine)` once. (One line in `Orchestrator.create`.)

- [ ] **Step 4: Run** → passes.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/storage/migrations/ src/runtime/orchestrator.py tests/test_migration_tool_call_status.py
git commit -m "feat(storage): idempotent migration backfilling tool_calls.status='auto'"
```

---

### P4-N — Final verification + bundle regen

**Files touched:** none new — verification only.

- [ ] Run full test suite: `pytest tests/ -v` exits 0.
- [ ] Run `pytest tests/ --collect-only -q | wc -l` and confirm new test count.
- [ ] Run linters/type checks per project conventions.
- [ ] Regenerate `dist/app.py` bundle: `python scripts/build_single_file.py`.
- [ ] Smoke-test in dev env: start a session that triggers a high-risk tool, observe pause, hit the API, observe resume.
- [ ] Smoke-test prod-env override: run same low-risk tool with `environment=production` and tool name in `prod_force_approval_tools` — confirm pause occurs.
- [ ] Update memory `MEMORY.md` if any new gotchas surfaced.

- [ ] Commit any bundle/doc updates:

```bash
git add dist/app.py
git commit -m "build: regen bundle after Phase 4 (gateway + HITL)"
```

---

## 6. Done Criteria

**Test suite:**
- `pytest tests/ -v` exits 0.
- 21 new tests added, all passing. No new skips/xfails.
- `pytest tests/test_gateway_*.py -v` exits 0 in isolation.

**Grep checks:**
- `grep -rn "from runtime.tools.gateway import wrap_tools" src/runtime/graph.py` → 1 hit.
- `grep -rn "interrupt(" src/runtime/tools/gateway.py` → exactly 1 hit (the require_approval branch).
- `grep -rn "pending_intervention" src/runtime/tools/` → 0 hits (gateway state lives on `tool_calls`, not `pending_intervention`).

**Behavioural checks:**
- Existing confidence-gate tests still pass (no regression in Phase-2 path).
- A session whose `environment="production"` AND whose tool name matches `prod_force_approval_tools` always pauses, regardless of declared risk.
- An existing pre-Phase-4 SQLite DB opens cleanly; running the migration once leaves all `tool_calls[].status == "auto"`.

**Operational:**
- `dist/app.py` builds and starts.
- Streamlit detail pane renders the approval card; clicking Approve/Reject triggers an HTTP call and a session refresh.

---

## 7. Risks & Mitigations

### R1 — Two sources of truth for risk: tool metadata vs. app config

**Risk:** A tool ships with `[risk: high]` in its description, but the app config sets `risk.default: low` for it. Which wins? If undisciplined, drift produces silent under-protection.

**Mitigation (locks the rule per P4.2):** App config is authoritative. Tool-side metadata (description suffix or future `Tool.meta.risk`) is purely advisory — a *suggestion* for the app author when wiring up the YAML. The gateway never reads tool metadata at runtime; it only consults `cfg.gateway.policy[tool_name]`. If the app config has no entry, default is `risk=low` / `action=auto` (preserves current behaviour). Document this clearly in `src/runtime/tools/policy.py` docstring and in the YAML example. A lint check (out of scope for P4) could later flag mismatches.

### R2 — Notification side channel

**Risk:** Where does `notify_on_execute` actually go? Webhook? Slack? Logged?

**Mitigation:** Phase 4 ships only the `Notifier` Protocol + a `LoggingNotifier` default that emits a structured log line. Apps that need Slack/PagerDuty wire a custom implementation by passing it to the orchestrator constructor (one extra kwarg). This keeps Phase 4 scope tight while remaining pluggable. Document the expected shape in the `Notifier` Protocol docstring.

### R3 — Approver authentication

**Risk:** P4 has no auth model. Anyone hitting `/sessions/{id}/approvals/{idx}` can claim to be `alice`.

**Mitigation:** P4 explicitly defers RBAC. The API accepts `approver: str` from the body and records it verbatim. Document in the API docstring: *"Phase 4 has no auth — `approver` is a self-claimed string, deployment behind a trusted ingress (mTLS / VPN) is assumed. RBAC lands in a future phase."* The `approver` field is a string, so adding auth later (e.g., reading from a JWT) is purely additive.

### R4 — Re-entrancy: closing the browser mid-approval

**Risk:** A user opens the approval card, closes the browser. Can a second user (or same user from a new tab) resume the session?

**Mitigation:** Yes, by design. The pending state lives in two places: (a) the LangGraph checkpointer — `interrupt()` keeps the thread paused regardless of which client started it; (b) the `Session.tool_calls[idx].status == "pending"` JSON in the DB — the UI polls this, so any new client renders the same approval card. The `POST /approvals/{idx}` endpoint is idempotent at the LangGraph level — only the first valid `Command(resume=...)` advances the thread. A second submit hits the `409 not pending` guard. The recovery path is therefore: any approver from any browser tab can complete the approval; no exclusivity lock needed for P4.

### R5 — Backward compat: legacy `tool_calls` rows lack the new fields

**Risk:** A user upgrades to Phase 4 with a non-empty `incidents.db`. Existing rows have `tool_calls` JSON entries with no `status`/`risk` keys. Pydantic hydration tolerates this (P4-A default), but the *raw JSON* in the DB is asymmetric, which makes audit queries painful.

**Mitigation:** P4-M ships an idempotent JSON-walk migration that backfills `status="auto"` on every legacy row. The orchestrator runs it once on startup. Test coverage: legacy row hydrates correctly; migration is idempotent; explicit non-default values are not clobbered.

### R6 — Interaction with Phase-2 confidence-gate `interrupt()`

**Risk:** Two distinct interrupt sites now exist (confidence gate, tool gateway). On resume, the orchestrator must dispatch the right payload to the right site.

**Mitigation:** Each interrupt payload carries a `kind` field (`"confidence_gate"` vs `"tool_approval"`). The orchestrator exposes two distinct resume methods (`resume_session` for the gate, `resume_with_approval` for the gateway), each emitting a payload shape the corresponding site expects. LangGraph's resume mechanism doesn't multiplex — only one interrupt is in flight per thread at a time, so dispatch is mechanical. Add an integration test: graph with both a confidence gate and a high-risk tool, ensure each pause/resume cycle uses the right method.

### R7 — Tool wrapper changes the agent's tool surface

**Risk:** `create_react_agent` introspects tool names/descriptions to build the LLM prompt. Wrapping tools must preserve `name`, `description`, and `args_schema` exactly — otherwise the LLM sees a different tool surface and the ReAct loop breaks.

**Mitigation:** `_wrap_one` constructs `StructuredTool.from_function` with `name=t.name, description=t.description, args_schema=t.args_schema`. A unit assertion in `test_gateway_wrap_auto.py` confirms the wrapped tool's `.name` matches the original. Manual smoke test in P4-N exercises the full ReAct loop end-to-end.

---

## 8. Bundling & Build

- `dist/app.py` is regenerated by `scripts/build_single_file.py` after final verification (P4-N). The bundler walks `src/runtime/` so the new `tools/` package is picked up automatically — no bundler changes expected. If the bundler missed any new module, fix in P4-N before commit.
- All new dependencies are Python stdlib (`fnmatch`, `logging`, `typing.Protocol`) plus already-vendored LangGraph (`langgraph.types.interrupt`, `Command`). No new packages required — see `~/.claude/rules/build.md`.

---

## 9. Self-Review Checklist (pre-execution)

- [ ] Every task has a failing-test step before implementation.
- [ ] Every task ends with a commit — no big-bang merges.
- [ ] No file is created that already exists with a different responsibility.
- [ ] The locked decisions (P4.1/P4.2/P4.3 + EXTRA) are reflected verbatim in the code surface.
- [ ] The mandated extra (prod-env override before risk lookup) has its own dedicated test file (P4-E).
- [ ] No new `*.md` files outside this plan; no documentation files created speculatively.
- [ ] No `apply_fix`-specific code paths added — the existing bespoke flow gets retired in a follow-up phase that re-expresses it as a `gateway.policy` entry.

---

## 10. Rationale: why these locked decisions

- **P4.1 hybrid** maps cleanly onto the existing graph: `auto` is the no-op; `notify` is a one-line side effect that doesn't change graph topology; `approve` reuses Phase 2's `interrupt()` machinery, so we don't re-invent suspension. Three tiers, three implementations, zero surprise.
- **P4.2 gateway-only** prevents skill prompts from racing the security policy. Skills evolve fast; risk policy must change deliberately. Single source = single review surface.
- **P4.3 inline audit** avoids a parallel `tool_audit` table that would duplicate `tool_calls`. The session row is already the audit log; extending it is cheaper than a join.
- **EXTRA prod-env override** is a one-function predicate that runs first. It cannot relax the policy — only tighten it — so it composes safely with the risk lookup.

---

## 11. Execution Handoff

This plan is consumable by `superpowers:executing-plans`. Tasks P4-A through P4-N are independent enough that P4-D and P4-F can be parallelised by a subagent dispatch (see `superpowers:subagent-driven-development`); the rest must run sequentially because each builds on the prior commit.

**Total tasks:** 14 (P4-A … P4-N)
**Total new tests:** 21
**Estimated effort:** 1 week single-engineer, 3 days with subagent parallelisation on P4-D/P4-F/P4-L.
