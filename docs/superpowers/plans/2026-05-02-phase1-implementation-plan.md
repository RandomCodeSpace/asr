# Phase 1 — Domain Extraction + Rename + Repository Split

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Synthesis note:** Three drafters were dispatched in parallel. Architect agent produced a complete plan (this content). Codex CLI couldn't read source files because Gemini broke out of plan mode mid-flight and renamed `src/orchestrator/` → `src/runtime/` before the codex run started — the working tree was reverted to `5315c1c` and Gemini's auto-execution discarded. Gemini's plan-mode draft (lines 1–258) confirmed the structural choices below: rename, repository split, examples/ layout, dist/ split. The architect's draft is the synthesized plan below.

**Goal:** Lift every incident-flavored type, field, config key, MCP tool, and skill out of the framework. Establish `examples/incident_management/` as the flagship example app. Rename framework-side `Incident → Session`. Split `IncidentRepository` into `SessionStore` (active) + `HistoryStore` (closed).

**Baseline:** commit `5315c1c`, 191 tests passing.

**Estimated effort:** 1 week.

---

## 1. Goal + Scope

Phase 1 extracts every incident-management concept from the generic framework layer and establishes `examples/incident_management/` as the self-contained flagship application. Concretely: the Pydantic `Incident` model is renamed `Session` in the framework (incident-specific fields migrate to `examples/incident_management/state.py` as `IncidentState(Session)`); `IncidentRepository` is split into `SessionStore` (active CRUD + vector writes) and `HistoryStore` (closed-session similarity search), both living in `src/runtime/storage/`; the `orchestrator.mcp_servers.incident` module moves verbatim to `examples/incident_management/mcp_server.py`; the four skills YAMLs and the app-level config keys (`incidents.*`, `orchestrator.severity_aliases`, `intervention.*`, `environments`) relocate to `examples/incident_management/`; and the source tree under `src/orchestrator/` is renamed `src/runtime/` with import paths updated throughout. The `dist/` bundle splits into `dist/app.py` (runtime) and `dist/apps/incident-management.py` (app entry). The Streamlit UI is updated to import from the example app's own module, keeping all UI behavior identical. The 191 existing tests keep passing after import-path surgery; no behavior changes.

**What does not change:** SQLAlchemy schema (`incidents` table name stays — it is the example app's table), FAISS/pgvector storage logic, LangGraph graph construction, skill YAML format, MCP tool contracts, embeddings stack, or any external API endpoint.

---

## 2. Target File Layout

```
src/
└── runtime/                            ← renamed from src/orchestrator/
    ├── __init__.py                     (empty, unchanged)
    ├── config.py                       ← strip incidents.*, orchestrator.severity_aliases,
    │                                      intervention.*, environments from AppConfig
    ├── state.py                        ← NEW: Session base model (generic fields only)
    ├── similarity.py                   (unchanged)
    ├── skill.py                        (unchanged)
    ├── llm.py                          (unchanged)
    ├── mcp_loader.py                   (unchanged)
    ├── api.py                          (unchanged, import paths updated)
    ├── graph.py                        ← Incident → Session, IncidentRepository → SessionStore
    ├── orchestrator.py                 ← façade; incident-flavored method names stay
    │                                      as thin shims delegating to generic names
    ├── storage/
    │   ├── __init__.py
    │   ├── engine.py                   (unchanged)
    │   ├── embeddings.py               (unchanged)
    │   ├── models.py                   ← IncidentRow table stays; SessionRow alias added
    │   ├── vector.py                   (unchanged)
    │   ├── session_store.py            ← NEW: active CRUD extracted from IncidentRepository
    │   ├── history_store.py            ← NEW: find_similar + list_recent extracted
    │   └── repository.py               ← becomes a thin compat shim:
    │                                      IncidentRepository = SessionStore + HistoryStore
    └── mcp_servers/
        ├── observability.py            (unchanged, import paths updated)
        ├── remediation.py              (unchanged, import paths updated)
        └── user_context.py             (unchanged, import paths updated)
        # incident.py REMOVED from here

examples/
└── incident_management/                ← Python package (underscore)
    ├── __init__.py
    ├── config.yaml                     ← app-level config (severity_aliases, intervention,
    │                                      environments, incidents.similarity_threshold)
    ├── state.py                        ← IncidentState(Session) + IncidentStatus type
    ├── mcp_server.py                   ← moved from src/orchestrator/mcp_servers/incident.py
    ├── skills/                         ← moved from config/skills/
    │   ├── _common/
    │   ├── intake/
    │   ├── triage/
    │   ├── deep_investigator/
    │   └── resolution/
    ├── ui.py                           ← moved from ui/streamlit_app.py (import paths updated)
    └── README.md

config/
└── config.yaml                        ← retains only framework keys; app keys removed;
                                          mcp.servers entry updated to new module path

dist/
├── app.py                             ← runtime bundle (was: orchestrator core + mcp servers)
└── apps/
    └── incident-management.py         ← app bundle (runtime + example app entry)

scripts/
└── build_single_file.py               ← updated SRC_ROOT, CORE_MODULE_ORDER, UI path,
                                          new build_incident_app() function
```

**Note on directory naming:** Python cannot import hyphenated package names. Use `examples/incident_management/` (underscore) for the directory. Human-facing labels (READMEs, CLI help) can use the colloquial `incident-management`.

---

## 3. Naming Map

| Old symbol / path | New symbol / path | Notes |
|---|---|---|
| `src/orchestrator/` (package) | `src/runtime/` | Directory rename; all `from orchestrator.X import Y` → `from runtime.X import Y` |
| `orchestrator.incident.Incident` | `runtime.state.Session` | Framework base; incident-flavored fields removed |
| `orchestrator.incident.IncidentStatus` | `examples.incident_management.state.IncidentStatus` | Domain Literal moved to app |
| `orchestrator.incident.Reporter` | `examples.incident_management.state.Reporter` | Domain model moved |
| `orchestrator.incident.ToolCall` | `runtime.state.ToolCall` | Generic; stays in framework |
| `orchestrator.incident.TokenUsage` | `runtime.state.TokenUsage` | Generic; stays in framework |
| `orchestrator.incident.AgentRun` | `runtime.state.AgentRun` | Generic; stays in framework |
| `orchestrator.incident._INC_ID_RE` | `examples.incident_management.state._INC_ID_RE` | App-specific |
| `orchestrator.incident._UTC_TS_FMT` | `runtime.state._UTC_TS_FMT` | Stays framework |
| `orchestrator.storage.repository.IncidentRepository` | `runtime.storage.session_store.SessionStore` + `runtime.storage.history_store.HistoryStore` | Split; `IncidentRepository` kept as shim in `repository.py` |
| `orchestrator.storage.models.IncidentRow` | `runtime.storage.models.IncidentRow` (alias `SessionRow`) | Table name `incidents` unchanged; alias added |
| `orchestrator.storage.models.Base` | `runtime.storage.models.Base` | Unchanged |
| `orchestrator.mcp_servers.incident` | `examples.incident_management.mcp_server` | Module path changes; `set_state` contract unchanged |
| `orchestrator.config.IncidentConfig` | `examples.incident_management.config.IncidentAppConfig` | Extracted from AppConfig |
| `orchestrator.config.InterventionConfig` | `examples.incident_management.config.InterventionConfig` | Extracted from AppConfig |
| `orchestrator.config.AppConfig.incidents` | removed from `AppConfig` | Lives in example app's config |
| `orchestrator.config.AppConfig.environments` | removed from `AppConfig` | Lives in example app's config |
| `orchestrator.config.AppConfig.intervention` | removed from `AppConfig` | Lives in example app's config |
| `orchestrator.config.OrchestratorConfig.severity_aliases` | removed from `OrchestratorConfig` | Lives in example app's `mcp_server.py` |
| `config/skills/` | `examples/incident_management/skills/` | Skill YAMLs moved |
| `config/config.yaml` (incidents/intervention/environments keys) | `examples/incident_management/config.yaml` | App keys extracted |
| `ui/streamlit_app.py` | `examples/incident_management/ui.py` | Moved; imports updated |
| `dist/app.py` (bundle) | `dist/app.py` (framework only) | Bundle scope narrows |
| `dist/ui.py` (bundle) | `dist/apps/incident-management.py` | Bundled app entry |
| `Orchestrator.get_incident` | `Orchestrator.get_session` + `get_incident` shim | Thin wrapper kept for UI compat |
| `Orchestrator.list_recent_incidents` | `Orchestrator.list_recent_sessions` + shim | Same |
| `Orchestrator.delete_incident` | `Orchestrator.delete_session` + shim | Same |
| `Orchestrator.start_investigation` | `Orchestrator.start_session` + shim | Same |
| `Orchestrator.stream_investigation` | `Orchestrator.stream_session` + shim | Same |
| `Orchestrator.resume_investigation` | `Orchestrator.resume_session` + shim | Same |
| `GraphState.incident` | `GraphState.session` (with `incident` alias kept during P1) | TypedDict field rename; bridge alias removed in Phase 2 |

---

## 4. Task Breakdown

Tasks are labeled **P1-A through P1-N**. Each task leaves the test suite green at its commit boundary. P1-A through P1-C are the structural foundation; all others build on them.

> **Implementer guidance:** dispatch each task to a fresh subagent (`ruflo-core:coder`) using `superpowers:subagent-driven-development`. Each task brief includes the goal, files touched, TDD steps with full code, and the commit message. After each task, verify `pytest -q` is green before proceeding.

### P1-A — Create `src/runtime/` package mirroring `src/orchestrator/`

Establish the new package name without deleting the old one. Both co-exist; this is the zero-breakage bridge step.

**Files:**
- Create: `src/runtime/__init__.py`
- Create: `src/runtime/storage/__init__.py`
- Create: `src/runtime/mcp_servers/__init__.py`

**Steps:**

- [ ] **1. Failing test** — create `tests/test_runtime_package.py`:

```python
"""Smoke test: runtime package is importable and exposes the same surface."""

def test_runtime_package_importable():
    import runtime  # noqa: F401


def test_runtime_config_importable():
    from runtime.config import AppConfig  # noqa: F401


def test_runtime_state_importable():
    from runtime.state import Session  # noqa: F401  (state.py created in P1-B)
```

- [ ] **2. Run** — `pytest tests/test_runtime_package.py -v` → expected `ModuleNotFoundError: No module named 'runtime'`.

- [ ] **3. Implementation** — create the three `__init__.py` files. Update `pyproject.toml` (or `setup.cfg`) to include `runtime`, `runtime.storage`, `runtime.mcp_servers` in the `packages` list.

```python
# src/runtime/__init__.py
"""Generic agent-orchestration runtime.

Application code lives in ``examples/<app-name>/``. Do not import
application-specific symbols here.
"""
```

(`storage/__init__.py` and `mcp_servers/__init__.py` are empty.)

- [ ] **4. Run** — first two assertions pass; third still fails (state.py not yet created in P1-B).

- [ ] **5. Commit** — `chore: scaffold src/runtime/ package skeleton`

### P1-B — Create `src/runtime/state.py` (generic `Session` base model)

Define the generic session model with only framework-owned fields.

**Files:**
- Create: `src/runtime/state.py`

**Steps:**

- [ ] **1. Add tests** to `tests/test_runtime_package.py`:

```python
def test_session_has_generic_fields():
    from runtime.state import Session
    s = Session(
        id="test-001",
        status="new",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
    )
    assert s.id == "test-001"
    assert s.agents_run == []
    assert s.tool_calls == []
    assert s.findings == {}
    assert s.pending_intervention is None
    assert s.token_usage.total_tokens == 0


def test_session_has_no_incident_fields():
    from runtime.state import Session
    fields = set(Session.model_fields.keys())
    incident_only = {"environment", "reporter", "query", "severity",
                     "category", "matched_prior_inc", "embedding",
                     "resolution", "tags"}
    leaked = incident_only & fields
    assert not leaked, f"Session leaks incident fields: {leaked}"
```

- [ ] **2. Run** → fail (`ModuleNotFoundError: No module named 'runtime.state'`).

- [ ] **3. Implementation** — create `src/runtime/state.py`:

```python
"""Generic session model — the framework's unit of work.

A ``Session`` is the in-progress (or archived) record of one agent run.
Applications extend this via subclassing::

    class IncidentState(Session):
        environment: str
        reporter: Reporter
        ...

``Session`` deliberately contains *no* domain-specific fields. Adding one
here is a framework regression — all domain fields belong in the example
app's ``state.py``.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_rationale: str | None = None
    signal: str | None = None


class Session(BaseModel):
    """Framework base session. Lifecycle + telemetry fields only.

    Applications subclass this and add domain fields. The framework only
    reads/writes the fields declared here.
    """
    id: str
    status: str  # values are app-defined; framework uses "deleted" for soft-delete
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    findings: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
```

- [ ] **4. Run** → all pass.

- [ ] **5. Commit** — `feat(runtime): add generic Session base model`

### P1-C — Create `examples/incident_management/state.py` (`IncidentState`)

Define `IncidentState(Session)` extending the base with all incident-specific fields.

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/incident_management/__init__.py`
- Create: `examples/incident_management/state.py`
- Create: `tests/test_incident_state.py`

**Steps:**

- [ ] **1. Failing tests** — create `tests/test_incident_state.py`:

```python
"""Tests for examples/incident_management/state.py."""
import re


def test_incident_state_importable():
    from examples.incident_management.state import IncidentState  # noqa


def test_incident_state_inherits_session():
    from runtime.state import Session
    from examples.incident_management.state import IncidentState
    assert issubclass(IncidentState, Session)


def test_incident_state_has_domain_fields():
    from examples.incident_management.state import IncidentState, Reporter
    inc = IncidentState(
        id="INC-20260502-001",
        status="new",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
        query="latency spike in payments",
        environment="production",
        reporter=Reporter(id="user-1", team="platform"),
    )
    assert inc.environment == "production"
    assert inc.severity is None
    assert inc.tags == []


def test_incident_status_values():
    from examples.incident_management.state import IncidentStatus
    import typing
    expected = {"new", "in_progress", "matched", "resolved",
                "escalated", "awaiting_input", "stopped", "deleted"}
    args = set(typing.get_args(IncidentStatus))
    assert args == expected


def test_id_format_validation():
    from examples.incident_management.state import _INC_ID_RE
    assert re.match(_INC_ID_RE, "INC-20260502-001")
    assert not re.match(_INC_ID_RE, "SESSION-001")
```

- [ ] **2. Run** → `ModuleNotFoundError: No module named 'examples'`.

- [ ] **3. Implementation**

`examples/__init__.py` (empty).

`examples/incident_management/__init__.py` (empty).

`examples/incident_management/state.py`:

```python
"""Incident-management domain state.

``IncidentState`` extends ``Session`` with all incident-specific fields.
The framework never imports this module; only example-app code does.
"""
from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, Field

from runtime.state import Session, TokenUsage, AgentRun, ToolCall  # noqa: F401

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

IncidentStatus = Literal[
    "new", "in_progress", "matched", "resolved",
    "escalated", "awaiting_input", "stopped", "deleted",
]


class Reporter(BaseModel):
    id: str
    team: str


class IncidentState(Session):
    """Incident-specific session fields."""
    query: str
    environment: str
    reporter: Reporter
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    severity: str | None = None
    category: str | None = None
    matched_prior_inc: str | None = None
    embedding: list[float] | None = None
    resolution: Any = None
```

Update `pyproject.toml` to include `examples`, `examples.incident_management`.

- [ ] **4. Run** → all pass.

- [ ] **5. Commit** — `feat(examples): add IncidentState extending Session`

### P1-D — Copy framework modules into `src/runtime/` with updated imports

Mechanical copy-and-rewrite step. No logic changes. After this task, both `src/orchestrator/` and `src/runtime/` exist as parallel packages; P1-H makes orchestrator a shim.

**Files:** Every `.py` in `src/orchestrator/` + subdirs gets copied to `src/runtime/` with import path rewrite.

**Steps:**

- [ ] **1. Add test** to `tests/test_runtime_package.py`:

```python
def test_graph_state_uses_session():
    from runtime.graph import GraphState
    import typing
    hints = typing.get_type_hints(GraphState)
    from runtime.state import Session
    assert hints.get("session") is Session, (
        f"GraphState.session should be Session, got {hints.get('session')}"
    )
```

- [ ] **2. Run** → fail (`runtime.graph` doesn't exist).

- [ ] **3. Implementation** — for each module in `src/orchestrator/`, copy to `src/runtime/`:

For each of `config.py`, `similarity.py`, `skill.py`, `llm.py`, `mcp_loader.py`, `api.py`, `storage/engine.py`, `storage/embeddings.py`, `storage/models.py`, `storage/vector.py`, `mcp_servers/observability.py`, `mcp_servers/remediation.py`, `mcp_servers/user_context.py`:
- Copy the file verbatim.
- Sed: `from orchestrator.X` → `from runtime.X`.
- Sed: `import orchestrator.X` → `import runtime.X`.

For `graph.py`:
- Copy the file.
- Update imports: `from orchestrator.incident import Incident, ToolCall, AgentRun, TokenUsage, _UTC_TS_FMT` → `from runtime.state import Session as Incident, ToolCall, AgentRun, TokenUsage, _UTC_TS_FMT`.
- Add bridge alias: `GraphState` keeps both `incident` and `session` fields:

```python
class GraphState(TypedDict, total=False):
    session: Session       # canonical name
    incident: Session      # compat alias — remove in Phase 2
    next_route: str | None
    last_agent: str | None
    gated_target: str | None
    error: str | None
```

For `orchestrator.py`:
- Copy.
- Update imports.
- (Generic method names + shims land in P1-K.)

For `storage/models.py`:
- Copy.
- Add at bottom: `SessionRow = IncidentRow  # generic alias`.

- [ ] **4. Run** → test passes; full suite still 191 (orchestrator path still primary).

- [ ] **5. Commit** — `refactor: copy framework modules to src/runtime/ with import rewrite`

### P1-E — Strip incident-specific keys from `runtime/config.py`; extract to example app

Remove `IncidentConfig`, `InterventionConfig`, `AppConfig.incidents`, `AppConfig.environments`, `AppConfig.intervention`, `OrchestratorConfig.severity_aliases` from `runtime/config.py`. Extract to example app.

**Files:**
- Modify: `src/runtime/config.py`
- Create: `examples/incident_management/config.py`
- Create: `examples/incident_management/config.yaml`
- Modify: `config/config.yaml` (strip incident-flavored sections)

**Steps:**

- [ ] **1. Failing tests** — append to `tests/test_runtime_package.py`:

```python
def test_runtime_config_no_incident_keys():
    from runtime.config import AppConfig
    fields = set(AppConfig.model_fields.keys())
    incident_keys = {"incidents", "intervention", "environments"}
    leaked = incident_keys & fields
    assert not leaked, f"AppConfig leaks domain keys: {leaked}"


def test_runtime_config_no_severity_aliases():
    from runtime.config import OrchestratorConfig
    fields = set(OrchestratorConfig.model_fields.keys())
    assert "severity_aliases" not in fields
```

Append to `tests/test_incident_state.py`:

```python
def test_incident_app_config_importable():
    from examples.incident_management.config import IncidentAppConfig  # noqa


def test_incident_app_config_has_required_keys():
    from examples.incident_management.config import IncidentAppConfig
    cfg = IncidentAppConfig()
    assert "production" in cfg.environments
    assert cfg.similarity_threshold == 0.2
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
```

- [ ] **2. Run** → failures on both files.

- [ ] **3. Implementation**

In `src/runtime/config.py`:
- Delete the `IncidentConfig` class.
- Delete the `InterventionConfig` class.
- Remove `incidents: IncidentConfig` from `AppConfig`.
- Remove `environments: list[str]` from `AppConfig`.
- Remove `intervention: InterventionConfig` from `AppConfig`.
- In `OrchestratorConfig`: remove `severity_aliases` field.

Create `examples/incident_management/config.py`:

```python
"""Incident-management application config.

Application-specific settings extracted from the framework's AppConfig.
Loaded separately via ``load_incident_app_config``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
import yaml


class IncidentAppConfig(BaseModel):
    """All incident-management keys extracted from the old framework AppConfig."""
    store_path: str = "incidents"
    similarity_threshold: float = 0.2
    similarity_method: Literal["keyword", "embedding"] = "keyword"
    confidence_threshold: float = 0.75
    escalation_teams: list[str] = Field(
        default_factory=lambda: [
            "platform-oncall", "data-oncall", "security-oncall",
        ],
    )
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"]
    )
    severity_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
            "critical": "high", "urgent": "high", "high": "high",
            "sev3": "medium", "p3": "medium", "moderate": "medium",
            "medium": "medium",
            "sev4": "low", "p4": "low", "info": "low",
            "informational": "low", "low": "low",
        }
    )


_APP_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_incident_app_config(path: str | Path | None = None) -> IncidentAppConfig:
    p = Path(path) if path else _APP_CONFIG_PATH
    if not p.exists():
        return IncidentAppConfig()
    raw = yaml.safe_load(p.read_text()) or {}
    return IncidentAppConfig(**raw)
```

Create `examples/incident_management/config.yaml`:

```yaml
# Incident-management application configuration.
# Framework keys (llm, mcp, storage, paths, orchestrator) live in config/config.yaml.
similarity_threshold: 0.2
similarity_method: keyword
confidence_threshold: 0.75
escalation_teams:
  - platform-oncall
  - data-oncall
  - security-oncall
environments: [production, staging, dev, local]
severity_aliases:
  sev1: high
  sev2: high
  p1: high
  p2: high
  critical: high
  urgent: high
  high: high
  sev3: medium
  p3: medium
  moderate: medium
  medium: medium
  sev4: low
  p4: low
  info: low
  informational: low
  low: low
```

Edit `config/config.yaml`:
- Remove `incidents:`, `intervention:`, `environments:` blocks.
- Remove `orchestrator.severity_aliases` if present.

- [ ] **4. Run** → tests pass.

- [ ] **5. Commit** — `refactor: strip domain config keys from runtime AppConfig; extract to examples/incident_management/config.py`

### P1-F — Move incident MCP server to `examples/incident_management/mcp_server.py`

The MCP server for incident management is app code. It should not live in the framework package.

**Files:**
- Create: `examples/incident_management/mcp_server.py`
- Modify: `config/config.yaml` (MCP server `module:` path)
- Modify: `src/runtime/orchestrator.py` (`_INCIDENT_MCP_MODULE` constant)
- Modify: `src/orchestrator/mcp_servers/incident.py` (becomes shim)
- Modify: `src/runtime/mcp_servers/incident.py` (becomes shim) → actually delete, since it shouldn't exist in `runtime/mcp_servers/` at all

**Steps:**

- [ ] **1. Failing test** — create `tests/test_incident_mcp_server.py`:

```python
"""Tests that the incident MCP server lives in the example app, not the framework."""


def test_incident_mcp_server_importable_from_example():
    from examples.incident_management.mcp_server import IncidentMCPServer  # noqa


def test_incident_mcp_server_has_three_tools():
    from examples.incident_management.mcp_server import IncidentMCPServer
    srv = IncidentMCPServer()
    tool_names = {t.name for t in srv.mcp._tool_manager._tools.values()}
    assert tool_names == {
        "lookup_similar_incidents",
        "create_incident",
        "update_incident",
    }


def test_framework_does_not_own_incident_mcp():
    """The framework's mcp_servers package must not contain incident.py."""
    import importlib.util
    spec = importlib.util.find_spec("runtime.mcp_servers.incident")
    assert spec is None, (
        "runtime.mcp_servers.incident should not exist; "
        "incident MCP server belongs in examples.incident_management.mcp_server"
    )
```

- [ ] **2. Run** → all three fail.

- [ ] **3. Implementation**

Create `examples/incident_management/mcp_server.py` — copy of `src/orchestrator/mcp_servers/incident.py` with these adjustments:
- Imports updated: `from runtime.storage.repository import IncidentRepository`.
- `_DEFAULT_SEVERITY_ALIASES` removed; default loaded from `IncidentAppConfig`.
- Module-level `mcp = _default_server.mcp` and `set_state` shim preserved (loader contract).

Full source (matches the architect's Section 4 P1-F snippet):

```python
"""FastMCP server: incident_management tools, backed by IncidentRepository.

Part of the incident-management example application. Framework code does
not import this module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fastmcp import FastMCP

from runtime.storage.repository import IncidentRepository
from examples.incident_management.config import load_incident_app_config


def normalize_severity(
    value: str | None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentMCPServer:
    """FastMCP server bound to a single IncidentRepository."""
    repository: IncidentRepository | None = None
    severity_aliases: dict[str, str] = field(
        default_factory=lambda: load_incident_app_config().severity_aliases
    )
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(
        self, *,
        repository: IncidentRepository,
        severity_aliases: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases

    def _require_repo(self) -> IncidentRepository:
        if self.repository is None:
            raise RuntimeError(
                "incident_management server not initialized — call configure() first"
            )
        return self.repository

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        repo = self._require_repo()
        hits = repo.find_similar(query=query, environment=environment, limit=5)
        return {"matches": [
            {"id": i.id, "summary": i.summary, "resolution": i.resolution,
             "score": round(s, 3)}
            for i, s in hits
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    reporter_id: str = "user-mock",
                                    reporter_team: str = "platform") -> dict:
        inc = self._require_repo().create(
            query=query, environment=environment,
            reporter_id=reporter_id, reporter_team=reporter_team,
        )
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        repo = self._require_repo()
        inc = repo.load(incident_id)
        if "status" in patch:
            inc.status = patch["status"]
        if "severity" in patch:
            inc.severity = normalize_severity(patch["severity"], self.severity_aliases)
        if "category" in patch:
            inc.category = patch["category"]
        if "summary" in patch:
            inc.summary = patch["summary"]
        if "tags" in patch:
            inc.tags = list(patch["tags"])
        if "matched_prior_inc" in patch:
            inc.matched_prior_inc = patch["matched_prior_inc"]
        if "resolution" in patch:
            inc.resolution = patch["resolution"]
        for key, value in patch.items():
            if key.startswith("findings_"):
                inc.findings[key[len("findings_"):]] = value
        repo.save(inc)
        return inc.model_dump()


_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(
    *,
    repository: IncidentRepository,
    severity_aliases: dict[str, str] | None = None,
) -> None:
    _default_server.configure(repository=repository, severity_aliases=severity_aliases)


async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str,
                          reporter_id: str = "user-mock",
                          reporter_team: str = "platform") -> dict:
    return await _default_server._tool_create_incident(
        query, environment, reporter_id, reporter_team
    )


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)
```

Update `config/config.yaml` MCP entry:

```yaml
mcp:
  servers:
  - name: local_inc
    transport: in_process
    module: examples.incident_management.mcp_server
    category: incident_management
```

Update `src/runtime/orchestrator.py`:

```python
_INCIDENT_MCP_MODULE = "examples.incident_management.mcp_server"
```

Convert `src/orchestrator/mcp_servers/incident.py` to a shim:

```python
"""Backward-compat shim. Import from examples.incident_management.mcp_server."""
from examples.incident_management.mcp_server import (  # noqa: F401
    IncidentMCPServer, normalize_severity, set_state, mcp,
    lookup_similar_incidents, create_incident, update_incident,
)
```

Delete `src/runtime/mcp_servers/incident.py` if it was created in P1-D (the framework runtime package must not own this).

- [ ] **4. Run** → all 191 + 3 new tests pass.

- [ ] **5. Commit** — `refactor: move incident MCP server to examples/incident_management/mcp_server.py`

### P1-G — Split `IncidentRepository` into `SessionStore` + `HistoryStore`

Core repository split. `SessionStore` owns active session lifecycle (create, load, save, delete, list). `HistoryStore` owns closed-session similarity search.

**Files:**
- Create: `src/runtime/storage/session_store.py`
- Create: `src/runtime/storage/history_store.py`
- Create: `src/runtime/storage/repository.py` (compat shim)
- Modify: `src/runtime/storage/models.py` (add `SessionRow` alias if not done in P1-D)

**Steps:**

- [ ] **1. Failing tests** — create `tests/test_session_store.py`:

```python
"""Tests for SessionStore (active CRUD) and HistoryStore (similarity search)."""
import pytest
from sqlalchemy import create_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.history_store import HistoryStore


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    e = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def store(engine):
    return SessionStore(engine=engine)


@pytest.fixture()
def history(engine):
    return HistoryStore(engine=engine, embedder=None, vector_store=None,
                        similarity_threshold=0.5)


def test_session_store_create_returns_incident(store):
    inc = store.create(
        query="payments latency", environment="production",
        reporter_id="user-1", reporter_team="platform",
    )
    assert inc.id.startswith("INC-")
    assert inc.status == "new"


def test_session_store_load_roundtrip(store):
    inc = store.create(query="t", environment="staging",
                       reporter_id="u", reporter_team="t")
    loaded = store.load(inc.id)
    assert loaded.id == inc.id and loaded.query == "t"


def test_session_store_save_updates(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    assert store.load(inc.id).status == "resolved"


def test_session_store_delete_soft(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    deleted = store.delete(inc.id)
    assert deleted.status == "deleted"
    assert deleted.deleted_at is not None


def test_session_store_list_excludes_deleted_by_default(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    store.delete(inc.id)
    active = store.list_recent(50)
    assert not any(i.id == inc.id for i in active)


def test_history_store_keyword_fallback(history, store):
    inc = store.create(query="payments timeout upstream",
                       environment="production",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    inc.summary = "restarted payments service"
    store.save(inc)
    results = history.find_similar(
        query="payments timeout", environment="production",
        status_filter="resolved", limit=5,
    )
    assert any(r[0].id == inc.id for r in results)


def test_session_store_is_separate_from_history_store():
    from runtime.storage.session_store import SessionStore
    from runtime.storage.history_store import HistoryStore
    assert SessionStore is not HistoryStore
```

- [ ] **2. Run** → `ModuleNotFoundError` for `runtime.storage.session_store`.

- [ ] **3. Implementation**

`src/runtime/storage/models.py` — at bottom add `SessionRow = IncidentRow`.

`src/runtime/storage/session_store.py` — extracted from `IncidentRepository`'s active-CRUD half. Full source per architect Section 4 P1-G.

`src/runtime/storage/history_store.py` — extracted from `IncidentRepository`'s `find_similar` half.

`src/runtime/storage/repository.py` — compat shim:

```python
"""Backward-compat shim.

``IncidentRepository`` is now composed of ``SessionStore`` (active CRUD) +
``HistoryStore`` (similarity search). Remove this shim in Phase 2.
"""
from __future__ import annotations
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy.engine import Engine

from runtime.storage.session_store import SessionStore
from runtime.storage.history_store import HistoryStore
from examples.incident_management.state import IncidentState as Incident


class IncidentRepository(SessionStore):
    """SessionStore + HistoryStore facade for back-compat."""

    def __init__(
        self,
        *,
        engine: Engine,
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        vector_path: Optional[str] = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.85,
        severity_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            engine=engine, embedder=embedder, vector_store=vector_store,
            vector_path=vector_path, vector_index_name=vector_index_name,
            distance_strategy=distance_strategy,
        )
        self.severity_aliases = severity_aliases or {}
        self._history = HistoryStore(
            engine=engine, embedder=embedder, vector_store=vector_store,
            similarity_threshold=similarity_threshold,
            distance_strategy=distance_strategy,
        )

    def find_similar(
        self, *,
        query: str, environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Incident, float]]:
        return self._history.find_similar(
            query=query, environment=environment,
            status_filter=status_filter,
            threshold=threshold, limit=limit,
        )
```

- [ ] **4. Run** `tests/test_session_store.py` → all pass.

- [ ] **5. Commit** — `feat(runtime): split IncidentRepository into SessionStore + HistoryStore`

### P1-H — Make `src/orchestrator/` a shim layer over `src/runtime/`

Every module in `src/orchestrator/` becomes a one-line re-export from `src/runtime/`. Preserves all existing test imports.

**Files:** Every `.py` in `src/orchestrator/` and subdirs.

**Steps:**

- [ ] **1. Test** — append to `tests/test_runtime_package.py`:

```python
def test_orchestrator_shims_still_importable():
    """All existing orchestrator imports must work via shims."""
    from orchestrator.config import AppConfig  # noqa
    from orchestrator.incident import Incident  # noqa
    from orchestrator.storage.repository import IncidentRepository  # noqa
    from orchestrator.storage.session_store import SessionStore  # noqa
    from orchestrator.storage.history_store import HistoryStore  # noqa
    from orchestrator.graph import GraphState  # noqa
    from orchestrator.orchestrator import Orchestrator  # noqa
```

- [ ] **2. Run** → fails on several imports.

- [ ] **3. Implementation** — rewrite each `src/orchestrator/*.py` as a shim that re-exports from `src/runtime/*`. Examples follow architect Section 4 P1-H.

- [ ] **4. Run** full suite → all 191 + new tests pass.

- [ ] **5. Commit** — `refactor: replace src/orchestrator/ modules with shim re-exports to src/runtime/`

### P1-I — Move skills directory to `examples/incident_management/skills/`

The four skill YAMLs (intake, triage, deep_investigator, resolution) are app code.

**Files:**
- Create: `examples/incident_management/skills/` (copy from `config/skills/`)
- Update: `config/config.yaml` `paths.skills_dir`

**Steps:**

- [ ] **1. Test** — append to `tests/test_incident_state.py`:

```python
def test_example_skills_dir_exists():
    from pathlib import Path
    skills_dir = Path("examples/incident_management/skills")
    assert skills_dir.is_dir()
    agents = [d.name for d in skills_dir.iterdir()
              if d.is_dir() and not d.name.startswith("_")]
    assert set(agents) >= {"intake", "triage", "deep_investigator", "resolution"}


def test_framework_skills_dir_is_empty_or_absent():
    from pathlib import Path
    skills_dir = Path("config/skills")
    if skills_dir.exists():
        agent_dirs = [d for d in skills_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")]
        assert not agent_dirs
```

- [ ] **2. Run** → second test fails.

- [ ] **3. Implementation** — `cp -r config/skills/ examples/incident_management/skills/`. Update `config/config.yaml`:

```yaml
paths:
  skills_dir: examples/incident_management/skills
  incidents_dir: incidents
```

After tests pass, delete agent dirs from `config/skills/` (keep `_common/` if referenced; otherwise remove the whole dir).

- [ ] **4. Run** → all pass.

- [ ] **5. Commit** — `refactor: move skills from config/skills/ to examples/incident_management/skills/`

### P1-J — Move Streamlit UI to `examples/incident_management/ui.py`

UI is app code. References incident-specific status palettes, severity colors.

**Files:**
- Create: `examples/incident_management/ui.py` (copy of `ui/streamlit_app.py`)
- Create: `examples/incident_management/__main__.py` (entrypoint)
- Convert: `ui/streamlit_app.py` to a shim

**Steps:**

- [ ] **1. Test** — append to `tests/test_incident_state.py`:

```python
def test_example_ui_importable():
    import importlib.util
    spec = importlib.util.find_spec("examples.incident_management.ui")
    assert spec is not None
```

- [ ] **2. Run** → fails.

- [ ] **3. Implementation** — copy `ui/streamlit_app.py` to `examples/incident_management/ui.py`, updating imports:

```python
# Old:
from orchestrator.config import load_config, AppConfig
from orchestrator.orchestrator import Orchestrator
from orchestrator.storage.repository import IncidentRepository
# (etc.)

# New:
from runtime.config import load_config, AppConfig
from runtime.orchestrator import Orchestrator
from runtime.storage.repository import IncidentRepository
from examples.incident_management.config import load_incident_app_config
```

Update functions that reference `cfg.incidents.*` / `cfg.intervention.*` / `cfg.environments` to use `load_incident_app_config()` (see architect Section 4 P1-J for explicit edits).

Create `examples/incident_management/__main__.py`:

```python
"""Entry point: python -m examples.incident_management"""
from examples.incident_management.ui import main
main()
```

Convert `ui/streamlit_app.py`:

```python
"""Shim: the UI now lives in examples/incident_management/ui.py"""
from examples.incident_management.ui import main  # noqa: F401
if __name__ == "__main__":
    main()
```

- [ ] **4. Run** → test passes.

- [ ] **5. Commit** — `refactor: move Streamlit UI to examples/incident_management/ui.py`

### P1-K — Generic method names on `Orchestrator` + shims

The public facade gains generic names while incident-flavored names stay as one-line shims for the UI during the migration window.

**Files:**
- Modify: `src/runtime/orchestrator.py`

**Steps:**

- [ ] **1. Test** — append to `tests/test_runtime_package.py`:

```python
def test_orchestrator_generic_methods_exist():
    from runtime.orchestrator import Orchestrator
    for m in ("start_session", "stream_session", "resume_session",
              "list_recent_sessions", "get_session", "delete_session"):
        assert hasattr(Orchestrator, m)


def test_orchestrator_incident_shims_exist():
    from runtime.orchestrator import Orchestrator
    for m in ("start_investigation", "stream_investigation", "resume_investigation",
              "list_recent_incidents", "get_incident", "delete_incident"):
        assert hasattr(Orchestrator, m)
```

- [ ] **2. Run** → fails.

- [ ] **3. Implementation** — in `src/runtime/orchestrator.py`, rename primary methods (`start_investigation → start_session`, etc.). Add one-line shims for backward compat. Update `Orchestrator.create()` to load `IncidentAppConfig` for `similarity_threshold`/`severity_aliases`/`escalation_teams` since `AppConfig` no longer carries them. See architect Section 4 P1-K for the explicit edit pattern.

- [ ] **4. Run** full suite → all 191 + new tests pass.

- [ ] **5. Commit** — `refactor(orchestrator): add generic session method names; keep incident shims`

### P1-L — Update `scripts/build_single_file.py` for new layout

Bundler must produce two artifacts: `dist/app.py` (runtime) and `dist/apps/incident-management.py` (app entry).

**Files:**
- Modify: `scripts/build_single_file.py`

**Steps:**

- [ ] **1. Test** — create `tests/test_build_script.py`:

```python
"""Build script smoke tests."""
import subprocess, sys, ast
from pathlib import Path


def test_build_succeeds():
    result = subprocess.run(
        [sys.executable, "scripts/build_single_file.py"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr


def test_dist_app_is_valid_python():
    src = Path("dist/app.py").read_text()
    ast.parse(src)


def test_dist_incident_app_is_valid_python():
    src = Path("dist/apps/incident-management.py").read_text()
    ast.parse(src)
```

- [ ] **2. Run** → some failures.

- [ ] **3. Implementation**

In `scripts/build_single_file.py`:
- Update `SRC_ROOT = Path("src/runtime")`.
- Add `EXAMPLES_ROOT = Path("examples/incident_management")`.
- Update `INTRA_IMPORT_RE` to match `runtime`, `orchestrator`, and `examples.incident_management`.
- Add `build_incident_app()` function that bundles runtime modules + example app modules.
- `build_ui()` reads from `examples/incident_management/ui.py`.
- `main()` calls `build_app()`, `build_incident_app()`, and `build_ui()`.

See architect Section 4 P1-L for the full skeleton.

- [ ] **4. Run** → all pass; `dist/app.py` and `dist/apps/incident-management.py` regenerate.

- [ ] **5. Commit** — `feat(build): update bundler for src/runtime/ + examples/incident_management/ layout`

### P1-M — Update existing test fixtures for removed `AppConfig` keys

Some tests construct `AppConfig` with `incidents=`, `intervention=`, `environments=` kwargs. After P1-E those fail.

**Files:**
- `tests/conftest.py` (if it has shared fixtures)
- Individual test files with broken fixtures

**Steps:**

- [ ] **1. Run full suite** — `pytest tests/ -x` to identify failures from removed keys.

- [ ] **2. For each failure**, apply one of:
  - Test only checks framework behavior: drop the incident-specific kwarg from the fixture.
  - Test checks incident-management behavior: replace with `app_cfg = IncidentAppConfig(...)` and read fields off it.

- [ ] **3. Run** full suite → 191+ tests pass.

- [ ] **4. Commit** — `fix(tests): update fixtures to remove incident-specific AppConfig keys`

### P1-N — Final verification + documentation

Verification + the example app's README. Not strictly TDD — sign-off task.

- [ ] **1. Run full suite** — `pytest tests/ -v`. Expected: 191 + ~35 new tests = 226+ passing.

- [ ] **2. Build bundle** — `python scripts/build_single_file.py` → `dist/app.py` + `dist/apps/incident-management.py` written.

- [ ] **3. Grep checks (must all return zero)**:

```bash
# Framework has zero incident terminology in non-comment Python
grep -rn 'severity\|IncidentStatus\|Reporter\|reporter_id\|reporter_team\|matched_prior_inc' \
  src/runtime/ --include='*.py' | grep -v '#'

# Framework config has zero domain keys
grep -n 'IncidentConfig\|InterventionConfig\|severity_aliases\|escalation_teams' \
  src/runtime/config.py

# No src/runtime/mcp_servers/incident.py
ls src/runtime/mcp_servers/incident.py 2>&1 | grep 'No such file'

# config/skills/ holds no agent dirs
find config/skills -mindepth 2 -name 'config.yaml' 2>/dev/null | wc -l   # → 0
```

- [ ] **4. Smoke run** — `python -m examples.incident_management` launches Streamlit; sidebar shows incidents; new investigation works.

- [ ] **5. Create `examples/incident_management/README.md`**:

```markdown
# Incident Management — example application

Flagship example for the `runtime` framework.

## Run

```bash
python -m examples.incident_management
```

## Configuration

- Framework config: `config/config.yaml`
- App config: `examples/incident_management/config.yaml`
- Skills: `examples/incident_management/skills/`

## Architecture

`IncidentState` extends `runtime.state.Session` with incident-specific fields.
`IncidentMCPServer` exposes the three core tools (create, update, lookup_similar).
All incident logic lives in this directory; the framework has no incident imports.
```

- [ ] **6. Commit** — `docs: add examples/incident_management/README.md; Phase 1 complete`

---

## 5. Sequencing and Dependencies

```
P1-A (scaffold runtime package)
  └─ P1-B (Session base model)
       └─ P1-C (IncidentState in example app)
            ├─ P1-D (copy framework modules to src/runtime/)
            │    └─ P1-E (strip domain config keys)              ← P1-D + P1-C
            │         └─ P1-F (move MCP server)                  ← P1-E
            │              └─ P1-G (split repository)            ← P1-F + P1-D
            │                   └─ P1-H (orchestrator shims)     ← P1-G
            │                        ├─ P1-K (generic methods)   ← P1-H
            │                        ├─ P1-I (move skills)       ← P1-H ‖ P1-J
            │                        └─ P1-J (move UI)           ← P1-H, P1-K
            └─ P1-L (update bundler)                              ← P1-J + P1-K
P1-M (update test fixtures)                                       ← P1-H ‖ P1-I, P1-J
P1-N (final verification)                                         ← all done
```

**Critical path:** A → B → C → D → E → F → G → H → K → (J + L parallel) → M → N

**Parallel-safe pairs:** (P1-I, P1-J), (P1-L, P1-M).

---

## 6. Risks and Mitigations

**R1: `GraphState` TypedDict field rename silently drops values.**
After renaming `incident → session` in `GraphState`, existing dict construction `{"incident": inc, ...}` would silently drop the value (TypedDict ignores unknown keys). Mitigation: keep both `session` and `incident` fields during P1; remove `incident` in Phase 2. Test guard: `test_graph_state_session_key_works`.

**R2: MCP loader hits stale module path.**
The loader resolves the module path from `config/config.yaml`. If the yaml entry isn't updated in the same commit as the move, the loader imports `orchestrator.mcp_servers.incident` (now a shim) which works, but the test `test_framework_does_not_own_incident_mcp` fails. Mitigation: update `config.yaml` and `_INCIDENT_MCP_MODULE` together in P1-F.

**R3: `HistoryStore._keyword_similar` cross-imports orchestrator.**
The original `_keyword_similar` imports `from orchestrator.similarity ...`. In `runtime`, this should be `from runtime.similarity ...`. Mitigation: use `runtime.*` everywhere in `runtime/`. Shims always direction `orchestrator → runtime`, never reverse.

**R4: Bundler regex misses new package prefixes.**
`INTRA_IMPORT_RE` currently matches `from orchestrator.X import Y`. Must also match `from runtime.X` and `from examples.incident_management.X` to strip them in the bundle. Mitigation: P1-L updates the regex; `test_dist_app_is_valid_python` catches output errors via `ast.parse`.

**R5: `Orchestrator.create()` reads removed config keys.**
Today's code reads `cfg.orchestrator.severity_aliases`, `cfg.intervention.escalation_teams`. After P1-E, these don't exist. Mitigation: P1-K updates `Orchestrator.create()` to load `IncidentAppConfig` and pass values from there. Tasks P1-E and P1-K must commit in close sequence.

**R6: Streamlit accesses `orch.cfg.environments` etc.**
After P1-E, `AppConfig` has no `environments`. UI crashes on `AttributeError`. Mitigation: P1-J updates `_load_metadata_dicts` to use `load_incident_app_config().environments`. `test_example_ui_importable` catches import-time errors.

**R7: Hyphenated package name.**
Python can't import `incident-management`. Use underscore directory name (`examples/incident_management/`). README labels can use hyphens for human readability.

**R8: Bundle name collision in `dist/`.**
`dist/app.py` and `dist/apps/incident-management.py` both expose `app` as a top-level. The dist UI loader does `from app import ...`. Mitigation: keep the bundle structure: each artifact is fully self-contained; the runtime bundle is named `app`, the app bundle is named `incident-management`. UI bundle in `dist/ui.py` continues to `from app import ...` (sibling of `app.py`).

**R9: Test fixtures in `conftest.py`.**
A shared conftest fixture builds `AppConfig` with old kwargs. P1-M handles this but the change can cascade many failures at once. Mitigation: run `pytest tests/ -x` after P1-E and check that the failures in P1-M are localized.

---

## 7. Done Criteria

**Test suite:**
- `pytest tests/ -v` exits 0.
- Total passing count is 191 + new tests added in this phase (target: 226+).
- Zero new skips/xfails.

**Grep checks (zero hits):**
- `grep -rn 'severity\|IncidentStatus\|Reporter' src/runtime/ --include='*.py' | grep -v '#'`
- `grep -n 'IncidentConfig\|InterventionConfig\|severity_aliases' src/runtime/config.py`
- `ls src/runtime/mcp_servers/incident.py 2>&1` → "No such file or directory"
- `find config/skills -mindepth 2 -name 'config.yaml' 2>/dev/null | wc -l` → 0

**Structural import checks (in fresh Python):**
- `from examples.incident_management.state import IncidentState` ✓
- `from runtime.state import Session` ✓
- `from runtime.storage.session_store import SessionStore` ✓
- `from runtime.storage.history_store import HistoryStore` ✓
- `from orchestrator.incident import Incident` ✓ (shim)
- `from orchestrator.storage.repository import IncidentRepository` ✓ (shim)

**Bundle:**
- `python scripts/build_single_file.py` exits 0.
- `dist/app.py` and `dist/apps/incident-management.py` are syntactically valid (`ast.parse`).

**Smoke (manual or with stub):**
- `python -m examples.incident_management` launches Streamlit, sidebar populates, a new investigation completes end-to-end.

---

## 8. Open Questions

These were resolved by the architect; flagged for follow-up if other reviewers disagree.

1. **Directory name underscore vs hyphen.** Underscore (`examples/incident_management/`) for Python imports. Hyphen used only in human-facing labels.
2. **`GraphState` bridge field.** Keep both `incident` and `session` during P1; remove `incident` in Phase 2.
3. **`HistoryStore` imports `IncidentState` directly.** Pragmatic compromise. True genericity is Phase 2 (extensible state schema).
4. **`severity_aliases` on `IncidentRepository` shim constructor.** Kept for call-site compat. Removed when shim is deleted in Phase 2.
5. **`similarity.py` location.** Keep in `src/runtime/` — it's a generic utility (keyword similarity scorer).
6. **`config/config.yaml` retains `paths.skills_dir` pointing at the example app.** Acceptable for Phase 1; framework-level skill-dir abstraction is Phase 2.

---

*End of Phase 1 plan.*
