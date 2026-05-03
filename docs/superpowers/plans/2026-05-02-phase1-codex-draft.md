# Phase 1 Implementation Plan: Domain Extraction, Session Rename, Repository Split

## 1. Goal + Scope

Phase 1 turns the current incident-shaped runtime into a generic session runtime and moves the incident-management domain into `examples/incident_management/`.

The framework landing shape for this repository should use the already-present `src/runtime/` package, not reintroduce `src/orchestrator/`. The old `orchestrator.*` import surface can remain only as a temporary compatibility shim during this phase, with tests proving the canonical imports are `runtime.*`.

In scope:

- Rename framework `Incident` model to generic `Session`.
- Keep only generic fields on framework `Session`: `id`, `status`, timestamps, `agents_run`, `tool_calls`, `findings`, `pending_intervention`, `user_inputs`, `token_usage`.
- Move incident fields into `examples.incident_management.state.IncidentState`: `query`, `environment`, `reporter`, `summary`, `tags`, `severity`, `category`, `matched_prior_inc`, `embedding`, `resolution`.
- Split `IncidentRepository` into:
  - `SessionStore`: active CRUD, soft delete, SQL row mapping, vector update writes.
  - `HistoryStore`: closed-session search, vector/keyword similarity, resolved-session filters.
- Move `incident_management` MCP server out of framework into `examples/incident_management/mcp_server.py`.
- Move incident app config and skills under `examples/incident_management/`.
- Keep existing UI working by making it the incident-management example UI, not framework UI.
- Update bundling to build two artifacts: `dist/framework.py` and `dist/apps/incident_management.py`.
- Keep all 191 tests passing throughout by adding shims first, then moving canonical imports.
- No new infra. Storage remains SQLAlchemy 2.x. Embedding remains Ollama `bge-m3` through existing embedder config. FAISS remains the local vector backend.

Out of scope:

- LangGraph checkpointer migration.
- Multi-session concurrency beyond renaming the public surface.
- Generic trigger registry.
- Generic tool gateway.
- App-extendable state-schema merge beyond a concrete `IncidentState(Session)` subclass.
- Database migrations that require external services.

Important current-state observation:

- The checkout already has `src/runtime/`, but several tests, config entries, and the bundler still reference `orchestrator.*` or `src/orchestrator`. First stabilization task is to make both canonical `runtime.*` and temporary `orchestrator.*` imports collect cleanly before deeper changes.

## 2. Target File Layout

```text
src/
  runtime/                              # framework package, no incident-management domain
    __init__.py
    state.py                            # SessionStatus, Reporter? no: Reporter moves to example; Session, ToolCall, AgentRun, TokenUsage
    config.py                           # framework-only config: llm, mcp, storage, paths, runtime/orchestrator signals
    graph.py                            # GraphState uses session key; app can pass IncidentState in Phase 1 through concrete store typing
    orchestrator.py                     # public generic facade; incident wrappers deprecated or moved to app
    api.py                              # generic /sessions routes; optional deprecated /incidents shim lives in example app only
    mcp_loader.py
    similarity.py
    skill.py
    llm.py
    storage/
      __init__.py
      models.py                         # SessionRow table, incident table alias only if needed for no-migration compatibility
      engine.py
      embeddings.py
      vector.py
      session_store.py                  # active CRUD: create/load/save/delete/list_recent/list_all
      history_store.py                  # find_similar, keyword fallback, vector read path
      repository.py                     # temporary compatibility alias only: IncidentRepository -> composed stores
    mcp_servers/
      __init__.py
      observability.py                  # still demo/local tool server; not incident state
      remediation.py                    # still demo/local tool server; may retain incident_id param until later app migration
      user_context.py
      incident.py                       # temporary import shim to examples.incident_management.mcp_server during P1 only

examples/
  __init__.py
  incident_management/                  # importable package; hyphenated directory is not runnable with -m
    __init__.py
    __main__.py                         # `python -m examples.incident_management`
    config.yaml                         # incident app config: severity_aliases, environments, intervention, app paths
    state.py                            # Reporter, IncidentState(Session), incident id minter helpers
    mcp_server.py                       # incident_management FastMCP server with lookup/create/update tools
    repository.py                       # optional thin app facade combining SessionStore + HistoryStore with incident names
    ui.py                               # Streamlit app moved from ui/streamlit_app.py or wrapper imports it
    skills/
      _common/
        confidence.md
        output.md
      intake/
        config.yaml
        system.md
      triage/
        config.yaml
        system.md
      deep_investigator/
        config.yaml
        system.md
      resolution/
        config.yaml
        system.md

ui/
  streamlit_app.py                      # compatibility wrapper: imports examples.incident_management.ui.main

config/
  config.yaml                           # framework-only sample/default, no incident keys
  config.yaml.example                   # framework-only example

dist/
  framework.py                          # bundled framework lib
  apps/
    incident_management.py              # bundled example app entrypoint
  ui.py                                 # optional compatibility wrapper if tests require it

scripts/
  build_single_file.py                  # knows runtime core and example app bundle
  migrate_jsonl_to_sql.py               # either moved to example or imports example state/store
  seed_demo_incidents.py                # moved to example or wrapper
```

Why `examples/incident_management` instead of `examples/incident-management`:

- Python cannot run `python -m examples.incident-management`.
- The hard constraint says runnable via `python -m examples.incident_management`.
- If the hyphenated path must exist for documentation continuity, keep it as a non-importable data mirror or symlink after the importable package exists, but canonical code must live in `examples/incident_management/`.

## 3. Naming Map

Framework model and state:

| Old | New | Location |
| --- | --- | --- |
| `runtime.incident.Incident` | `runtime.state.Session` | framework |
| `runtime.incident.IncidentStatus` | `runtime.state.SessionStatus` | framework |
| `runtime.incident.ToolCall` | `runtime.state.ToolCall` | framework |
| `runtime.incident.TokenUsage` | `runtime.state.TokenUsage` | framework |
| `runtime.incident.AgentRun` | `runtime.state.AgentRun` | framework |
| `runtime.incident.Reporter` | `examples.incident_management.state.Reporter` | example app |
| `Incident.query` | `IncidentState.query` | example app |
| `Incident.environment` | `IncidentState.environment` | example app |
| `Incident.reporter` | `IncidentState.reporter` | example app |
| `Incident.summary` | `IncidentState.summary` | example app |
| `Incident.tags` | `IncidentState.tags` | example app |
| `Incident.severity` | `IncidentState.severity` | example app |
| `Incident.category` | `IncidentState.category` | example app |
| `Incident.matched_prior_inc` | `IncidentState.matched_prior_inc` | example app |
| `Incident.embedding` | `IncidentState.embedding` | example app, likely still unused |
| `Incident.resolution` | `IncidentState.resolution` | example app |
| `runtime.incident._UTC_TS_FMT` | `runtime.state.UTC_TS_FMT` | framework |
| `runtime.incident._INC_ID_RE` | `examples.incident_management.state.INC_ID_RE` | example app |

Repository and storage:

| Old | New | Location |
| --- | --- | --- |
| `IncidentRepository` | `SessionStore` | `runtime.storage.session_store` |
| `IncidentRepository.find_similar` | `HistoryStore.find_similar` | `runtime.storage.history_store` |
| `IncidentRepository._keyword_similar` | `HistoryStore._keyword_similar` | `runtime.storage.history_store` |
| `IncidentRepository._next_id` | `SessionStore._next_id` with injected minter | framework plus example app id minter |
| `IncidentRow` | `SessionRow` | `runtime.storage.models` |
| `incidents` DB table | `sessions` table | canonical; keep `incidents` alias only if migration-free compatibility is required |
| `incident_id` params in framework store | `session_id` | framework |
| `inc_id` locals in framework | `session_id` | framework |
| `list_recent_incidents` | `list_recent_sessions` | framework |
| `get_incident` | `get_session` | framework |
| `delete_incident` | `delete_session` | framework |

MCP and tools:

| Old | New | Location |
| --- | --- | --- |
| `runtime.mcp_servers.incident` | `examples.incident_management.mcp_server` | example app |
| `IncidentMCPServer` | `IncidentManagementMCPServer` | example app |
| `lookup_similar_incidents` | unchanged tool name | example app compatibility |
| `create_incident` | unchanged tool name | example app compatibility |
| `update_incident` | unchanged tool name | example app compatibility |
| `set_state(repository=...)` | `set_state(session_store=..., history_store=..., app_config=...)` | example app |
| `_DEFAULT_SEVERITY_ALIASES` | app config `severity_aliases` plus local default | example app |

Config:

| Old | New |
| --- | --- |
| `AppConfig.incidents` | removed from framework; `IncidentManagementConfig.similarity_threshold` in example |
| `IncidentConfig` | `IncidentAppConfig` in example |
| `cfg.environments` | example app config |
| `cfg.intervention` | example app config for Phase 1; generic HITL config comes Phase 4 |
| `cfg.orchestrator.severity_aliases` | example app config |
| `paths.incidents_dir` | framework `paths.data_dir` or storage URL only; example can define `incidents_dir` |
| `storage.metadata.url=sqlite:///incidents/incidents.db` | framework default `sqlite:///sessions/sessions.db`; example overrides to `incidents/incidents.db` |
| `storage.vector.collection_name=incidents` | framework default `sessions`; example overrides `incidents` |

Graph and public events:

| Old | New |
| --- | --- |
| `GraphState.incident` | `GraphState.session` |
| `start_investigation` | `start_session` |
| `stream_investigation` | `stream_session` |
| `resume_investigation` | `resume_session` |
| `investigation_started` | `session_started` |
| `investigation_completed` | `session_completed` |
| event key `incident_id` | event key `session_id` in framework; example can map to `incident_id` for UI compatibility |
| prompt line `Incident <ID>` | generic framework prompt `Session <ID>`; incident skills can still say Incident because they are app code |

Compatibility shims to keep temporarily:

- `runtime/incident.py` re-exports `Session as Incident` only until all tests and app imports move.
- `runtime/storage/repository.py` re-exports or composes `IncidentRepository` for renamed tests during P1.
- `runtime/mcp_servers/incident.py` imports from `examples.incident_management.mcp_server`.
- `orchestrator` top-level package shim imports from `runtime` if tests still reference old imports.
- Deprecated `Orchestrator.start_investigation`, `stream_investigation`, `resume_investigation`, `get_incident`, `list_recent_incidents`, `delete_incident` call generic methods and emit old event names only through the example UI/API wrapper.

## 4. Task Breakdown

Each task below is intended to be a small green commit. The tests named in "passing test" should pass before moving to the next task. Full suite stays green after every task.

### P1-A: Stabilize Package Imports and Baseline Collection

Files affected:

- `src/orchestrator/__init__.py` or `orchestrator/__init__.py`
- `src/orchestrator/*.py` shim modules if needed
- `pyproject.toml`
- `tests/test_import_surfaces.py`

Failing test:

```python
# tests/test_import_surfaces.py
def test_runtime_is_canonical_package():
    import runtime
    from runtime.config import AppConfig
    from runtime.orchestrator import Orchestrator

    assert runtime is not None
    assert AppConfig.__name__ == "AppConfig"
    assert Orchestrator.__name__ == "Orchestrator"


def test_orchestrator_import_surface_is_temporary_alias():
    from orchestrator.config import AppConfig as OldAppConfig
    from runtime.config import AppConfig

    assert OldAppConfig is AppConfig
```

Implementation:

```python
# src/orchestrator/__init__.py
"""Temporary compatibility package for pre-runtime imports."""
from runtime import *  # noqa: F401,F403
```

```python
# src/orchestrator/config.py
from runtime.config import *  # noqa: F401,F403
```

Create equivalent one-line shims for modules still imported by tests:

```text
api.py
graph.py
incident.py
llm.py
mcp_loader.py
similarity.py
skill.py
orchestrator.py
storage/__init__.py
storage/engine.py
storage/embeddings.py
storage/models.py
storage/repository.py
storage/vector.py
mcp_servers/__init__.py
mcp_servers/incident.py
mcp_servers/observability.py
mcp_servers/remediation.py
mcp_servers/user_context.py
```

`pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/runtime", "src/orchestrator"]
```

Passing test:

```bash
python -m pytest --collect-only -q
python -m pytest tests/test_import_surfaces.py -q
```

Commit message:

```text
test: stabilize runtime and orchestrator import surfaces
```

### P1-B: Introduce Generic Session Model Beside Incident

Files affected:

- `src/runtime/state.py`
- `src/runtime/incident.py`
- `tests/test_session_model.py`
- `tests/test_incident_model.py`

Failing test:

```python
# tests/test_session_model.py
import pydantic
import pytest

from runtime.state import AgentRun, Session, SessionStatus, TokenUsage, ToolCall


def test_session_minimal_construction_has_only_generic_defaults():
    session = Session(
        id="ses_01HX0000000000000000000000",
        status="new",
        created_at="2026-05-02T19:00:00Z",
        updated_at="2026-05-02T19:00:00Z",
    )

    assert session.id.startswith("ses_")
    assert session.agents_run == []
    assert session.tool_calls == []
    assert session.findings == {}
    assert session.pending_intervention is None
    assert session.user_inputs == []
    assert session.deleted_at is None
    assert session.token_usage == TokenUsage()
    assert "query" not in Session.model_fields
    assert "environment" not in Session.model_fields
    assert "severity" not in Session.model_fields
    assert "reporter" not in Session.model_fields


def test_session_status_rejects_unknown_value():
    with pytest.raises(pydantic.ValidationError):
        Session(id="s1", status="invalid", created_at="t", updated_at="t")


def test_agent_run_and_tool_call_remain_generic():
    run = AgentRun(agent="agent", started_at="t0", ended_at="t1", summary="ok")
    call = ToolCall(agent="agent", tool="tool", args={"a": 1}, result={"b": 2}, ts="t1")

    assert run.token_usage.total_tokens == 0
    assert call.tool == "tool"
```

Implementation:

```python
# src/runtime/state.py
"""Generic runtime session state."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

SessionStatus = Literal[
    "new",
    "in_progress",
    "matched",
    "resolved",
    "escalated",
    "awaiting_input",
    "stopped",
    "deleted",
]


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
    id: str
    status: SessionStatus
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    findings: dict[str, Any] = Field(default_factory=dict)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
```

Compatibility:

```python
# src/runtime/incident.py
"""Temporary compatibility exports for incident-era imports.

Canonical framework imports live in runtime.state. IncidentState lives in
examples.incident_management.state after P1-D.
"""
from runtime.state import (  # noqa: F401
    UTC_TS_FMT as _UTC_TS_FMT,
    AgentRun,
    Session as Incident,
    SessionStatus as IncidentStatus,
    TokenUsage,
    ToolCall,
)
```

Passing test:

```bash
python -m pytest tests/test_session_model.py tests/test_incident_model.py -q
```

Commit message:

```text
feat: add generic Session state model
```

### P1-C: Add IncidentState Extension in Example App

Files affected:

- `examples/__init__.py`
- `examples/incident_management/__init__.py`
- `examples/incident_management/state.py`
- `tests/test_incident_state_example.py`

Failing test:

```python
# tests/test_incident_state_example.py
from examples.incident_management.state import INC_ID_RE, IncidentState, Reporter
from runtime.state import Session


def test_incident_state_extends_session_with_domain_fields():
    inc = IncidentState(
        id="INC-20260502-001",
        status="new",
        created_at="2026-05-02T19:00:00Z",
        updated_at="2026-05-02T19:00:00Z",
        query="API latency spike",
        environment="production",
        reporter=Reporter(id="u1", team="platform"),
    )

    assert isinstance(inc, Session)
    assert inc.query == "API latency spike"
    assert inc.environment == "production"
    assert inc.reporter.team == "platform"
    assert inc.summary == ""
    assert inc.tags == []
    assert inc.resolution is None
    assert INC_ID_RE.match(inc.id)
```

Implementation:

```python
# examples/__init__.py
"""Example applications for the runtime framework."""
```

```python
# examples/incident_management/__init__.py
"""Incident-management flagship example app."""
```

```python
# examples/incident_management/state.py
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from runtime.state import Session

INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")


class Reporter(BaseModel):
    id: str
    team: str


class IncidentState(Session):
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

Passing test:

```bash
python -m pytest tests/test_incident_state_example.py tests/test_session_model.py -q
```

Commit message:

```text
feat: move incident state into example app
```

### P1-D: Rename Storage Row and Add SessionStore Active CRUD

Files affected:

- `src/runtime/storage/models.py`
- `src/runtime/storage/session_store.py`
- `src/runtime/storage/__init__.py`
- `tests/test_session_store.py`
- `tests/test_incident_store.py`
- `tests/test_storage_repository.py`

Failing test:

```python
# tests/test_session_store.py
import pytest

from examples.incident_management.state import IncidentState, Reporter
from runtime.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.models import Base, SessionRow
from runtime.storage.session_store import SessionStore


def _incident_factory(*, id, status, created_at, updated_at, **payload):
    return IncidentState(
        id=id,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        query=payload["query"],
        environment=payload["environment"],
        reporter=Reporter(
            id=payload.get("reporter_id", "user-mock"),
            team=payload.get("reporter_team", "platform"),
        ),
    )


@pytest.fixture
def store(tmp_path):
    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(engine)
    return SessionStore(
        engine=engine,
        state_type=IncidentState,
        create_state=_incident_factory,
        id_pattern=r"^INC-\d{8}-\d{3}$",
        id_error="expected INC-YYYYMMDD-NNN",
    )


def test_create_load_save_delete_active_session(store):
    inc = store.create(
        query="redis oom",
        environment="production",
        reporter_id="u1",
        reporter_team="platform",
    )

    assert inc.id.startswith("INC-")
    loaded = store.load(inc.id)
    assert loaded.query == "redis oom"
    assert loaded.environment == "production"
    assert loaded.reporter.id == "u1"

    loaded.summary = "edited"
    store.save(loaded)
    assert store.load(inc.id).summary == "edited"

    deleted = store.delete(inc.id)
    assert deleted.status == "deleted"
    assert deleted.deleted_at is not None


def test_session_row_is_canonical_table_model():
    assert SessionRow.__tablename__ == "sessions"
```

Implementation:

```python
# src/runtime/storage/models.py
class SessionRow(Base):
    __tablename__ = "sessions"
    # same columns as current IncidentRow for Phase 1, because app state is
    # stored in the same hybrid relational-plus-JSON shape.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment: Mapped[str | None] = mapped_column(String, nullable=True)
    reporter_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reporter_team: Mapped[str | None] = mapped_column(String, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    matched_prior_inc: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    agents_run: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    findings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pending_intervention: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    user_inputs: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_sessions_status_env_active", "status", "environment",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
        Index("ix_sessions_created_at_active", "created_at",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
    )


IncidentRow = SessionRow
```

```python
# src/runtime/storage/session_store.py
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlSession

from runtime.state import AgentRun, Session, TokenUsage, ToolCall
from runtime.storage.models import SessionRow


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now().strftime("%Y%m%d")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _deserialize_resolution(raw: Optional[str]):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class SessionStore:
    def __init__(
        self,
        *,
        engine: Engine,
        state_type: type[Session] = Session,
        create_state: Callable[..., Session] | None = None,
        id_prefix: str = "ses",
        id_pattern: str | None = None,
        id_error: str | None = None,
        embedder: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        vector_path: str | None = None,
        vector_index_name: str = "sessions",
    ) -> None:
        self.engine = engine
        self.state_type = state_type
        self.create_state = create_state
        self.id_prefix = id_prefix
        self.id_re = re.compile(id_pattern) if id_pattern else None
        self.id_error = id_error or "invalid session id"
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_path = vector_path
        self.vector_index_name = vector_index_name

    def _next_id(self, db: SqlSession) -> str:
        if self.id_prefix == "INC":
            prefix = f"INC-{_today_str()}-"
            rows = db.execute(select(SessionRow.id).where(SessionRow.id.like(f"{prefix}%"))).scalars().all()
            max_seq = 0
            for row_id in rows:
                try:
                    max_seq = max(max_seq, int(row_id.rsplit("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
            return f"{prefix}{max_seq + 1:03d}"
        return f"{self.id_prefix}_{_now().strftime('%Y%m%d%H%M%S%f')}"

    def _validate_id(self, session_id: str) -> None:
        if self.id_re and not self.id_re.match(session_id):
            raise ValueError(f"Invalid session id {session_id!r}; {self.id_error}")

    def create(self, **payload: Any) -> Session:
        with SqlSession(self.engine) as db:
            now = _now()
            session_id = self._next_id(db)
            state = self._make_state(session_id=session_id, now=now, payload=payload)
            row = self._state_to_row(state)
            db.add(row)
            db.commit()
            db.refresh(row)
            state = self._row_to_state(row)
        self._add_vector(state)
        return state

    def load(self, session_id: str) -> Session:
        self._validate_id(session_id)
        with SqlSession(self.engine) as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                raise FileNotFoundError(session_id)
            return self._row_to_state(row)

    def save(self, state: Session) -> None:
        self._validate_id(state.id)
        state.updated_at = _iso(_now())
        with SqlSession(self.engine) as db:
            existing = db.get(SessionRow, state.id)
            prior_text = self.embedding_text_from_row(existing) if existing is not None else ""
            data = self._state_to_row_dict(state)
            if existing is None:
                db.add(SessionRow(**data))
            else:
                for key, value in data.items():
                    setattr(existing, key, value)
            db.commit()
        self._refresh_vector(state, prior_text=prior_text)

    def delete(self, session_id: str) -> Session:
        with SqlSession(self.engine) as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                raise FileNotFoundError(session_id)
            if row.status != "deleted":
                row.status = "deleted"
                row.deleted_at = _now()
                row.pending_intervention = None
            db.commit()
            db.refresh(row)
            return self._row_to_state(row)

    def list_all(self, *, include_deleted: bool = False) -> list[Session]:
        with SqlSession(self.engine) as db:
            stmt = select(SessionRow)
            if not include_deleted:
                stmt = stmt.where(SessionRow.deleted_at.is_(None))
            return [self._row_to_state(r) for r in db.execute(stmt).scalars().all()]

    def list_recent(self, limit: int = 20, *, include_deleted: bool = False) -> list[Session]:
        with SqlSession(self.engine) as db:
            stmt = select(SessionRow)
            if not include_deleted:
                stmt = stmt.where(SessionRow.deleted_at.is_(None))
            stmt = stmt.order_by(desc(SessionRow.created_at), desc(SessionRow.id)).limit(limit)
            return [self._row_to_state(r) for r in db.execute(stmt).scalars().all()]

    def embedding_text(self, state: Session) -> str:
        return str(getattr(state, "query", "") or "").strip()

    def embedding_text_from_row(self, row: SessionRow | None) -> str:
        return "" if row is None else (row.query or "").strip()
```

The rest of `SessionStore` should copy the existing row mapping and vector persistence helpers, renamed from incident to session, with `_row_to_state()` validating `state_type`.

Passing test:

```bash
python -m pytest tests/test_session_store.py tests/test_incident_store.py tests/test_storage_repository.py -q
```

Commit message:

```text
feat: add SessionStore for active session CRUD
```

### P1-E: Extract HistoryStore for Closed-Session Search

Files affected:

- `src/runtime/storage/history_store.py`
- `src/runtime/storage/session_store.py`
- `src/runtime/storage/repository.py`
- `tests/test_history_store.py`
- `tests/test_storage_find_similar.py`

Failing test:

```python
# tests/test_history_store.py
import pytest

from examples.incident_management.state import IncidentState, Reporter
from runtime.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


def _factory(*, id, status, created_at, updated_at, **payload):
    return IncidentState(
        id=id,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        query=payload["query"],
        environment=payload["environment"],
        reporter=Reporter(id=payload.get("reporter_id", "u"), team=payload.get("reporter_team", "t")),
    )


@pytest.fixture
def stores(tmp_path):
    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(engine)
    active = SessionStore(
        engine=engine,
        state_type=IncidentState,
        create_state=_factory,
        id_prefix="INC",
        id_pattern=r"^INC-\d{8}-\d{3}$",
    )
    history = HistoryStore(session_store=active, similarity_threshold=0.0)
    return active, history


def test_history_store_finds_only_closed_matching_environment(stores):
    active, history = stores
    resolved = active.create(query="api latency production", environment="production")
    resolved.status = "resolved"
    resolved.summary = "scaled api"
    active.save(resolved)

    unresolved = active.create(query="api latency production", environment="production")
    unresolved.status = "in_progress"
    active.save(unresolved)

    other_env = active.create(query="api latency production", environment="staging")
    other_env.status = "resolved"
    active.save(other_env)

    hits = history.find_similar(query="api latency", environment="production", limit=5)
    assert [item.id for item, score in hits] == [resolved.id]
```

Implementation:

```python
# src/runtime/storage/history_store.py
from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from runtime.state import Session
from runtime.storage.session_store import SessionStore


class HistoryStore:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        embedder: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.85,
    ) -> None:
        self.session_store = session_store
        self.embedder = embedder if embedder is not None else session_store.embedder
        self.vector_store = vector_store if vector_store is not None else session_store.vector_store
        self.distance_strategy = distance_strategy
        self.similarity_threshold = similarity_threshold

    def find_similar(
        self,
        *,
        query: str,
        environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Session, float]]:
        if self.vector_store is None or self.embedder is None:
            return self._keyword_similar(
                query=query,
                environment=environment,
                status_filter=status_filter,
                threshold=threshold,
                limit=limit,
            )
        from runtime.storage.vector import distance_to_similarity
        cutoff = self.similarity_threshold if threshold is None else threshold
        vec = self.embedder.embed_query(query)
        raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit * 4)
        out: list[tuple[Session, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(float(distance), self.distance_strategy)
            if score < cutoff:
                continue
            session_id = doc.metadata.get("id")
            if session_id is None:
                continue
            try:
                state = self.session_store.load(session_id)
            except (FileNotFoundError, ValueError):
                continue
            if (
                getattr(state, "environment", None) != environment
                or state.status != status_filter
                or state.deleted_at is not None
            ):
                continue
            out.append((state, score))
            if len(out) >= limit:
                break
        return out

    def _keyword_similar(self, *, query, environment, status_filter, threshold, limit):
        from runtime.similarity import KeywordSimilarity, find_similar

        candidates_state = [
            state for state in self.session_store.list_all()
            if getattr(state, "environment", None) == environment
            and state.status == status_filter
            and state.deleted_at is None
        ]
        candidates = [
            {
                "id": state.id,
                "text": " ".join(
                    str(part)
                    for part in (
                        getattr(state, "query", ""),
                        getattr(state, "summary", ""),
                        " ".join(getattr(state, "tags", []) or []),
                    )
                    if part
                ),
                "session": state,
            }
            for state in candidates_state
        ]
        results = find_similar(
            query=query,
            candidates=candidates,
            text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(candidate["session"], float(score)) for candidate, score in results]
```

Compatibility repository:

```python
# src/runtime/storage/repository.py
"""Compatibility facade for the former IncidentRepository."""
from __future__ import annotations

from examples.incident_management.repository import IncidentRepository
from runtime.storage.history_store import HistoryStore
from runtime.storage.session_store import SessionStore

__all__ = ["IncidentRepository", "SessionStore", "HistoryStore"]
```

Passing test:

```bash
python -m pytest tests/test_history_store.py tests/test_storage_find_similar.py tests/test_storage_repository.py -q
```

Commit message:

```text
feat: split closed-session search into HistoryStore
```

### P1-F: Add Incident Example Repository Facade

Files affected:

- `examples/incident_management/repository.py`
- `tests/test_incident_repository_example.py`
- `tests/test_storage_repository.py`

Failing test:

```python
# tests/test_incident_repository_example.py
from examples.incident_management.repository import IncidentRepository
from examples.incident_management.state import IncidentState
from runtime.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.models import Base


def test_incident_repository_preserves_old_create_and_find_similar_contract(tmp_path):
    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(engine)
    repo = IncidentRepository(engine=engine, embedder=None, similarity_threshold=0.0)

    inc = repo.create(query="api latency production", environment="production", reporter_id="u", reporter_team="t")
    assert isinstance(inc, IncidentState)
    assert inc.id.startswith("INC-")

    inc.status = "resolved"
    inc.summary = "scaled api"
    repo.save(inc)

    hits = repo.find_similar(query="api latency", environment="production", limit=1)
    assert hits[0][0].id == inc.id
```

Implementation:

```python
# examples/incident_management/repository.py
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy.engine import Engine

from examples.incident_management.state import INC_ID_RE, IncidentState, Reporter
from runtime.storage.history_store import HistoryStore
from runtime.storage.session_store import SessionStore


def _create_incident_state(*, id, status, created_at, updated_at, **payload) -> IncidentState:
    return IncidentState(
        id=id,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        query=payload["query"],
        environment=payload["environment"],
        reporter=Reporter(
            id=payload.get("reporter_id", "user-mock"),
            team=payload.get("reporter_team", "platform"),
        ),
    )


class IncidentRepository:
    """Incident app facade preserving the old repository contract."""

    def __init__(
        self,
        *,
        engine: Engine,
        embedder: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        vector_path: str | None = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.85,
        severity_aliases: dict[str, str] | None = None,
    ) -> None:
        self.session_store = SessionStore(
            engine=engine,
            state_type=IncidentState,
            create_state=_create_incident_state,
            id_prefix="INC",
            id_pattern=INC_ID_RE.pattern,
            id_error="expected INC-YYYYMMDD-NNN",
            embedder=embedder,
            vector_store=vector_store,
            vector_path=vector_path,
            vector_index_name=vector_index_name,
        )
        self.history_store = HistoryStore(
            session_store=self.session_store,
            embedder=embedder,
            vector_store=vector_store,
            distance_strategy=distance_strategy,
            similarity_threshold=similarity_threshold,
        )
        self.severity_aliases = severity_aliases or {}

    def create(self, *, query: str, environment: str, reporter_id: str = "user-mock", reporter_team: str = "platform"):
        return self.session_store.create(
            query=query,
            environment=environment,
            reporter_id=reporter_id,
            reporter_team=reporter_team,
        )

    def load(self, incident_id: str):
        return self.session_store.load(incident_id)

    def save(self, incident: IncidentState) -> None:
        self.session_store.save(incident)

    def delete(self, incident_id: str):
        return self.session_store.delete(incident_id)

    def list_all(self, *, include_deleted: bool = False):
        return self.session_store.list_all(include_deleted=include_deleted)

    def list_recent(self, limit: int = 20, *, include_deleted: bool = False):
        return self.session_store.list_recent(limit=limit, include_deleted=include_deleted)

    def find_similar(self, *, query: str, environment: str, status_filter: str = "resolved", threshold=None, limit: int = 5):
        return self.history_store.find_similar(
            query=query,
            environment=environment,
            status_filter=status_filter,
            threshold=threshold,
            limit=limit,
        )
```

Passing test:

```bash
python -m pytest tests/test_incident_repository_example.py tests/test_storage_repository.py tests/test_incident_store.py -q
```

Commit message:

```text
feat: provide incident repository as example app facade
```

### P1-G: Move Incident MCP Server to Example App

Files affected:

- `examples/incident_management/mcp_server.py`
- `src/runtime/mcp_servers/incident.py`
- `tests/test_mcp_incident_server.py`
- `tests/test_mcp_loader.py`

Failing test:

```python
# tests/test_mcp_incident_server.py
from examples.incident_management.mcp_server import (
    IncidentManagementMCPServer,
    create_incident,
    lookup_similar_incidents,
    set_state,
    update_incident,
)


def test_mcp_module_name_is_example_app():
    import examples.incident_management.mcp_server as mod

    assert mod.mcp.name == "incident_management"
```

Implementation:

```python
# examples/incident_management/mcp_server.py
from __future__ import annotations

from dataclasses import dataclass, field

from fastmcp import FastMCP

from examples.incident_management.repository import IncidentRepository

DEFAULT_SEVERITY_ALIASES: dict[str, str] = {
    "sev1": "high",
    "sev2": "high",
    "p1": "high",
    "p2": "high",
    "critical": "high",
    "urgent": "high",
    "high": "high",
    "sev3": "medium",
    "p3": "medium",
    "moderate": "medium",
    "medium": "medium",
    "sev4": "low",
    "p4": "low",
    "info": "low",
    "informational": "low",
    "low": "low",
}


def normalize_severity(value: str | None, aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentManagementMCPServer:
    repository: IncidentRepository | None = None
    severity_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SEVERITY_ALIASES))
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(self, *, repository: IncidentRepository, severity_aliases: dict[str, str] | None = None) -> None:
        self.repository = repository
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases

    def _require_repo(self) -> IncidentRepository:
        if self.repository is None:
            raise RuntimeError("incident_management server not initialized")
        return self.repository

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        hits = self._require_repo().find_similar(query=query, environment=environment, limit=5)
        return {
            "matches": [
                {"id": incident.id, "summary": incident.summary, "resolution": incident.resolution, "score": round(score, 3)}
                for incident, score in hits
            ]
        }

    async def _tool_create_incident(self, query: str, environment: str, reporter_id: str = "user-mock", reporter_team: str = "platform") -> dict:
        incident = self._require_repo().create(
            query=query,
            environment=environment,
            reporter_id=reporter_id,
            reporter_team=reporter_team,
        )
        return incident.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        repo = self._require_repo()
        incident = repo.load(incident_id)
        if "status" in patch:
            incident.status = patch["status"]
        if "severity" in patch:
            incident.severity = normalize_severity(patch["severity"], self.severity_aliases)
        if "category" in patch:
            incident.category = patch["category"]
        if "summary" in patch:
            incident.summary = patch["summary"]
        if "tags" in patch:
            incident.tags = list(patch["tags"])
        if "matched_prior_inc" in patch:
            incident.matched_prior_inc = patch["matched_prior_inc"]
        if "resolution" in patch:
            incident.resolution = patch["resolution"]
        for key, value in patch.items():
            if key.startswith("findings_"):
                incident.findings[key[len("findings_"):]] = value
        repo.save(incident)
        return incident.model_dump()


_default_server = IncidentManagementMCPServer()
mcp = _default_server.mcp


def set_state(*, repository: IncidentRepository, severity_aliases: dict[str, str] | None = None) -> None:
    _default_server.configure(repository=repository, severity_aliases=severity_aliases)


async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str, reporter_id: str = "user-mock", reporter_team: str = "platform") -> dict:
    return await _default_server._tool_create_incident(query, environment, reporter_id, reporter_team)


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)
```

Shim:

```python
# src/runtime/mcp_servers/incident.py
"""Temporary shim for the incident-management example MCP server."""
from examples.incident_management.mcp_server import *  # noqa: F401,F403
```

Passing test:

```bash
python -m pytest tests/test_mcp_incident_server.py tests/test_mcp_loader.py -q
```

Commit message:

```text
feat: move incident MCP server into example app
```

### P1-H: Move Incident Config and Skills to Example App

Files affected:

- `examples/incident_management/config.py`
- `examples/incident_management/config.yaml`
- `examples/incident_management/skills/**`
- `src/runtime/config.py`
- `config/config.yaml`
- `config/config.yaml.example`
- `tests/test_config.py`
- `tests/test_config_loader.py`
- `tests/test_incident_app_config.py`

Failing test:

```python
# tests/test_incident_app_config.py
from pathlib import Path

from examples.incident_management.config import load_incident_app_config
from runtime.config import load_config


def test_framework_config_has_no_incident_specific_keys():
    cfg = load_config("config/config.yaml")
    assert not hasattr(cfg, "incidents")
    assert not hasattr(cfg, "environments")
    assert not hasattr(cfg, "intervention")
    assert "severity_aliases" not in cfg.orchestrator.model_dump()


def test_incident_example_config_carries_domain_keys():
    cfg = load_incident_app_config(Path("examples/incident_management/config.yaml"))
    assert cfg.environments == ["production", "staging", "dev", "local"]
    assert cfg.similarity_threshold > 0
    assert cfg.severity_aliases["sev1"] == "high"
    assert "platform-oncall" in cfg.escalation_teams
    assert cfg.skills_dir.endswith("examples/incident_management/skills")
```

Implementation:

```python
# examples/incident_management/config.py
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class IncidentAppConfig(BaseModel):
    environments: list[str] = Field(default_factory=lambda: ["production", "staging", "dev", "local"])
    similarity_threshold: float = 0.85
    severity_aliases: dict[str, str] = Field(default_factory=dict)
    confidence_threshold: float = 0.75
    escalation_teams: list[str] = Field(default_factory=lambda: ["platform-oncall", "data-oncall", "security-oncall"])
    skills_dir: str = "examples/incident_management/skills"
    incidents_dir: str = "incidents"


def load_incident_app_config(path: str | Path) -> IncidentAppConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return IncidentAppConfig(**raw)
```

`examples/incident_management/config.yaml`:

```yaml
environments:
  - production
  - staging
  - dev
  - local
similarity_threshold: 0.2
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
confidence_threshold: 0.75
escalation_teams:
  - platform-oncall
  - data-oncall
  - security-oncall
skills_dir: examples/incident_management/skills
incidents_dir: incidents
```

Framework `config/config.yaml`:

```yaml
storage:
  metadata:
    url: "sqlite:///sessions/sessions.db"
  vector:
    backend: faiss
    path: "sessions/faiss"
    collection_name: "sessions"
    distance_strategy: cosine
llm:
  default: workhorse
  providers:
    ollama_cloud:
      kind: ollama
      base_url: https://ollama.com
      api_key: ${OLLAMA_API_KEY}
    ollama_local:
      kind: ollama
      base_url: http://localhost:11434
    azure:
      kind: azure_openai
      endpoint: ${AZURE_ENDPOINT}
      api_version: 2024-08-01-preview
      api_key: ${AZURE_OPENAI_KEY}
  models:
    workhorse:
      provider: ollama_cloud
      model: gpt-oss:120b
      temperature: 0.0
    cheap:
      provider: ollama_cloud
      model: gpt-oss:20b
      temperature: 0.2
    smart:
      provider: azure
      model: gpt-4o
      deployment: gpt-4o
      temperature: 0.0
  embedding:
    provider: ollama_local
    model: bge-m3
    dim: 1024
mcp:
  servers: []
paths:
  skills_dir: ""
  data_dir: sessions
orchestrator:
  entry_agent: intake
  signals: ["success", "failed", "needs_input"]
```

Passing test:

```bash
python -m pytest tests/test_config.py tests/test_config_loader.py tests/test_incident_app_config.py -q
```

Commit message:

```text
feat: move incident config and skills into example app
```

### P1-I: Generic Orchestrator Session Facade With Incident Compatibility Wrappers

Files affected:

- `src/runtime/orchestrator.py`
- `src/runtime/graph.py`
- `tests/test_orchestrator.py`
- `tests/test_e2e.py`
- `tests/test_resume.py`

Failing test:

```python
# tests/test_orchestrator_sessions.py
import pytest

from runtime.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths
from runtime.orchestrator import Orchestrator


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(
                name="local_inc",
                transport="in_process",
                module="examples.incident_management.mcp_server",
                category="incident_management",
            ),
        ]),
        paths=Paths(skills_dir="examples/incident_management/skills", data_dir=str(tmp_path)),
    )


@pytest.mark.asyncio
async def test_generic_session_methods_exist_and_incident_wrappers_delegate(cfg):
    orch = await Orchestrator.create_incident_management(cfg)
    try:
        session_id = await orch.start_session(
            payload={
                "query": "api latency",
                "environment": "production",
                "reporter_id": "u",
                "reporter_team": "platform",
            }
        )
        assert session_id.startswith("INC-")
        assert orch.get_session(session_id)["id"] == session_id
        assert orch.get_incident(session_id)["id"] == session_id
        assert orch.list_recent_sessions()
        assert orch.list_recent_incidents()
    finally:
        await orch.aclose()
```

Implementation:

```python
# src/runtime/orchestrator.py
class Orchestrator:
    def __init__(self, cfg: AppConfig, store: SessionStore, history: HistoryStore, ...):
        self.store = store
        self.history = history

    @classmethod
    async def create(cls, cfg: AppConfig) -> "Orchestrator":
        # generic only; use SessionStore with Session and no incident MCP config
        ...

    @classmethod
    async def create_incident_management(cls, cfg: AppConfig, app_config_path: str = "examples/incident_management/config.yaml") -> "Orchestrator":
        from examples.incident_management.config import load_incident_app_config
        from examples.incident_management.repository import IncidentRepository

        app_cfg = load_incident_app_config(app_config_path)
        engine = build_engine(...)
        Base.metadata.create_all(engine)
        embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
        vector_store = build_vector_store(cfg.storage.vector, embedder, engine)
        repo = IncidentRepository(
            engine=engine,
            embedder=embedder,
            vector_store=vector_store,
            vector_path=cfg.storage.vector.path if cfg.storage.vector.backend == "faiss" else None,
            vector_index_name=cfg.storage.vector.collection_name,
            distance_strategy=cfg.storage.vector.distance_strategy,
            similarity_threshold=app_cfg.similarity_threshold,
            severity_aliases=app_cfg.severity_aliases,
        )
        # configure example MCP server through importlib using the exact configured module name
        ...
```

Generic methods:

```python
    def get_session(self, session_id: str) -> dict:
        return self.store.load(session_id).model_dump()

    def list_recent_sessions(self, limit: int = 20) -> list[dict]:
        return [state.model_dump() for state in self.store.list_recent(limit)]

    def delete_session(self, session_id: str) -> dict:
        return self.store.delete(session_id).model_dump()

    async def start_session(self, *, payload: dict) -> str:
        state = self.store.create(**payload)
        await self.graph.ainvoke(GraphState(session=state, next_route=None, last_agent=None, error=None))
        return state.id
```

Deprecated wrappers:

```python
    def get_incident(self, incident_id: str) -> dict:
        return self.get_session(incident_id)

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        return self.list_recent_sessions(limit)

    def delete_incident(self, incident_id: str) -> dict:
        return self.delete_session(incident_id)

    async def start_investigation(self, *, query: str, environment: str, reporter_id: str = "user-mock", reporter_team: str = "platform") -> str:
        return await self.start_session(
            payload={
                "query": query,
                "environment": environment,
                "reporter_id": reporter_id,
                "reporter_team": reporter_team,
            }
        )
```

Passing test:

```bash
python -m pytest tests/test_orchestrator_sessions.py tests/test_orchestrator.py tests/test_e2e.py tests/test_resume.py -q
```

Commit message:

```text
feat: expose generic session facade with incident compatibility wrappers
```

### P1-J: Rename GraphState Key From Incident to Session Safely

Files affected:

- `src/runtime/graph.py`
- `tests/test_build_graph.py`
- `tests/test_gate.py`
- `tests/test_graph_helpers.py`
- `tests/test_agent_node.py`

Failing test:

```python
# tests/test_graph_state_session_key.py
from typing import get_type_hints

from runtime.graph import GraphState
from runtime.state import Session


def test_graph_state_uses_session_key_not_incident_key():
    hints = get_type_hints(GraphState)
    assert "session" in hints
    assert hints["session"] is Session
    assert "incident" not in hints
```

Implementation:

```python
# src/runtime/graph.py
from runtime.state import AgentRun, Session, TokenUsage, ToolCall, UTC_TS_FMT


class GraphState(TypedDict, total=False):
    session: Session
    next_route: str | None
    last_agent: str | None
    gated_target: str | None
    error: str | None
```

Mechanical edits:

```text
state["incident"] -> state["session"]
incident variable names in framework-only functions -> session
inc_id -> session_id
_format_agent_input -> generic labels:
  Session <id>
  Status: <status>
```

Preserve app prompt quality by adding an app hook only if needed:

```python
def _format_agent_input(session: Session) -> str:
    base = f"Session {session.id}\nStatus: {session.status}\n"
    if hasattr(session, "environment"):
        base += f"Environment: {getattr(session, 'environment')}\n"
    if hasattr(session, "query"):
        base += f"Query: {getattr(session, 'query')}\n"
    ...
```

Do not define both keys in `GraphState`. A `TypedDict` alias such as `IncidentGraphState = GraphState` is acceptable for test migration, but having both `incident` and `session` keys invites silent stale-state bugs.

Passing test:

```bash
python -m pytest tests/test_graph_state_session_key.py tests/test_build_graph.py tests/test_gate.py tests/test_graph_helpers.py tests/test_agent_node.py -q
```

Commit message:

```text
refactor: rename graph state incident key to session
```

### P1-K: Move UI Into Incident Example and Keep Wrapper Working

Files affected:

- `examples/incident_management/ui.py`
- `ui/streamlit_app.py`
- `tests/test_ui_import.py`

Failing test:

```python
# tests/test_ui_import.py
def test_example_ui_exports_main():
    from examples.incident_management import ui

    assert callable(ui.main)


def test_legacy_streamlit_entrypoint_delegates_to_example_ui():
    import ui.streamlit_app as legacy
    from examples.incident_management.ui import main

    assert legacy.main is main
```

Implementation:

```python
# ui/streamlit_app.py
"""Compatibility Streamlit entrypoint for the incident-management example."""
from examples.incident_management.ui import main


if __name__ == "__main__":
    main()
```

In `examples/incident_management/ui.py`, move the existing UI code and update imports:

```python
from examples.incident_management.config import load_incident_app_config
from examples.incident_management.repository import IncidentRepository
from runtime.config import AppConfig, MetadataConfig, load_config
from runtime.orchestrator import Orchestrator
```

Repository creation:

```python
def _make_repository(cfg: AppConfig, app_cfg: IncidentAppConfig) -> IncidentRepository:
    url = cfg.storage.metadata.url
    engine = build_engine(MetadataConfig(url=url, pool_size=cfg.storage.metadata.pool_size, echo=cfg.storage.metadata.echo))
    ...
    return IncidentRepository(
        engine=engine,
        embedder=embedder,
        vector_store=vector_store,
        vector_path=cfg.storage.vector.path if cfg.storage.vector.backend == "faiss" else None,
        vector_index_name=cfg.storage.vector.collection_name,
        distance_strategy=cfg.storage.vector.distance_strategy,
        similarity_threshold=app_cfg.similarity_threshold,
    )
```

Keep visible UI labels incident-flavored because this is the incident app.

Passing test:

```bash
python -m pytest tests/test_ui_import.py tests/test_build_single_file.py -q
```

Manual smoke:

```bash
streamlit run ui/streamlit_app.py
streamlit run examples/incident_management/ui.py
```

Commit message:

```text
refactor: move Streamlit UI into incident example app
```

### P1-L: Add `python -m examples.incident_management` Entrypoint

Files affected:

- `examples/incident_management/__main__.py`
- `examples/incident_management/api.py` if API wrapper is useful
- `tests/test_incident_example_main.py`

Failing test:

```python
# tests/test_incident_example_main.py
import subprocess
import sys


def test_incident_example_module_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "examples.incident_management", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "incident-management" in result.stdout
    assert "streamlit" in result.stdout
```

Implementation:

```python
# examples/incident_management/__main__.py
from __future__ import annotations

import argparse
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.incident_management")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["streamlit"],
        help="Run the incident-management Streamlit UI",
    )
    args = parser.parse_args(argv)
    if args.command == "streamlit":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "examples/incident_management/ui.py"])
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Passing test:

```bash
python -m pytest tests/test_incident_example_main.py -q
python -m examples.incident_management --help
```

Commit message:

```text
feat: add incident example module entrypoint
```

### P1-M: Rebuild Bundler for Framework and App Artifacts

Files affected:

- `scripts/build_single_file.py`
- `tests/test_build_single_file.py`
- `dist/framework.py`
- `dist/apps/incident_management.py`
- optional `dist/ui.py`

Failing test:

```python
# tests/test_build_single_file.py
def test_build_produces_framework_and_incident_app():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    framework = repo_root / "dist" / "framework.py"
    app = repo_root / "dist" / "apps" / "incident_management.py"
    framework.unlink(missing_ok=True)
    app.unlink(missing_ok=True)

    result = subprocess.run(
        [sys.executable, "scripts/build_single_file.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert framework.exists()
    assert app.exists()

    framework_src = framework.read_text()
    app_src = app.read_text()
    assert "class Session" in framework_src
    assert "class SessionStore" in framework_src
    assert "class HistoryStore" in framework_src
    assert "class IncidentState" not in framework_src
    assert "class IncidentState" in app_src
    assert "class IncidentManagementMCPServer" in app_src
    assert "from runtime." not in framework_src
    assert "from examples.incident_management" not in framework_src
```

Implementation:

```python
# scripts/build_single_file.py
SRC_ROOT = Path("src/runtime")
EXAMPLE_ROOT = Path("examples/incident_management")
OUT_FRAMEWORK = Path("dist/framework.py")
OUT_INCIDENT_APP = Path("dist/apps/incident_management.py")

FRAMEWORK_MODULE_ORDER = [
    "state.py",
    "config.py",
    "similarity.py",
    "skill.py",
    "llm.py",
    "storage/models.py",
    "storage/engine.py",
    "storage/embeddings.py",
    "storage/vector.py",
    "storage/session_store.py",
    "storage/history_store.py",
    "mcp_servers/observability.py",
    "mcp_servers/remediation.py",
    "mcp_servers/user_context.py",
    "mcp_loader.py",
    "graph.py",
    "orchestrator.py",
    "api.py",
]

INCIDENT_APP_MODULE_ORDER = [
    "state.py",
    "config.py",
    "repository.py",
    "mcp_server.py",
    "ui.py",
    "__main__.py",
]
```

Regex hardening:

```python
INTRA_RUNTIME_IMPORT_RE = re.compile(
    r"^\s*from\s+runtime(?:\.[\w.]+)?\s+import\s+(?:\([^)]*\)|.*?)$",
    re.MULTILINE | re.DOTALL,
)

INTRA_EXAMPLE_IMPORT_RE = re.compile(
    r"^\s*from\s+examples\.incident_management(?:\.[\w.]+)?\s+import\s+(?:\([^)]*\)|.*?)$",
    re.MULTILINE | re.DOTALL,
)
```

Critical mitigation: app bundle must either include framework code first or import `framework.py` as a sibling. Prefer sibling import to avoid name collisions:

```python
def build_incident_app() -> None:
    # Strip only intra-example imports.
    # Rewrite `from runtime.x import Y` to `from framework import Y`.
```

Passing test:

```bash
python -m pytest tests/test_build_single_file.py -q
python scripts/build_single_file.py
python -m py_compile dist/framework.py dist/apps/incident_management.py
```

Commit message:

```text
build: split framework and incident app bundles
```

### P1-N: Remove Incident Terms From Framework and Run End-to-End Verification

Files affected:

- `src/runtime/**`
- `tests/**`
- `README.md` if present
- `pyproject.toml`
- all previously touched files

Failing test:

```python
# tests/test_framework_domain_cleanliness.py
import pathlib


FORBIDDEN = [
    "Incident",
    "incident",
    "incidents",
    "INC-",
    "investigation",
    "severity_aliases",
    "escalation_teams",
    "environments",
]


def test_runtime_package_has_no_incident_domain_terms_outside_shims():
    root = pathlib.Path("src/runtime")
    allowed = {
        pathlib.Path("src/runtime/incident.py"),
        pathlib.Path("src/runtime/mcp_servers/incident.py"),
        pathlib.Path("src/runtime/storage/repository.py"),
    }
    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text()
        for term in FORBIDDEN:
            if term in text:
                offenders.append(f"{path}:{term}")
    assert offenders == []
```

Implementation:

- Rename docstrings and comments in `src/runtime` from incident to session.
- Update `pyproject.toml` description:

```toml
description = "Generic agent orchestration runtime with example apps"
requires-python = ">=3.14"
```

- Update test imports to canonical modules once shims are proven:

```text
orchestrator.config -> runtime.config
orchestrator.graph -> runtime.graph
orchestrator.storage.* -> runtime.storage.*
orchestrator.mcp_servers.incident -> examples.incident_management.mcp_server
runtime.incident -> runtime.state or examples.incident_management.state
```

- Keep shim tests for deprecated import surface until removal is scheduled.

Passing test:

```bash
python -m pytest --collect-only -q
python -m pytest -q
python scripts/build_single_file.py
python -m py_compile dist/framework.py dist/apps/incident_management.py
python -m examples.incident_management --help
```

Commit message:

```text
chore: verify phase1 domain extraction complete
```

## 5. Sequencing & Dependencies Graph

```text
P1-A import stabilization
  |
  v
P1-B generic Session model
  |
  v
P1-C IncidentState example extension
  |
  v
P1-D SessionStore active CRUD
  |
  v
P1-E HistoryStore closed search
  |
  v
P1-F incident repository facade
  |
  +-----------------------------+
  |                             |
  v                             v
P1-G example MCP server       P1-H example config and skills
  |                             |
  +-------------+---------------+
                |
                v
P1-I generic Orchestrator facade plus wrappers
                |
                v
P1-J GraphState incident -> session key
                |
                v
P1-K UI moves to example app
                |
                v
P1-L python -m examples.incident_management
                |
                v
P1-M split dist bundler
                |
                v
P1-N cleanup, grep gates, full smoke
```

Parallelizable work after P1-F:

- P1-G MCP move and P1-H config/skills move can be done independently if both use the incident repository facade.
- P1-K UI move can start after P1-I exposes stable wrappers.
- P1-M should wait until file locations are final.

Do not start P1-J before P1-I. Renaming the `GraphState` key without generic facade wrappers risks breaking every graph, gate, resume, and UI test at once.

## 6. Risks + Mitigations

Bundler import-strip regex can eat too much.

- Current `INTRA_IMPORT_RE` uses `re.DOTALL` with a permissive `.*?` and can consume across unrelated lines when parenthesized imports or comments appear nearby.
- Mitigation: use separate runtime and example regexes; add fixture tests with single-line imports, parenthesized imports, comments between imports, and imports inside docstrings. Verify `ast.parse` and `py_compile` on both bundles.

Dist-bundle name collisions.

- If `framework.py` and `incident_management.py` both inline classes named `Session`, `AgentRun`, `Config`, or `main`, later definitions can shadow earlier ones.
- Mitigation: make app bundle import `from framework import ...` instead of inlining framework again. If a single fully self-contained app artifact is required, prefix module section comments and add tests that instantiate both `Session` and `IncidentState` from the bundle.

Soft-shim chain breakage.

- There may be three import paths during P1: `orchestrator.*`, `runtime.incident`, and canonical `runtime.state` / `examples.incident_management.*`.
- Mitigation: add explicit shim tests in P1-A and remove shims only in a later cleanup phase. Do not let production config point at shim modules; config should point at canonical example modules.

Test fixture cascades.

- Many tests build `AppConfig(paths=Paths(skills_dir="config/skills", incidents_dir=tmp_path))`. Removing `incidents_dir` in one commit will cascade failures through API, orchestrator, resume, and E2E tests.
- Mitigation: add `Paths.data_dir` while keeping `incidents_dir` as a deprecated property or accepted alias until tests migrate. Move fixtures gradually to `examples/incident_management/skills`.

GraphState TypedDict alias semantics.

- `IncidentGraphState = GraphState` does not preserve old key names; it only aliases the same type. Tests using `GraphState(incident=...)` may still run at runtime because `TypedDict` is just `dict`, but graph nodes reading `session` will fail with `KeyError`.
- Mitigation: update all call sites atomically in P1-J and add a test asserting `"incident" not in get_type_hints(GraphState)`. Avoid runtime fallback from `incident` to `session`, because that hides missed migrations.

Generic `Session` losing app fields during row mapping.

- If `SessionStore._row_to_state()` defaults to `Session`, incident fields such as `query`, `environment`, and `reporter` disappear after reload. The graph prompt and MCP updates then break.
- Mitigation: require `state_type` and `create_state` for app-specific stores; test that `IncidentRepository.load()` returns `IncidentState` and preserves all domain fields.

ID minter ownership.

- Framework should not know `INC-YYYYMMDD-NNN`, but existing tests and UI depend on it.
- Mitigation: `SessionStore` accepts an injected prefix/minter or app factory. Incident repository configures `id_prefix="INC"` and validation regex. Generic framework default uses `ses_...`.

Closed-session semantics.

- Existing `find_similar` filters `status_filter="resolved"`. Roadmap says `HistoryStore` is for closed-session search, but statuses also include `stopped`, `escalated`, and `deleted`.
- Mitigation: Phase 1 preserves behavior with `status_filter="resolved"`. Add an open question on whether `HistoryStore` should default to all terminal non-deleted statuses later.

Database table rename migration.

- Moving from `incidents` to `sessions` can orphan existing local SQLite data.
- Mitigation: for P1 tests, create fresh tables. For developer continuity, optionally keep `IncidentRow = SessionRow` alias but do not map two SQLAlchemy classes to the same table. A real data migration can be a separate script if production data exists.

MCP singleton identity.

- Existing orchestrator uses `importlib.import_module(_INCIDENT_MCP_MODULE).set_state(...)` to configure the same module instance the loader imports. Moving modules can recreate the old bug where one singleton is configured and another is loaded.
- Mitigation: compare configured server module strings exactly and import that exact module. Tests should invoke a loaded MCP tool through `load_tools()` after `Orchestrator.create_incident_management()`.

UI remains incident-flavored by design.

- The goal is not to genericize the existing UI. Moving it under the example means visible labels like "Recent INCs" are acceptable.
- Mitigation: keep `ui/streamlit_app.py` as a wrapper so current commands still work.

Config removal can break environment interpolation tests.

- `load_config()` currently interpolates `${OLLAMA_API_KEY}` and external MCP secrets from `config/config.yaml`.
- Mitigation: keep interpolation behavior unchanged. Move only domain keys. Tests should use fixture YAMLs that do not require real secrets unless env vars are set in test.

Python version mismatch.

- Roadmap invariant says Python 3.14. `pyproject.toml` currently says `>=3.11`.
- Mitigation: update `requires-python = ">=3.14"` in P1-N after CI image compatibility is confirmed. If CI still runs 3.11, this change must be coordinated with CI before commit.

`examples/incident-management` vs `examples/incident_management`.

- Current files live under a hyphenated directory, but the hard constraint requires `python -m examples.incident_management`.
- Mitigation: create the importable underscore package and move/copy skills there. Leave the hyphenated directory only if documentation needs it, but do not put importable Python code there.

## 7. Done Criteria

Test and smoke criteria:

```bash
python -m pytest --collect-only -q
# Expected: 191 tests collected, 0 collection errors.

python -m pytest -q
# Expected: 191 passed, with only explicitly marked external-provider smoke skips if env vars are absent.

python scripts/build_single_file.py
python -m py_compile dist/framework.py dist/apps/incident_management.py
python -m examples.incident_management --help
```

UI smoke:

```bash
streamlit run ui/streamlit_app.py
streamlit run examples/incident_management/ui.py
```

API smoke:

```bash
ASR_CONFIG=config/config.yaml python -m uvicorn runtime.api:get_app --factory --port 37776
```

Example app smoke:

```bash
ASR_CONFIG=config/config.yaml python -m examples.incident_management --help
```

Concrete grep checks:

```bash
rg -n "Incident|incident|incidents|INC-|investigation|severity_aliases|escalation_teams|environments" src/runtime \
  -g '*.py' \
  -g '!incident.py' \
  -g '!mcp_servers/incident.py' \
  -g '!storage/repository.py'
# Expected: no matches.

rg -n "from orchestrator|orchestrator\\." src tests examples ui scripts config -g '*.py' -g '*.yaml'
# Expected: only explicit compatibility-shim tests and legacy wrapper docs.

rg -n "IncidentRepository" src/runtime -g '*.py'
# Expected: only storage/repository.py compatibility shim, if retained.

rg -n "class IncidentState|incident_management|lookup_similar_incidents|update_incident" examples/incident_management
# Expected: matches present.

rg -n "bge-m3" config examples
# Expected: embedding config still uses bge-m3.
```

Bundle checks:

```bash
test -f dist/framework.py
test -f dist/apps/incident_management.py
rg -n "class SessionStore|class HistoryStore|class Session" dist/framework.py
rg -n "class IncidentState|class IncidentManagementMCPServer" dist/apps/incident_management.py
! rg -n "class IncidentState|IncidentManagementMCPServer|severity_aliases" dist/framework.py
```

Behavior checks:

- `IncidentRepository` tests remain largely unchanged; imports may move to `examples.incident_management.repository`.
- `lookup_similar_incidents`, `create_incident`, and `update_incident` still work through FastMCP and direct-call shims.
- Existing UI can start, list recent incidents, start an investigation, resume a paused incident, and soft-delete an incident.
- `SessionStore` can create/load/save/delete generic sessions without importing `examples.incident_management`.
- `HistoryStore` can perform keyword fallback without an embedder and vector similarity when FAISS plus embedder are configured.
- Dist bundle rebuild creates framework and app artifacts and both parse cleanly.

## 8. Open Questions for Synthesis

1. Should Phase 1 physically rename the SQL table from `incidents` to `sessions`, or keep the old table name for migration-free compatibility and only rename the ORM class? The clean architecture answer is `sessions`; the low-risk answer is table alias until a migration phase.

2. Should `HistoryStore` default to `status_filter="resolved"` for behavior parity, or define "closed" as `resolved | stopped | escalated`? P1 should preserve `resolved`; roadmap language suggests broader semantics later.

3. Should framework `Orchestrator.create()` be fully runnable with no skills and no MCP servers, or should a framework app always provide skills? The roadmap says framework runs without the example app and exposes lifecycle plus tools when wired up, so empty skills should not crash.

4. How long should `orchestrator.*` compatibility imports remain? Keeping them through P1 lowers risk, but they weaken the "runtime is canonical" signal.

5. Should `runtime.mcp_servers.observability`, `remediation`, and `user_context` also move into the incident example now? They are not all strictly incident-state servers, but current skills use them as the incident app's tool environment. Leaving them in framework may still smell domain/demo-specific.

6. Is `matched` a generic `SessionStatus` or an incident/dedup-specific status? It exists today and tests likely expect it. Phase 1 can keep it to avoid behavior churn, but Phase 7 may want a generic `merged` or subscriber state.

7. Should the example app own deprecated event names (`investigation_started`, `incident_id`) while framework emits `session_started` and `session_id`, or should P1 keep event names unchanged for UI stability? The plan keeps wrappers, but the exact boundary should be reviewed.

8. Should skills under `config/skills` be removed entirely or kept as wrappers/symlinks to `examples/incident_management/skills` until docs catch up?

9. Current repo state already has modified files and partial `runtime` migration. Before executing this plan, decide whether to incorporate those edits as baseline or first land a stabilization commit.

STATUS: DRAFT_COMPLETE
Summary: Phase 1 is staged as 14 TDD tasks, P1-A through P1-N, extracting incident code into `examples.incident_management` while preserving compatibility until tests, UI, and dist bundles are green.
