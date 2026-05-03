# Phase 2 — Extensible State + LangGraph Checkpointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the framework's hard-coded coupling to `examples.incident_management.state.IncidentState` with a config-resolved `Generic[StateT]` parameterisation, and replace the hand-rolled `pending_intervention` + `build_resume_graph` resume mechanism with the official LangGraph checkpointer (`SqliteSaver` for dev, `PostgresSaver` for prod) plus `interrupt()` / `Command(resume=...)`.

**Architecture:** Each app declares `runtime.state_class: <dotted.path>` in YAML; `Orchestrator.create()` resolves that class via `importlib`, parameterises `Orchestrator[StateT]`, `SessionStore[StateT]`, `HistoryStore[StateT]`, and compiles the LangGraph with a checkpointer pool wired to `cfg.storage.metadata.url` (separate connection pool, WAL on for SQLite). The gate node both raises `interrupt(payload)` and dual-writes the same payload to `Session.pending_intervention` for UI back-compat. P1's `IncidentRepository` shim, `runtime/incident.py`, and `GraphState.incident` bridge alias are deleted as part of this phase.

**Tech Stack:** `langgraph>=0.6` (pinned in `pyproject.toml`), `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres`, `psycopg[binary]>=3.2` (already a top-level dep), `sqlalchemy>=2.0`, `pydantic>=2.9`, `pytest-asyncio`.

---

## 1. Goal + Scope

Phase 1 finished with the framework cleaned of incident-flavoured *fields* and *config keys*, but the framework still has structural coupling to the example app:

- `runtime.storage.history_store.HistoryStore` directly imports `examples.incident_management.state.IncidentState` to hydrate rows.
- `runtime.graph.GraphState` carries a `session: Session` *and* an `incident: Session` bridge alias to avoid breaking the gate node, MCP servers, and tests in one mechanical sweep.
- `runtime.incident` (a shim re-exporting `Session` as `Incident`) and `runtime.storage.repository.IncidentRepository` (a `SessionStore + HistoryStore` facade) still exist to keep callers compiling.
- `runtime.config.Paths.skills_dir` defaults to `examples/incident_management/skills` — an example-app path baked into the framework default.
- The gate node persists `pending_intervention` to the Session row and `Orchestrator.resume_session` invokes a hand-rolled `build_resume_graph` rather than LangGraph's native `interrupt()` + `Command(resume=...)`.

Phase 2 closes those gaps:

1. Add a `runtime.state_class` config key that lets each app inject its own `Session` subclass.
2. Parameterise the framework's stateful surfaces (`Orchestrator`, `SessionStore`, `HistoryStore`, `build_graph`) on `Generic[StateT]`.
3. Wire LangGraph's official checkpointers (`SqliteSaver`/`PostgresSaver`) to the same DB URL as the metadata store, using a *separate* connection pool with WAL on for SQLite.
4. Migrate gate to `interrupt()`; replace `build_resume_graph` with `Command(resume=...)`; keep dual-writing `pending_intervention` so the existing Streamlit UI doesn't break.
5. Delete the P1 carry-over debt: `runtime/incident.py`, `IncidentRepository` shim, `GraphState.incident` bridge alias, `Paths.skills_dir` framework default.

**In scope:** framework-side state extensibility, official checkpointer wiring, gate/resume migration, P1 debt cleanup, end-to-end resume-after-cold-restart test.

**Out of scope:** multi-session concurrency / async-loop-per-session (locked for Phase 3), custom checkpointer back-ends beyond SQLite/Postgres, distributed checkpointing, websocket UI streaming.

## 2. Target Architecture After Phase 2

```
src/runtime/
  config.py              # +RuntimeConfig{state_class}; Paths.skills_dir default = None
  state.py               # unchanged: Session base
  graph.py               # GraphState[StateT]; no `incident` bridge; gate uses interrupt()
  orchestrator.py        # Orchestrator[StateT]; checkpointer wired; resume via Command()
  checkpointer.py        # NEW — make_checkpointer(cfg) factory
  storage/
    session_store.py     # SessionStore[StateT] — accepts state_cls in __init__
    history_store.py     # HistoryStore[StateT] — accepts state_cls in __init__
    models.py            # unchanged
    engine.py            # unchanged
    embeddings.py        # unchanged
    vector.py            # unchanged
    __init__.py          # drop IncidentRepository export
    repository.py        # DELETED
  incident.py            # DELETED
  mcp_servers/
    incident.py          # DELETED (shim was a P1 leftover)

examples/incident_management/
  state.py               # IncidentState(Session) — unchanged
  config.py              # IncidentAppConfig — unchanged
  config.yaml            # +runtime.state_class, +paths.skills_dir
  mcp_server.py          # uses SessionStore + HistoryStore directly (no IncidentRepository)
  skills/                # unchanged

tests/
  test_runtime_config.py            # NEW — RuntimeConfig + state_class resolution
  test_state_resolver.py            # NEW — importlib resolution, error paths
  test_checkpointer.py              # NEW — sqlite + postgres make_checkpointer
  test_orchestrator_generic.py      # NEW — Generic[StateT] parametrisation
  test_session_store_generic.py     # NEW — SessionStore hydrates configured class
  test_history_store_generic.py     # NEW — HistoryStore hydrates configured class
  test_resume.py                    # MODIFIED — interrupt() + Command(resume) flow
  test_gate.py                      # MODIFIED — interrupt assertion + dual-write
  test_resume_cold_restart.py       # NEW — cold-restart resume e2e
  test_runtime_package.py           # MODIFIED — drop GraphState.incident assertion
```

## 3. Naming Map / API Changes

| Before                                                        | After                                                       |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| `runtime.config.AppConfig` (no `runtime` field)               | `AppConfig.runtime: RuntimeConfig`                          |
| `runtime.config.Paths.skills_dir = "config/skills"`           | `Paths.skills_dir: str \| None = None`                      |
| `runtime.storage.repository.IncidentRepository`               | DELETED — callers use `SessionStore` + `HistoryStore`       |
| `runtime.incident` (shim)                                     | DELETED                                                     |
| `runtime.graph.GraphState{ session, incident }`               | `class GraphState(TypedDict, Generic[StateT])` — no alias   |
| `Orchestrator(... resume_graph: ...)`                         | `Orchestrator[StateT]` — no `resume_graph`; uses checkpointer |
| `SessionStore(engine, ...)`                                   | `SessionStore[StateT](engine, *, state_cls: type[StateT], ...)` |
| `HistoryStore(engine, ...)`                                   | `HistoryStore[StateT](engine, *, state_cls: type[StateT], ...)` |
| `build_graph(*, cfg, skills, store, registry)`                | `build_graph[StateT](*, cfg, skills, store, registry, state_cls, checkpointer)` |
| `build_resume_graph(...)`                                     | DELETED — orchestrator calls `graph.invoke(Command(resume=...), config)` |
| Gate node only writes `inc.pending_intervention`              | Gate node calls `interrupt(payload)` AND writes `pending_intervention` |
| `Orchestrator.resume_investigation(...)`                      | unchanged signature; internally uses `Command(resume=...)`  |

## 4. Task Breakdown

### P2-A — Add `RuntimeConfig` to `AppConfig` with `state_class` field

Introduce a top-level `runtime` config block; default `state_class` to `runtime.state.Session` so apps with no override still work.

**Files:**
- Modify: `src/runtime/config.py`
- Create: `tests/test_runtime_config.py`

**Steps:**

- [ ] **1. Failing test** — create `tests/test_runtime_config.py`:

```python
"""Tests for RuntimeConfig added in P2-A."""
import pytest
import yaml


def test_runtime_config_has_state_class_default():
    from runtime.config import RuntimeConfig
    cfg = RuntimeConfig()
    assert cfg.state_class == "runtime.state.Session"


def test_app_config_has_runtime_block():
    from runtime.config import AppConfig
    fields = set(AppConfig.model_fields.keys())
    assert "runtime" in fields


def test_runtime_config_loads_from_yaml(tmp_path):
    from runtime.config import load_app_config
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "llm": {"providers": {"primary": {"kind": "stub"}}, "embedding": {"backend": "stub"}},
        "mcp": {"servers": []},
        "runtime": {"state_class": "examples.incident_management.state.IncidentState"},
    }))
    cfg = load_app_config(yaml_path)
    assert cfg.runtime.state_class == "examples.incident_management.state.IncidentState"


def test_runtime_state_class_required_string_format():
    from runtime.config import RuntimeConfig
    with pytest.raises(Exception):
        RuntimeConfig(state_class="not-a-dotted-path")
```

- [ ] **2. Run** → fail (`AttributeError: module 'runtime.config' has no attribute 'RuntimeConfig'`).

- [ ] **3. Implementation** — append to `src/runtime/config.py` (above `class AppConfig`):

```python
_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


class RuntimeConfig(BaseModel):
    """Framework-level runtime knobs.

    ``state_class`` is the dotted import path of the application's
    ``Session`` subclass. The framework loads it lazily at orchestrator
    construction time via :func:`importlib.import_module`. Default is
    the bare framework ``Session`` so apps with no subclass keep
    working.
    """
    state_class: str = "runtime.state.Session"

    @field_validator("state_class")
    @classmethod
    def _validate_dotted_path(cls, v: str) -> str:
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(
                f"state_class must be a dotted import path "
                f"(e.g. 'examples.foo.state.FooState'), got {v!r}"
            )
        return v
```

Add to `AppConfig`:

```python
class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: Paths = Field(default_factory=Paths)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # NEW
```

Add `from pydantic import field_validator` to imports if not already present.

- [ ] **4. Run tests** → `tests/test_runtime_config.py` passes; existing `tests/test_config*` still pass.

- [ ] **5. Commit** — `feat(runtime): add RuntimeConfig with state_class field`

---

### P2-B — Resolve state class at orchestrator construction (`importlib` + `getattr`)

A reusable resolver that the orchestrator and storage will call. Errors with a useful message if the dotted path is wrong, and validates that the resolved class is a subclass of `Session`.

**Files:**
- Create: `src/runtime/state_resolver.py`
- Create: `tests/test_state_resolver.py`

**Steps:**

- [ ] **1. Failing test** — create `tests/test_state_resolver.py`:

```python
"""Tests for runtime.state_resolver — dotted-path → state class."""
import pytest


def test_resolves_default_session():
    from runtime.state_resolver import resolve_state_class
    from runtime.state import Session
    cls = resolve_state_class("runtime.state.Session")
    assert cls is Session


def test_resolves_incident_state():
    from runtime.state_resolver import resolve_state_class
    from examples.incident_management.state import IncidentState
    cls = resolve_state_class("examples.incident_management.state.IncidentState")
    assert cls is IncidentState


def test_rejects_non_session_subclass():
    with pytest.raises(TypeError, match="must be a Session subclass"):
        from runtime.state_resolver import resolve_state_class
        resolve_state_class("builtins.dict")


def test_rejects_missing_module():
    with pytest.raises(ImportError, match="cannot import"):
        from runtime.state_resolver import resolve_state_class
        resolve_state_class("does.not.exist.Whatever")


def test_rejects_missing_attr():
    with pytest.raises(AttributeError, match="has no attribute"):
        from runtime.state_resolver import resolve_state_class
        resolve_state_class("runtime.state.NoSuchClass")
```

- [ ] **2. Run** → fail (`ModuleNotFoundError: No module named 'runtime.state_resolver'`).

- [ ] **3. Implementation** — create `src/runtime/state_resolver.py`:

```python
"""Dotted-path → ``Session`` subclass resolver.

The framework never imports application state directly; it reads
``cfg.runtime.state_class`` and resolves it lazily via this helper at
``Orchestrator.create()`` time.
"""
from __future__ import annotations

import importlib
from typing import Type

from runtime.state import Session


def resolve_state_class(dotted_path: str) -> Type[Session]:
    """Resolve ``module.path.ClassName`` → the class.

    Raises:
        ImportError: if the module cannot be imported.
        AttributeError: if the class is not in the module.
        TypeError: if the resolved attribute is not a ``Session`` subclass.
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"state_class={dotted_path!r} is not a dotted path "
            f"(expected 'module.path.ClassName')"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"cannot import module {module_path!r} for state_class "
            f"{dotted_path!r}: {e}"
        ) from e
    if not hasattr(module, class_name):
        raise AttributeError(
            f"module {module_path!r} has no attribute {class_name!r} "
            f"(state_class={dotted_path!r})"
        )
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, Session)):
        raise TypeError(
            f"state_class={dotted_path!r} resolved to {cls!r}, which "
            f"must be a Session subclass"
        )
    return cls
```

- [ ] **4. Run tests** → all 5 in `test_state_resolver.py` pass.

- [ ] **5. Commit** — `feat(runtime): add state_resolver for dotted-path lookups`

---

### P2-C — Parameterise `Orchestrator[StateT]`, `SessionStore[StateT]`, `HistoryStore[StateT]`

Add `Generic[StateT]` to the three primary stateful surfaces and thread the resolved class through the constructors. The class is used at row→object hydration time inside `_row_to_session`.

**Files:**
- Modify: `src/runtime/orchestrator.py`
- Modify: `src/runtime/storage/session_store.py`
- Modify: `src/runtime/storage/history_store.py`
- Create: `tests/test_session_store_generic.py`
- Create: `tests/test_history_store_generic.py`
- Create: `tests/test_orchestrator_generic.py`

**Steps:**

- [ ] **1. Failing tests** — create `tests/test_session_store_generic.py`:

```python
"""Tests for SessionStore[StateT] generic parametrisation (P2-C)."""
import pytest
from sqlalchemy import create_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.state import Session


@pytest.fixture()
def engine(tmp_path):
    e = create_engine(
        f"sqlite:///{tmp_path}/t.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(e)
    return e


def test_session_store_accepts_state_cls(engine):
    store = SessionStore(engine=engine, state_cls=Session)
    assert store._state_cls is Session


def test_session_store_hydrates_to_configured_class(engine):
    from examples.incident_management.state import IncidentState, Reporter
    store = SessionStore(engine=engine, state_cls=IncidentState)
    inc = IncidentState(
        id="INC-20260502-001",
        status="new",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
        query="latency", environment="production",
        reporter=Reporter(id="u1", team="p"),
    )
    store.save(inc)
    loaded = store.load("INC-20260502-001")
    assert isinstance(loaded, IncidentState)
    assert loaded.environment == "production"


def test_session_store_default_state_cls_is_session(engine):
    """No explicit state_cls → SessionStore defaults to bare Session."""
    store = SessionStore(engine=engine)
    assert store._state_cls is Session
```

Create `tests/test_history_store_generic.py`:

```python
"""Tests for HistoryStore[StateT] generic parametrisation (P2-C)."""
import pytest
from sqlalchemy import create_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.history_store import HistoryStore


@pytest.fixture()
def engine(tmp_path):
    e = create_engine(
        f"sqlite:///{tmp_path}/t.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(e)
    return e


def test_history_store_accepts_state_cls(engine):
    from examples.incident_management.state import IncidentState
    h = HistoryStore(engine=engine, state_cls=IncidentState)
    assert h._state_cls is IncidentState


def test_history_store_no_direct_incident_state_import():
    """Regression: HistoryStore must not import IncidentState at module scope."""
    import runtime.storage.history_store as mod
    src = open(mod.__file__).read()
    assert "examples.incident_management" not in src, (
        "HistoryStore leaks example-app import"
    )
```

Create `tests/test_orchestrator_generic.py`:

```python
"""Tests for Orchestrator[StateT] (P2-C)."""
import pytest


@pytest.mark.asyncio
async def test_orchestrator_resolves_default_session(stub_cfg_factory):
    from runtime.orchestrator import Orchestrator
    from runtime.state import Session
    cfg = stub_cfg_factory()  # no runtime.state_class override
    async with await Orchestrator.create(cfg) as orch:
        assert orch.state_cls is Session


@pytest.mark.asyncio
async def test_orchestrator_resolves_incident_state(stub_cfg_factory):
    from runtime.orchestrator import Orchestrator
    from examples.incident_management.state import IncidentState
    cfg = stub_cfg_factory(
        state_class="examples.incident_management.state.IncidentState",
    )
    async with await Orchestrator.create(cfg) as orch:
        assert orch.state_cls is IncidentState
```

Add `stub_cfg_factory` fixture to `tests/conftest.py` (or extend it) — see `tests/test_orchestrator.py:24` for the existing pattern using `Paths(skills_dir=..., incidents_dir=...)`.

- [ ] **2. Run** → all three new test files fail with `state_cls` constructor errors.

- [ ] **3. Implementation** — `src/runtime/storage/session_store.py`:

```python
from typing import Generic, Optional, Type, TypeVar
from runtime.state import Session

StateT = TypeVar("StateT", bound=Session)


class SessionStore(Generic[StateT]):
    """Active session/incident lifecycle store, parametrised on Session subclass."""

    def __init__(
        self,
        *,
        engine: Engine,
        state_cls: Type[StateT] = Session,  # type: ignore[assignment]
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        vector_path: Optional[str] = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
    ) -> None:
        self._engine = engine
        self._state_cls = state_cls
        self._embedder = embedder
        self._vector_store = vector_store
        self._vector_path = vector_path
        self._vector_index_name = vector_index_name
        self._distance_strategy = distance_strategy

    def _row_to_session(self, row) -> StateT:
        """Hydrate a SessionRow → the configured ``state_cls`` instance."""
        return self._state_cls.model_validate(row, from_attributes=True)
```

Apply analogous changes to `src/runtime/storage/history_store.py`. The current `_row_to_session` uses `IncidentState.model_validate` — replace `IncidentState` with `self._state_cls` and remove the `from examples.incident_management.state import IncidentState as Incident` import.

`src/runtime/orchestrator.py` — add the type var and parametrise:

```python
from typing import Generic, TypeVar
from runtime.state import Session
from runtime.state_resolver import resolve_state_class

StateT = TypeVar("StateT", bound=Session)


class Orchestrator(Generic[StateT]):
    state_cls: type[StateT]

    def __init__(self, cfg: AppConfig, store: SessionStore[StateT],
                 history: HistoryStore[StateT],
                 skills: dict[str, Skill], registry: ToolRegistry, graph,
                 exit_stack: AsyncExitStack,
                 state_cls: type[StateT],
                 app_cfg: IncidentAppConfig | None = None):
        self.state_cls = state_cls
        # ... (no resume_graph param anymore — see P2-I)
```

In `Orchestrator.create()`:

```python
state_cls = resolve_state_class(cfg.runtime.state_class)
store = SessionStore(engine=engine, state_cls=state_cls, ...)
history = HistoryStore(engine=engine, state_cls=state_cls, ...)
```

- [ ] **4. Run tests** → `pytest tests/test_session_store_generic.py tests/test_history_store_generic.py tests/test_orchestrator_generic.py` all pass; existing storage and orchestrator tests still pass (they default to `Session`).

- [ ] **5. Commit** — `feat(runtime): parameterise Orchestrator/SessionStore/HistoryStore on StateT`

---

### P2-D — Drop `GraphState.incident` bridge alias

P1 added `GraphState.incident` alongside `GraphState.session` to keep gate, MCP servers, and tests working during the rename. P2 removes the alias and migrates all readers to `state["session"]`.

**Files:**
- Modify: `src/runtime/graph.py`
- Modify: `src/runtime/orchestrator.py`
- Modify: `src/runtime/mcp_servers/incident.py` (if any reads `state["incident"]`)
- Modify: `examples/incident_management/mcp_server.py` (gate-related callbacks)
- Modify: `tests/test_runtime_package.py`
- Modify: any other test that asserts on `state["incident"]` (likely `test_gate.py`, `test_build_graph.py`)

**Steps:**

- [ ] **1. Failing test** — modify `tests/test_runtime_package.py`:

```python
def test_graph_state_has_no_incident_alias():
    """P2-D: the bridge alias is gone; only `session` remains."""
    from runtime.graph import GraphState
    import typing
    hints = typing.get_type_hints(GraphState)
    assert "session" in hints
    assert "incident" not in hints, (
        f"GraphState.incident bridge alias must be removed; got hints {hints}"
    )
```

- [ ] **2. Run** → fail (alias still there).

- [ ] **3. Implementation** — in `src/runtime/graph.py`, remove the `incident: Session` line from `class GraphState(TypedDict, total=False)` and grep+replace `state["incident"]` → `state["session"]` (and `.get("incident")` → `.get("session")`) throughout `runtime/graph.py`. Repeat the grep+replace in `src/runtime/orchestrator.py`, `examples/incident_management/mcp_server.py`, `tests/test_gate.py`, `tests/test_build_graph.py`, `tests/test_resume.py`. Re-read each diff to confirm semantics didn't shift (the alias was a literal duplicate of `session`; readers should be unaffected).

- [ ] **4. Run tests** — full suite. Fix any straggler readers caught by failing tests.

- [ ] **5. Commit** — `refactor(runtime): drop GraphState.incident bridge alias`

---

### P2-E — Decouple `HistoryStore` from `IncidentState` import

This is mostly already enforced by P2-C's import-scan test, but make sure module-level imports are scrubbed and `find_similar` returns `list[StateT]` (not `list[IncidentState]`).

**Files:**
- Modify: `src/runtime/storage/history_store.py`
- Modify: `tests/test_storage_find_similar.py`

**Steps:**

- [ ] **1. Failing test** — extend `tests/test_history_store_generic.py`:

```python
def test_find_similar_returns_state_cls_instances(engine):
    """find_similar hydrates to the configured state_cls, not Incident-specific."""
    from examples.incident_management.state import IncidentState, Reporter
    from runtime.storage.session_store import SessionStore
    from runtime.storage.history_store import HistoryStore

    sstore = SessionStore(engine=engine, state_cls=IncidentState)
    sstore.save(IncidentState(
        id="INC-20260502-002", status="resolved",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
        query="payments slow", environment="production",
        reporter=Reporter(id="u1", team="p"),
        summary="resolved by restart",
    ))
    h = HistoryStore(engine=engine, state_cls=IncidentState,
                     similarity_threshold=0.0)
    matches = h.find_similar(query="payments slow", environment="production")
    assert all(isinstance(m, IncidentState) for m in matches)
```

- [ ] **2. Run** → may already pass if P2-C was thorough; if not, fix.

- [ ] **3. Implementation** — verify `runtime/storage/history_store.py` has no `from examples.` imports at module scope. Annotate `find_similar` return type as `list[StateT]`. Confirm no string `"IncidentState"` references remain (`grep -n IncidentState src/runtime/storage/history_store.py` → empty).

- [ ] **4. Run tests** → pass.

- [ ] **5. Commit** — `refactor(runtime): scrub HistoryStore of IncidentState coupling`

---

### P2-F — Add SQLite checkpointer wired to `storage.metadata.url` + WAL

Introduce `src/runtime/checkpointer.py`. The factory inspects `cfg.storage.metadata.url`, branches on `sqlite:` vs `postgresql:`, and returns the appropriate `BaseCheckpointSaver`. SQLite gets `journal_mode=WAL` + `check_same_thread=False`. Postgres branch is stubbed in this task and filled in P2-G.

**Files:**
- Create: `src/runtime/checkpointer.py`
- Create: `tests/test_checkpointer.py`
- Modify: `pyproject.toml` (add `langgraph-checkpoint-sqlite>=2.0`)

**Steps:**

- [ ] **1. Failing test** — create `tests/test_checkpointer.py`:

```python
"""Tests for runtime.checkpointer factory (P2-F, P2-G)."""
import pytest
import sqlite3
from pathlib import Path


def test_make_checkpointer_sqlite(tmp_path):
    from runtime.checkpointer import make_checkpointer
    from runtime.config import AppConfig, StorageConfig, MetadataConfig, LLMConfig, ProviderConfig, EmbeddingConfig
    from runtime.config import MCPConfig

    db_path = tmp_path / "ckpt.db"
    cfg = AppConfig(
        llm=LLMConfig(
            providers={"primary": ProviderConfig(kind="stub")},
            embedding=EmbeddingConfig(backend="stub"),
        ),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(url=f"sqlite:///{db_path}")),
    )
    saver, cleanup = make_checkpointer(cfg)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        assert isinstance(saver, SqliteSaver)
        # Verify WAL was set on the underlying connection.
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal", f"expected wal, got {mode}"
    finally:
        cleanup()


def test_make_checkpointer_uses_separate_pool(tmp_path):
    """The checkpointer connection must not be the same object as the
    metadata-store SQLAlchemy engine — they're separate pools."""
    from runtime.checkpointer import make_checkpointer
    from runtime.config import AppConfig, StorageConfig, MetadataConfig, LLMConfig, ProviderConfig, EmbeddingConfig, MCPConfig
    from runtime.storage.engine import build_engine

    db_path = tmp_path / "ckpt.db"
    cfg = AppConfig(
        llm=LLMConfig(
            providers={"primary": ProviderConfig(kind="stub")},
            embedding=EmbeddingConfig(backend="stub"),
        ),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(url=f"sqlite:///{db_path}")),
    )
    engine = build_engine(cfg.storage.metadata)
    saver, cleanup = make_checkpointer(cfg)
    try:
        # Exercise both: write a row through the engine, then a
        # checkpoint through the saver, in either order — neither
        # should deadlock with WAL.
        with engine.begin() as c:
            c.exec_driver_sql("CREATE TABLE IF NOT EXISTS probe(x INT)")
            c.exec_driver_sql("INSERT INTO probe VALUES (1)")
        # SqliteSaver creates its own tables on first use.
        from langgraph.checkpoint.base import empty_checkpoint
        cfg_dict = {"configurable": {"thread_id": "t1"}}
        saver.put(cfg_dict, empty_checkpoint(), {}, {})
    finally:
        cleanup()
        engine.dispose()


def test_make_checkpointer_unsupported_url_raises():
    from runtime.checkpointer import make_checkpointer
    from runtime.config import AppConfig, StorageConfig, MetadataConfig, LLMConfig, ProviderConfig, EmbeddingConfig, MCPConfig
    cfg = AppConfig(
        llm=LLMConfig(
            providers={"primary": ProviderConfig(kind="stub")},
            embedding=EmbeddingConfig(backend="stub"),
        ),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(url="mysql://x/y")),
    )
    with pytest.raises(ValueError, match="unsupported"):
        make_checkpointer(cfg)
```

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** — add to `pyproject.toml` `[project].dependencies`:

```
"langgraph-checkpoint-sqlite>=2.0",
```

Run `pip install -e .[dev]` then `npm-style audit` per `~/.claude/rules/security.md` (Python: `pip-audit`).

Create `src/runtime/checkpointer.py`:

```python
"""LangGraph checkpointer factory.

Reuses ``cfg.storage.metadata.url`` so checkpointer state and incident
metadata live in the same database. A *separate* connection pool is
created so the two never deadlock — for SQLite this means
``check_same_thread=False`` and ``journal_mode=WAL`` on the dedicated
connection. The Postgres branch uses ``psycopg.ConnectionPool`` for
the same reason.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Tuple
from urllib.parse import urlparse

from langgraph.checkpoint.base import BaseCheckpointSaver

from runtime.config import AppConfig


def _sqlite_path_from_url(url: str) -> str:
    """Extract the on-disk path from a sqlite SQLAlchemy URL.

    Accepts both ``sqlite:///abs/path.db`` and the relative
    ``sqlite:///./path.db`` forms.
    """
    parsed = urlparse(url)
    # SQLAlchemy uses sqlite:///<path>; urlparse puts the path in `.path`
    # with a leading slash, so we strip it carefully.
    path = parsed.path
    if path.startswith("/") and len(path) > 1 and path[1] != "/":
        # Keep the leading / for absolute paths; sqlite3 handles both.
        return path
    return path.lstrip("/")


def make_checkpointer(cfg: AppConfig) -> Tuple[BaseCheckpointSaver, Callable[[], None]]:
    """Build a checkpointer for the configured metadata DB.

    Returns:
        ``(saver, cleanup)``. The caller is responsible for invoking
        ``cleanup()`` at orchestrator shutdown.
    """
    url = cfg.storage.metadata.url
    if url.startswith("sqlite:"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = _sqlite_path_from_url(url)
        # Ensure parent dir exists (orchestrator may run before any
        # storage write that would have created it).
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Dedicated connection — separate from SQLAlchemy's pool.
        conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        saver = SqliteSaver(conn)
        saver.setup()
        return saver, conn.close

    if url.startswith("postgresql:") or url.startswith("postgres:"):
        # Filled in P2-G.
        from runtime.checkpointer_postgres import make_postgres_checkpointer
        return make_postgres_checkpointer(url)

    raise ValueError(
        f"unsupported checkpointer URL scheme {url!r} — expected sqlite or postgresql"
    )
```

- [ ] **4. Run tests** → SQLite tests pass; postgres test in P2-G.

- [ ] **5. Commit** — `feat(runtime): add SQLite checkpointer factory with WAL`

---

### P2-G — Add Postgres checkpointer (dev/prod parity)

Mirror P2-F for `PostgresSaver`. Connection pool via `psycopg.ConnectionPool`. Skip the integration test if no `LANGGRAPH_PG_TEST_URL` env var is set (so CI without Postgres still passes).

**Files:**
- Create: `src/runtime/checkpointer_postgres.py`
- Modify: `tests/test_checkpointer.py` (add gated postgres test)
- Modify: `pyproject.toml` (add `langgraph-checkpoint-postgres>=2.0`)

**Steps:**

- [ ] **1. Failing test** — append to `tests/test_checkpointer.py`:

```python
import os


@pytest.mark.skipif(
    not os.environ.get("LANGGRAPH_PG_TEST_URL"),
    reason="LANGGRAPH_PG_TEST_URL not set; postgres checkpointer not exercised in CI",
)
def test_make_checkpointer_postgres():
    from runtime.checkpointer import make_checkpointer
    from runtime.config import AppConfig, StorageConfig, MetadataConfig, LLMConfig, ProviderConfig, EmbeddingConfig, MCPConfig
    from langgraph.checkpoint.postgres import PostgresSaver

    cfg = AppConfig(
        llm=LLMConfig(
            providers={"primary": ProviderConfig(kind="stub")},
            embedding=EmbeddingConfig(backend="stub"),
        ),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(
            url=os.environ["LANGGRAPH_PG_TEST_URL"],
        )),
    )
    saver, cleanup = make_checkpointer(cfg)
    try:
        assert isinstance(saver, PostgresSaver)
    finally:
        cleanup()
```

- [ ] **2. Run** → skipped or fail.

- [ ] **3. Implementation** — add to `pyproject.toml`:

```
"langgraph-checkpoint-postgres>=2.0",
```

Create `src/runtime/checkpointer_postgres.py`:

```python
"""Postgres checkpointer wrapper.

Uses ``psycopg.ConnectionPool`` (separate from SQLAlchemy's pool) so
checkpointer writes don't contend with metadata writes on the same
connection. The pool's lifecycle is bound to the orchestrator's
``AsyncExitStack`` via the returned cleanup callable.
"""
from __future__ import annotations

from typing import Callable, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver


def make_postgres_checkpointer(url: str) -> Tuple[BaseCheckpointSaver, Callable[[], None]]:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    # Translate SQLAlchemy URL → libpq connection string. SQLAlchemy
    # accepts ``postgresql+psycopg://...`` while psycopg wants the
    # bare ``postgresql://...``. Strip the dialect suffix if present.
    if "+" in url.split("://", 1)[0]:
        scheme, rest = url.split("://", 1)
        url = f"postgresql://{rest}"

    pool = ConnectionPool(
        conninfo=url, max_size=4, kwargs={"autocommit": True},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver, pool.close
```

Add `psycopg-pool>=3.2` to `pyproject.toml` if not already a transitive dep of `psycopg[binary]`.

- [ ] **4. Run tests** — local SQLite test passes; postgres test skipped (or run if env var set).

- [ ] **5. Commit** — `feat(runtime): add Postgres checkpointer wrapper`

---

### P2-H — Migrate gate node to `interrupt()` + dual-write Session

The gate node must do two things:

1. Persist `pending_intervention` onto the Session row (back-compat for the existing Streamlit UI).
2. Raise `interrupt(payload)` so LangGraph itself pauses, the checkpointer captures the state, and `graph.get_state(thread_id)` returns the same payload to UI tools that adopt the LangGraph API.

The framework graph compiles with `checkpointer=...` and per-thread invocation always passes `config={"configurable": {"thread_id": session.id}}`.

**Files:**
- Modify: `src/runtime/graph.py` (gate node body)
- Modify: `tests/test_gate.py`

**Steps:**

- [ ] **1. Failing test** — append to `tests/test_gate.py`:

```python
@pytest.mark.asyncio
async def test_gate_dual_writes_pending_intervention_and_interrupts(tmp_path):
    """When confidence is low, gate must:
       1. Set Session.pending_intervention on disk.
       2. Raise interrupt() so the checkpointer pauses execution.
    """
    from langgraph.errors import GraphInterrupt
    # ... build a minimal cfg with a stub agent that emits low confidence ...
    cfg = _make_cfg(tmp_path, threshold=0.75)
    inc = _seed_session_with_low_confidence_run(cfg, confidence=0.42)

    saver, cleanup = make_checkpointer(cfg)
    try:
        graph = await build_graph(
            cfg=cfg, skills=skills, store=store,
            registry=registry, state_cls=IncidentState,
            checkpointer=saver,
        )
        with pytest.raises(GraphInterrupt) as exc_info:
            await graph.ainvoke(
                {"session": inc, "next_route": "resolution",
                 "last_agent": "deep_investigator"},
                config={"configurable": {"thread_id": inc.id}},
            )
    finally:
        cleanup()

    # Assertion 1: interrupt payload matches.
    interrupt_value = exc_info.value.args[0]
    assert interrupt_value["reason"] == "low_confidence"
    assert interrupt_value["confidence"] == approx(0.42)

    # Assertion 2: same payload is on the Session row on disk.
    reloaded = store.load(inc.id)
    assert reloaded.pending_intervention is not None
    assert reloaded.pending_intervention["reason"] == "low_confidence"
    assert reloaded.pending_intervention["confidence"] == approx(0.42)


@pytest.mark.asyncio
async def test_gate_clears_intervention_and_does_not_interrupt_on_pass(tmp_path):
    """High confidence → no interrupt, pending_intervention cleared if set."""
    cfg = _make_cfg(tmp_path, threshold=0.5)
    inc = _seed_session_with_high_confidence_run(cfg, confidence=0.9)
    inc.pending_intervention = {"stale": True}
    store.save(inc)

    saver, cleanup = make_checkpointer(cfg)
    try:
        graph = await build_graph(
            cfg=cfg, skills=skills, store=store,
            registry=registry, state_cls=IncidentState,
            checkpointer=saver,
        )
        # Should NOT raise; should route through.
        await graph.ainvoke(
            {"session": inc, "next_route": "resolution",
             "last_agent": "deep_investigator"},
            config={"configurable": {"thread_id": inc.id}},
        )
    finally:
        cleanup()

    reloaded = store.load(inc.id)
    assert reloaded.pending_intervention is None
```

- [ ] **2. Run** → fails (gate doesn't yet call `interrupt`).

- [ ] **3. Implementation** — in `src/runtime/graph.py`, locate the `gate` node body. Current shape (post-P1, with `incident` alias removed in P2-D):

```python
async def gate(state: GraphState) -> dict:
    session = state["session"]
    upstream = state.get("last_agent")
    intended_target = state.get("next_route")
    try:
        session = store.load(session.id)
    except FileNotFoundError:
        pass
    upstream_run = _latest_run_for(session, upstream)
    upstream_conf = upstream_run.confidence if upstream_run else None
    if upstream_conf is not None and upstream_conf < threshold:
        session.status = "awaiting_input"
        payload = {
            "reason": "low_confidence",
            "confidence": upstream_conf,
            "threshold": threshold,
            "upstream_agent": upstream,
            "summary": upstream_run.summary if upstream_run else "",
            "rationale": upstream_run.confidence_rationale if upstream_run else "",
            "options": ["accept", "retry", "escalate"],
            "escalation_teams": app_cfg.escalation_teams,
            "intended_target": intended_target,
        }
        session.pending_intervention = payload
        store.save(session)
        # NEW in P2-H:
        from langgraph.types import interrupt
        interrupt(payload)
    # Confidence passed → clear stale intervention if any.
    if session.pending_intervention is not None:
        session.pending_intervention = None
        store.save(session)
    return {"session": session}
```

Note that `interrupt()` raises a `GraphInterrupt`; the row save must happen *before* the `interrupt()` call so the back-compat write is persisted even on the unwind.

- [ ] **4. Run tests** → both gate tests pass.

- [ ] **5. Commit** — `feat(runtime): gate node raises interrupt() and dual-writes Session`

---

### P2-I — Replace `build_resume_graph` with `Command(resume=...)` path

Delete `build_resume_graph`. `Orchestrator.resume_session` (renamed from `resume_investigation` in any later phase but kept as-is here for surface stability) becomes:

```python
async def resume_session(self, session_id: str, decision: dict):
    config = {"configurable": {"thread_id": session_id}}
    return await self.graph.ainvoke(Command(resume=decision), config=config)
```

The `Orchestrator.__init__` no longer takes `resume_graph`. `Orchestrator.create()` no longer compiles a separate resume graph.

**Files:**
- Modify: `src/runtime/graph.py` (delete `build_resume_graph`)
- Modify: `src/runtime/orchestrator.py` (drop `resume_graph` param + the second `build_*` call)
- Modify: `tests/test_resume.py`

**Steps:**

- [ ] **1. Failing test** — modify `tests/test_resume.py`:

```python
@pytest.mark.asyncio
async def test_resume_uses_command_resume(tmp_path):
    """P2-I: resume invokes the same graph via Command(resume=...)."""
    from langgraph.types import Command
    cfg = _make_cfg(tmp_path)
    async with await Orchestrator.create(cfg) as orch:
        # Drive an investigation until gate triggers.
        sid = await orch.start_investigation(
            query="payments slow", environment="production",
        )
        # gate fired; session is awaiting_input.
        sess = orch.store.load(sid)
        assert sess.status == "awaiting_input"
        assert sess.pending_intervention is not None

        # Resume via the public API.
        result = await orch.resume_investigation(sid, {"decision": "retry"})

        # The session should have advanced past the gate.
        sess2 = orch.store.load(sid)
        assert sess2.pending_intervention is None
        assert sess2.status != "awaiting_input"


def test_orchestrator_has_no_resume_graph_attr():
    """P2-I: the bespoke resume graph is gone."""
    import runtime.orchestrator as mod
    src = open(mod.__file__).read()
    assert "build_resume_graph" not in src
    assert "self.resume_graph" not in src


def test_graph_module_has_no_build_resume_graph():
    import runtime.graph as mod
    assert not hasattr(mod, "build_resume_graph")
```

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** —

In `src/runtime/graph.py`: delete the entire `build_resume_graph` function. Update the docstring on `_make_router_factory` if it referenced both builders.

In `src/runtime/orchestrator.py`:

```python
# remove `from runtime.graph import build_resume_graph`
from runtime.graph import build_graph, GraphState
from langgraph.types import Command

class Orchestrator(Generic[StateT]):
    def __init__(self, cfg, store, history, skills, registry, graph,
                 exit_stack, state_cls, app_cfg=None):
        # no resume_graph
        ...

    @classmethod
    async def create(cls, cfg):
        # ... build saver via make_checkpointer ...
        saver, ckpt_cleanup = make_checkpointer(cfg)
        stack.callback(ckpt_cleanup)
        graph = await build_graph(
            cfg=cfg, skills=skills, store=store, registry=registry,
            state_cls=state_cls, checkpointer=saver,
        )
        # no resume_graph build
        return cls(cfg, store, history, skills, registry, graph,
                   stack, state_cls, app_cfg=app_cfg)

    async def resume_investigation(self, session_id: str, decision: dict):
        config = {"configurable": {"thread_id": session_id}}
        return await self.graph.ainvoke(Command(resume=decision), config=config)
```

Delete `_resume_with_input` if it exists as a private helper — `Command(resume=...)` does the work.

- [ ] **4. Run tests** → all `test_resume.py` cases pass.

- [ ] **5. Commit** — `feat(runtime): replace build_resume_graph with Command(resume=...)`

---

### P2-J — Delete `runtime/incident.py`, orchestrator MCP shim, `IncidentRepository`

Hard delete the three P1 carry-over files. Update every importer.

**Files:**
- Delete: `src/runtime/incident.py`
- Delete: `src/runtime/mcp_servers/incident.py`
- Delete: `src/runtime/storage/repository.py`
- Modify: `src/runtime/storage/__init__.py` (drop `IncidentRepository` export)
- Modify: `src/runtime/orchestrator.py` (already migrated in P2-C/I, verify no stale imports)
- Modify: `examples/incident_management/mcp_server.py` (use `SessionStore` + `HistoryStore`)
- Modify: any test importing the deleted modules (`tests/test_storage_repository.py` → rename to `test_storage_repo_compat.py` or delete)

**Steps:**

- [ ] **1. Failing test** — append to `tests/test_runtime_package.py`:

```python
def test_runtime_incident_module_gone():
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("runtime.incident")


def test_storage_repository_module_gone():
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("runtime.storage.repository")


def test_storage_init_does_not_export_incident_repository():
    import runtime.storage as s
    assert not hasattr(s, "IncidentRepository")
```

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** —

```bash
git rm src/runtime/incident.py
git rm src/runtime/storage/repository.py
git rm src/runtime/mcp_servers/incident.py
```

In `src/runtime/storage/__init__.py`, remove the `IncidentRepository` line and the `repository` import.

For `examples/incident_management/mcp_server.py`, the `set_state(repository=...)` setter today receives an `IncidentRepository`. Change to:

```python
def set_state(*, store: SessionStore, history: HistoryStore,
              app_cfg: IncidentAppConfig) -> None:
    global _store, _history, _app_cfg
    _store, _history, _app_cfg = store, history, app_cfg
```

And update `Orchestrator.create()` to pass `store=` and `history=` instead of the legacy `repository=`.

- [ ] **4. Run tests** — full suite. Migrate or delete `tests/test_storage_repository.py` (most assertions should already be covered by `tests/test_session_store.py` after P1).

- [ ] **5. Commit** — `chore(runtime): delete IncidentRepository / runtime.incident / mcp incident shim`

---

### P2-K — Replace `Paths.skills_dir` framework default with `None`; require apps to set

The current default `skills_dir = "examples/incident_management/skills"` (or `"config/skills"` per the alternate snapshot) bakes an example-app path into the framework. P2-K makes the framework demand that apps configure their own skills path.

**Files:**
- Modify: `src/runtime/config.py`
- Modify: `examples/incident_management/config.yaml`
- Modify: `src/runtime/orchestrator.py` (raise on `None`)
- Modify: any test that constructs `Paths(...)` without `skills_dir`

**Steps:**

- [ ] **1. Failing test** — extend `tests/test_runtime_config.py`:

```python
def test_paths_skills_dir_default_is_none():
    from runtime.config import Paths
    assert Paths().skills_dir is None


@pytest.mark.asyncio
async def test_orchestrator_create_requires_skills_dir(tmp_path):
    from runtime.orchestrator import Orchestrator
    from runtime.config import AppConfig, Paths, MCPConfig, LLMConfig, ProviderConfig, EmbeddingConfig
    cfg = AppConfig(
        llm=LLMConfig(providers={"primary": ProviderConfig(kind="stub")},
                      embedding=EmbeddingConfig(backend="stub")),
        mcp=MCPConfig(servers=[]),
        paths=Paths(),  # skills_dir=None
    )
    with pytest.raises(ValueError, match="paths.skills_dir"):
        await Orchestrator.create(cfg)
```

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** — in `src/runtime/config.py`:

```python
class Paths(BaseModel):
    skills_dir: str | None = None
    incidents_dir: str = "incidents"
```

In `src/runtime/orchestrator.py`'s `create()`, before `load_all_skills(...)`:

```python
if cfg.paths.skills_dir is None:
    raise ValueError(
        "paths.skills_dir must be set in app config — the framework "
        "no longer ships a default. Each application owns its skills "
        "directory."
    )
```

In `examples/incident_management/config.yaml`, set `paths.skills_dir: examples/incident_management/skills` explicitly. Likewise update tests in `tests/test_orchestrator.py`, `tests/test_resume.py`, `tests/test_migration_script.py` that today rely on a string default — they all already pass `Paths(skills_dir=..., ...)` per the codebase, so this should be a no-op for tests, but verify with grep.

- [ ] **4. Run tests** — full suite passes.

- [ ] **5. Commit** — `refactor(runtime): drop framework default for paths.skills_dir`

---

### P2-L — End-to-end test: cold restart with `pending_intervention` → resume works

The whole point of the checkpointer is that a process crash + cold restart can resume mid-graph. This test seeds a paused session, kills the orchestrator, instantiates a new one against the same DB, and resumes via `Command(resume=...)`.

**Files:**
- Create: `tests/test_resume_cold_restart.py`

**Steps:**

- [ ] **1. Failing test** — create `tests/test_resume_cold_restart.py`:

```python
"""P2-L: cold-restart resume — checkpointer state survives orchestrator teardown."""
import pytest
from langgraph.types import Command


@pytest.mark.asyncio
async def test_cold_restart_resume(tmp_path, monkeypatch):
    from runtime.orchestrator import Orchestrator
    from runtime.config import (
        AppConfig, Paths, MetadataConfig, StorageConfig,
        MCPConfig, LLMConfig, ProviderConfig, EmbeddingConfig,
        RuntimeConfig,
    )

    db_path = tmp_path / "incidents.db"
    skills_dir = "examples/incident_management/skills"
    cfg = AppConfig(
        llm=LLMConfig(
            providers={"primary": ProviderConfig(kind="stub")},
            embedding=EmbeddingConfig(backend="stub"),
        ),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(url=f"sqlite:///{db_path}")),
        paths=Paths(skills_dir=skills_dir, incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )

    # --- Process 1: drive until gate fires ---
    async with await Orchestrator.create(cfg) as orch1:
        sid = await orch1.start_investigation(
            query="payments slow", environment="production",
        )
        sess = orch1.store.load(sid)
        assert sess.status == "awaiting_input"
        assert sess.pending_intervention is not None
    # orch1 is fully torn down (FastMCP clients closed, checkpointer
    # connection closed). The DB on disk still has the checkpoint.

    # --- Process 2: cold-instantiate and resume ---
    async with await Orchestrator.create(cfg) as orch2:
        # The graph object is fresh; the checkpointer has rehydrated.
        result = await orch2.resume_investigation(sid, {"decision": "retry"})
        sess2 = orch2.store.load(sid)
        assert sess2.pending_intervention is None
        assert sess2.status != "awaiting_input"
```

- [ ] **2. Run** → should pass after P2-A through P2-K. If the test fails, the gap is almost certainly in `Orchestrator.create()` not threading the same `thread_id` (== `session.id`) through `start_investigation` and `resume_investigation`.

- [ ] **3. Implementation** — verify in `Orchestrator`:

```python
async def start_investigation(self, *, query, environment, ...):
    # ...
    sess = self.state_cls(... id=new_id ...)
    self.store.save(sess)
    config = {"configurable": {"thread_id": sess.id}}
    try:
        await self.graph.ainvoke({"session": sess}, config=config)
    except GraphInterrupt:
        pass  # gate fired; session row is already updated
    return sess.id
```

`resume_investigation` already uses the same `thread_id`. The test exercises end-to-end.

- [ ] **4. Run tests** → passes.

- [ ] **5. Commit** — `test(runtime): cold-restart resume e2e`

---

### P2-M — Bundle / dist updates for new modules

The dist bundle (`dist/app.py`) is regenerated from `src/runtime/`. The new modules (`checkpointer.py`, `checkpointer_postgres.py`, `state_resolver.py`) plus the new `langgraph-checkpoint-sqlite` / `langgraph-checkpoint-postgres` packages must be picked up by the bundler.

**Files:**
- Modify: `scripts/build_dist.py` (or whatever produces `dist/app.py`)
- Modify: `dist/app.py` (regenerate)
- Modify: `pyproject.toml` `[tool.hatch.build.targets.wheel]` packages list (already includes `src/runtime`)

**Steps:**

- [ ] **1. Locate the bundler** — `find . -name 'build_dist*' -o -name 'bundle*'` (the project memory file references a 7-file payload; the bundler should be self-evident).

- [ ] **2. Verify** that the bundler walks `src/runtime/` and includes `*.py` recursively. New modules pick up automatically; if it has an explicit allow-list, add `runtime.checkpointer`, `runtime.checkpointer_postgres`, `runtime.state_resolver`.

- [ ] **3. Verify** that `langgraph_checkpoint_sqlite` and `langgraph_checkpoint_postgres` are pip-installable in the target deployment env. Per `~/.claude/rules/build.md`, both must be vendored into the offline wheels cache.

- [ ] **4. Smoke test** — `python dist/app.py` (or whatever the entrypoint is) instantiates the orchestrator without import errors.

- [ ] **5. Commit** — `chore(dist): regenerate bundle for P2 modules and checkpointer deps`

---

### P2-N — Final verification, README updates if requested

Per `~/.claude/CLAUDE.md` §1, do not write `README.md` proactively. Only update existing docs if the user asks. This task is verification-only.

**Files:** None modified by default.

**Steps:**

- [ ] **1. Full suite** — `pytest -q tests/`; expect ≥ baseline test count + ~12 new tests added across this phase.

- [ ] **2. Lint** — `ruff check src/ tests/`; `pyright src/ tests/`.

- [ ] **3. Audit** — `pip-audit` per `~/.claude/rules/security.md`. Resolve any High/Critical CVEs introduced by the new langgraph-checkpoint-* packages.

- [ ] **4. Grep checks** — confirm cleanup is complete:
  - `grep -rn "from runtime.incident" src/ tests/ examples/` → empty.
  - `grep -rn "IncidentRepository" src/ tests/ examples/` → empty (or only in deleted-test references).
  - `grep -rn "build_resume_graph" src/ tests/` → empty.
  - `grep -rn 'state\["incident"\]' src/ tests/ examples/` → empty.
  - `grep -rn "examples.incident_management" src/runtime/` → empty.

- [ ] **5. Smoke run** — start the Streamlit example UI; trigger an investigation that hits the gate; restart the process; resume the session; assert the UI hand-off works.

- [ ] **6. Commit** — only if any straggler grep results were fixed: `chore(p2): final cleanup pass`.

## 5. Sequencing and Dependencies

```
                    P2-A  RuntimeConfig
                      │
                      ▼
                    P2-B  state_resolver
                      │
                      ▼
        ┌─────────── P2-C  Orchestrator/SessionStore/HistoryStore[StateT]
        │             │
        │             ├── P2-D  drop GraphState.incident alias
        │             │
        │             └── P2-E  scrub HistoryStore IncidentState import
        │
        ▼
    P2-F  SqliteSaver factory ──── P2-G  PostgresSaver factory
        │                              │
        └────────────┬─────────────────┘
                     ▼
                 P2-H  gate uses interrupt() + dual-write
                     │
                     ▼
                 P2-I  resume via Command(resume=...)
                     │
                     ▼
                 P2-J  delete incident.py / IncidentRepository / mcp shim
                     │
                     ▼
                 P2-K  Paths.skills_dir = None
                     │
                     ▼
                 P2-L  cold-restart e2e
                     │
                     ▼
                 P2-M  dist bundle update
                     │
                     ▼
                 P2-N  verification
```

Independent within a level: P2-D / P2-E can run after P2-C in parallel; P2-F and P2-G can be split across two subagents but P2-G's test is gated on env so most CI pressure is on P2-F.

## 6. Risks and Mitigations

**R1 — SQLite WAL with two pools (metadata SQLAlchemy engine + checkpointer sqlite3 connection) deadlocks on the same file.**
Mitigation: WAL is exactly the journal mode that allows concurrent readers + one writer per process. Both pools must use it. The checkpointer factory sets `PRAGMA journal_mode=WAL` on its dedicated connection. SQLAlchemy's `build_engine` already uses `connect_args={"check_same_thread": False}` (per `runtime/storage/engine.py` post-P1). If the engine doesn't enable WAL, add an `event.listen("connect", ...)` hook in `engine.py` setting `journal_mode=WAL` on first connect — but verify in P2-F's tests before assuming.

**R2 — `Generic[StateT]` is erased at runtime; type checkers see `StateT` but `isinstance` checks see `Session`.**
Mitigation: never `isinstance(x, StateT)` — use `self._state_cls` (the concrete class stashed in `__init__`) for runtime checks. The plan does exactly that in `_row_to_session`. Pyright surfaces the generic relationship for callers; runtime correctness is enforced by `state_resolver.resolve_state_class`'s `issubclass(cls, Session)` check.

**R3 — `interrupt()` semantics differ from the hand-rolled gate: the framework's old gate was idempotent (re-entered with the same state and decided again from scratch); LangGraph's `interrupt()` resumes inside the same node with `Command(resume=...)` returning the user's decision as the *value of the `interrupt()` call*.**
Mitigation: gate captures `decision = interrupt(payload)` so post-resume execution lands on the next line. The pre-resume save of `pending_intervention` is preserved; the post-resume continuation must clear it. Test this flow in `test_gate.py` AND in the cold-restart e2e (P2-L).

**R4 — Streamlit UI back-compat: the existing UI polls `Session.pending_intervention` from the store. If we *only* emit `interrupt()` and rely on `graph.get_state(thread_id)`, the UI breaks.**
Mitigation: the locked design dual-writes — gate persists `pending_intervention` to the row *and* raises `interrupt()`. P2-H tests both. UI continues to work unchanged; future UI refactors can adopt the LangGraph state API.

**R5 — Import cycle: `runtime.checkpointer` imports `runtime.config`, and `runtime.orchestrator` imports both. `runtime.config` must not import `runtime.checkpointer`.**
Mitigation: keep `RuntimeConfig` purely declarative (string field). The factory lives in a separate module that the orchestrator imports lazily inside `create()`. No risk if rules are followed.

**R6 — Air-gapped install of `langgraph-checkpoint-sqlite` / `-postgres`.**
Mitigation: per `~/.claude/rules/build.md`, both packages must be vendored into the offline wheels cache during P2-M. The test for P2-G is skipped without an explicit `LANGGRAPH_PG_TEST_URL` so CI passes without Postgres.

**R7 — `Command(resume=...)` requires the same `checkpointer` instance LangGraph used to pause; passing a fresh saver in process 2 (P2-L) reads from the same on-disk DB but is a *different* `SqliteSaver` object.**
Mitigation: this is exactly what LangGraph's persistence model supports — checkpointers are stateless wrappers around the DB. P2-L's test asserts the cold-restart path. If it fails, the bug is a transient `setup()` race, not the API model.

**R8 — Resolved `state_class` mismatch with rows already saved under the old class.**
Mitigation: P2 doesn't change row schemas. `_row_to_session` calls `state_cls.model_validate(row, from_attributes=True)`; any `IncidentState` row hydrates cleanly into `IncidentState` (which is what the example app's config resolves to). Apps that reconfigure to a different class take responsibility for migration — out of scope for P2.

## 7. Done Criteria

- [ ] All tasks P2-A through P2-N committed.
- [ ] Test count: P1 baseline + ≥12 new tests (P2-A: 4, P2-B: 5, P2-C: 7, P2-D: 1, P2-E: 1, P2-F: 3, P2-G: 1, P2-H: 2, P2-I: 3, P2-J: 3, P2-K: 2, P2-L: 1).
- [ ] `pytest -q tests/` passes; postgres test skipped without env var.
- [ ] `ruff check src/ tests/` clean; `pyright src/ tests/` clean.
- [ ] `pip-audit` no High/Critical.
- [ ] Grep checks (P2-N step 4) all empty.
- [ ] Smoke run: cold-restart resume works in the Streamlit UI.
- [ ] `dist/app.py` regenerated; offline build still produces the artifact.

## 8. Open Questions

- **Multi-session lifecycle.** Phase 3's locked answer (per the prior conversation) is task-per-session asyncio loops. P2 doesn't need that — `Orchestrator.create()` builds one graph with one checkpointer for the lifetime of the process; concurrent sessions multiplex on `thread_id` only. Defer the lifecycle work to Phase 3.
- **Custom checkpointer back-ends.** Some teams may want Redis or DynamoDB checkpointers. The factory already raises on unsupported URL schemes; adding a registry hook is a Phase 4 concern, not P2.
- **State migration when `runtime.state_class` changes mid-deployment.** Out of scope; treat the dotted path as immutable for the lifetime of an incident database. Document this in the example app config comment.
- **`SessionStore` API split.** Some methods (e.g. `_next_id`) are hard-coded to the `INC-YYYYMMDD-NNN` format — fine for the incident app but framework-inappropriate. Phase 3 should let apps inject an `IdGenerator` strategy. Tracked, not done in P2.
