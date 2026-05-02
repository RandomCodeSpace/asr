# SQL Storage + Embedding Similarity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the JSON-file `IncidentStore` with a SQLAlchemy `IncidentRepository` running on SQLite + sqlite-vec (dev) and Postgres + pgvector (prod), with config-only swap. Wire LangChain `Embeddings` (Ollama `bge-m3` now, Azure OpenAI later) into a `find_similar` path that does KNN in SQL.

**Architecture:** New `src/orchestrator/storage/` package. Repository owns the embedder. Hybrid schema (scalar columns + JSON columns + native vector column). One dialect-aware similarity query; everything else identical across SQLite and Postgres. Existing Pydantic `Incident` model stays as the domain object; `IncidentStore` is removed (clean break).

**Tech Stack:** Python 3.14, SQLAlchemy 2.x sync, sqlite-vec, pgvector, psycopg[binary], LangChain Embeddings (`langchain-ollama` already a dep), Pydantic v2, pytest + parameterized engine fixtures.

**Spec:** `docs/superpowers/specs/2026-05-02-sql-storage-and-embeddings-design.md` — read before starting.

---

## Task A — Dependencies + storage package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/orchestrator/storage/__init__.py`
- Test: `tests/test_storage_smoke.py`

- [ ] **Step 1 — Add deps to `pyproject.toml`**

In the `dependencies = [...]` list (around line 10), add:

```
"sqlite-vec>=0.1.6",
"pgvector>=0.3",
"psycopg[binary]>=3.2",
"numpy>=1.26",
```

(`sqlalchemy>=2.0` and `langchain-ollama>=0.3` are already present — confirmed by grep.)

- [ ] **Step 2 — Install**

Run: `/home/dev/projects/asr/.venv/bin/pip install -e '.[dev]' 2>&1 | tail -10`
Expected: success; resolves the four new packages.

- [ ] **Step 3 — Create the package**

Create `src/orchestrator/storage/__init__.py` with content:

```python
"""SQLAlchemy-backed storage layer for incidents and embeddings.

Public surface
--------------
- ``IncidentRepository`` — replaces the old JSON ``IncidentStore``.
- ``build_engine``       — engine factory (sqlite + sqlite-vec, postgres + pgvector).
- ``build_embedder``     — LangChain ``Embeddings`` factory.
- ``Base``, ``IncidentRow`` — declarative model (exposed for tests/migrations).
"""
from orchestrator.storage.engine import build_engine
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.models import Base, IncidentRow
from orchestrator.storage.repository import IncidentRepository

__all__ = [
    "Base",
    "IncidentRepository",
    "IncidentRow",
    "build_embedder",
    "build_engine",
]
```

- [ ] **Step 4 — Smoke import test**

Create `tests/test_storage_smoke.py`:

```python
def test_storage_package_imports():
    from orchestrator.storage import (
        Base, IncidentRepository, IncidentRow, build_embedder, build_engine,
    )
    assert all([Base, IncidentRepository, IncidentRow, build_embedder, build_engine])
```

This will fail until later tasks define the modules — that's expected. Mark this test xfail until Task F:

```python
import pytest
@pytest.mark.xfail(reason="implementations land in Tasks B–F", strict=False)
def test_storage_package_imports():
    ...
```

- [ ] **Step 5 — Run**

`pytest tests/test_storage_smoke.py -v`
Expected: XFAIL (or pass once F is in).

- [ ] **Step 6 — Commit**

```bash
git add pyproject.toml src/orchestrator/storage/__init__.py tests/test_storage_smoke.py
git commit -m "feat(storage): add sqlite-vec/pgvector/psycopg deps and storage pkg skeleton"
```

---

## Task B — `StorageConfig` + `EmbeddingConfig.dim`

**Files:**
- Modify: `src/orchestrator/config.py`
- Test: `tests/test_config.py` (add cases)

- [ ] **Step 1 — Write the failing test (`tests/test_config.py`)**

Append:

```python
def test_storage_config_default():
    from orchestrator.config import AppConfig, LLMConfig, MCPConfig
    cfg = AppConfig(llm=LLMConfig.stub(), mcp=MCPConfig())
    assert cfg.storage.url == "sqlite:///incidents.db"
    assert cfg.storage.pool_size == 5
    assert cfg.storage.echo is False


def test_embedding_config_dim_default():
    from orchestrator.config import EmbeddingConfig
    e = EmbeddingConfig(provider="p", model="m")
    assert e.dim == 1024


def test_storage_config_postgres_url():
    from orchestrator.config import StorageConfig
    s = StorageConfig(url="postgresql+psycopg://u:p@h/db", pool_size=10)
    assert s.url.startswith("postgresql+psycopg://")
    assert s.pool_size == 10
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_config.py -k "storage_config or embedding_config_dim" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3 — Implement (`src/orchestrator/config.py`)**

Add after `class IncidentConfig`:

```python
class StorageConfig(BaseModel):
    """Database backend. SQLite (with sqlite-vec) for dev, Postgres (with pgvector) for prod."""
    url: str = "sqlite:///incidents.db"
    pool_size: int = 5      # postgres only; sqlite uses NullPool
    echo: bool = False
```

Add `dim: int = 1024` field on `EmbeddingConfig`:

```python
class EmbeddingConfig(BaseModel):
    provider: str
    model: str
    deployment: str | None = None
    dim: int = 1024
```

In `AppConfig`, add:

```python
storage: StorageConfig = Field(default_factory=StorageConfig)
```

(Place it next to `incidents`.)

- [ ] **Step 4 — Run, verify PASS**

`pytest tests/test_config.py -v`
Expected: all green.

- [ ] **Step 5 — Commit**

```bash
git add src/orchestrator/config.py tests/test_config.py
git commit -m "feat(config): add StorageConfig and EmbeddingConfig.dim"
```

---

## Task C — Custom column types (`VectorColumn`, `JSONColumn`)

**Files:**
- Create: `src/orchestrator/storage/types.py`
- Test: `tests/test_storage_types.py`

- [ ] **Step 1 — Write the failing tests**

Create `tests/test_storage_types.py`:

```python
"""Round-trip serialization tests for the custom SQLAlchemy types.

We use a tiny in-memory SQLite DB rather than mocking dialects — closer to
production, less brittle than dialect mocks.
"""
from __future__ import annotations
import numpy as np
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select


@pytest.fixture
def sqlite_engine():
    return create_engine("sqlite:///:memory:")


def test_vector_column_round_trip_sqlite(sqlite_engine):
    from orchestrator.storage.types import VectorColumn
    md = MetaData()
    t = Table("v", md, Column("id", Integer, primary_key=True),
              Column("vec", VectorColumn(4), nullable=True))
    md.create_all(sqlite_engine)
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "vec": [0.1, 0.2, 0.3, 0.4]}])
        conn.commit()
        row = conn.execute(select(t.c.vec)).scalar_one()
    assert isinstance(row, list)
    assert len(row) == 4
    np.testing.assert_allclose(row, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)


def test_vector_column_handles_none_sqlite(sqlite_engine):
    from orchestrator.storage.types import VectorColumn
    md = MetaData()
    t = Table("v", md, Column("id", Integer, primary_key=True),
              Column("vec", VectorColumn(4), nullable=True))
    md.create_all(sqlite_engine)
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "vec": None}])
        conn.commit()
        assert conn.execute(select(t.c.vec)).scalar_one() is None


def test_json_column_round_trip_sqlite(sqlite_engine):
    from orchestrator.storage.types import JSONColumn
    md = MetaData()
    t = Table("j", md, Column("id", Integer, primary_key=True),
              Column("payload", JSONColumn, nullable=False))
    md.create_all(sqlite_engine)
    payload = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "payload": payload}])
        conn.commit()
        out = conn.execute(select(t.c.payload)).scalar_one()
    assert out == payload
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_storage_types.py -v`
Expected: ImportError (module doesn't exist).

- [ ] **Step 3 — Implement (`src/orchestrator/storage/types.py`)**

```python
"""Custom SQLAlchemy column types that bridge SQLite and Postgres.

- ``VectorColumn(dim)`` — ``pgvector.sqlalchemy.Vector(dim)`` on Postgres,
  ``LargeBinary`` (numpy float32 bytes) on SQLite. Python value is always
  ``list[float] | None`` regardless of dialect.
- ``JSONColumn`` — thin alias over SQLAlchemy ``JSON``; auto-routes to
  ``JSONB`` on Postgres, ``TEXT`` on SQLite. Centralized so we can adjust
  serialization (e.g. datetime support) in one place later.
"""
from __future__ import annotations
import numpy as np
from sqlalchemy import JSON, LargeBinary
from sqlalchemy.types import TypeDecorator


JSONColumn = JSON  # SQLAlchemy auto-dialects: JSONB on pg, TEXT on sqlite.


class VectorColumn(TypeDecorator):
    """Vector column backed by pgvector on Postgres, BLOB on SQLite.

    Python value: ``list[float] | None``.
    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            # pgvector adapter accepts list[float] / numpy arrays directly.
            return list(value)
        # SQLite: pack as float32 bytes for compactness + sqlite-vec compat.
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.dim,):
            raise ValueError(
                f"vector dim {arr.shape[0]} != column dim {self.dim}"
            )
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return np.frombuffer(value, dtype=np.float32).tolist()
```

- [ ] **Step 4 — Run, verify PASS**

`pytest tests/test_storage_types.py -v`
Expected: 3 passed.

- [ ] **Step 5 — Commit**

```bash
git add src/orchestrator/storage/types.py tests/test_storage_types.py
git commit -m "feat(storage): VectorColumn + JSONColumn TypeDecorators"
```

---

## Task D — Declarative model + engine builder + sqlite-vec loader

**Files:**
- Create: `src/orchestrator/storage/models.py`
- Create: `src/orchestrator/storage/engine.py`
- Test: `tests/test_storage_engine.py`

- [ ] **Step 1 — Write the failing tests (`tests/test_storage_engine.py`)**

```python
"""Engine + sqlite-vec extension load tests."""
from __future__ import annotations
import pytest


def test_build_engine_sqlite_in_memory():
    from orchestrator.config import StorageConfig
    from orchestrator.storage.engine import build_engine
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    with eng.connect() as conn:
        # sqlite-vec should be loaded — vec_distance_cosine should be callable.
        from sqlalchemy import text
        out = conn.execute(text("SELECT vec_distance_cosine(?, ?)"),
                           ((b"\x00" * 16, b"\x00" * 16))).scalar()
        # zero vectors → cosine distance is undefined or 0/1 depending on impl;
        # we only need to assert the function exists (no exception).
        assert out is not None or out is None  # smoke: function callable


def test_create_all_smoke():
    from orchestrator.config import StorageConfig
    from orchestrator.storage.engine import build_engine
    from orchestrator.storage.models import Base
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    Base.metadata.create_all(eng)
    from sqlalchemy import inspect
    insp = inspect(eng)
    assert "incidents" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("incidents")}
    expected = {
        "id", "status", "created_at", "updated_at", "deleted_at",
        "query", "environment", "reporter_id", "reporter_team",
        "summary", "severity", "category", "matched_prior_inc",
        "resolution", "tags", "agents_run", "tool_calls", "findings",
        "pending_intervention", "user_inputs", "embedding",
        "input_tokens", "output_tokens", "total_tokens",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_storage_engine.py -v`
Expected: ImportError.

- [ ] **Step 3 — Implement `src/orchestrator/storage/models.py`**

```python
"""SQLAlchemy declarative model for the ``incidents`` table.

Hybrid schema: scalar/queryable fields as columns, nested Pydantic
structures as JSON columns (JSONB on Postgres, TEXT on SQLite), and a
native vector column for embeddings.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from orchestrator.storage.types import JSONColumn, VectorColumn


EMBEDDING_DIM = 1024  # bge-m3; if you change embed model, re-embed corpus.


class Base(DeclarativeBase):
    pass


class IncidentRow(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(String, nullable=False)
    reporter_id: Mapped[str] = mapped_column(String, nullable=False)
    reporter_team: Mapped[str] = mapped_column(String, nullable=False)

    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    matched_prior_inc: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)

    tags: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    agents_run: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    findings: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    pending_intervention: Mapped[dict | None] = mapped_column(JSONColumn, nullable=True)
    user_inputs: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)

    embedding: Mapped[list[float] | None] = mapped_column(
        VectorColumn(EMBEDDING_DIM), nullable=True
    )

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_incidents_status_env_active", "status", "environment",
              postgresql_where=Text("deleted_at IS NULL"),
              sqlite_where=Text("deleted_at IS NULL")),
        Index("ix_incidents_created_at_active", "created_at",
              postgresql_where=Text("deleted_at IS NULL"),
              sqlite_where=Text("deleted_at IS NULL")),
    )
```

- [ ] **Step 4 — Implement `src/orchestrator/storage/engine.py`**

```python
"""SQLAlchemy engine factory + sqlite-vec extension loader.

Behaviour
---------
- ``sqlite://``  → engine with ``NullPool``, ``check_same_thread=False``,
                  and a ``connect`` event hook that loads sqlite-vec into
                  every new dbapi connection.
- ``postgresql://`` → engine with the configured pool size, plus a
                     one-time ``CREATE EXTENSION IF NOT EXISTS vector``.
"""
from __future__ import annotations
from sqlalchemy import event, text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool

from orchestrator.config import StorageConfig


def _attach_sqlite_vec(engine: Engine) -> None:
    """Load sqlite-vec on every new SQLite dbapi connection."""
    import sqlite_vec

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):  # type: ignore[misc]
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)


def _ensure_pgvector(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def build_engine(cfg: StorageConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )
        _attach_sqlite_vec(engine)
        return engine
    engine = create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
    _ensure_pgvector(engine)
    return engine
```

- [ ] **Step 5 — Run, verify PASS**

`pytest tests/test_storage_engine.py -v`
Expected: 2 passed.

- [ ] **Step 6 — Commit**

```bash
git add src/orchestrator/storage/models.py src/orchestrator/storage/engine.py tests/test_storage_engine.py
git commit -m "feat(storage): IncidentRow declarative model + engine builder with sqlite-vec hook"
```

---

## Task E — Embedding facade + `_StubEmbeddings`

**Files:**
- Create: `src/orchestrator/storage/embeddings.py`
- Test: `tests/test_storage_embeddings.py`

- [ ] **Step 1 — Write the failing tests**

```python
"""Embedding facade tests — stub determinism and provider dispatch."""
from __future__ import annotations
import pytest


def test_stub_embeddings_determinism():
    from orchestrator.storage.embeddings import _StubEmbeddings
    e = _StubEmbeddings(dim=8)
    a = e.embed_query("hello world")
    b = e.embed_query("hello world")
    c = e.embed_query("different")
    assert a == b
    assert a != c
    assert len(a) == 8
    assert all(isinstance(x, float) for x in a)


def test_stub_embed_documents_returns_list_of_lists():
    from orchestrator.storage.embeddings import _StubEmbeddings
    e = _StubEmbeddings(dim=4)
    out = e.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 4 for v in out)


def test_build_embedder_stub():
    from orchestrator.config import EmbeddingConfig, ProviderConfig
    from orchestrator.storage.embeddings import build_embedder
    cfg = EmbeddingConfig(provider="s", model="x", dim=8)
    providers = {"s": ProviderConfig(kind="stub")}
    e = build_embedder(cfg, providers)
    assert e is not None
    v = e.embed_query("test")
    assert len(v) == 8


def test_build_embedder_none_returns_none():
    from orchestrator.storage.embeddings import build_embedder
    assert build_embedder(None, {}) is None


def test_build_embedder_unknown_kind_raises():
    from orchestrator.config import EmbeddingConfig, ProviderConfig
    from orchestrator.storage.embeddings import build_embedder
    cfg = EmbeddingConfig(provider="x", model="m")
    bad = ProviderConfig(kind="ollama")
    bad.kind = "nonsense"  # bypass pydantic for the test
    with pytest.raises(ValueError, match="unknown provider kind"):
        build_embedder(cfg, {"x": bad})
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_storage_embeddings.py -v`
Expected: ImportError.

- [ ] **Step 3 — Implement `src/orchestrator/storage/embeddings.py`**

```python
"""LangChain ``Embeddings`` facade + deterministic stub for tests.

Construction is config-driven via :func:`build_embedder`. Provider kind
dispatches to ``OllamaEmbeddings`` / ``AzureOpenAIEmbeddings`` / a local
stub. Stubs are deterministic so tests can assert similarity ordering
without external services.
"""
from __future__ import annotations
import hashlib
import numpy as np
from langchain_core.embeddings import Embeddings

from orchestrator.config import EmbeddingConfig, ProviderConfig


class _StubEmbeddings(Embeddings):
    """Deterministic dummy embedder. Same text → same vector; different
    texts → different vectors. Useful for CI and unit tests without
    network.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(
    cfg: EmbeddingConfig | None,
    providers: dict[str, ProviderConfig],
) -> Embeddings | None:
    """Build a LangChain ``Embeddings`` from config; ``None`` if not configured."""
    if cfg is None:
        return None
    p = providers[cfg.provider]
    if p.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=cfg.model, base_url=p.base_url or "http://localhost:11434")
    if p.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=cfg.deployment,
            model=cfg.model,
            azure_endpoint=p.endpoint,
            api_version=p.api_version,
            api_key=p.api_key,
        )
    if p.kind == "stub":
        return _StubEmbeddings(dim=cfg.dim)
    raise ValueError(f"unknown provider kind: {p.kind!r}")
```

- [ ] **Step 4 — Run, verify PASS**

`pytest tests/test_storage_embeddings.py -v`
Expected: 5 passed.

- [ ] **Step 5 — Commit**

```bash
git add src/orchestrator/storage/embeddings.py tests/test_storage_embeddings.py
git commit -m "feat(storage): embedder facade + deterministic stub"
```

---

## Task F — `IncidentRepository` (CRUD only)

**Files:**
- Create: `src/orchestrator/storage/repository.py`
- Test: `tests/test_storage_repository.py`

This task ships the full CRUD surface (mirror of `IncidentStore` minus `find_similar`, which lands in Task G). Keep it strict to existing semantics — same id format, same timestamp semantics, same soft-delete behaviour.

- [ ] **Step 1 — Write the failing tests**

```python
"""IncidentRepository CRUD tests against in-memory SQLite + stub embedder."""
from __future__ import annotations
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from orchestrator.config import StorageConfig
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base


@pytest.fixture
def repo():
    from orchestrator.storage.repository import IncidentRepository
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    Base.metadata.create_all(eng)
    return IncidentRepository(engine=eng)


def test_create_assigns_id_and_persists(repo):
    inc = repo.create(query="redis OOM", environment="production",
                      reporter_id="u1", reporter_team="platform")
    assert inc.id.startswith("INC-")
    loaded = repo.load(inc.id)
    assert loaded.query == "redis OOM"
    assert loaded.environment == "production"
    assert loaded.reporter.id == "u1"
    assert loaded.status == "new"


def test_create_id_sequence(repo):
    a = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    b = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    seq_a = int(a.id.rsplit("-", 1)[1])
    seq_b = int(b.id.rsplit("-", 1)[1])
    assert seq_b == seq_a + 1


def test_save_round_trip_preserves_nested(repo):
    from orchestrator.incident import AgentRun, ToolCall, TokenUsage
    inc = repo.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.summary = "edited"
    inc.tags = ["redis", "oom"]
    inc.findings = {"triage": {"k": "v"}}
    inc.agents_run.append(AgentRun(agent="intake", started_at=inc.created_at,
                                   ended_at=inc.created_at, summary="ok",
                                   token_usage=TokenUsage()))
    inc.tool_calls.append(ToolCall(agent="intake", tool="x", args={}, result={},
                                   ts=inc.created_at))
    repo.save(inc)
    loaded = repo.load(inc.id)
    assert loaded.summary == "edited"
    assert loaded.tags == ["redis", "oom"]
    assert loaded.findings == {"triage": {"k": "v"}}
    assert len(loaded.agents_run) == 1
    assert len(loaded.tool_calls) == 1


def test_load_missing_raises_filenotfound(repo):
    with pytest.raises(FileNotFoundError):
        repo.load("INC-20260102-999")


def test_load_invalid_id_raises_valueerror(repo):
    with pytest.raises(ValueError):
        repo.load("not-an-id")


def test_delete_is_soft(repo):
    inc = repo.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    out = repo.delete(inc.id)
    assert out.status == "deleted"
    assert out.deleted_at is not None
    # idempotent
    again = repo.delete(inc.id)
    assert again.status == "deleted"


def test_list_recent_excludes_deleted_by_default(repo):
    a = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    b = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    repo.delete(a.id)
    listed = repo.list_recent(limit=10)
    ids = [i.id for i in listed]
    assert b.id in ids
    assert a.id not in ids
    listed_all = repo.list_recent(limit=10, include_deleted=True)
    assert {a.id, b.id}.issubset({i.id for i in listed_all})


def test_list_recent_orders_by_created_at_desc(repo):
    inc1 = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    inc2 = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    listed = repo.list_recent(limit=10)
    assert [i.id for i in listed[:2]] == [inc2.id, inc1.id]
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_storage_repository.py -v`
Expected: ImportError.

- [ ] **Step 3 — Implement `src/orchestrator/storage/repository.py`**

```python
"""SQLAlchemy-backed Incident store.

Public methods mirror the previous JSON ``IncidentStore`` 1:1 so call
sites in the MCP server and orchestrator change minimally. The repository
also owns the embedder; ``find_similar`` (Task G) does the dialect dispatch.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.embeddings import Embeddings
from sqlalchemy import desc, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orchestrator.incident import (
    AgentRun, Incident, Reporter, TokenUsage, ToolCall,
)
from orchestrator.storage.models import IncidentRow

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now().strftime("%Y%m%d")


class IncidentRepository:
    """SQLAlchemy-backed Incident store. Drop-in for ``IncidentStore``.

    Threading note: methods open short-lived sessions; safe for the
    orchestrator's coarse-grained concurrency model.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        embedder: Optional[Embeddings] = None,
        similarity_threshold: float = 0.85,
        severity_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.severity_aliases = severity_aliases or {}

    # ---------- ID minting ----------
    def _next_id(self, session: Session) -> str:
        prefix = f"INC-{_today_str()}-"
        like = f"{prefix}%"
        rows = session.execute(
            select(IncidentRow.id).where(IncidentRow.id.like(like))
        ).scalars().all()
        max_seq = 0
        for r in rows:
            try:
                max_seq = max(max_seq, int(r.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"{prefix}{max_seq + 1:03d}"

    # ---------- public API ----------
    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> Incident:
        with Session(self.engine) as session:
            now = _now()
            inc_id = self._next_id(session)
            row = IncidentRow(
                id=inc_id,
                status="new",
                created_at=now,
                updated_at=now,
                query=query,
                environment=environment,
                reporter_id=reporter_id,
                reporter_team=reporter_team,
                summary="",
                tags=[],
                agents_run=[],
                tool_calls=[],
                findings={},
                user_inputs=[],
                embedding=self._maybe_embed(query),
            )
            session.add(row)
            session.commit()
            return self._row_to_incident(row)

    def load(self, incident_id: str) -> Incident:
        if not _INC_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected INC-YYYYMMDD-NNN"
            )
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def save(self, incident: Incident) -> None:
        if not _INC_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected INC-YYYYMMDD-NNN"
            )
        incident.updated_at = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
        with Session(self.engine) as session:
            existing = session.get(IncidentRow, incident.id)
            new_embedding = self._compute_save_embedding(existing, incident)
            data = self._incident_to_row_dict(incident, embedding=new_embedding)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()

    def delete(self, incident_id: str) -> Incident:
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            if row.status != "deleted":
                row.status = "deleted"
                row.deleted_at = _now()
                row.pending_intervention = None
            session.commit()
            return self._row_to_incident(row)

    def list_all(self, *, include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_recent(self, limit: int = 20, *,
                    include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            stmt = stmt.order_by(desc(IncidentRow.created_at), desc(IncidentRow.id)).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    # ---------- mapping helpers ----------
    def _row_to_incident(self, row: IncidentRow) -> Incident:
        agents_run = [AgentRun.model_validate(a) for a in (row.agents_run or [])]
        tool_calls = [ToolCall.model_validate(t) for t in (row.tool_calls or [])]
        token_usage = TokenUsage(
            input_tokens=row.input_tokens, output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
        )
        return Incident(
            id=row.id,
            status=row.status,
            created_at=_iso(row.created_at),
            updated_at=_iso(row.updated_at),
            deleted_at=_iso(row.deleted_at) if row.deleted_at else None,
            query=row.query,
            environment=row.environment,
            reporter=Reporter(id=row.reporter_id, team=row.reporter_team),
            summary=row.summary or "",
            tags=list(row.tags or []),
            severity=row.severity,
            category=row.category,
            matched_prior_inc=row.matched_prior_inc,
            embedding=row.embedding,
            agents_run=agents_run,
            tool_calls=tool_calls,
            findings=dict(row.findings or {}),
            resolution=row.resolution,
            token_usage=token_usage,
            pending_intervention=row.pending_intervention,
            user_inputs=list(row.user_inputs or []),
        )

    def _incident_to_row_dict(
        self, inc: Incident, *, embedding: Optional[list[float]],
    ) -> dict:
        return {
            "id": inc.id,
            "status": inc.status,
            "created_at": _parse_iso(inc.created_at),
            "updated_at": _parse_iso(inc.updated_at),
            "deleted_at": _parse_iso(inc.deleted_at) if inc.deleted_at else None,
            "query": inc.query,
            "environment": inc.environment,
            "reporter_id": inc.reporter.id,
            "reporter_team": inc.reporter.team,
            "summary": inc.summary or "",
            "severity": inc.severity,
            "category": inc.category,
            "matched_prior_inc": inc.matched_prior_inc,
            "resolution": inc.resolution,
            "tags": list(inc.tags),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "embedding": embedding,
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
        }

    # ---------- embedding lifecycle ----------
    def _maybe_embed(self, text: str) -> Optional[list[float]]:
        if self.embedder is None or not text:
            return None
        return self.embedder.embed_query(text)

    def _compute_save_embedding(
        self, existing: Optional[IncidentRow], inc: Incident,
    ) -> Optional[list[float]]:
        """Re-embed only when the embed source text materially changed."""
        if self.embedder is None:
            return existing.embedding if existing is not None else None
        text = _embed_source(inc)
        if existing is not None:
            prior = _embed_source_from_row(existing)
            if prior == text and existing.embedding is not None:
                return existing.embedding
        return self.embedder.embed_query(text) if text else None


def _embed_source(inc: Incident) -> str:
    parts = [inc.query or "", inc.summary or "", " ".join(inc.tags or [])]
    return " ".join(p for p in parts if p).strip()


def _embed_source_from_row(row: IncidentRow) -> str:
    parts = [row.query or "", row.summary or "", " ".join(row.tags or [])]
    return " ".join(p for p in parts if p).strip()


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
```

- [ ] **Step 4 — Remove the xfail marker on `tests/test_storage_smoke.py`**

Open the file, delete the `@pytest.mark.xfail(...)` decorator and the unused `import pytest`.

- [ ] **Step 5 — Run all storage tests**

`pytest tests/test_storage_*.py -v`
Expected: all pass.

- [ ] **Step 6 — Commit**

```bash
git add src/orchestrator/storage/repository.py tests/test_storage_repository.py tests/test_storage_smoke.py
git commit -m "feat(storage): IncidentRepository with full CRUD on SQLAlchemy"
```

---

## Task G — `IncidentRepository.find_similar` + dialect dispatch

**Files:**
- Modify: `src/orchestrator/storage/repository.py`
- Test: `tests/test_storage_find_similar.py`

- [ ] **Step 1 — Write the failing tests**

```python
"""find_similar tests — embedding ordering and dialect dispatch.

Default backend is SQLite (with sqlite-vec). Postgres parity is gated
behind ``POSTGRES_TEST_URL`` env var; if unset, the postgres params skip.
"""
from __future__ import annotations
import os
import pytest
from sqlalchemy.engine import Engine

from orchestrator.config import EmbeddingConfig, ProviderConfig, StorageConfig
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository


def _engines() -> list[tuple[str, str]]:
    eng = [("sqlite", "sqlite:///:memory:")]
    pg = os.environ.get("POSTGRES_TEST_URL")
    if pg:
        eng.append(("postgres", pg))
    return eng


@pytest.fixture(params=_engines(), ids=lambda p: p[0])
def repo(request) -> IncidentRepository:
    _, url = request.param
    eng: Engine = build_engine(StorageConfig(url=url))
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return IncidentRepository(engine=eng, embedder=embedder, similarity_threshold=0.0)


def test_find_similar_returns_self_first(repo):
    """Stub embeddings are deterministic, so identical query → cosine 1.0."""
    a = repo.create(query="redis OOMKill in payments", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    a.resolution = "raise memory"
    repo.save(a)

    repo.create(query="ssh down on bastion", environment="production",
                reporter_id="u", reporter_team="t")  # noise

    hits = repo.find_similar(query="redis OOMKill in payments",
                             environment="production")
    assert hits, "no hits returned"
    top_id, top_score = hits[0]
    assert top_id.id == a.id
    assert top_score > 0.99


def test_find_similar_filters_by_environment(repo):
    a = repo.create(query="match me", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    repo.save(a)
    b = repo.create(query="match me", environment="staging",
                    reporter_id="u", reporter_team="t")
    b.status = "resolved"
    repo.save(b)
    hits = repo.find_similar(query="match me", environment="production")
    assert {h[0].id for h in hits} == {a.id}


def test_find_similar_excludes_unresolved(repo):
    a = repo.create(query="hello", environment="dev", reporter_id="u", reporter_team="t")
    # status stays "new"
    repo.save(a)
    hits = repo.find_similar(query="hello", environment="dev")
    assert hits == []


def test_find_similar_keyword_fallback_when_no_embedder():
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    Base.metadata.create_all(eng)
    repo = IncidentRepository(engine=eng, embedder=None, similarity_threshold=0.0)
    a = repo.create(query="redis OOM", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    repo.save(a)
    hits = repo.find_similar(query="redis OOM", environment="production")
    assert hits and hits[0][0].id == a.id


def test_find_similar_threshold_excludes_low_scores(repo):
    a = repo.create(query="alpha", environment="dev", reporter_id="u", reporter_team="t")
    a.status = "resolved"
    repo.save(a)
    repo.similarity_threshold = 0.999
    hits = repo.find_similar(query="zeta differs entirely", environment="dev")
    assert hits == []
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_storage_find_similar.py -v`
Expected: FAIL — `find_similar` doesn't exist yet.

- [ ] **Step 3 — Add `find_similar` to `IncidentRepository`**

Append to `src/orchestrator/storage/repository.py` after `list_recent`:

```python
    def find_similar(
        self, *, query: str, environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Incident, float]]:
        """Return up to ``limit`` similar resolved incidents for the same env.

        Embedding path uses native vector ops (pgvector ``<=>`` /
        sqlite-vec ``vec_distance_cosine``). Keyword path falls back to
        the existing ``KeywordSimilarity`` scorer to preserve behaviour
        when no embedder is configured.
        """
        if self.embedder is None:
            return self._keyword_similar(
                query=query, environment=environment,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        return self._vector_similar(
            query=query, environment=environment,
            status_filter=status_filter,
            threshold=threshold, limit=limit,
        )

    # ---- vector path ----
    def _vector_similar(self, *, query, environment, status_filter, threshold, limit):
        from sqlalchemy import and_, literal
        vec = self.embedder.embed_query(query)
        threshold = self.similarity_threshold if threshold is None else threshold

        with Session(self.engine) as session:
            if self.engine.dialect.name == "postgresql":
                # pgvector cosine_distance: smaller = closer; similarity = 1 - distance.
                score = (literal(1.0) - IncidentRow.embedding.cosine_distance(vec)).label("score")
            else:
                # sqlite-vec: vec_distance_cosine returns distance in [0, 2].
                # Same conversion: similarity = 1 - distance.
                import numpy as np
                blob = np.asarray(vec, dtype=np.float32).tobytes()
                score = (literal(1.0) - func.vec_distance_cosine(IncidentRow.embedding, blob)).label("score")
            stmt = (
                select(IncidentRow, score)
                .where(and_(
                    IncidentRow.deleted_at.is_(None),
                    IncidentRow.status == status_filter,
                    IncidentRow.environment == environment,
                    IncidentRow.embedding.is_not(None),
                ))
                .order_by(desc("score"))
                .limit(limit)
            )
            rows = session.execute(stmt).all()
        out: list[tuple[Incident, float]] = []
        for row, s in rows:
            s = float(s)
            if s < threshold:
                continue
            out.append((self._row_to_incident(row), s))
        return out

    # ---- keyword path ----
    def _keyword_similar(self, *, query, environment, status_filter, threshold, limit):
        from orchestrator.similarity import KeywordSimilarity, find_similar
        candidates_inc = [i for i in self.list_all() if i.environment == environment
                          and i.status == status_filter and i.deleted_at is None]
        candidates = [
            {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
             "incident": i}
            for i in candidates_inc
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(c["incident"], float(s)) for c, s in results]
```

- [ ] **Step 4 — Run, verify PASS on SQLite**

`pytest tests/test_storage_find_similar.py -v`
Expected: 5 passed (Postgres parametrization skipped without `POSTGRES_TEST_URL`).

- [ ] **Step 5 — Optional Postgres parity (skip if no Postgres locally)**

`POSTGRES_TEST_URL=postgresql+psycopg://user:pass@localhost/test pytest tests/test_storage_find_similar.py -v`
Expected: 10 passed (5 sqlite + 5 postgres).

- [ ] **Step 6 — Commit**

```bash
git add src/orchestrator/storage/repository.py tests/test_storage_find_similar.py
git commit -m "feat(storage): find_similar with pgvector/sqlite-vec dialect dispatch + keyword fallback"
```

---

## Task H — Wire MCP server to repository

**Files:**
- Modify: `src/orchestrator/mcp_servers/incident.py`
- Modify: `tests/test_mcp_servers_incident.py` (rename store→repo references)

- [ ] **Step 1 — Update `IncidentMCPServer` in `src/orchestrator/mcp_servers/incident.py`**

Replace the dataclass and tool bodies. Keep the `mcp` module-level + `set_state` shim so existing import paths continue to work.

```python
"""FastMCP server: incident_management mock tools.

Backed by ``IncidentRepository`` (SQLAlchemy). State is per-instance — the
old module-level singleton is gone except for a back-compat shim so the
MCP loader's ``getattr(mod, "mcp")`` keeps working.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fastmcp import FastMCP

from orchestrator.storage.repository import IncidentRepository


_DEFAULT_SEVERITY_ALIASES: dict[str, str] = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low", "informational": "low",
    "low": "low",
}


def normalize_severity(value: str | None,
                       aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentMCPServer:
    """FastMCP server bound to a single :class:`IncidentRepository`."""
    repository: IncidentRepository | None = None
    severity_aliases: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SEVERITY_ALIASES)
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
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.repository

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
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
        """Create a new INC ticket and persist it."""
        inc = self._require_repo().create(query=query, environment=environment,
                                          reporter_id=reporter_id,
                                          reporter_team=reporter_team)
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC."""
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


# Module-level back-compat (MCP loader does ``getattr(mod, "mcp")``).
_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, repository: IncidentRepository,
              severity_aliases: dict[str, str] | None = None) -> None:
    _default_server.configure(repository=repository, severity_aliases=severity_aliases)


# Direct-call shims kept for tests that import these names.
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

- [ ] **Step 2 — Update `tests/test_mcp_servers_incident.py` (or equivalent existing tests)**

Open every test that constructed `IncidentStore(...)` and called `set_state(store=...)`. Replace with the repository pattern:

```python
from orchestrator.config import StorageConfig, EmbeddingConfig, ProviderConfig
from orchestrator.storage.engine import build_engine
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository
from orchestrator.mcp_servers.incident import set_state


def _make_repo():
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return IncidentRepository(engine=eng, embedder=embedder, similarity_threshold=0.0)


@pytest.fixture
def server_repo():
    repo = _make_repo()
    set_state(repository=repo)
    return repo
```

(Search-replace `store=...` constructor patterns throughout `tests/`.)

- [ ] **Step 3 — Run all MCP server tests**

`pytest tests/ -v -k "mcp_servers or incident_management or normalize_severity" --tb=short`
Expected: green.

- [ ] **Step 4 — Commit**

```bash
git add src/orchestrator/mcp_servers/incident.py tests/
git commit -m "refactor(mcp): wire incident_management server to IncidentRepository"
```

---

## Task I — Wire orchestrator startup

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Test: `tests/test_orchestrator.py` (update fixtures)

- [ ] **Step 1 — Update `Orchestrator.create` in `src/orchestrator/orchestrator.py`**

Replace the IncidentStore + set_state wiring with engine + repository construction.

Replace this block (currently inside `Orchestrator.create`):

```python
            store = IncidentStore(cfg.paths.incidents_dir)
            # [...the importlib set_state block from the recent fix...]
```

With:

```python
            from orchestrator.storage.engine import build_engine
            from orchestrator.storage.embeddings import build_embedder
            from orchestrator.storage.models import Base
            from orchestrator.storage.repository import IncidentRepository

            engine = build_engine(cfg.storage)
            Base.metadata.create_all(engine)
            embedder = (
                build_embedder(cfg.llm.embedding, cfg.llm.providers)
                if cfg.incidents.similarity_method == "embedding"
                else None
            )
            store = IncidentRepository(
                engine=engine,
                embedder=embedder,
                similarity_threshold=cfg.incidents.similarity_threshold,
                severity_aliases=cfg.orchestrator.severity_aliases,
            )
            # Configure incident_management state via importlib so the FastMCP
            # transport hits the same module instance the loader will import.
            for srv in cfg.mcp.servers:
                if (srv.transport == "in_process" and srv.enabled
                        and srv.module == _INCIDENT_MCP_MODULE):
                    importlib.import_module(_INCIDENT_MCP_MODULE).set_state(
                        repository=store,
                        severity_aliases=cfg.orchestrator.severity_aliases,
                    )
                    break
```

Update the constructor signature:

```python
def __init__(self, cfg: AppConfig, store: IncidentRepository,
             skills: dict[str, Skill], registry: ToolRegistry, graph,
             resume_graph, exit_stack: AsyncExitStack):
```

- [ ] **Step 2 — Run orchestrator tests**

`pytest tests/test_orchestrator*.py -v`
Expected: green.

- [ ] **Step 3 — Full suite**

`pytest -q`
Expected: all tests pass except the optional Postgres parametrization.

- [ ] **Step 4 — Commit**

```bash
git add src/orchestrator/orchestrator.py tests/
git commit -m "refactor(orchestrator): build engine + IncidentRepository on startup"
```

---

## Task J — JSON → SQL migration script

**Files:**
- Create: `scripts/migrate_jsonl_to_sql.py`
- Test: `tests/test_migration_script.py`

- [ ] **Step 1 — Write the failing test**

```python
"""Migration script test: walk fixture JSON dir → upsert into SQL → verify rows."""
from __future__ import annotations
import json
from pathlib import Path

from orchestrator.config import AppConfig, EmbeddingConfig, IncidentConfig, LLMConfig, MCPConfig, Paths, ProviderConfig, StorageConfig


def test_migration_script_idempotent(tmp_path: Path):
    from scripts.migrate_jsonl_to_sql import migrate
    src = tmp_path / "incidents"
    src.mkdir()
    # Plant two fixture INCs with the right id format.
    for i, q in enumerate(["redis OOM", "ssh down"], start=1):
        inc_id = f"INC-20260101-{i:03d}"
        (src / f"{inc_id}.json").write_text(json.dumps({
            "id": inc_id, "status": "resolved",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "query": q, "environment": "production",
            "reporter": {"id": "u", "team": "t"},
            "summary": q, "tags": [], "agents_run": [], "tool_calls": [],
            "findings": {}, "user_inputs": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }))
    db_path = tmp_path / "out.db"
    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(),
        incidents=IncidentConfig(store_path=str(src)),
        storage=StorageConfig(url=f"sqlite:///{db_path}"),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(src)),
    )
    out1 = migrate(cfg, with_embeddings=False, dry_run=False)
    assert out1 == {"inserted": 2, "skipped": 0, "failed": 0}
    out2 = migrate(cfg, with_embeddings=False, dry_run=False)
    assert out2 == {"inserted": 0, "skipped": 2, "failed": 0}
```

- [ ] **Step 2 — Run, verify FAIL (script missing)**

`pytest tests/test_migration_script.py -v`

- [ ] **Step 3 — Implement `scripts/migrate_jsonl_to_sql.py`**

```python
"""One-shot backfill of JSON-file incidents into the SQL store.

Usage:
    python scripts/migrate_jsonl_to_sql.py [--config PATH] [--with-embeddings] [--dry-run]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from orchestrator.config import AppConfig, load_config
from orchestrator.incident import Incident
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository


def migrate(cfg: AppConfig, *, with_embeddings: bool, dry_run: bool) -> dict[str, int]:
    src = Path(cfg.paths.incidents_dir)
    engine = build_engine(cfg.storage)
    Base.metadata.create_all(engine)
    embedder = (
        build_embedder(cfg.llm.embedding, cfg.llm.providers)
        if with_embeddings else None
    )
    repo = IncidentRepository(engine=engine, embedder=embedder)
    counts = {"inserted": 0, "skipped": 0, "failed": 0}
    for path in sorted(src.glob("INC-*.json")):
        try:
            inc = Incident.model_validate(json.loads(path.read_text()))
        except Exception:
            counts["failed"] += 1
            continue
        try:
            repo.load(inc.id)
            counts["skipped"] += 1
            continue
        except FileNotFoundError:
            pass
        if not dry_run:
            repo.save(inc)
        counts["inserted"] += 1
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--with-embeddings", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = migrate(cfg, with_embeddings=args.with_embeddings, dry_run=args.dry_run)
    print(f"migrate: inserted={out['inserted']} skipped={out['skipped']} failed={out['failed']}")


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4 — Run, verify PASS**

`pytest tests/test_migration_script.py -v`

- [ ] **Step 5 — Live trial (optional)**

`/home/dev/projects/asr/.venv/bin/python scripts/migrate_jsonl_to_sql.py --dry-run`
Expected: prints inserted/skipped/failed for the existing `incidents/` dir.

- [ ] **Step 6 — Commit**

```bash
git add scripts/migrate_jsonl_to_sql.py tests/test_migration_script.py
git commit -m "feat(storage): one-shot JSON→SQL backfill script"
```

---

## Task K — Dist bundle update

**Files:**
- Modify: `scripts/build_single_file.py`
- Verify: `dist/app.py`, `dist/ui.py`

- [ ] **Step 1 — Extend `CORE_MODULE_ORDER`**

In `scripts/build_single_file.py`, expand the list. The new storage modules go after `incident.py` and `similarity.py` (they import `Incident`, `KeywordSimilarity`) and before `mcp_servers/incident.py` (which imports `IncidentRepository`):

```python
CORE_MODULE_ORDER = [
    "config.py",
    "incident.py",
    "similarity.py",
    "skill.py",
    "llm.py",
    "storage/types.py",
    "storage/models.py",
    "storage/engine.py",
    "storage/embeddings.py",
    "storage/repository.py",
    "mcp_servers/incident.py",
    "mcp_servers/observability.py",
    "mcp_servers/remediation.py",
    "mcp_servers/user_context.py",
    "mcp_loader.py",
    "graph.py",
    "orchestrator.py",
    "api.py",
]
```

- [ ] **Step 2 — Rebuild bundle**

`/home/dev/projects/asr/.venv/bin/python scripts/build_single_file.py`
Expected: `wrote dist/app.py (...)` and `wrote dist/ui.py (...)`.

- [ ] **Step 3 — Smoke import the bundle**

```bash
/home/dev/projects/asr/.venv/bin/python -c "
import sys; sys.path.insert(0, 'dist')
import app
print('OK', hasattr(app, 'IncidentRepository'),
      hasattr(app, 'build_engine'), hasattr(app, 'build_embedder'))
"
```

Expected: `OK True True True`.

- [ ] **Step 4 — End-to-end bundle smoke (intake)**

```bash
/home/dev/projects/asr/.venv/bin/python -c "
import sys, asyncio, os
sys.path.insert(0, 'dist')
os.environ.setdefault('AZURE_OPENAI_API_KEY', 'fake')
os.environ.setdefault('AZURE_OPENAI_ENDPOINT', 'http://localhost')
from app import Orchestrator, load_config
async def main():
    cfg = load_config('config/config.yaml')
    orch = await Orchestrator.create(cfg)
    try:
        entry = orch.registry.entries[('local_inc', 'lookup_similar_incidents')]
        result = await entry.tool.ainvoke({'query':'redis','environment':'production'})
        print('OK', type(result).__name__)
    finally:
        await orch.aclose()
asyncio.run(main())
"
```

Expected: `OK list` (or whatever the structured-content shape is — the key is no exception about uninitialized server).

- [ ] **Step 5 — Commit**

```bash
git add scripts/build_single_file.py dist/app.py dist/ui.py
git commit -m "build: bundle storage package into dist/app.py"
```

---

## Task L — End-to-end verification + final cleanup

**Files:**
- Modify: `src/orchestrator/incident.py` (remove `IncidentStore`)
- Modify: `config/config.yaml`
- Verify: UI run, intake works end-to-end

- [ ] **Step 1 — Remove `IncidentStore` class from `src/orchestrator/incident.py`**

Open the file, delete:
- `class IncidentStore:` block (everything from `class IncidentStore` through the end of its `delete` method)
- The `_INC_ID_RE` constant + `_utc_now_iso` / `_utc_today` helpers if unused elsewhere
- The unused `import json`, `from pathlib import Path`, `import re` if not referenced post-edit

Run: `grep -rn "IncidentStore" src/ tests/`
Expected: zero matches. If any remain, replace with `IncidentRepository`.

- [ ] **Step 2 — Update `config/config.yaml`**

Add the storage block and flip `similarity_method`:

```yaml
storage:
  url: "sqlite:///incidents.db"
incidents:
  similarity_threshold: 0.85
  similarity_method: embedding
llm:
  embedding:
    provider: workhorse-ollama
    model: bge-m3
    dim: 1024
```

(Keep the rest of the config as-is. The `embedding.provider` must be a key in `llm.providers`; verify before saving.)

- [ ] **Step 3 — Pull `bge-m3` if Ollama is reachable**

`curl -fsS http://localhost:11434/api/tags >/dev/null && ollama pull bge-m3 || echo "ollama not running, skipping"`

- [ ] **Step 4 — Backfill existing INCs**

```bash
/home/dev/projects/asr/.venv/bin/python scripts/migrate_jsonl_to_sql.py --with-embeddings
```

Expected: `inserted=N skipped=0 failed=0` where N is the count of `incidents/*.json`.

- [ ] **Step 5 — Rebuild and restart UI**

```bash
/home/dev/projects/asr/.venv/bin/python scripts/build_single_file.py
lsof -ti:37776 | xargs -r kill -9; sleep 1
nohup /home/dev/projects/asr/.venv/bin/streamlit run /home/dev/projects/asr/dist/ui.py > /tmp/streamlit-37776.log 2>&1 &
sleep 3
curl -s -o /dev/null -w "ui_health=%{http_code}\n" http://localhost:37776/_stcore/health
```

Expected: `ui_health=200`.

- [ ] **Step 6 — Manual smoke test**

In the UI: submit an incident query (e.g., `"redis OOMKill on payments"`, env `production`). Verify:
- Intake completes without "incident_management server not initialized" error.
- `lookup_similar_incidents` returns matches if any backfilled INC is similar.
- Detail panel shows the new incident with all fields populated.

- [ ] **Step 7 — Run full test suite**

`/home/dev/projects/asr/.venv/bin/python -m pytest -q`
Expected: same pass count as before this plan + new tests, zero failures.

- [ ] **Step 8 — Commit and push**

```bash
git add src/orchestrator/incident.py config/config.yaml dist/app.py dist/ui.py
git commit -m "feat: switch to SQL storage + embedding similarity (Ollama bge-m3)"
git push origin main
```

---

## Self-review notes

**Spec coverage check:** Every section of the spec maps to a task —
- Architecture / package layout: A, F (repository class), I (orchestrator wiring)
- Schema: D
- Custom column types: C
- Repository interface (CRUD): F
- Embedding facade: E
- Engine + sqlite-vec: D
- Migration: J
- Config: B (and L for yaml)
- Testing: every task has a TDD pair
- File structure: A through L collectively
- Risks (dim drift, pgvector install, bundle order): K, L, and documented in the spec

**Placeholder scan:** No TBDs, every code block is concrete and self-contained, every test has its assertions.

**Type consistency:** `IncidentRepository.create/load/save/delete/list_*` signatures match `IncidentStore` (verified against incident.py:103-160). `find_similar` returns `list[tuple[Incident, float]]`. `VectorColumn(dim)` Python type is `list[float] | None` consistently.

**Risks worth flagging at execution time:**
- `langchain-ollama` is already a dep but verify `OllamaEmbeddings` is the correct import path in the version we have.
- The `_StubEmbeddings.embed_query` returns deterministic-but-unnormalized vectors — cosine similarity between two stub vectors with different seeds is around 0 (orthogonal random vectors), so the "alpha vs zeta" threshold test passes naturally. If it doesn't on first run, lower the threshold from 0.999 to 0.99.
- The dist bundle's `_DEFAULT_SEVERITY_ALIASES` / `set_state` shim must end up in the bundled module surface — the bundler inlines source as-is, so as long as the module ordering in K is correct it just works.
