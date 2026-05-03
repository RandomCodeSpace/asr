# Metadata + Vector Store Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the just-shipped sqlite-vec/pgvector design into a clean separation: metadata in SQL, vectors in a LangChain `VectorStore` (FAISS for dev, PGVector for prod). Drops ~95 LOC of custom wrappers and the ctypes sqlite-vec hack.

**Architecture:** `IncidentRepository` keeps its public API. Internally it now talks to two stores: a SQLAlchemy `Engine` (pure metadata) and a LangChain `VectorStore` (vectors). Construction is config-driven via `build_engine(cfg.storage.metadata)` and `build_vector_store(cfg.storage.vector, embedder, engine)`.

**Tech Stack:** Python 3.14, SQLAlchemy 2.x sync, `faiss-cpu`, `langchain-community` (FAISS), `langchain-postgres` (PGVector), Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-05-02-metadata-vector-split-design.md` — read before starting.

**Starting commit:** `195ee3d` (the unified-table design, currently live).

---

## Task M1 — Config schema refactor

**Files:**
- Modify: `src/orchestrator/config.py`
- Modify: `tests/test_config.py`
- Modify: `config/config.yaml`

- [ ] **Step 1 — Failing tests in `tests/test_config.py`**

Replace existing `test_storage_config_default` / `test_storage_config_postgres_url` with:

```python
def test_storage_metadata_default():
    from orchestrator.config import AppConfig, LLMConfig, MCPConfig
    cfg = AppConfig(llm=LLMConfig.stub(), mcp=MCPConfig())
    assert cfg.storage.metadata.url == "sqlite:///incidents/incidents.db"
    assert cfg.storage.metadata.pool_size == 5
    assert cfg.storage.metadata.echo is False


def test_storage_vector_default():
    from orchestrator.config import AppConfig, LLMConfig, MCPConfig
    cfg = AppConfig(llm=LLMConfig.stub(), mcp=MCPConfig())
    assert cfg.storage.vector.backend == "faiss"
    assert cfg.storage.vector.path == "incidents/faiss"
    assert cfg.storage.vector.collection_name == "incidents"
    assert cfg.storage.vector.distance_strategy == "cosine"


def test_vector_backend_pgvector():
    from orchestrator.config import VectorConfig
    v = VectorConfig(backend="pgvector", collection_name="incidents")
    assert v.backend == "pgvector"


def test_vector_backend_none():
    from orchestrator.config import VectorConfig
    v = VectorConfig(backend="none")
    assert v.backend == "none"


def test_vector_backend_invalid_rejected():
    from orchestrator.config import VectorConfig
    import pytest
    with pytest.raises(Exception):
        VectorConfig(backend="qdrant")
```

- [ ] **Step 2 — Run, verify FAIL**

`pytest tests/test_config.py -v` → AttributeError on the nested fields.

- [ ] **Step 3 — Refactor `config.py`**

Replace the existing `StorageConfig` class with:

```python
from typing import Literal


class MetadataConfig(BaseModel):
    """Relational store for incident metadata. SQLite (dev) or Postgres (prod)."""
    url: str = "sqlite:///incidents/incidents.db"
    pool_size: int = 5      # postgres only; sqlite uses NullPool
    echo: bool = False


VectorBackend = Literal["faiss", "pgvector", "none"]
DistanceStrategy = Literal["cosine", "euclidean", "inner_product"]


class VectorConfig(BaseModel):
    """Vector store backing. FAISS (dev) or PGVector (prod) or none (keyword-only)."""
    backend: VectorBackend = "faiss"
    path: str = "incidents/faiss"           # faiss-only; created if missing
    collection_name: str = "incidents"      # FAISS index_name / PGVector collection_name
    distance_strategy: DistanceStrategy = "cosine"


class StorageConfig(BaseModel):
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
```

- [ ] **Step 4 — Update `config/config.yaml`**

Replace the flat `storage.url` with the nested form:

```yaml
storage:
  metadata:
    url: "sqlite:///incidents/incidents.db"
  vector:
    backend: faiss
    path: "incidents/faiss"
    collection_name: "incidents"
    distance_strategy: cosine
```

- [ ] **Step 5 — Run**

`pytest tests/test_config.py -v` → 5 new tests + existing all pass.

- [ ] **Step 6 — Commit**

```bash
git add src/orchestrator/config.py tests/test_config.py config/config.yaml
git commit -m "refactor(config): split storage into metadata + vector"
```

---

## Task M2 — Drop VectorColumn / sqlite-vec / embedding column

**Files:**
- Modify: `pyproject.toml` (deps)
- Delete: `tests/test_storage_types.py`
- Modify: `src/orchestrator/storage/types.py` (remove VectorColumn; remove JSONColumn alias entirely)
- Modify: `src/orchestrator/storage/models.py` (drop embedding column + EMBEDDING_DIM; replace JSONColumn with `JSON`)
- Modify: `src/orchestrator/storage/engine.py` (drop sqlite-vec hook + ctypes path + pgvector extension code)
- Modify: `tests/test_storage_engine.py` (drop sqlite-vec smoke)

- [ ] **Step 1 — Update `pyproject.toml`**

In `dependencies = [...]`:
- Remove `"sqlite-vec>=0.1.6"`
- Remove `"pgvector>=0.3"`
- Add `"faiss-cpu>=1.8"`
- Add `"langchain-community>=0.3"`
- Add `"langchain-postgres>=0.0.12"`

Run `/home/dev/projects/asr/.venv/bin/pip install -e '.[dev]'`. Then `pip uninstall -y sqlite-vec pgvector` (the editable install won't auto-uninstall removed deps).

- [ ] **Step 2 — Delete `tests/test_storage_types.py`**

`git rm tests/test_storage_types.py`

- [ ] **Step 3 — Reduce `src/orchestrator/storage/types.py` to a stub or delete**

The file is now empty (no JSONColumn alias either). Either delete it entirely, or leave a one-line shim for forward-compat:

```python
"""(Deprecated.) VectorColumn / JSONColumn lived here in a prior design.

Kept as an empty module to avoid breaking ``from orchestrator.storage.types
import …`` references during the refactor cycle. Delete after one cycle.
"""
```

Recommendation: delete the file. `git rm src/orchestrator/storage/types.py`.

- [ ] **Step 4 — Update `src/orchestrator/storage/models.py`**

Remove:
- `from orchestrator.storage.types import JSONColumn, VectorColumn`
- `EMBEDDING_DIM = 1024`
- `embedding: Mapped[list[float] | None] = mapped_column(VectorColumn(EMBEDDING_DIM), nullable=True)`

Replace:
- All `JSONColumn` usages → `JSON`
- Add `from sqlalchemy import JSON` import

After the edit, `IncidentRow` has all original columns minus `embedding`. Indexes unchanged.

- [ ] **Step 5 — Update `src/orchestrator/storage/engine.py`**

Strip the sqlite-vec block:

```python
"""SQLAlchemy engine factory.

Sync engine for SQLite (dev) or Postgres (prod). No vector-extension
loading — vectors live in a separate LangChain VectorStore (see
:mod:`orchestrator.storage.vector`).
"""
from __future__ import annotations
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool

from orchestrator.config import MetadataConfig


def build_engine(cfg: MetadataConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        return create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )
    return create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
```

- [ ] **Step 6 — Update `tests/test_storage_engine.py`**

Drop `test_build_engine_sqlite_in_memory` (no more `vec_distance_cosine` smoke). Keep `test_create_all_smoke` but adjust the expected column set: drop `embedding` from the `expected` set.

- [ ] **Step 7 — Update `IncidentRepository` (temporary fix)**

In `src/orchestrator/storage/repository.py`:
- Remove `from orchestrator.storage.models import IncidentRow` references to `embedding` column.
- Remove `_maybe_embed` and `_compute_save_embedding` methods (vector handling moves to Task M4).
- Remove `_vector_similar` method entirely.
- `_keyword_similar` remains as the only `find_similar` path for now.
- `find_similar` body becomes: always delegate to `_keyword_similar`.
- Constructor still accepts `embedder` (kept for Task M4 wiring), but it's unused for now.

This breaks any tests that asserted vector ordering — those are in `tests/test_storage_find_similar.py`. Mark them xfail with `reason="vector path lands in Task M4"`.

- [ ] **Step 8 — Run**

`pytest -q` → existing tests pass; vector-similarity tests xfail (expected).

- [ ] **Step 9 — Commit**

```bash
git add -u
git commit -m "refactor(storage): drop VectorColumn + sqlite-vec + embedding column"
```

---

## Task M3 — VectorStore factory (`storage/vector.py`)

**Files:**
- Create: `src/orchestrator/storage/vector.py`
- Create: `tests/test_storage_vector.py`

- [ ] **Step 1 — Failing tests**

```python
"""build_vector_store factory + FAISS persistence + stub-embedder smoke."""
from __future__ import annotations
import pytest
from pathlib import Path

from orchestrator.config import (
    EmbeddingConfig, ProviderConfig, VectorConfig,
)
from orchestrator.storage.embeddings import build_embedder


def _stub_embedder(dim: int = 8):
    return build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=dim),
        {"s": ProviderConfig(kind="stub")},
    )


def test_build_vector_store_none():
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="none")
    assert build_vector_store(cfg, _stub_embedder()) is None


def test_build_vector_store_faiss_roundtrip(tmp_path: Path):
    from langchain_core.documents import Document
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="faiss", path=str(tmp_path / "vs"),
                       collection_name="t", distance_strategy="cosine")
    embedder = _stub_embedder()
    vs = build_vector_store(cfg, embedder)
    assert vs is not None
    vs.add_documents(
        [Document(page_content="hello world", metadata={"id": "INC-1"})],
        ids=["INC-1"],
    )
    # Persist + reload to assert disk round-trip:
    vs.save_local(folder_path=cfg.path, index_name=cfg.collection_name)
    vs2 = build_vector_store(cfg, embedder)  # loads from disk
    hits = vs2.similarity_search_with_score("hello world", k=1)
    assert hits and hits[0][0].metadata["id"] == "INC-1"


def test_build_vector_store_faiss_delete(tmp_path: Path):
    from langchain_core.documents import Document
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="faiss", path=str(tmp_path / "vs"),
                       collection_name="t")
    embedder = _stub_embedder()
    vs = build_vector_store(cfg, embedder)
    vs.add_documents([Document(page_content="x", metadata={"id": "A"}),
                      Document(page_content="y", metadata={"id": "B"})],
                     ids=["A", "B"])
    vs.delete(ids=["A"])
    hits = vs.similarity_search_with_score("x", k=5)
    ids = {h[0].metadata.get("id") for h in hits}
    assert "A" not in ids
```

- [ ] **Step 2 — Implement `src/orchestrator/storage/vector.py`**

```python
"""LangChain ``VectorStore`` factory.

Backends
--------
- ``faiss``    → ``langchain_community.vectorstores.FAISS`` (file-backed, dev).
- ``pgvector`` → ``langchain_postgres.PGVector`` (DB-backed, prod). Reuses
                 the metadata SQLAlchemy engine — same Postgres connection.
- ``none``     → ``None``; caller falls back to keyword similarity.

FAISS persistence: callers invoke :meth:`vector_store.save_local` after
each mutation. The factory loads from disk if a saved index exists at the
configured ``path``; otherwise it constructs an empty index by seeding
with a placeholder doc and immediately deleting it (LangChain's FAISS
constructor doesn't accept an empty docstore).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from orchestrator.config import VectorConfig


_PLACEHOLDER_ID = "__seed__"


def _faiss_distance_strategy(name: str):
    from langchain_community.vectorstores.utils import DistanceStrategy
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
    }[name]


def _pgvector_distance_strategy(name: str):
    from langchain_postgres.vectorstores import DistanceStrategy
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN,
        "inner_product": DistanceStrategy.INNER_PRODUCT,
    }[name]


def _build_faiss(cfg: VectorConfig, embedder: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    folder = Path(cfg.path)
    index_file = folder / f"{cfg.collection_name}.faiss"
    if index_file.exists():
        return FAISS.load_local(
            folder_path=str(folder),
            index_name=cfg.collection_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
            distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
        )
    folder.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(
        [Document(page_content=_PLACEHOLDER_ID, metadata={"id": _PLACEHOLDER_ID})],
        embedding=embedder,
        ids=[_PLACEHOLDER_ID],
        distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
    )
    vs.delete(ids=[_PLACEHOLDER_ID])
    return vs


def _build_pgvector(cfg: VectorConfig, embedder: Embeddings,
                    engine) -> VectorStore:
    from langchain_postgres import PGVector
    return PGVector(
        embeddings=embedder,
        collection_name=cfg.collection_name,
        connection=engine,
        distance_strategy=_pgvector_distance_strategy(cfg.distance_strategy),
        use_jsonb=True,
    )


def build_vector_store(
    cfg: VectorConfig,
    embedder: Optional[Embeddings],
    metadata_engine=None,
) -> Optional[VectorStore]:
    if cfg.backend == "none" or embedder is None:
        return None
    if cfg.backend == "faiss":
        return _build_faiss(cfg, embedder)
    if cfg.backend == "pgvector":
        if metadata_engine is None:
            raise ValueError(
                "pgvector backend requires metadata_engine (the SQLAlchemy "
                "engine, used as the connection)"
            )
        return _build_pgvector(cfg, embedder, metadata_engine)
    raise ValueError(f"unknown vector backend: {cfg.backend!r}")


def distance_to_similarity(distance: float, strategy: str) -> float:
    """Normalize a backend-native distance to a similarity in roughly [0, 1].

    - cosine: ``1 - distance``  (LangChain returns cosine-distance ∈ [0, 2])
    - inner_product: ``distance`` is already a similarity (already + dot product).
    - euclidean: ``1 / (1 + distance)`` — monotonic, compressed to (0, 1].
    """
    if strategy == "cosine":
        return 1.0 - distance
    if strategy == "inner_product":
        return distance
    if strategy == "euclidean":
        return 1.0 / (1.0 + distance)
    raise ValueError(f"unknown distance strategy: {strategy!r}")
```

- [ ] **Step 3 — Run**

`pytest tests/test_storage_vector.py -v` → 3 passed.

- [ ] **Step 4 — Commit**

```bash
git add src/orchestrator/storage/vector.py tests/test_storage_vector.py
git commit -m "feat(storage): VectorStore factory (FAISS + PGVector branches)"
```

---

## Task M4 — Refactor `IncidentRepository` to use VectorStore

**Files:**
- Modify: `src/orchestrator/storage/repository.py`
- Modify: `tests/test_storage_find_similar.py` (un-xfail, parameterize)
- Modify: `tests/test_storage_repository.py` (none expected, but verify after)

- [ ] **Step 1 — Constructor + lifecycle**

In `IncidentRepository.__init__`, add `vector_store: Optional[VectorStore] = None` and `vector_path: Optional[str] = None` (used to call `save_local` after writes when the backend is FAISS).

Detect FAISS by `hasattr(vector_store, "save_local")`. If present, save after each mutation.

```python
def _persist_vector(self) -> None:
    if self.vector_store is not None and hasattr(self.vector_store, "save_local") \
            and self.vector_path is not None:
        # Inferred collection_name = last path segment is fine for tests.
        from pathlib import Path
        folder = Path(self.vector_path)
        folder.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(folder_path=str(folder),
                                     index_name=self._vector_index_name)
```

Pass `vector_index_name` (the collection_name) into the constructor too. Default `"incidents"`.

- [ ] **Step 2 — `create / save / delete` write to vector store**

```python
def create(self, ...):
    ...
    session.commit()
    inc = self._row_to_incident(row)
    self._add_vector(inc)
    return inc

def save(self, incident):
    ...
    session.commit()
    self._refresh_vector(incident, prior_text=prior_embed_source)

def delete(self, incident_id):
    ...
    # vectors stay; find_similar filters by deleted_at
    return self._row_to_incident(row)
```

Vector helpers:

```python
def _add_vector(self, inc: Incident) -> None:
    if self.vector_store is None or self.embedder is None:
        return
    text = _embed_source(inc)
    if not text:
        return
    from langchain_core.documents import Document
    self.vector_store.add_documents(
        [Document(page_content=text, metadata={"id": inc.id})],
        ids=[inc.id],
    )
    self._persist_vector()

def _refresh_vector(self, inc: Incident, *, prior_text: str | None) -> None:
    if self.vector_store is None or self.embedder is None:
        return
    text = _embed_source(inc)
    if not text:
        return
    if prior_text == text:
        return
    self.vector_store.delete(ids=[inc.id])
    from langchain_core.documents import Document
    self.vector_store.add_documents(
        [Document(page_content=text, metadata={"id": inc.id})],
        ids=[inc.id],
    )
    self._persist_vector()
```

`save()` reads the existing row's embed source BEFORE writing the update, so it can pass `prior_text`.

- [ ] **Step 3 — `find_similar` rewrite**

```python
def find_similar(self, *, query, environment, status_filter="resolved",
                 threshold=None, limit=5):
    if self.vector_store is None or self.embedder is None:
        return self._keyword_similar(query=query, environment=environment,
                                     status_filter=status_filter,
                                     threshold=threshold, limit=limit)
    threshold = self.similarity_threshold if threshold is None else threshold
    vec = self.embedder.embed_query(query)
    raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit*4)
    out: list[tuple[Incident, float]] = []
    for doc, distance in raw:
        score = distance_to_similarity(float(distance), self._distance_strategy)
        if score < threshold:
            continue
        inc_id = doc.metadata.get("id")
        if inc_id is None:
            continue
        try:
            inc = self.load(inc_id)
        except (FileNotFoundError, ValueError):
            continue
        if inc.environment != environment or inc.status != status_filter \
                or inc.deleted_at is not None:
            continue
        out.append((inc, score))
        if len(out) >= limit:
            break
    return out
```

`_distance_strategy` is a constructor arg defaulted to `"cosine"`. Imports: `from orchestrator.storage.vector import distance_to_similarity`.

- [ ] **Step 4 — Update `tests/test_storage_find_similar.py`**

Remove `xfail` markers. Refactor the fixture to build a FAISS-backed repo:

```python
@pytest.fixture
def repo(tmp_path):
    from orchestrator.config import (
        EmbeddingConfig, MetadataConfig, ProviderConfig, VectorConfig,
    )
    from orchestrator.storage.embeddings import build_embedder
    from orchestrator.storage.engine import build_engine
    from orchestrator.storage.models import Base
    from orchestrator.storage.repository import IncidentRepository
    from orchestrator.storage.vector import build_vector_store

    db = tmp_path / "test.db"
    eng = build_engine(MetadataConfig(url=f"sqlite:///{db}"))
    Base.metadata.create_all(eng)

    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=8),
        {"s": ProviderConfig(kind="stub")},
    )
    vec_path = str(tmp_path / "vs")
    vs_cfg = VectorConfig(backend="faiss", path=vec_path,
                          collection_name="t", distance_strategy="cosine")
    vs = build_vector_store(vs_cfg, embedder)

    return IncidentRepository(
        engine=eng, embedder=embedder, vector_store=vs,
        vector_path=vec_path, vector_index_name="t",
        distance_strategy="cosine", similarity_threshold=0.0,
    )
```

The test bodies stay the same — they assert ordering, not absolute distance. The `> 0.99` self-match assertion holds for cosine because the stub embedder is deterministic and the query exactly matches the stored content.

- [ ] **Step 5 — Run**

`pytest tests/test_storage_find_similar.py -v tests/test_storage_repository.py -v` → all pass.

- [ ] **Step 6 — Full suite**

`pytest -q` → no regressions.

- [ ] **Step 7 — Commit**

```bash
git add src/orchestrator/storage/repository.py tests/
git commit -m "refactor(storage): IncidentRepository uses LangChain VectorStore"
```

---

## Task M5 — Wire orchestrator + migration script

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Modify: `scripts/migrate_jsonl_to_sql.py`
- Modify: `tests/test_migration_script.py`

- [ ] **Step 1 — Update `Orchestrator.create`**

Replace the engine/repo block with:

```python
from orchestrator.storage.vector import build_vector_store

engine = build_engine(MetadataConfig(
    url=_metadata_url(cfg),
    pool_size=cfg.storage.metadata.pool_size,
    echo=cfg.storage.metadata.echo,
))
Base.metadata.create_all(engine)
embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
vector_store = build_vector_store(cfg.storage.vector, embedder, engine)
store = IncidentRepository(
    engine=engine,
    embedder=embedder,
    vector_store=vector_store,
    vector_path=cfg.storage.vector.path if cfg.storage.vector.backend == "faiss" else None,
    vector_index_name=cfg.storage.vector.collection_name,
    distance_strategy=cfg.storage.vector.distance_strategy,
    similarity_threshold=cfg.incidents.similarity_threshold,
    severity_aliases=cfg.orchestrator.severity_aliases,
)
```

`_storage_url` helper renamed `_metadata_url`; reads `cfg.storage.metadata.url`.

- [ ] **Step 2 — Update migration script**

`scripts/migrate_jsonl_to_sql.py`'s `migrate(cfg, ...)` builds the same engine + vector store stack as the orchestrator, then iterates JSON files. When `--with-embeddings`, the repository's `save()` path automatically writes both metadata and vector entries (since vector_store is non-None and embedder is configured).

- [ ] **Step 3 — Run**

`pytest tests/test_migration_script.py -v` → passes (idempotent counts unchanged).

- [ ] **Step 4 — Full suite**

`pytest -q` → 185+ pass, 0 failures.

- [ ] **Step 5 — Commit**

```bash
git add src/orchestrator/orchestrator.py scripts/migrate_jsonl_to_sql.py tests/test_migration_script.py
git commit -m "refactor(orchestrator,scripts): wire vector_store into runtime + migration"
```

---

## Task M6 — Bundle update + dist smoke

**Files:**
- Modify: `scripts/build_single_file.py`
- Verify: `dist/app.py`, `dist/ui.py`

- [ ] **Step 1 — Update `CORE_MODULE_ORDER`**

Replace the storage list with:

```python
"storage/models.py",      # types.py is gone
"storage/engine.py",
"storage/embeddings.py",
"storage/vector.py",      # NEW
"storage/repository.py",
```

(`storage/types.py` is no longer in the list because the file is deleted.)

- [ ] **Step 2 — Rebuild**

`python scripts/build_single_file.py` → expect both `dist/app.py` and `dist/ui.py` written.

- [ ] **Step 3 — Smoke import**

```bash
python -c "
import sys; sys.path.insert(0, 'dist')
import app
print('OK',
      hasattr(app, 'IncidentRepository'),
      hasattr(app, 'build_engine'),
      hasattr(app, 'build_embedder'),
      hasattr(app, 'build_vector_store'))
"
```

Expected: `OK True True True True`.

- [ ] **Step 4 — End-to-end smoke**

```bash
python -c "
import sys, asyncio, os
sys.path.insert(0, 'dist')
os.environ.setdefault('AZURE_OPENAI_API_KEY', 'fake')
os.environ.setdefault('AZURE_OPENAI_ENDPOINT', 'http://localhost')
from app import Orchestrator, load_config
async def main():
    cfg = load_config('config/config.yaml')
    orch = await Orchestrator.create(cfg)
    try:
        recent = orch.list_recent_incidents(limit=3)
        print('OK', len(recent), 'recent INCs')
    finally:
        await orch.aclose()
asyncio.run(main())
"
```

Expected: `OK <N> recent INCs` (no errors). `N` matches what's in `incidents/incidents.db`.

- [ ] **Step 5 — Commit**

```bash
git add scripts/build_single_file.py dist/app.py dist/ui.py
git commit -m "build: bundle vector store factory into dist/app.py"
```

---

## Task M7 — Final integration: config flip, UI restart, push

**Files:**
- Modify: `config/config.yaml` (flip embedding model, set similarity_method)
- Verify: UI live with vector search

- [ ] **Step 1 — Probe Ollama and set the embedding model**

Verify Ollama has `bge-m3:latest` (curl `http://localhost:11434/api/tags`). If yes:

In `config/config.yaml`, under `llm:`:
- Change embedding `provider:` to a local Ollama provider (look for one that has `base_url: http://localhost:11434` in `llm.providers`; if absent, add a `workhorse-ollama` entry with `kind: ollama, base_url: http://localhost:11434`).
- Set `model: bge-m3`.
- Set `dim: 1024`.

Under `incidents:`:
- Set `similarity_method: embedding` (this no-op flag becomes meaningful again once we restore the gate; if not restored in this plan, leave at whatever value — vector path is taken whenever vector_store is not None).

- [ ] **Step 2 — Re-embed historical INCs**

```bash
/home/dev/projects/asr/.venv/bin/python scripts/migrate_jsonl_to_sql.py --with-embeddings
```

Expected: prior 28 INCs were already inserted; this run re-saves them (the `save()` path's embed-source check sees prior=None and computes a new vector; writes to FAISS). The `incidents/faiss` directory grows.

(If migration script doesn't update existing rows by default, add a quick `python -c` that walks `repo.list_all(include_deleted=True)` and calls `repo.save(inc)` on each.)

- [ ] **Step 3 — Rebuild bundle (post-config-edit)**

`python scripts/build_single_file.py`

- [ ] **Step 4 — Restart UI**

```bash
lsof -ti:37776 | xargs -r kill -9
sleep 1
nohup .venv/bin/streamlit run dist/ui.py > /tmp/streamlit-37776.log 2>&1 &
sleep 4
curl -s -o /dev/null -w "ui_health=%{http_code}\n" http://localhost:37776/_stcore/health
```

Expected: `ui_health=200`.

- [ ] **Step 5 — Manual smoke test**

In the browser at `http://localhost:37776`, submit a new incident query close to one of the backfilled ones. Verify `lookup_similar_incidents` returns matches with non-zero scores (vector path active).

- [ ] **Step 6 — Final pytest**

`pytest -q` → 185+ pass, no regressions.

- [ ] **Step 7 — Commit + push**

```bash
git add config/config.yaml dist/app.py dist/ui.py
git commit -m "feat: enable vector similarity (Ollama bge-m3 + FAISS)"
git push origin main
```

---

## Self-review

**Spec coverage:** every spec section maps to a task — config (M1), drop legacy (M2), vector factory (M3), repository refactor (M4), runtime wiring (M5), bundle (M6), final integration (M7).

**Placeholder scan:** no TBDs; every code block is concrete with imports.

**Type consistency:** `IncidentRepository.find_similar` returns `list[tuple[Incident, float]]` (unchanged); `VectorStore.similarity_search_with_score_by_vector` returns `list[tuple[Document, float]]` (LangChain stable); `distance_to_similarity` is the consistent normalizer used in M3 and M4.

**Risks at execution time:**
- LangChain's FAISS may need `allow_dangerous_deserialization=True` — already in the factory.
- `langchain_postgres` may pin SQLAlchemy <2.0 in older releases; verify version compat at install time. If conflict, pin `langchain-postgres>=0.0.12` (which is SQLA 2.0-compatible).
- `langchain_community.vectorstores.utils.DistanceStrategy` and `langchain_postgres.vectorstores.DistanceStrategy` are different enums — the factory imports them separately by intent.
