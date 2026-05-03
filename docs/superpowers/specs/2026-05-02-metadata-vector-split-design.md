# Metadata + Vector Store Split — Design

**Status:** approved 2026-05-02
**Supersedes:** the unified-table approach from `2026-05-02-sql-storage-and-embeddings-design.md`. That landed as commit `195ee3d`; this design refactors it.
**Author:** brainstorm session, ak.nitrr13

---

## Goal

Replace the sqlite-vec/pgvector custom-column path with a clean separation: **metadata in SQL, vectors in a LangChain `VectorStore`**.

- Dev: SQLite (metadata) + FAISS (vectors)
- Prod: Postgres (metadata) + PGVector via `langchain_postgres` (vectors, in the same Postgres instance)
- Same code path; only config changes between environments.

## Why this over the unified-column design

- **No `sqlite-vec` C extension**, no ctypes hack to call `sqlite3_load_extension` (pyenv Python 3.14 doesn't ship `SQLITE_ENABLE_LOAD_EXTENSION`). Just `pip install faiss-cpu` — pre-built wheel, no system dep.
- **No custom SQLAlchemy `VectorColumn` `TypeDecorator`** with dialect-aware bind/result. `langchain_community.vectorstores.FAISS` and `langchain_postgres.PGVector` are off-the-shelf.
- **No dialect dispatch** in `find_similar` (`<=>` vs `vec_distance_cosine`). One `VectorStore.similarity_search_with_score_by_vector(vec, k)` call.
- **Plug-replaceable backends.** New LangChain `VectorStore` integration (LanceDB, Qdrant, Chroma, ...) becomes a one-line branch in `build_vector_store` — no schema migration.

Trade-offs accepted:
- Two-phase writes (metadata + vector) without a single transaction in the FAISS dev case. Best-effort consistency; orphaned vector entries get filtered out at hydrate time.
- Vector-side filtering is portability-limited (FAISS has no native `WHERE`; PGVector accepts a JSONB filter). We **overshoot k×4 then filter in Python** instead of pushing filters into each backend's dialect.

## Non-goals

- Async LangChain APIs. The orchestrator's coarse-grained concurrency is fine with sync `VectorStore` calls.
- Multi-vector per incident (separate query / summary / resolution vectors). Single embedding per INC.
- Hard-delete or vector-side soft-delete sync. Soft-delete stays SQL-only; `find_similar` filters `deleted_at IS NULL` after hydrating.

## Architecture

```
                 IncidentRepository
                ┌───────────────────────────┐
                │  metadata_engine: Engine  │ ← SQL, no embedding column
                │  vector_store:   VectorStore │ ← LangChain VectorStore
                │  embedder:       Embeddings │ ← unchanged
                └─────┬───────────────┬─────┘
                      │               │
            ┌─────────▼────┐   ┌──────▼─────────────┐
            │ SQLite/PG    │   │ FAISS / PGVector   │
            │ (incidents)  │   │ (langchain wrappers)│
            └──────────────┘   └────────────────────┘
```

`IncidentRow` becomes a pure metadata table — no `embedding` column. The vector store has its own backing store:
- **FAISS:** flat L2/IP index file + pickled docstore in `cfg.storage.vector.path/`. Single directory, two files.
- **PGVector:** two side tables in the same Postgres DB (`langchain_pg_collection`, `langchain_pg_embedding`).

## Config

```yaml
storage:
  metadata:
    url: "sqlite:///incidents/incidents.db"   # postgresql+psycopg://... in prod
    pool_size: 5
    echo: false
  vector:
    backend: faiss            # faiss | pgvector | none
    path: "incidents/faiss"   # faiss-only, dir; created if missing
    collection_name: "incidents"   # used by both (FAISS index_name, PGVector collection_name)
    distance_strategy: cosine  # cosine | euclidean | inner_product

llm:
  embedding:
    provider: workhorse        # name in llm.providers; provider.kind in {ollama, azure_openai, stub}
    model: bge-m3
    dim: 1024
```

`storage.vector.backend = "none"` disables vector search; `find_similar` falls back to keyword similarity. Useful for skipping the vector dependency entirely (e.g., minimal deployments or stub mode).

## Component changes

### `src/orchestrator/storage/types.py`
- **Drop:** `VectorColumn` (~50 LOC). Tests for `VectorColumn` deleted with it.
- **Drop:** `JSONColumn` alias (1 line — use `sqlalchemy.JSON` directly throughout).

### `src/orchestrator/storage/models.py`
- **Drop:** `embedding: Mapped[list[float] | None]` column on `IncidentRow`.
- **Drop:** `EMBEDDING_DIM` constant.
- **Drop:** `VectorColumn`/`JSONColumn` imports; replace `JSONColumn` usages with `JSON`.

### `src/orchestrator/storage/engine.py`
- **Drop:** `_attach_sqlite_vec`, ctypes path, the entire sqlite-vec extension-loading code.
- **Drop:** `_ensure_pgvector` (LangChain PGVector handles `CREATE EXTENSION` itself if missing).
- `build_engine(cfg)` becomes plain SQLAlchemy: `create_engine(url, ...)` with `NullPool` for sqlite, `pool_size` for postgres.

### `src/orchestrator/storage/vector.py` (new)
```python
def build_vector_store(
    cfg: VectorConfig,
    embedder: Embeddings | None,
    metadata_engine: Engine | None = None,
) -> VectorStore | None:
    """Construct a LangChain VectorStore from config.

    - faiss     → langchain_community.vectorstores.FAISS  (file-backed, dev)
    - pgvector  → langchain_postgres.PGVector             (DB-backed, prod)
    - none      → None  (caller falls back to keyword similarity)
    """
```

FAISS branch: `FAISS.load_local(path)` if exists, else `FAISS.from_texts(["__init__"], embedder, ids=["__init__"])` placeholder + immediate delete (LangChain's FAISS doesn't support empty construction in older versions; if newer version supports it, simplify). Persisted via `vector_store.save_local(path, index_name=cfg.collection_name)` after every write.

PGVector branch: `PGVector(embeddings=embedder, collection_name=..., connection=metadata_engine, distance_strategy=DistanceStrategy.COSINE, use_jsonb=True)`. Reuses the metadata engine (same Postgres connection, no extra config).

### `src/orchestrator/storage/repository.py`
- Constructor: `__init__(*, engine, embedder=None, vector_store=None, vector_path=None, similarity_threshold=..., severity_aliases=...)`.
- `create()`: insert SQL row, then `vector_store.add_texts([_embed_source(inc)], metadatas=[{"id": inc.id, "environment": ..., "status": ...}], ids=[inc.id])`. If FAISS, follow with `save_local`.
- `save()`: update SQL row. If embed source diverged from prior, `vector_store.delete([inc.id])` + `add_texts(...)` + `save_local`.
- `delete()`: soft-delete SQL row only. Vector entry stays (filtered out at find time).
- `find_similar()`:
  ```python
  if self.vector_store is None or self.embedder is None:
      return self._keyword_similar(...)
  vec = self.embedder.embed_query(query)
  raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit*4)
  out = []
  for doc, distance in raw:
      score = self._distance_to_similarity(distance)
      if score < threshold:
          continue
      try:
          inc = self.load(doc.metadata["id"])
      except FileNotFoundError:
          continue
      if inc.environment != environment or inc.status != status_filter or inc.deleted_at is not None:
          continue
      out.append((inc, score))
      if len(out) >= limit:
          break
  return out
  ```
  `_distance_to_similarity` per `cfg.storage.vector.distance_strategy`:
    - cosine: `1.0 - distance`
    - inner_product: `distance` (already a similarity)
    - euclidean: `1.0 / (1.0 + distance)` (compressed mapping)

- **Drop:** `_vector_similar` dialect dispatch. **Drop:** the `from sqlalchemy import and_, func, literal` imports that supported it.

### `src/orchestrator/config.py`
- New `MetadataConfig` (just the existing `StorageConfig` fields renamed) and `VectorConfig`.
- `StorageConfig` becomes a container: `metadata: MetadataConfig`, `vector: VectorConfig`.
- Existing `cfg.storage.url` references → `cfg.storage.metadata.url`. Backward-compat shim NOT provided — clean break.

### `src/orchestrator/orchestrator.py`
- `Orchestrator.create()` builds engine via `build_engine(cfg.storage.metadata)`, then `vector_store = build_vector_store(cfg.storage.vector, embedder, engine)`, then constructs `IncidentRepository(engine=..., embedder=..., vector_store=...)`.
- `_storage_url` helper updated to read `cfg.storage.metadata.url` and derive sqlite URL from `cfg.paths.incidents_dir` when default.

### `src/orchestrator/mcp_servers/incident.py`
- Unchanged. The repository's public API stays the same; only its internals shift.

### `scripts/migrate_jsonl_to_sql.py`
- Now writes both metadata SQL row AND vector store entry when `--with-embeddings`.
- Without `--with-embeddings`: SQL only; vectors get backfilled later via a re-embed run (just re-run `--with-embeddings`; `repo.save()` detects missing vector and adds it).

### `scripts/build_single_file.py`
- `CORE_MODULE_ORDER` updated: drop nothing (storage/* still bundled), but the contents of types.py shrink and a new `storage/vector.py` is added in dep order between `storage/embeddings.py` and `storage/repository.py`.

## Dependencies

**Add:**
- `faiss-cpu>=1.8` — pre-built wheel, ~4MB, no system deps
- `langchain-community>=0.3` — for `FAISS` vectorstore (likely already transitive)
- `langchain-postgres>=0.0.12` — for `PGVector` vectorstore

**Remove:**
- `sqlite-vec`
- `pgvector` (the Python adapter — only needed for the SQLAlchemy `Vector(N)` column type, which we drop)

**Keep:**
- `sqlalchemy>=2.0`, `psycopg[binary]>=3.2`, `numpy`, `langchain-ollama` (existing)

## Tests

- All existing storage tests stay valid for the metadata side. `test_storage_types.py` is **deleted** (no more `VectorColumn`).
- `test_storage_engine.py` stripped of sqlite-vec smoke; only the `create_all` smoke remains.
- New `tests/test_storage_vector.py`:
  - FAISS in `tmp_path`: round-trip add → search → delete → save_local → load_local.
  - Stub embedder; vector parity asserts ordering not absolute distance.
- `test_storage_find_similar.py`: same test bodies, parameterized over `[faiss_repo, pgvector_repo_if_url_set]`.
- `test_storage_repository.py`: unchanged contracts; fixture switches to `vector_store=None` (keyword fallback) for CRUD-only tests, and a new fixture for the vector-aware tests.
- Migration script test: parameterized with/without embeddings, verifies vector entries when enabled.

## Risks / open follow-ups

- **FAISS persistence frequency.** Saving after every write is fine at our scale (incidents are infrequent). At higher write rate, batch save on shutdown via `aclose`.
- **`allow_dangerous_deserialization=True`** is required by FAISS load. The pickle is OURS — we control the on-disk file. Documented; not a concern for the deploy threat model.
- **Soft-delete and vector entries.** Vectors stay even after soft-delete; `find_similar` filters at hydrate time. If prod corpus ever grows large enough that the orphan vectors hurt search quality, add a cleanup job (out of scope here).
- **Distance/similarity normalization** differs per backend; `_distance_to_similarity` centralizes the conversion. Test asserts cross-backend ordering is identical.
- **PGVector requires the `pgvector` Postgres extension server-side.** Air-gapped deploy needs the extension installed in the Postgres image. Documented as a runtime dep.

## Out of scope

- Re-embed CLI command (use the migration script's `--with-embeddings` rerun pattern).
- Tuning FAISS index types (`IndexFlatL2` is fine for our scale; HNSW/IVF tuning later).
- Hybrid keyword+vector ranking (the existing keyword fallback covers the no-embedder case; no need to compose them).
