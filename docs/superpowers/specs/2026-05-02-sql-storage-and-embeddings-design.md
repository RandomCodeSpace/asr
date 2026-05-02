# SQL Storage + Embedding Similarity — Design

**Status:** approved 2026-05-02
**Author:** brainstorm session, ak.nitrr13
**Replaces:** JSON-file `IncidentStore`, keyword-only `lookup_similar_incidents`

---

## Goal

Replace the JSON-on-disk `IncidentStore` with a SQLAlchemy-backed `IncidentRepository` that runs on **SQLite + sqlite-vec** (dev) and **Postgres + pgvector** (prod), with config-only swap. Wire a LangChain-`Embeddings`-backed similarity path that uses **Ollama `bge-m3`** locally and swaps to **Azure OpenAI** in prod via the same config switch already used for chat models.

## Non-goals

- No async-SQLAlchemy migration. Sync sessions are sufficient at our scale.
- No Alembic. Schema lifecycle is `Base.metadata.create_all(engine)` until prod schema actually churns.
- No vector DB beside the relational DB (chromadb / faiss / weaviate). The whole point is one store.
- No multi-tenancy, no RLS, no historic-vector versioning. Single-model assumption: switching the embedding model requires re-embedding the corpus.
- No incremental schema changes from this design — this is a clean break (with a one-shot backfill from existing JSON files).

## Architecture

```
                Orchestrator + MCP tools (interface unchanged)
                                │
                                ▼
                  ┌────────────────────────────┐
                  │  IncidentRepository        │   replaces IncidentStore
                  │  (Pydantic ↔ Row mapping,  │
                  │   embedder ownership)      │
                  └─────┬─────────────────┬────┘
                        │                 │
              ┌─────────▼─────┐   ┌───────▼─────────┐
              │ EmbeddingFacade│   │ SQLAlchemy core │
              │ (LangChain)   │   │   engine        │
              └────────────────┘   └─┬─────────────┬─┘
                                     │             │
                              ┌──────▼─────┐ ┌────▼──────┐
                              │ SQLite     │ │ Postgres  │
                              │ +sqlite-vec│ │ +pgvector │
                              └────────────┘ └───────────┘
```

The `incident_management` MCP server's tool surface (`lookup_similar_incidents`, `create_incident`, `update_incident`) is unchanged. Only the underlying state holder swaps from `IncidentStore` to `IncidentRepository`. The MCP server's `configure(...)` method takes a repository instead of a store.

## Schema

One table — `incidents` — with scalar columns for queryable fields, JSON columns for nested Pydantic structures, and a real vector column for embeddings.

```sql
CREATE TABLE incidents (
    id                       TEXT        PRIMARY KEY,
    status                   TEXT        NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL,
    updated_at               TIMESTAMPTZ NOT NULL,
    deleted_at               TIMESTAMPTZ NULL,            -- soft delete
    query                    TEXT        NOT NULL,
    environment              TEXT        NOT NULL,
    reporter_id              TEXT        NOT NULL,
    reporter_team            TEXT        NOT NULL,
    summary                  TEXT        NOT NULL DEFAULT '',
    severity                 TEXT        NULL,
    category                 TEXT        NULL,
    matched_prior_inc        TEXT        NULL,
    resolution               TEXT        NULL,
    -- nested-as-JSON (JSONB on pg, TEXT on sqlite, both via SQLAlchemy JSON):
    tags                     JSON        NOT NULL DEFAULT '[]',
    agents_run               JSON        NOT NULL DEFAULT '[]',
    tool_calls               JSON        NOT NULL DEFAULT '[]',
    findings                 JSON        NOT NULL DEFAULT '{}',
    pending_intervention     JSON        NULL,
    user_inputs              JSON        NOT NULL DEFAULT '[]',
    -- vector (1024 = bge-m3):
    embedding                VECTOR(1024)/BLOB NULL,
    -- denormalized for cheap list queries:
    input_tokens             INTEGER     NOT NULL DEFAULT 0,
    output_tokens            INTEGER     NOT NULL DEFAULT 0,
    total_tokens             INTEGER     NOT NULL DEFAULT 0
);
```

**Indexes**

- `incidents(status, environment) WHERE deleted_at IS NULL` — `find_similar` prefilter.
- `incidents(created_at DESC) WHERE deleted_at IS NULL` — `list_recent`.
- Postgres only: `USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)` once the corpus is large enough (>~1k rows). Skip until then.
- SQLite: no special vector index. `vec_distance_cosine()` over a prefiltered candidate set is fast enough at this scale.

**Why hybrid (scalars + JSON) and not normalized child tables?**

`agents_run`, `tool_calls`, and `findings` are always read whole alongside the parent incident. Normalizing into `agents_run` / `tool_calls` tables would force eager joins on every `load()` and complicate `save()` (delete-and-reinsert child rows on every update). JSON columns preserve the existing Pydantic models with no domain-layer change.

## Custom column types

Two SQLAlchemy `TypeDecorator`s in `storage/types.py`:

### `VectorColumn(dim: int)`

Single declarative column type that maps to:

- **Postgres** → `pgvector.sqlalchemy.Vector(dim)`. `bind_param` and `result_value` round-trip a Python `list[float]`.
- **SQLite** → `LargeBinary` storing `numpy.asarray(vec, dtype=np.float32).tobytes()`. `process_result_value` converts back via `np.frombuffer(b, dtype=np.float32).tolist()`.

The Python value is always `list[float] | None` regardless of dialect.

### `JSONColumn`

Thin shim around SQLAlchemy `JSON()` so we keep one symbol and can adjust serialization (e.g., for datetime objects) in one place. Dialect routing is automatic — `JSONB` on Postgres, `TEXT` on SQLite.

## Repository interface

`IncidentRepository` (in `storage/repository.py`) replaces `IncidentStore`. Same public methods, plus one:

```python
class IncidentRepository:
    def __init__(self, engine: Engine, *,
                 embedder: Embeddings | None = None,
                 similarity_threshold: float = 0.85,
                 severity_aliases: dict[str, str] | None = None) -> None: ...

    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> Incident: ...

    def load(self, incident_id: str) -> Incident: ...
    def save(self, incident: Incident) -> None: ...
    def delete(self, incident_id: str) -> Incident: ...           # soft delete
    def list_all(self, *, include_deleted: bool = False) -> list[Incident]: ...
    def list_recent(self, limit: int = 20,
                    *, include_deleted: bool = False) -> list[Incident]: ...

    # New:
    def find_similar(self, *, query: str, environment: str,
                     status_filter: str = "resolved",
                     threshold: float | None = None,
                     limit: int = 5) -> list[tuple[Incident, float]]: ...
```

**Pydantic ↔ row mapping**

`save()` serializes nested fields via `model.model_dump()` (Pydantic v2). `load()` validates via `Incident.model_validate(row_dict)`. We don't use SQLAlchemy ORM relationships for nested data — JSON columns handle it. Scalar columns map 1:1 to Pydantic fields.

**Embedding lifecycle inside the repository**

- `create(query=...)`: if `self.embedder` is set, `vec = embedder.embed_query(query)` and persist on insert.
- `save(inc)`: re-embed when `query` or `summary` changes from the on-disk row. To detect change, `save()` reads the existing row's `query` + `summary` columns first (one round-trip), compares, and embeds only on diff. Cheap and avoids re-embed on every status flip.
- `find_similar(query=...)`: `vec = embedder.embed_query(query)`. KNN dispatched per dialect (see below). Result mapped back to `(Incident, score)` tuples.

**Similarity dispatch (the only dialect-aware code)**

```python
def find_similar(self, *, query, environment, ...) -> list[tuple[Incident, float]]:
    if self.embedder is None:
        # Fallback path for keyword similarity_method.
        return self._keyword_similar(query=query, environment=environment, ...)
    vec = self.embedder.embed_query(query)
    if self.engine.dialect.name == "postgresql":
        # pgvector: 1 - (embedding <=> :vec)  →  cosine similarity in [-1, 1]
        score_col = (literal(1.0) - IncidentRow.embedding.cosine_distance(vec))
    else:
        # sqlite-vec: 1 - vec_distance_cosine(embedding, :vec)
        score_col = literal(1.0) - func.vec_distance_cosine(IncidentRow.embedding, _vec_to_blob(vec))
    stmt = (
        select(IncidentRow, score_col.label("score"))
        .where(IncidentRow.deleted_at.is_(None))
        .where(IncidentRow.status == status_filter)
        .where(IncidentRow.environment == environment)
        .where(IncidentRow.embedding.is_not(None))
        .order_by(desc("score"))
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    threshold = threshold if threshold is not None else self.similarity_threshold
    return [(self._row_to_incident(r), s) for r, s in rows if s >= threshold]
```

`_vec_to_blob` is the SQLite-side serializer (numpy `tobytes`). On Postgres the vec literal goes through `pgvector`'s built-in adapter.

`_keyword_similar` keeps the existing `find_similar(query=..., scorer=KeywordSimilarity())` path — preserves backwards compatibility for `similarity_method = "keyword"` and works fully offline without an embedder.

## Embedding facade

`storage/embeddings.py` exposes one factory:

```python
def build_embedder(cfg: EmbeddingConfig | None,
                   providers: dict[str, ProviderConfig]) -> Embeddings | None:
    if cfg is None:
        return None
    p = providers[cfg.provider]
    if p.kind == "ollama":
        return OllamaEmbeddings(model=cfg.model, base_url=p.base_url)
    if p.kind == "azure_openai":
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

`_StubEmbeddings` returns deterministic vectors (hash → seed → `np.random.rand(dim)`). Used for tests and `similarity_method = "keyword"` paths where an embedder is technically still constructed.

`cfg.llm.embedding` already exists; we add a `dim: int = 1024` field for stub-embedder sizing and as a sanity-check assertion against the live model.

## Engine + sqlite-vec extension loading

`storage/engine.py` builds the SQLAlchemy engine from `cfg.storage`:

```python
def build_engine(cfg: StorageConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(cfg.url, poolclass=NullPool, echo=cfg.echo,
                               connect_args={"check_same_thread": False})
        @event.listens_for(engine, "connect")
        def _load_sqlite_vec(dbapi_conn, _):
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)
        return engine
    return create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
```

`pgvector` doesn't need a `connect` hook — it's a server-side extension assumed installed. Engine builder runs `CREATE EXTENSION IF NOT EXISTS vector` once on startup for Postgres deployments.

## Config

New `StorageConfig` in `config.py`:

```python
class StorageConfig(BaseModel):
    url: str = "sqlite:///incidents.db"
    pool_size: int = 5     # postgres only
    echo: bool = False
```

`AppConfig.storage: StorageConfig` field added with default. `IncidentConfig.store_path` is repurposed as the JSON-backfill source path (used only by the migration script). `EmbeddingConfig.dim: int = 1024` added.

Example config:

```yaml
storage:
  url: "sqlite:///incidents.db"
llm:
  embedding:
    provider: workhorse-ollama
    model: bge-m3
    dim: 1024
incidents:
  similarity_method: embedding
```

## Migration from JSON files

One-shot script `scripts/migrate_jsonl_to_sql.py`:

```
usage: migrate_jsonl_to_sql.py [--config PATH] [--with-embeddings] [--dry-run]
```

Flow:

1. Load `AppConfig`, build engine + repository (with embedder if `--with-embeddings`).
2. Walk `cfg.incidents.store_path` for `*.json` files.
3. Validate each via `Incident.model_validate(json.loads(...))`.
4. Skip if `repo.load(inc.id)` succeeds (idempotent).
5. Insert via raw SQL (not `repo.create`, which would mint a fresh id and run embed on `query` only). Bypass for backfill: write the row as-is, then if `--with-embeddings`, embed `query + summary + " ".join(tags)` and update.
6. Print summary: `inserted=N skipped=M failed=K`.

The script is committed but never auto-run; ops triggers it explicitly.

## Test strategy

- **Default test backend: in-memory SQLite + stub embedder.** All 157 existing tests run unchanged after `IncidentStore` references are migrated to `IncidentRepository`.
- **Postgres parity tests:** parameterized over `[sqlite_engine, postgres_engine]` fixtures. Postgres engine constructed only when `POSTGRES_TEST_URL` env var is set; otherwise the parameter is skipped.
- **Vector-ops parity:** seed corpus of 5 incidents, compute embeddings via stub, assert top-3 ids are identical between SQLite and Postgres (when both are runnable).
- **Migration script:** fixture directory of 3 JSON incidents (subset of real ones), run the script in `--dry-run` and live mode, assert row count + key field equality.
- **Real Ollama smoke test:** `tests/test_embeddings_ollama.py`, marked `@pytest.mark.integration`, skipped unless `OLLAMA_BASE_URL` is reachable.
- **Lifecycle test:** create → embed → save → find_similar → delete (soft) → list (excluded by default).

## File structure

**New**
- `src/orchestrator/storage/__init__.py` — public exports
- `src/orchestrator/storage/types.py` — `VectorColumn`, `JSONColumn`
- `src/orchestrator/storage/models.py` — `Base`, `IncidentRow`
- `src/orchestrator/storage/engine.py` — `build_engine` + sqlite-vec loader
- `src/orchestrator/storage/repository.py` — `IncidentRepository`
- `src/orchestrator/storage/embeddings.py` — `build_embedder`, `_StubEmbeddings`
- `scripts/migrate_jsonl_to_sql.py` — backfill
- `tests/test_storage_engine.py`
- `tests/test_storage_repository.py`
- `tests/test_storage_vector_parity.py` (parameterized over engines)
- `tests/test_storage_migration.py`

**Modified**
- `src/orchestrator/config.py` — add `StorageConfig`, `EmbeddingConfig.dim`
- `src/orchestrator/orchestrator.py` — build engine + repository, pass to MCP server `configure()`
- `src/orchestrator/mcp_servers/incident.py` — replace `_require_store` with `_require_repo`; `configure(*, repository, ...)`; tools delegate to repository
- `src/orchestrator/incident.py` — keep `Incident` Pydantic model; remove `IncidentStore` (or leave as deprecated alias for one cycle if call-sites migrate gradually — but the spec preference is rename + cut)
- `scripts/build_single_file.py` — extend `CORE_MODULE_ORDER` to include `storage/types.py`, `storage/models.py`, `storage/engine.py`, `storage/embeddings.py`, `storage/repository.py` (in dep order), and the new module imports
- `config/config.yaml` — add `storage:` block, set `similarity_method: embedding`, `embedding.dim: 1024`
- `tests/test_incident_*.py` — replace `IncidentStore` references with `IncidentRepository`

## Dependencies

New (vendored for air-gapped):

- `sqlalchemy>=2.0` — ORM core
- `sqlite-vec` — pure-pip wheel; ships the C extension; loaded per-connection
- `pgvector` — Python adapter (`from pgvector.sqlalchemy import Vector`)
- `psycopg[binary]` — Postgres driver (binary keeps install simple; can switch to `psycopg-c` later)
- `langchain-ollama` — verify if not already pulled transitively; the chat side already uses Ollama
- `numpy` — vector serialization (likely already transitive via langchain)

Verify via `pip install` round-trip on a clean venv before merge.

## Risk + open follow-ups (documented, not blocking)

- **Embedding-dim drift on provider switch:** if we move from Ollama bge-m3 (1024) to Azure text-embedding-3-large (3072), all rows must re-embed. Mitigation: a `re-embed` ops command that walks all rows. Out of scope for this spec.
- **Soft-delete semantics across `find_similar`:** `deleted_at IS NULL` filter is on the prefilter SQL. Tested explicitly.
- **Postgres-extension install in air-gapped:** `pgvector` install path must be documented in `rules/build.md` follow-up; image pinning by digest. Not in scope for this spec.
- **Bundle build:** the `scripts/build_single_file.py` extension must list the new storage modules in dep order. Verify the dist bundle imports cleanly post-bundle as part of plan-phase smoke testing.
- **Findings-by-agent JSON shape:** the existing `inc.findings: dict[str, Any]` is portable as-is; no schema lock needed yet.

## Out of scope

- Async repository, async sessions, async tests.
- Vector index tuning (IVFFlat lists, HNSW). Defer until corpus > ~1k rows.
- Cross-environment similarity (today similarity prefilters by environment; intentional).
- Multi-embedding-per-incident (e.g., separate query / summary / resolution vectors).
- Re-embed-all CLI command.
- HTTP API surface changes — purely an internals swap.
