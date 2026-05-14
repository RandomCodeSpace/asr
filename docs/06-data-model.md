# 06 — Data model

## Storage backends in use

| Concern | Backend | Default URL/path | Source |
|---|---|---|---|
| Session metadata | SQLAlchemy (SQLite default; Postgres optional via `asr[postgres]`) | `sqlite:////tmp/asr.db` | `src/runtime/storage/models.py`, `engine.py`, `session_store.py` |
| Vector similarity | FAISS (filesystem) | `/tmp/asr-faiss/` | `src/runtime/storage/vector.py`, `embeddings.py` |
| LangGraph checkpoints | `langgraph-checkpoint-sqlite` (default) or `langgraph-checkpoint-postgres` | Same SQLite DB as session metadata | `src/runtime/checkpointer.py` |
| Per-step events | SQLAlchemy `session_events` table | Same SQLite DB | `src/runtime/storage/event_log.py` |
| Lessons (auto-learning) | SQLAlchemy `session_lessons` table | Same SQLite DB | `src/runtime/storage/lesson_store.py` |
| Dedup retractions | SQLAlchemy `dedup_retractions` table | Same SQLite DB | `storage/session_store.py:un_duplicate` |
| Trigger idempotency keys | SQLAlchemy `trigger_idempotency_keys` table | Same SQLite DB | `src/runtime/triggers/idempotency.py` |
| Memory layers (incident_management) | Filesystem JSON / YAML | `incidents/{kg,releases,playbooks}/` (or seed bundle) | `examples/incident_management/asr/*_store.py` |

All SQLAlchemy concerns share the **same engine**
(`storage.metadata.url`). One DB, one connection pool, four
logical tables.

---

## Entities

### `IncidentRow` — primary table

Source: `src/runtime/storage/models.py`.

```python
class IncidentRow(Base):
    __tablename__ = "incidents"
    id: str                          # PK; format: "<PREFIX>-YYYYMMDD-NNN"
    status: str                      # new | in_progress | resolved | escalated |
                                     # needs_review | awaiting_input | error |
                                     # stopped | duplicate
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None      # soft delete
    query: str
    environment: str
    reporter_id: str                 # incident-shaped column; apps without
    reporter_team: str               # the concept ignore (round-trip omits)
    summary: str
    severity: str | None             # incident-shaped column
    category: str | None             # incident-shaped column
    matched_prior_inc: str | None    # FK to another row; dedup linkage
    resolution: str | None
    tags: list[str]                  # JSON
    agents_run: list[AgentRun]       # JSON; append-only audit
    tool_calls: list[ToolCall]       # JSON; append-only audit
    findings: dict[str, Any]         # JSON; per-agent finding bag
    pending_intervention: dict | None # JSON; gate node payload when paused
    user_inputs: list[str]           # JSON
    input_tokens: int                # accumulated TokenUsage
    output_tokens: int
    total_tokens: int
    parent_session_id: str | None    # dedup linkage to confirmed parent
    dedup_rationale: str | None      # stage-2 LLM rationale text
    extra_fields: dict[str, Any]     # JSON; per-app extension bag
    version: int                     # optimistic concurrency token
```

**Why so many incident-shaped columns?** History — the framework was
born incident-management-shaped. v1.1 (DEC-005) lifted the runtime
out of the incident shape, but renaming the schema columns would
have required a destructive migration. The columns are tolerated: an
app whose `Session` subclass doesn't declare `severity` or `reporter`
just leaves those columns NULL (round-trip silently omits them per
`_row_to_incident`).

The v1.5-B generic-noun pass (DEC-008) renamed local variables and
docstrings but **left the SQLAlchemy columns alone** — they would
require a migration. See `docs/DESIGN.md` § 8.2 for rationale.

### `EventRow` — per-step telemetry

Source: `src/runtime/storage/models.py`, `event_log.py`.

```python
class EventRow(Base):
    __tablename__ = "session_events"
    id: int                          # autoincrement
    session_id: str                  # FK to incidents.id
    kind: EventKind                  # tool_invoked | gate_fired |
                                     # agent_started | agent_finished |
                                     # confidence_emitted | route_decided |
                                     # status_changed | lesson_extracted | ...
    payload: dict                    # JSON; per-event shape
    ts: datetime
```

Append-only. Every meaningful boundary in the runtime emits a row.

### `LessonRow` — auto-learning corpus

Source: `src/runtime/storage/models.py`, `lesson_store.py`.

```python
class LessonRow(Base):
    __tablename__ = "session_lessons"
    id: int
    source_session_id: str           # FK to incidents.id
    title: str
    body: str                        # extracted narrative
    embedding: list[float] | None    # JSON; for similarity lookup
    metadata: dict                   # JSON
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None      # soft delete (intake's "still relevant?" gate)
```

Built by `LessonExtractor` at session finalize; refreshed nightly by
`LessonRefresher` for sessions resolved manually after the fact.

### `DedupRetractionRow` — operator un-duplicate audit

Source: `src/runtime/storage/models.py`, `session_store.py:un_duplicate`.

```python
class DedupRetractionRow(Base):
    __tablename__ = "dedup_retractions"
    id: int
    session_id: str
    original_match_id: str
    retracted_at: datetime
    retracted_by: str | None
    note: str | None
```

### `TriggerIdempotencyRow`

Source: `src/runtime/triggers/idempotency.py`.

```python
class TriggerIdempotencyRow(Base):
    __tablename__ = "trigger_idempotency_keys"
    trigger_name: str                # PK part 1
    key: str                         # PK part 2 (Idempotency-Key header)
    session_id: str                  # session minted by the original request
    created_at: datetime
```

Inference: rows expire opportunistically per `idempotency_ttl_hours`
on each trigger config.

---

## Pydantic models (in-memory; round-trip via `extra_fields`)

The `Session` base class (`src/runtime/state.py:70-117`) corresponds
roughly to the typed columns on `IncidentRow`. Apps subclass to add
domain fields:

```python
class IncidentState(Session):
    query: str
    environment: str
    reporter: Reporter
    summary: str
    tags: list[str]
    severity: str | None
    category: str | None
    matched_prior_inc: str | None
    resolution: Any
    memory: MemoryLayerState         # ASR memory bundle (read-only)

class CodeReviewState(Session):
    pr: PullRequest
    review_findings: list[ReviewFinding]
    overall_recommendation: Literal["approve", "request_changes", "comment"] | None
    review_summary: str
    review_token_budget: int
```

Round-trip pattern (`SessionStore._row_to_incident` /
`_incident_to_row_dict`):

- For each field declared on the state class:
  - If `IncidentRow` has a typed column for it → write to that column
  - Else → write to `extra_fields` JSON
- On load, fields with typed columns hydrate from those columns;
  everything else reads from `extra_fields[name]`.

This keeps row schema migrations rare — apps freely add domain
fields without touching the row schema.

---

## Relationships

```
incidents (PK: id)
    │
    ├──< session_events.session_id (one-to-many, append-only)
    │
    ├──< session_lessons.source_session_id (one-to-many, soft-deletable)
    │
    ├──< dedup_retractions.session_id (one-to-many)
    │
    ├──> incidents.parent_session_id (self-FK; dedup linkage)
    │
    └──> incidents.matched_prior_inc (self-FK; legacy linkage)

trigger_idempotency_keys (PK: trigger_name + key)
    │
    └──> incidents.id (loose ref; not enforced FK)

LangGraph checkpointer state
    └─ keyed by `configurable.thread_id`
       (= session_id by default; bumped to "<sid>:retry-N" on retry)
```

---

## Migrations

Source: `src/runtime/storage/migrations.py` (~210 lines).

The framework runs **idempotent JSON-walk migrations** at orchestrator
boot, not Alembic. Pre-existing rows get their new fields filled with
defaults so the audit history reads consistently after a schema
extension.

Two named migrations exist (Inference: based on tests +
`migrations.py` content):

- `migrate_tool_calls_audit` — added when Phase 4 introduced the
  risk-rated gateway audit fields (`risk`, `status`, `approver`,
  `approved_at`, `approval_rationale`). Walks every `tool_calls`
  JSON and fills missing audit fields with their pydantic defaults.
- `migrate_extra_fields` (Inference) — for the v1.1 decoupling
  (DEC-005) extension column.

There is no Alembic / SQLAlchemy migration framework — schema
changes are additive (new column, new table) and rely on
`Base.metadata.create_all(engine)` at boot for new tables. **Risk:
destructive schema changes (drop column, change type, rename)
require a hand-rolled migration script.**

---

## Persistence assumptions

- **Single writer per session** — enforced by `SessionLockRegistry`
  (`src/runtime/locks.py`); `SessionBusy` raised on contention.
- **Optimistic concurrency on save** — every `SessionStore.save`
  bumps `version` and rejects stale-version writes with
  `StaleVersionError`. Caller's contract is reload + retry.
- **Append-only audit logs** — `agents_run`, `tool_calls`,
  `session_events` are never updated in place (the gateway DOES
  update individual `tool_calls[idx]` for status transitions, but
  the rest of the row stays pristine).
- **Soft delete** — `deleted_at` column on `IncidentRow` and
  `LessonRow`. Hard delete is rare; the `delete_session` API is a
  soft delete + vector-store removal.
- **Dual write for pending intervention** — both LangGraph
  checkpoint AND `IncidentRow.pending_intervention` are written
  when a gate pauses, so dashboards reading the relational row
  stay accurate.
- **No cross-session transactions** — the framework doesn't model
  workflows that span multiple sessions (the `parent_session_id`
  link is the only inter-session reference, and it's a passive
  pointer).
- **Retry creates a new langgraph thread** — `Orchestrator.retry_session`
  bumps the `active_thread_id` (e.g. `INC-…:retry-2`); the
  original thread's checkpoint stays at the failed state so the
  retry runs fresh.

---

## Vector index

FAISS is the default (`vector.backend: faiss`); pgvector and "none"
are also supported (`src/runtime/storage/vector.py`). Vectors are
written through on every `SessionStore.save` so the index stays
aligned with the row table.

Index is keyed on `session_id`; each row carries a single embedding
of `_embed_source` (the session's query text, falling back to
`extra_fields["query"]`).

---

## Backup / restore

Inference: not formally documented. Practical recovery:

- **SQLite**: copy `/tmp/asr.db` (and `*-wal`, `*-shm` if mid-write).
- **FAISS**: copy `/tmp/asr-faiss/` directory.
- The two MUST be backed up together — a vector index pointing at
  rows that no longer exist will surface "ghost" similar-incidents
  matches. The reverse (rows without vectors) silently degrades
  similarity to "no matches".
