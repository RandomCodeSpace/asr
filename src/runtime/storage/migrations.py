"""Idempotent migrations for the JSON-shaped row payloads.

Fills the per-call audit fields on :class:`runtime.state.ToolCall` for
legacy rows. The risk-rated tool gateway uses five optional audit fields:

  * ``risk``          — ``"low" | "medium" | "high" | None``
  * ``status``        — ``ToolStatus`` literal (default ``"executed"``)
  * ``approver``      — operator id, set when status in {approved, rejected}
  * ``approved_at``   — ISO-8601 timestamp of the decision
  * ``approval_rationale`` — free-text justification

Older rows in the ``incidents.tool_calls`` JSON column lack these
fields. Pydantic hydrates the missing keys with their defaults at read
time so reading is already back-compat — but the on-disk JSON still
shows the legacy shape until something rewrites the row.

This migration walks every session, normalises the JSON-shaped
``tool_calls`` list to the current audit schema, and saves the row back
when (and only when) at least one entry changed. Idempotent — running
twice is safe (the second pass is a no-op because every row already
has the fields).

The function operates on the row's JSON list directly (not via the
``ToolCall`` Pydantic model) so we don't accidentally widen the
migration's contract — for example, dropping unknown extra keys via
Pydantic's ``extra='ignore'`` would silently delete forward-compat
fields in a downgrade scenario. JSON-walk is conservative: only fill
what's missing; leave everything else alone.
"""
from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlSession

from runtime.storage.models import IncidentRow

# Columns added after the initial schema. Each entry is
# ``(column_name, sql_type, default_clause_or_None)``. SQLite ``ADD
# COLUMN`` cannot add a non-nullable column without a constant default,
# so every entry here is nullable — Pydantic hydrates the missing keys
# at read time. Append-only: never reorder, never delete. Removing a
# column needs a separate destructive migration with explicit sign-off.
_FORWARD_COLUMNS: list[tuple[str, str]] = [
    ("parent_session_id", "VARCHAR"),  # dedup linkage
    ("dedup_rationale", "TEXT"),       # LLM rationale
    ("extra_fields", "JSON"),          # generic round-trip tunnel
]
_FORWARD_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, table, column) — mirrors models.IncidentRow.__table_args__.
    ("ix_incidents_parent_session_id", "incidents", "parent_session_id"),
]

# Default audit fields. Mirrors the Pydantic defaults on
# :class:`runtime.state.ToolCall`. Keep these in sync — a divergence
# means rows hydrated post-migration would carry different defaults
# than rows hydrated via the Pydantic constructor, which would surface
# as subtle test flakes long after the migration ran.
_AUDIT_DEFAULTS: dict[str, Any] = {
    "status": "executed",
    "risk": None,
    "approver": None,
    "approved_at": None,
    "approval_rationale": None,
}


def _fill_audit_fields(tc: dict[str, Any]) -> bool:
    """Mutate ``tc`` in place, filling any missing audit field with its
    default. Returns ``True`` when at least one key was added.

    Existing values (including explicit ``None`` already on the row)
    are left untouched — this is the idempotency guarantee.
    """
    changed = False
    for key, default in _AUDIT_DEFAULTS.items():
        if key not in tc:
            tc[key] = default
            changed = True
    return changed


def _normalise_tool_calls_list(
    tool_calls: Iterable[Any] | None,
) -> tuple[list[Any], bool]:
    """Walk a session's tool_calls JSON list, fill missing audit fields.

    Returns ``(new_list, changed)``. Non-dict entries (corrupt rows)
    are passed through unchanged — the migration is not a validator.
    """
    if not tool_calls:
        return [], False
    new: list[Any] = []
    changed = False
    for tc in tool_calls:
        if isinstance(tc, dict):
            # Copy so we don't mutate caller-owned data accidentally.
            tc_copy = dict(tc)
            if _fill_audit_fields(tc_copy):
                changed = True
            new.append(tc_copy)
        else:
            new.append(tc)
    return new, changed


def migrate_tool_calls_audit(engine: Engine) -> dict[str, int]:
    """Walk every session's ``tool_calls`` and fill missing audit fields.

    Idempotent — running on a freshly-migrated DB is a no-op.

    Returns a small stats dict::

        {"sessions_scanned": N, "sessions_updated": M, "rows_filled": K}

    where ``rows_filled`` is the count of individual ToolCall entries
    that received at least one default. Useful for ops dashboards and
    post-migration verification.
    """
    scanned = 0
    updated = 0
    filled = 0
    with SqlSession(engine) as session:
        rows = session.query(IncidentRow).all()
        for row in rows:
            scanned += 1
            new_list, changed = _normalise_tool_calls_list(row.tool_calls)
            if changed:
                # Count individual entries that gained at least one
                # field. Cheap re-walk — rows.tool_calls is already in
                # memory.
                for old, new in zip(row.tool_calls or [], new_list):
                    if isinstance(old, dict) and isinstance(new, dict):
                        if any(k not in old for k in _AUDIT_DEFAULTS):
                            filled += 1
                row.tool_calls = new_list
                updated += 1
        if updated:
            session.commit()
    return {
        "sessions_scanned": scanned,
        "sessions_updated": updated,
        "rows_filled": filled,
    }


def migrate_add_lesson_table(engine: Engine) -> dict[str, int]:
    """M5: create the ``session_lessons`` table if missing. Idempotent.

    Older databases predating M5 lack this table; we use
    ``Base.metadata.create_all`` scoped to the lesson table so the
    DDL is generated by SQLAlchemy (handles SQLite / Postgres / etc.)
    rather than handwritten ALTER statements. Running on a freshly-
    created database is a no-op (``create_all`` checks existence).

    Returns ``{"tables_added": N}``.
    """
    from runtime.storage.models import Base, SessionLessonRow

    inspector = inspect(engine)
    if "session_lessons" in inspector.get_table_names():
        return {"tables_added": 0}
    Base.metadata.create_all(
        engine,
        tables=[SessionLessonRow.__table__],  # pyright: ignore[reportArgumentType]
    )
    return {"tables_added": 1}


def migrate_add_session_columns(engine: Engine) -> dict[str, int]:
    """Add post-initial columns to ``incidents`` if missing. Idempotent.

    Older on-disk databases may lack ``extra_fields``,
    ``parent_session_id``, or ``dedup_rationale``; SQLAlchemy's read-side
    query then errors with ``no such column``. This walker uses
    ``PRAGMA table_info`` (via SQLAlchemy's ``inspect``) to detect
    missing columns and adds each one nullable. Running on a freshly-
    migrated DB is a no-op.

    Returns ``{"columns_added": N, "indexes_added": M}``.
    """
    inspector = inspect(engine)
    if "incidents" not in inspector.get_table_names():
        # Fresh DB; ``Base.metadata.create_all`` already produced the
        # full schema. Nothing to backfill.
        return {"columns_added": 0, "indexes_added": 0}
    existing_cols = {c["name"] for c in inspector.get_columns("incidents")}
    existing_idx = {i["name"] for i in inspector.get_indexes("incidents")}
    added_cols = 0
    added_idx = 0
    with engine.begin() as conn:
        for col, sql_type in _FORWARD_COLUMNS:
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE incidents ADD COLUMN {col} {sql_type}"))
                added_cols += 1
        for idx_name, table, col in _FORWARD_INDEXES:
            if idx_name in existing_idx:
                continue
            # If the column itself was just added (or already present)
            # the index is safe to create now.
            cols_after = {c["name"] for c in inspect(conn).get_columns(table)}
            if col in cols_after:
                conn.execute(text(f"CREATE INDEX {idx_name} ON {table} ({col})"))
                added_idx += 1
    return {"columns_added": added_cols, "indexes_added": added_idx}
