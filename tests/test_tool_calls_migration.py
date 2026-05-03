"""Tests for the idempotent migration that fills the per-call audit
fields on legacy ``tool_calls`` JSON rows.

The migration walks every session in the metadata DB, fills missing
audit fields (``status="executed"``, ``risk=None``, ...) on each
``ToolCall`` entry, and saves the row back. The 3 cases below cover:

  1. A legacy row (no audit fields) -> defaults are filled.
  2. A new row (already has audit fields) -> unchanged.
  3. A mixed batch -> both kinds correct + counters reflect activity.

Each test seeds the DB directly via raw inserts so we don't depend
on the live ``ToolCall`` Pydantic constructor (which would inject the
defaults *before* the migration runs and silently mask any regression).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session as SqlSession

from runtime.storage import migrate_tool_calls_audit
from runtime.storage.engine import build_engine
from runtime.storage.models import Base, IncidentRow
from runtime.config import MetadataConfig


def _make_engine(tmp_path):
    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(engine)
    return engine


def _seed_row(engine, *, session_id: str, tool_calls: list) -> None:
    """Insert a single IncidentRow with a hand-built tool_calls JSON
    payload so we control exactly which keys are present."""
    now = datetime.now(timezone.utc)
    with SqlSession(engine) as session:
        row = IncidentRow(
            id=session_id,
            status="resolved",
            created_at=now,
            updated_at=now,
            query="seeded",
            environment="dev",
            reporter_id="u",
            reporter_team="t",
            tags=[],
            agents_run=[],
            tool_calls=tool_calls,
            findings={},
            user_inputs=[],
        )
        session.add(row)
        session.commit()


def _load_tool_calls(engine, session_id: str) -> list:
    with SqlSession(engine) as session:
        row = session.get(IncidentRow, session_id)
        assert row is not None
        return list(row.tool_calls or [])


# ---------------------------------------------------------------------------
# Case 1 — legacy row gets audit defaults
# ---------------------------------------------------------------------------


def test_migration_fills_missing_audit_fields_on_legacy_row(tmp_path):
    """A pre-Phase-4 row (no ``status`` / ``risk``) must end up with
    ``status="executed"`` and ``risk=None`` after the migration."""
    engine = _make_engine(tmp_path)
    legacy = {
        "agent": "intake",
        "tool": "create_incident",
        "args": {"q": "x"},
        "result": {"id": "INC-1"},
        "ts": "2026-05-02T00:00:00Z",
    }
    _seed_row(engine, session_id="INC-20260502-001", tool_calls=[legacy])

    stats = migrate_tool_calls_audit(engine)

    assert stats["sessions_scanned"] == 1
    assert stats["sessions_updated"] == 1
    assert stats["rows_filled"] == 1

    rows = _load_tool_calls(engine, "INC-20260502-001")
    assert len(rows) == 1
    rec = rows[0]
    assert rec["status"] == "executed"
    assert rec["risk"] is None
    assert rec["approver"] is None
    assert rec["approved_at"] is None
    assert rec["approval_rationale"] is None
    # Existing fields survive untouched.
    assert rec["agent"] == "intake"
    assert rec["tool"] == "create_incident"
    assert rec["args"] == {"q": "x"}


# ---------------------------------------------------------------------------
# Case 2 — new row passes through unchanged
# ---------------------------------------------------------------------------


def test_migration_leaves_rows_with_audit_fields_unchanged(tmp_path):
    """A row already carrying the audit fields must be
    untouched. ``rows_filled`` is 0; the row dict is byte-equal."""
    engine = _make_engine(tmp_path)
    new_shape = {
        "agent": "resolution",
        "tool": "apply_fix",
        "args": {"target": "payments-svc"},
        "result": {"ok": True},
        "ts": "2026-05-02T00:01:00Z",
        "status": "approved",
        "risk": "high",
        "approver": "alice",
        "approved_at": "2026-05-02T00:01:05Z",
        "approval_rationale": "blast radius small",
    }
    _seed_row(engine, session_id="INC-20260502-002", tool_calls=[new_shape])

    stats = migrate_tool_calls_audit(engine)

    assert stats["sessions_scanned"] == 1
    # No update needed; sessions_updated is 0 since nothing changed.
    assert stats["sessions_updated"] == 0
    assert stats["rows_filled"] == 0

    rows = _load_tool_calls(engine, "INC-20260502-002")
    assert len(rows) == 1
    assert rows[0] == new_shape, (
        "Migration must be a no-op on rows that already have audit fields"
    )

    # Idempotency — second pass is also a no-op.
    stats2 = migrate_tool_calls_audit(engine)
    assert stats2["sessions_updated"] == 0
    assert stats2["rows_filled"] == 0


# ---------------------------------------------------------------------------
# Case 3 — mixed batch: legacy + new + multi-entry rows
# ---------------------------------------------------------------------------


def test_migration_handles_mixed_batch_correctly(tmp_path):
    """A DB containing legacy + already-migrated + multi-entry rows must
    end up uniformly audit-shaped, and the stats dict must report
    only the rows that genuinely changed."""
    engine = _make_engine(tmp_path)

    legacy = {
        "agent": "intake", "tool": "create_incident",
        "args": {}, "result": None, "ts": "2026-05-02T00:00:00Z",
    }
    new_shape = {
        "agent": "resolution", "tool": "apply_fix",
        "args": {}, "result": {"ok": True}, "ts": "2026-05-02T00:00:01Z",
        "status": "approved", "risk": "high",
        "approver": "alice", "approved_at": "2026-05-02T00:00:02Z",
        "approval_rationale": None,
    }
    # A row containing one legacy + one new entry — the migration must
    # touch both, the counter increments by 1 for the row + 1 for each
    # filled entry.
    mixed_legacy = {
        "agent": "triage", "tool": "lookup_similar_incidents",
        "args": {"q": "y"}, "result": [], "ts": "2026-05-02T00:01:00Z",
    }
    mixed_new = {
        "agent": "triage", "tool": "update_incident",
        "args": {"id": "INC-1", "patch": {"summary": "x"}},
        "result": {"ok": True}, "ts": "2026-05-02T00:01:30Z",
        "status": "executed_with_notify", "risk": "medium",
        "approver": None, "approved_at": None, "approval_rationale": None,
    }
    _seed_row(engine, session_id="INC-20260502-003", tool_calls=[legacy])
    _seed_row(engine, session_id="INC-20260502-004", tool_calls=[new_shape])
    _seed_row(engine, session_id="INC-20260502-005",
              tool_calls=[mixed_legacy, mixed_new])
    # A session with no tool calls at all — must scan but not update.
    _seed_row(engine, session_id="INC-20260502-006", tool_calls=[])

    stats = migrate_tool_calls_audit(engine)

    assert stats["sessions_scanned"] == 4
    # Three sessions had tool_calls; one was already migrated and one
    # was empty -> 2 sessions updated.
    assert stats["sessions_updated"] == 2
    # Two individual entries gained defaults: legacy + mixed_legacy.
    assert stats["rows_filled"] == 2

    # Spot-check each shape end-state.
    after_legacy = _load_tool_calls(engine, "INC-20260502-003")[0]
    assert after_legacy["status"] == "executed"
    assert after_legacy["risk"] is None

    after_new = _load_tool_calls(engine, "INC-20260502-004")[0]
    assert after_new == new_shape

    mixed_rows = _load_tool_calls(engine, "INC-20260502-005")
    assert len(mixed_rows) == 2
    assert mixed_rows[0]["status"] == "executed"  # legacy filled
    assert mixed_rows[1]["status"] == "executed_with_notify"  # already set

    empty_rows = _load_tool_calls(engine, "INC-20260502-006")
    assert empty_rows == []
