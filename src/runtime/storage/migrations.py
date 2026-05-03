"""Idempotent migrations for the JSON-shaped row payloads.

P4-M — fill the per-call audit fields added in P4-D for legacy rows.

The Phase-4 risk-rated tool gateway (P4-A..H) extended :class:`runtime.state.ToolCall`
with five optional audit fields:

  * ``risk``          — ``"low" | "medium" | "high" | None``
  * ``status``        — ``ToolStatus`` literal (default ``"executed"``)
  * ``approver``      — operator id, set when status in {approved, rejected}
  * ``approved_at``   — ISO-8601 timestamp of the decision
  * ``approval_rationale`` — free-text justification

Pre-Phase-4 rows in the ``incidents.tool_calls`` JSON column lack these
fields. Pydantic hydrates the missing keys with their defaults at read
time so reading is already back-compat — but the on-disk JSON still
shows the legacy shape until something rewrites the row.

This migration walks every session, normalises the JSON-shaped
``tool_calls`` list to the post-P4-D schema, and saves the row back
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

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlSession

from runtime.storage.models import IncidentRow

# Default audit fields. Mirrors the Pydantic defaults on
# :class:`runtime.state.ToolCall` (P4-D). Keep these in sync — a divergence
# means rows hydrated post-migration would carry different defaults than
# rows hydrated via the Pydantic constructor, which would surface as
# subtle test flakes long after the migration ran.
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
