"""Garbage-collect orphaned LangGraph checkpoints.

When ``Orchestrator.retry_session`` rebinds a session to a new
``thread_id`` (e.g. ``INC-1:retry-1``), the original ``INC-1`` thread's
checkpoint becomes orphaned — no code path will ever resume it. Over
time these accumulate. ``gc_orphaned_checkpoints`` removes any
checkpoint whose ``thread_id`` does not reference an active session
(or a known retry suffix).

This is intentionally conservative: only checkpoints whose thread_id
prefix matches no live session row at all are removed.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError


def gc_orphaned_checkpoints(engine: Engine) -> int:
    """Remove orphaned checkpoint rows; return count removed.

    Returns 0 if the ``checkpoints`` table doesn't exist (fresh DB,
    LangGraph checkpointer has not yet bootstrapped its schema).
    """
    with engine.begin() as conn:
        live_ids = {row[0] for row in conn.execute(
            text("SELECT id FROM incidents")
        )}
        try:
            rows = conn.execute(text(
                "SELECT DISTINCT thread_id FROM checkpoints"
            )).all()
        except OperationalError:
            return 0
        # thread_id may be ``INC-1`` or ``INC-1:retry-N`` — strip suffix.
        orphans = []
        for (tid,) in rows:
            base = tid.split(":")[0] if tid else tid
            if base not in live_ids:
                orphans.append(tid)
        for tid in orphans:
            conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :tid"),
                {"tid": tid},
            )
        return len(orphans)
