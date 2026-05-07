"""Append-only session event log.

Events drive the status finalizer's inference (e.g. a registered
``<terminal_tool>`` event appearing in the log -> session reached
the corresponding terminal status). They are never mutated or
deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from runtime.storage.models import SessionEventRow


@dataclass(frozen=True)
class SessionEvent:
    """Immutable view of one row in the event log."""
    seq: int
    session_id: str
    kind: str
    payload: dict
    ts: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """Append-only log of session events.

    Events drive the status finalizer's inference (e.g. a registered
    ``<terminal_tool>`` event appearing in the log -> session reached
    the corresponding terminal status). They are never mutated or
    deleted.
    """

    def __init__(self, *, engine: Engine) -> None:
        self.engine = engine

    def append(self, session_id: str, kind: str, payload: dict) -> None:
        """Append a new event row. Never mutates existing rows."""
        with Session(self.engine) as s:
            with s.begin():
                s.add(SessionEventRow(
                    session_id=session_id,
                    kind=kind,
                    payload=dict(payload),
                    ts=_now(),
                ))

    def iter_for(self, session_id: str) -> Iterator[SessionEvent]:
        """Yield events for ``session_id`` in monotonic insertion order."""
        with Session(self.engine) as s:
            stmt = (
                select(SessionEventRow)
                .where(SessionEventRow.session_id == session_id)
                .order_by(SessionEventRow.seq)
            )
            for row in s.execute(stmt).scalars():
                yield SessionEvent(
                    seq=row.seq,
                    session_id=row.session_id,
                    kind=row.kind,
                    payload=row.payload,
                    ts=row.ts,
                )
