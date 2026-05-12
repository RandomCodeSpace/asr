"""Append-only session event log.

Events drive the status finalizer's inference (e.g. a registered
``<terminal_tool>`` event appearing in the log -> session reached
the corresponding terminal status). They are never mutated or
deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Literal, get_args

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from runtime.storage.models import SessionEventRow

# M2 (per-step telemetry): stable kind vocabulary for the event log.
# Adding a new kind without updating callers is intentional — but
# emitting a kind outside this Literal is a typo and raises at
# record() time so the typo doesn't silently pollute the log.
EventKind = Literal[
    "agent_started",
    "agent_finished",
    "tool_invoked",
    "confidence_emitted",
    "route_decided",
    "gate_fired",
    "status_changed",
    "lesson_extracted",
]

_VALID_EVENT_KINDS: frozenset[str] = frozenset(get_args(EventKind))


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

    def record(
        self,
        session_id: str,
        kind: EventKind,
        **payload: Any,
    ) -> None:
        """Convenience over ``append`` for the common kwargs shape.

        ``record(sid, "tool_invoked", tool="x", latency_ms=12)`` is
        equivalent to ``append(sid, "tool_invoked", {"tool": "x",
        "latency_ms": 12})`` but validates ``kind`` against the
        :data:`EventKind` Literal at call time — a typo is a hard
        failure, not a silently-malformed row.
        """
        if kind not in _VALID_EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {kind!r}; allowed: "
                f"{sorted(_VALID_EVENT_KINDS)}"
            )
        self.append(session_id, kind, payload)

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
