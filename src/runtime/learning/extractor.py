"""M5: lesson extractor — distills a terminal session's event log +
final session row into a :class:`SessionLessonRow` suitable for the
:class:`LessonStore` corpus.

Pure data-flow: walks ``event_log.iter_for(session.id)`` for tool calls,
reads ``session.agents_run`` for the final confidence + summary, and
composes a canonical ``embedding_text`` string the vector backend
embeds for retrieval. The same input session + event log always
produces the same ``embedding_text`` (modulo the ``created_at``
timestamp and uuid id) so M7's idempotency check can compare
``embedding_text`` to decide whether a re-extract is needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from runtime.state import Session
from runtime.storage.event_log import EventLog
from runtime.storage.models import SessionLessonRow


EXTRACTOR_VERSION = "1"


def _project_signals(session: Session) -> dict[str, Any]:
    """Carve a JSON-safe dict of categorical signals out of the
    session's ``extra_fields``. Used as the lesson row's queryable
    ``signals`` column — the intake runner can SQL-filter by these
    later.

    The framework is domain-neutral: every str / int / float /
    bool value in ``extra_fields`` becomes a signal. Apps that
    want richer filterability declare their state-class schema and
    the relevant keys flow through automatically.
    """
    extra = session.extra_fields or {}
    out: dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, (str, int, float, bool)) and v is not None:
            out[k] = v
    return out


def _project_tool_sequence(event_log: EventLog, session_id: str) -> list[dict]:
    """Walk the event log; produce a small ``[{tool, args_summary,
    result_kind}]`` list for every ``tool_invoked`` event in order."""
    seq: list[dict] = []
    for ev in event_log.iter_for(session_id):
        if ev.kind != "tool_invoked":
            continue
        seq.append({
            "tool": ev.payload.get("tool"),
            "args_summary": ev.payload.get("args", {}),
            "result_kind": ev.payload.get("result_kind"),
        })
    return seq


def _compose_embedding_text(
    session: Session,
    status: str,
    tool_sequence: list[dict],
    confidence_final: Optional[float],
) -> str:
    """Canonical embedding source. Same inputs -> identical string.

    Form: ``<to_agent_input>\\n\\nOutcome: <status>\\nKey tools:
    [<t1>, <t2>]\\nConfidence: <conf>``. Kept stable across releases
    so M7 can detect unchanged rows without re-embedding.
    """
    tools = [t.get("tool") for t in tool_sequence if t.get("tool")]
    return (
        f"{session.to_agent_input()}\n\n"
        f"Outcome: {status}\n"
        f"Key tools: {tools}\n"
        f"Confidence: {confidence_final}"
    )


class LessonExtractor:
    """Distills a terminal session into a :class:`SessionLessonRow`.

    Pure-function class — no I/O. The caller (orchestrator M4 hook or
    M7 batch refresher) is responsible for persisting the row via
    :class:`LessonStore.add` and emitting a ``lesson_extracted``
    event.
    """

    @staticmethod
    def extract(
        *,
        session: Session,
        event_log: EventLog,
        terminal_statuses: frozenset[str] | None = None,
    ) -> Optional[SessionLessonRow]:
        """Return a :class:`SessionLessonRow` for a terminal session,
        or ``None`` when the session is not in a terminal status.

        ``terminal_statuses`` is the configured terminal-status set
        (typically every name in ``cfg.orchestrator.statuses`` whose
        ``terminal=True``). When ``None``, no status check is applied
        and the extractor produces a row for any session — useful
        for tests that synthesise a pre-resolved session.
        """
        if terminal_statuses is not None and session.status not in terminal_statuses:
            return None

        tool_sequence = _project_tool_sequence(event_log, session.id)
        signals = _project_signals(session)
        confidence_final: Optional[float] = None
        outcome_summary = ""
        if session.agents_run:
            last_run = session.agents_run[-1]
            confidence_final = last_run.confidence
            outcome_summary = last_run.summary

        embedding_text = _compose_embedding_text(
            session,
            session.status,
            tool_sequence,
            confidence_final,
        )

        row = SessionLessonRow(
            id=str(uuid4()),
            source_session_id=session.id,
            created_at=datetime.now(timezone.utc),
            signals=signals,
            tool_sequence=tool_sequence,
            outcome_status=session.status,
            outcome_summary=outcome_summary,
            confidence_final=confidence_final,
            embedding_text=embedding_text,
            provenance={
                "kind": "auto",
                "model": "bge-m3",
                "extractor_version": EXTRACTOR_VERSION,
            },
        )
        # Emit the lesson_extracted event alongside the row so callers
        # need not duplicate the bookkeeping. Telemetry failures are
        # logged and dropped — the row is still returned.
        try:
            event_log.record(
                session.id, "lesson_extracted",
                lesson_id=row.id,
                outcome_status=row.outcome_status,
            )
        except Exception:  # noqa: BLE001 — telemetry must not block extraction
            import logging
            logging.getLogger("runtime.learning.extractor").debug(
                "event_log.record(lesson_extracted) failed", exc_info=True,
            )
        return row
