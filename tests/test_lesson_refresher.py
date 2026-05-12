"""M7 — nightly LessonRefresher.

Three tests cover the contract:
- `test_run_once_refreshes_recent_lessons`: seed 3 terminal sessions
  inside the window, `run_once()` produces 3 lesson rows.
- `test_idempotent_on_unchanged`: a second `run_once()` with the same
  extractor version is a no-op (no duplicate rows).
- `test_scheduler_starts_and_stops_cleanly`: `start(loop)` then
  `stop()` on the LessonRefresher mirrors the watchdog start/stop
  pattern without raising or leaving an APScheduler running.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as SqlaSession

from runtime.learning.scheduler import LessonRefresher
from runtime.state import AgentRun
from runtime.storage import EventLog, LessonStore, SessionLessonRow
from runtime.storage.models import Base, IncidentRow


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'refr.db'}")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def event_log(engine):
    return EventLog(engine=engine)


def _seed_terminal_session(engine, *, sid: str, status: str = "resolved") -> None:
    """Insert a terminal IncidentRow with a stub agent_run on it.
    LessonExtractor reads agents_run for confidence + summary, so we
    set them too.
    """
    now = datetime.now(timezone.utc)
    agent_run = AgentRun(
        agent="resolution",
        started_at="2026-05-12T00:00:00Z",
        ended_at="2026-05-12T00:00:05Z",
        summary=f"resolved {sid}",
        confidence=0.88,
        signal="success",
    )
    with SqlaSession(engine) as s:
        with s.begin():
            s.add(IncidentRow(
                id=sid,
                status=status,
                created_at=now,
                updated_at=now,
                query=f"q-{sid}",
                environment="dev",
                reporter_id="u", reporter_team="t",
                agents_run=[agent_run.model_dump()],
            ))


def _make_refresher(engine, event_log) -> LessonRefresher:
    """Build a refresher wired to an in-process LessonStore (no
    vector store) so run_once writes rows but skips embeddings."""
    store = LessonStore(engine=engine, vector_store=None)
    return LessonRefresher(
        engine=engine,
        lesson_store=store,
        event_log=event_log,
        terminal_statuses=frozenset({"resolved", "escalated"}),
        cron="0 3 * * *",
        window_days=7,
    )


# ===================================================================
# run_once writes rows
# ===================================================================

def test_run_once_refreshes_recent_lessons(engine, event_log):
    for sid in ("INC-A", "INC-B", "INC-C"):
        _seed_terminal_session(engine, sid=sid)

    refresher = _make_refresher(engine, event_log)
    stats = refresher.run_once()
    assert stats.sessions_scanned == 3
    assert stats.lessons_added == 3
    assert stats.lessons_skipped == 0

    with SqlaSession(engine) as s:
        rows = s.execute(select(SessionLessonRow)).scalars().all()
    assert len(rows) == 3
    assert {r.source_session_id for r in rows} == {"INC-A", "INC-B", "INC-C"}
    for r in rows:
        assert r.outcome_status == "resolved"
        assert r.provenance["extractor_version"] == "1"


# ===================================================================
# Idempotency: second run is a no-op
# ===================================================================

def test_idempotent_on_unchanged(engine, event_log):
    _seed_terminal_session(engine, sid="INC-1")
    refresher = _make_refresher(engine, event_log)

    first = refresher.run_once()
    assert first.lessons_added == 1
    assert first.lessons_skipped == 0

    second = refresher.run_once()
    assert second.lessons_added == 0, "second pass must NOT duplicate rows"
    assert second.lessons_skipped == 1
    assert second.sessions_scanned == 1

    with SqlaSession(engine) as s:
        rows = s.execute(select(SessionLessonRow)).scalars().all()
    assert len(rows) == 1, "no duplicate rows after second run_once"


# ===================================================================
# run_once skips non-terminal + out-of-window sessions
# ===================================================================

def test_run_once_skips_non_terminal(engine, event_log):
    _seed_terminal_session(engine, sid="INC-OPEN", status="in_progress")
    _seed_terminal_session(engine, sid="INC-DONE", status="resolved")
    refresher = _make_refresher(engine, event_log)
    stats = refresher.run_once()
    # Only the resolved row counts; in_progress is filtered.
    assert stats.sessions_scanned == 1
    assert stats.lessons_added == 1


# ===================================================================
# Scheduler start/stop lifecycle (APScheduler-driven)
# ===================================================================

def test_scheduler_starts_and_stops_cleanly(engine, event_log):
    """LessonRefresher.start(loop) arms the APScheduler; stop() shuts
    it down. Mirrors the ApprovalWatchdog lifecycle contract."""
    refresher = _make_refresher(engine, event_log)
    assert not refresher.is_running

    # Spin up an event loop on a background thread so we can mimic
    # the OrchestratorService boot pattern.
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, name="refresher_test_loop", daemon=True)
    t.start()
    try:
        refresher.start(loop)
        assert refresher.is_running

        # Idempotent: second start is a no-op.
        refresher.start(loop)
        assert refresher.is_running

        # Stop on the loop thread.
        fut = asyncio.run_coroutine_threadsafe(refresher.stop(), loop)
        fut.result(timeout=2.0)
        assert not refresher.is_running

        # Idempotent: second stop is a no-op (no exception).
        fut2 = asyncio.run_coroutine_threadsafe(refresher.stop(), loop)
        fut2.result(timeout=2.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        loop.close()
