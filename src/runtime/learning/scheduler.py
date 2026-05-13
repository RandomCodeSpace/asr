"""M7: nightly batch refresher for the lesson corpus.

Runs an APScheduler ``AsyncIOScheduler`` that fires on
:attr:`FrameworkAppConfig.lesson_refresh_cron` (default ``0 3 * * *`` —
03:00 UTC daily). On each tick it walks the recently-terminated
sessions inside the configured window, dispatches
:class:`LessonExtractor.extract` for any that don't already have a
current-version lesson row, and persists the result via the existing
:class:`LessonStore`.

Idempotency contract: rerunning :meth:`run_once` after a previous
successful pass produces zero new rows (the source_session_id +
``provenance.extractor_version`` pair is unique-by-content). When the
extractor version bumps in a future release, the refresher writes a
fresh row — older lessons stay queryable (append-only corpus).

Tests drive the refresher synchronously via :meth:`run_once`; the
cron loop only exists to fire ``run_once`` on a schedule.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlaSession

from runtime.learning.extractor import EXTRACTOR_VERSION, LessonExtractor
from runtime.storage.event_log import EventLog
from runtime.storage.lesson_store import LessonStore
from runtime.storage.models import IncidentRow, SessionLessonRow

_log = logging.getLogger("runtime.learning.scheduler")


@dataclass
class RefreshStats:
    """Outcome of a single :meth:`LessonRefresher.run_once` invocation."""

    sessions_scanned: int = 0
    lessons_added: int = 0
    lessons_skipped: int = 0


class LessonRefresher:
    """Nightly refresher for the lesson corpus.

    Constructor wires the three collaborators (engine, lesson_store,
    event_log) so the cron tick can run without touching the global
    orchestrator. Mirrors the
    :class:`runtime.tools.approval_watchdog.ApprovalWatchdog`
    start/stop shape: ``start(loop)`` is idempotent and returns
    immediately; ``stop()`` is a graceful shutdown.

    The actual work happens in :meth:`run_once`, which tests call
    synchronously. The APScheduler-driven cron job is a thin wrapper
    around the same method.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        lesson_store: LessonStore,
        event_log: EventLog,
        terminal_statuses: frozenset[str],
        cron: str = "0 3 * * *",
        window_days: int = 7,
    ) -> None:
        self.engine = engine
        self.lesson_store = lesson_store
        self.event_log = event_log
        self.terminal_statuses = terminal_statuses
        self.cron = cron
        self.window_days = window_days
        self._scheduler: Optional[object] = None
        # Mirror of ApprovalWatchdog's idempotency flag.
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Scheduler lifecycle (cron entry point).
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start an :class:`AsyncIOScheduler` on ``loop`` that fires
        :meth:`run_once` per :attr:`cron`. Idempotent — a second call
        with the same instance returns immediately.

        Called from ``OrchestratorService.start()`` on the service's
        background loop.
        """
        if self._scheduler is not None:
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        async def _arm() -> None:
            self._stopped = False
            scheduler = AsyncIOScheduler(timezone="UTC", event_loop=loop)
            trigger = CronTrigger.from_crontab(self.cron, timezone="UTC")
            scheduler.add_job(
                self._run_once_async,
                trigger=trigger,
                id="lesson_refresher",
                replace_existing=True,
            )
            scheduler.start()
            self._scheduler = scheduler

        fut = asyncio.run_coroutine_threadsafe(_arm(), loop)
        fut.result(timeout=5.0)

    async def stop(self) -> None:
        """Shut the scheduler down. Idempotent and safe to call before
        :meth:`start` or after a previous :meth:`stop`."""
        if self._stopped:
            return
        self._stopped = True
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is None:
            return
        try:
            # AsyncIOScheduler.shutdown is sync but the underlying job
            # cleanup happens on the loop.
            scheduler.shutdown(wait=False)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:  # noqa: BLE001
            _log.warning(
                "LessonRefresher.stop: scheduler shutdown raised",
                exc_info=True,
            )

    async def close(self) -> None:
        """Alias for :meth:`stop`. Provided so callers using
        ``async with`` patterns read naturally."""
        await self.stop()

    # ------------------------------------------------------------------
    # Work — the cron tick + synchronous test entry point.
    # ------------------------------------------------------------------

    async def _run_once_async(self) -> RefreshStats:
        """APScheduler-callable wrapper around :meth:`run_once`."""
        return self.run_once()

    def run_once(self) -> RefreshStats:
        """One refresh pass.

        Walks ``incidents`` for sessions whose ``status`` is in
        :attr:`terminal_statuses` and whose ``updated_at`` falls within
        the last :attr:`window_days`. For each session:

        * Skip if a SessionLessonRow with the current
          ``EXTRACTOR_VERSION`` already exists for ``source_session_id``.
        * Otherwise call :meth:`LessonExtractor.extract` and persist
          via :meth:`LessonStore.add`.

        Returns a :class:`RefreshStats` summary.
        """
        stats = RefreshStats()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.window_days)

        with SqlaSession(self.engine) as s:
            stmt = (
                select(IncidentRow)
                .where(IncidentRow.deleted_at.is_(None))
                .where(IncidentRow.updated_at >= cutoff)
            )
            for row in s.execute(stmt).scalars():
                if row.status not in self.terminal_statuses:
                    continue
                stats.sessions_scanned += 1
                if self._has_current_lesson(s, row.id):
                    stats.lessons_skipped += 1
                    continue
                try:
                    inc = self._row_to_session(row)
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "LessonRefresher: failed to hydrate session %s; skipping",
                        row.id, exc_info=True,
                    )
                    continue
                lesson = LessonExtractor.extract(
                    session=inc,
                    event_log=self.event_log,
                )
                if lesson is None:
                    continue
                try:
                    self.lesson_store.add(lesson)
                    stats.lessons_added += 1
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "LessonRefresher: lesson_store.add failed for %s; "
                        "row stays unwritten this pass",
                        row.id, exc_info=True,
                    )
        _log.info(
            "lesson refresher tick: scanned=%d added=%d skipped=%d",
            stats.sessions_scanned, stats.lessons_added, stats.lessons_skipped,
        )
        return stats

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _has_current_lesson(
        self, session: SqlaSession, source_session_id: str,
    ) -> bool:
        """True iff a lesson row with the CURRENT extractor_version
        already exists for ``source_session_id``. Older version rows
        do NOT block — the refresher rewrites when the version bumps.
        """
        stmt = (
            select(SessionLessonRow)
            .where(SessionLessonRow.source_session_id == source_session_id)
        )
        for row in session.execute(stmt).scalars():
            prov = row.provenance or {}
            if prov.get("extractor_version") == EXTRACTOR_VERSION:
                return True
        return False

    def _row_to_session(self, row: IncidentRow):
        """Hydrate a minimal :class:`runtime.state.Session` from a row.

        Reuses :class:`SessionStore`'s converter so the extractor sees
        the same shape it would in the orchestrator finalize hook.
        """
        from runtime.storage.session_store import SessionStore

        # ``state_cls=None`` lets the converter default to the bare
        # framework ``Session`` — the extractor only reads fields
        # declared on the base class (id, status, agents_run,
        # extra_fields, to_agent_input).
        converter = SessionStore(engine=self.engine)
        return converter._row_to_incident(row)
