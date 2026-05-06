"""Pending-approval timeout watchdog.

A high-risk tool call enters ``langgraph.types.interrupt()`` and the
session sits in ``awaiting_input`` indefinitely. Without a watchdog
the slot leaks against ``OrchestratorService.max_concurrent_sessions``
forever — the cap eventually starves out new traffic.

The :class:`ApprovalWatchdog` is an asyncio task that runs on the
service's background loop. Every ``poll_interval_seconds`` it:

  1. Snapshots the in-flight session registry.
  2. For each session whose row has ``status="awaiting_input"``,
     scans ``tool_calls`` for entries with ``status="pending_approval"``
     whose ``ts`` is older than ``approval_timeout_seconds``.
  3. Resumes each such session via ``Command(resume={"decision":
     "timeout", "approver": "system", "rationale": "approval window
     expired"})``. The wrapped tool's resume path updates the audit
     row to ``status="timeout"``.

Failures during polling (DB hiccup, malformed row) are logged and
swallowed so a single bad session cannot kill the watchdog.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from runtime.locks import SessionBusy  # noqa: TCH001 — needed at runtime for except clause

if TYPE_CHECKING:
    from runtime.service import OrchestratorService

logger = logging.getLogger(__name__)

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Sessions whose status is in this set are *not* candidates for the
# watchdog — either they never paused for approval, or they have already
# moved past it. ``awaiting_input`` is the only status produced by
# ``langgraph.types.interrupt()`` while a high-risk gate is open.
_TERMINAL_STATUSES = frozenset({
    "resolved", "stopped", "escalated", "duplicate", "deleted", "error",
})


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 ``YYYY-MM-DDTHH:MM:SSZ`` ts back into UTC.

    Returns ``None`` for malformed values; callers treat that as
    "skip this row" so the watchdog never crashes on a bad audit
    record.
    """
    if not ts:
        return None
    try:
        # Replace trailing 'Z' so ``fromisoformat`` accepts it on
        # Python <3.11. The format is fixed by ``_UTC_TS_FMT`` so this
        # round-trips cleanly.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


class ApprovalWatchdog:
    """Background asyncio task that resumes stale pending-approval sessions.

    Owned by :class:`runtime.service.OrchestratorService`; started in
    ``OrchestratorService.start()`` and stopped in ``shutdown()``. The
    task runs on the service's background loop so it shares the same
    checkpointer / SQLite engine / FastMCP transports the live
    sessions are using.
    """

    def __init__(
        self,
        service: "OrchestratorService",
        *,
        approval_timeout_seconds: int,
        poll_interval_seconds: float = 60.0,
    ) -> None:
        self._service = service
        self._approval_timeout_seconds = approval_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule the watchdog onto ``loop``. Idempotent.

        Must be called from a thread that is not the loop's own thread —
        the typical caller is :meth:`OrchestratorService.start`. Returns
        immediately; the polling coroutine runs in the background.
        """
        if self._task is not None and not self._task.done():
            return

        async def _arm() -> None:
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(
                self._run(), name="approval_watchdog",
            )

        fut = asyncio.run_coroutine_threadsafe(_arm(), loop)
        fut.result(timeout=5.0)

    async def stop(self) -> None:
        """Signal the polling loop to exit and await termination.

        Runs on the loop thread (called from ``OrchestratorService._close_*``
        helpers). Idempotent — a no-op when the watchdog never started.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task  # LOCAL variable — guards against concurrent stop() calls
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                try:
                    await task  # drain LOCAL task ref; suppresses CancelledError
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._stop_event = None

    async def _run(self) -> None:
        """Polling loop. Runs until ``_stop_event`` is set."""
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("approval watchdog tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                # Expected — wakes the loop every ``poll_interval_seconds``.
                continue

    async def _tick(self) -> None:
        """One scan + resume pass. Visible for tests via ``run_once``."""
        await self.run_once()

    async def run_once(self) -> int:
        """Single scan pass. Returns the number of sessions resumed.

        Exposed publicly so tests can drive the watchdog
        deterministically without waiting on the polling cadence.
        """
        orch = getattr(self._service, "_orch", None)
        if orch is None:
            return 0
        registry = dict(self._service._registry)
        if not registry:
            return 0
        now = datetime.now(timezone.utc)
        resumed = 0
        for session_id in list(registry.keys()):
            try:
                inc = orch.store.load(session_id)
            except Exception:  # noqa: BLE001
                continue
            status = getattr(inc, "status", None)
            if status in _TERMINAL_STATUSES:
                continue
            if status != "awaiting_input":
                # Only sessions paused on a high-risk gate are watchdog
                # candidates. ``in_progress`` / ``new`` are still
                # actively running on the loop.
                continue
            stale = self._find_stale_pending(inc, now)
            if not stale:
                continue
            # No is_locked() peek here — try_acquire (inside
            # _resume_with_timeout) is the single contention check, so
            # there is no TOCTOU window between check and acquire. The
            # SessionBusy handler below fires on real contention.
            try:
                await self._resume_with_timeout(orch, session_id)
                resumed += 1
            except SessionBusy:
                logger.debug(
                    "approval watchdog: session %s SessionBusy at resume, skipping",
                    session_id,
                )
                continue
            except Exception:  # noqa: BLE001
                logger.exception(
                    "approval watchdog: resume failed for session %s",
                    session_id,
                )
        return resumed

    def _find_stale_pending(self, inc: Any, now: datetime) -> list[int]:
        """Return indices of ``pending_approval`` ToolCalls older than the
        configured timeout."""
        out: list[int] = []
        tool_calls = getattr(inc, "tool_calls", []) or []
        threshold = self._approval_timeout_seconds
        for idx, tc in enumerate(tool_calls):
            if getattr(tc, "status", None) != "pending_approval":
                continue
            ts = _parse_iso(getattr(tc, "ts", None))
            if ts is None:
                continue
            age = (now - ts).total_seconds()
            if age >= threshold:
                out.append(idx)
        return out

    async def _resume_with_timeout(
        self, orch: Any, session_id: str,
    ) -> None:
        """Resume the paused graph with a synthetic timeout decision.

        Uses ``Command(resume=...)`` against the same ``thread_id`` the
        approval API would use — the wrap_tool resume path updates the
        audit row to ``status="timeout"`` automatically.

        Per D-18: the ``ainvoke`` call is wrapped in
        ``orch._locks.try_acquire(session_id)`` so a concurrent user-
        driven turn cannot interleave checkpoint writes for the same
        ``thread_id``. If the lock is already held, ``try_acquire``
        raises ``SessionBusy`` immediately (no waiting); the caller
        (``run_once``) catches that and skips the tick — this is how
        the watchdog tolerates a busy session without piling up.
        """
        from langgraph.types import Command  # local: heavy import

        decision_payload = {
            "decision": "timeout",
            "approver": "system",
            "rationale": "approval window expired",
        }
        async with orch._locks.try_acquire(session_id):
            await orch.graph.ainvoke(
                Command(resume=decision_payload),
                config=orch._thread_config(session_id),
            )
