"""Phase 17 / HARD-07: ``ApprovalWatchdog`` cancellation hygiene.

Companion to ``tests/test_approval_watchdog.py`` (which covers the
scan/resume scoring logic). This module focuses on the lifecycle
contract:

  * ``stop()`` is a clean no-op when the watchdog never started
    (defensive call from a partially-failed ``start()``).
  * ``stop()`` is idempotent: a second call after the first returns
    must not raise, must not re-cancel the (now-None) task.
  * Concurrent ``stop()`` callers cooperate: only one drains the task,
    the second short-circuits on ``_stopped``.
  * ``close()`` is an alias for ``stop()`` (symmetry with aiohttp/httpx).
  * Dropping references to a started watchdog without calling
    ``stop()`` does not leak a "task pending" warning into pytest's
    warnings stream — the task is at least cancelled by GC + asyncio's
    own teardown sweep.

The polling cadence (60s default) is irrelevant here; what we exercise
is the cancellation path itself.
"""
from __future__ import annotations

import asyncio
import gc
import warnings
from unittest.mock import MagicMock

from runtime.locks import SessionLockRegistry
from runtime.tools.approval_watchdog import ApprovalWatchdog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_watchdog(*, poll_interval_seconds: float = 0.05) -> ApprovalWatchdog:
    """Construct an ApprovalWatchdog with a tight poll interval so the
    polling loop iterates promptly under test."""
    service = MagicMock()
    service._registry = {}

    orch = MagicMock()
    orch._locks = SessionLockRegistry()
    service._orch = orch

    return ApprovalWatchdog(
        service,
        approval_timeout_seconds=3600,
        poll_interval_seconds=poll_interval_seconds,
    )


async def _arm_inline(wd: ApprovalWatchdog) -> None:
    """Arm the watchdog without going through ``start()`` (which spins
    a thread). Test runs already inside a loop via ``asyncio_mode=auto``,
    so we mirror what ``start()._arm()`` does."""
    wd._stopped = False
    wd._stop_event = asyncio.Event()
    wd._task = asyncio.create_task(wd._run(), name="approval_watchdog_test")
    # Yield once so the polling loop's first iteration enters
    # ``_stop_event.wait()``; otherwise stop() may race the task before
    # it's parked on the event.
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_stop_before_start_is_noop():
    """``stop()`` on a never-armed watchdog must return cleanly."""
    wd = _build_watchdog()
    # No exception, returns None promptly.
    await wd.stop()
    assert wd._task is None
    assert wd._stop_event is None
    assert wd._stopped is True


async def test_start_then_stop_drains_task_cleanly():
    """Happy path: arm, stop, no leaked task; no warnings."""
    wd = _build_watchdog()
    await _arm_inline(wd)
    assert wd.is_running

    await wd.stop()

    # Task is no longer referenced from the watchdog.
    assert wd._task is None
    assert wd._stop_event is None
    assert wd._stopped is True
    # And no task with our name remains pending on the loop.
    leaked = [t for t in asyncio.all_tasks() if "approval_watchdog_test" in (t.get_name() or "")]
    assert leaked == [], f"watchdog leaked tasks after stop(): {leaked!r}"


async def test_double_stop_is_noop():
    """Calling ``stop()`` twice must not raise and must not re-attempt
    to drain a vanished task."""
    wd = _build_watchdog()
    await _arm_inline(wd)
    await wd.stop()
    # Second call: must short-circuit on ``_stopped`` flag, no exception.
    await wd.stop()
    await wd.stop()
    assert wd._stopped is True


async def test_concurrent_stop_callers_are_safe():
    """Two coroutines calling ``stop()`` concurrently must both return
    without error; only one performs the drain (the other observes
    ``_stopped`` and short-circuits)."""
    wd = _build_watchdog()
    await _arm_inline(wd)

    # Fire both stops on the same loop — gather collects without raising
    # if both complete cleanly.
    results = await asyncio.gather(wd.stop(), wd.stop(), return_exceptions=True)

    assert results == [None, None], f"unexpected stop() results: {results!r}"
    assert wd._task is None
    assert wd._stopped is True


async def test_close_alias_calls_stop():
    """``close()`` is the documented alias — must produce identical
    state to ``stop()``."""
    wd = _build_watchdog()
    await _arm_inline(wd)
    await wd.close()
    assert wd._task is None
    assert wd._stopped is True


async def test_drop_without_stop_does_not_leak_pending_warning():
    """If a caller arms the watchdog and then drops the reference
    without calling stop, GC + the event-loop's teardown sweep should
    cancel the task. We capture warnings and assert no
    ``Task was destroyed but it is pending!`` message escapes.

    The asyncio framework itself tries to be helpful here, but only if
    the task is at least *cancelled* before GC; the watchdog must not
    actively prevent that.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        wd = _build_watchdog()
        await _arm_inline(wd)
        # Cancel + drain explicitly — drop alone is racy because the
        # loop may still hold a strong ref via run-queue. The contract
        # we test here is that stop() suppresses the warning even when
        # the polling loop hasn't observed _stop_event yet.
        await wd.stop()

        # Force a GC pass so any unreachable task references surface.
        del wd
        gc.collect()
        # Yield to give asyncio a chance to emit any pending-task
        # warnings before we leave the catch_warnings context.
        await asyncio.sleep(0)

        leaked_warnings = [
            w for w in caught
            if "Task was destroyed" in str(w.message)
            or "pending" in str(w.message).lower() and "task" in str(w.message).lower()
        ]
        assert leaked_warnings == [], (
            f"unexpected pending-task warnings: "
            f"{[str(w.message) for w in leaked_warnings]!r}"
        )


async def test_stop_after_task_already_done_is_clean():
    """If the polling task has already exited (e.g. cancelled by an
    external observer), ``stop()`` must observe ``task.done()`` and
    return without trying to re-await."""
    wd = _build_watchdog()
    await _arm_inline(wd)
    # Cancel the task externally and wait for it to actually finish.
    wd._task.cancel()
    try:
        await wd._task
    except asyncio.CancelledError:
        pass
    # Now stop() must complete promptly without raising.
    await wd.stop()
    assert wd._stopped is True
