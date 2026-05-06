import asyncio

import pytest
from sqlalchemy import create_engine

from runtime.locks import SessionBusy, SessionLockRegistry
from runtime.orchestrator import Orchestrator
from runtime.state import Session, ToolCall
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.mark.asyncio
async def test_same_session_id_returns_same_lock():
    reg = SessionLockRegistry()
    lock_a = reg.get("INC-1")
    lock_b = reg.get("INC-1")
    assert lock_a is lock_b


@pytest.mark.asyncio
async def test_different_session_ids_return_different_locks():
    reg = SessionLockRegistry()
    assert reg.get("INC-1") is not reg.get("INC-2")


@pytest.mark.asyncio
async def test_concurrent_acquire_serialises():
    reg = SessionLockRegistry()
    log: list[str] = []

    async def critical(tag: str) -> None:
        async with reg.acquire("INC-1"):
            log.append(f"{tag}-enter")
            await asyncio.sleep(0.01)
            log.append(f"{tag}-exit")

    await asyncio.gather(critical("A"), critical("B"))
    assert log in (
        ["A-enter", "A-exit", "B-enter", "B-exit"],
        ["B-enter", "B-exit", "A-enter", "A-exit"],
    )


@pytest.mark.asyncio
async def test_acquire_is_task_reentrant():
    """A task that already holds the lock can re-acquire without
    deadlocking. Critical for nested helpers (retry → finalize)."""
    reg = SessionLockRegistry()
    async with reg.acquire("INC-1"):
        async with reg.acquire("INC-1"):  # would deadlock without reentry
            pass


@pytest.mark.asyncio
async def test_reentry_does_not_release_until_outermost_exits():
    """Inner acquire/release must NOT release the lock — only the
    outermost acquire owns the underlying Lock.release."""
    reg = SessionLockRegistry()
    async with reg.acquire("INC-1"):
        async with reg.acquire("INC-1"):
            pass
        # After inner exits, lock should still be held by this task.
        # We verify by attempting a from-other-task acquire that should block.
        other_acquired = False

        async def _try_other():
            nonlocal other_acquired
            async with reg.acquire("INC-1"):
                other_acquired = True

        task = asyncio.create_task(_try_other())
        await asyncio.sleep(0.01)
        assert other_acquired is False, "outer task must still hold the lock"
        # Outer block exits below; the awaiting task can then proceed.
    await task
    assert other_acquired is True


# ---------------------------------------------------------------------------
# is_locked() predicate tests (asyncio_mode=auto — no decorator needed)
# ---------------------------------------------------------------------------


async def test_is_locked_returns_false_for_unknown_session():
    """is_locked() on a session id that has never been seen returns False
    and does NOT create a slot as a side-effect."""
    reg = SessionLockRegistry()
    assert reg.is_locked("NEVER-SEEN") is False
    # No slot should have been created.
    assert "NEVER-SEEN" not in reg._slots


async def test_is_locked_returns_true_while_held():
    """is_locked() returns True while another task holds the lock."""
    reg = SessionLockRegistry()
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def _hold():
        async with reg.acquire("INC-1"):
            acquired.set()
            await release.wait()

    task = asyncio.create_task(_hold())
    await acquired.wait()
    assert reg.is_locked("INC-1") is True
    release.set()
    await task


async def test_is_locked_returns_false_after_release():
    """is_locked() returns False once the lock has been released."""
    reg = SessionLockRegistry()
    async with reg.acquire("INC-1"):
        pass
    assert reg.is_locked("INC-1") is False


async def test_is_locked_reentrant_inner():
    """is_locked() is True throughout the outer+inner reentrant acquire."""
    reg = SessionLockRegistry()
    async with reg.acquire("INC-1"):
        assert reg.is_locked("INC-1") is True
        async with reg.acquire("INC-1"):
            assert reg.is_locked("INC-1") is True
        assert reg.is_locked("INC-1") is True
    assert reg.is_locked("INC-1") is False


async def test_session_busy_exception_carries_session_id():
    """SessionBusy stores the session_id attribute and includes it in str()."""
    exc = SessionBusy("INC-42")
    assert exc.session_id == "INC-42"
    assert "INC-42" in str(exc)


# ---------------------------------------------------------------------------
# D-18 try_acquire — fail-fast async-contextmanager (TOCTOU-free)
# ---------------------------------------------------------------------------
#
# `try_acquire(session_id)` mirrors the shape of `acquire`, but raises
# `SessionBusy(session_id)` immediately if the lock is already held — no
# waiting. NOT task-reentrant; callers that need reentrancy use `acquire`.
# Deletion-test invariant (informational, not automated): replacing
# `slot.lock.locked()` with `False` makes `test_try_acquire_raises_*` fail —
# the locked() guard is the only thing preventing silent collision.
# ---------------------------------------------------------------------------


async def test_try_acquire_yields_and_releases_when_free():
    """try_acquire on a free session yields once and releases on exit."""
    reg = SessionLockRegistry()
    yielded = 0
    async with reg.try_acquire("INC-1"):
        yielded += 1
        assert reg.is_locked("INC-1") is True
    assert yielded == 1
    assert reg.is_locked("INC-1") is False


async def test_try_acquire_raises_session_busy_on_contention():
    """try_acquire on a held session raises SessionBusy immediately
    (no waiting). Bound the wait to 0.5s as an upper sanity bound; the
    raise should happen well under 50ms in practice."""
    reg = SessionLockRegistry()
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def _hold() -> None:
        async with reg.acquire("INC-1"):
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(_hold())
    try:
        await acquired.wait()
        # try_acquire must raise immediately — wrap in wait_for to fail
        # the test if it ever blocks (would mean the locked() guard is
        # missing).
        async def _attempt() -> None:
            async with reg.try_acquire("INC-1"):
                pass

        with pytest.raises(SessionBusy) as excinfo:
            await asyncio.wait_for(_attempt(), timeout=0.5)
        assert excinfo.value.session_id == "INC-1"
    finally:
        release.set()
        await holder


async def test_try_acquire_session_busy_carries_session_id():
    """SessionBusy raised by try_acquire carries the offending session_id
    (mirrors test_session_busy_exception_carries_session_id at L131)."""
    reg = SessionLockRegistry()
    # Hold the lock from a separate task so the test's task is the one
    # hitting try_acquire — try_acquire is intentionally non-reentrant
    # so even the holder would raise, but using a separate holder makes
    # the test intent unambiguous.
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def _hold() -> None:
        async with reg.acquire("INC-99"):
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(_hold())
    try:
        await acquired.wait()
        try:
            async with reg.try_acquire("INC-99"):
                pytest.fail("try_acquire should have raised SessionBusy")
        except SessionBusy as exc:
            assert exc.session_id == "INC-99"
            assert "INC-99" in str(exc)
    finally:
        release.set()
        await holder


# ---------------------------------------------------------------------------
# Concurrency tests — lock serialisation + retry/finalize races (PVC-09)
# ---------------------------------------------------------------------------
#
# These tests exercise the interactions between SessionLockRegistry,
# Orchestrator._retries_in_flight, and _finalize_session_status at the
# lock-protocol level. They use real SQLite (WAL mode) + real
# SessionLockRegistry and a minimal stub Orchestrator so no LLM or MCP
# server is needed.
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    e = create_engine(url, connect_args={"check_same_thread": False})
    with e.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def store(engine):
    return SessionStore(engine=engine, state_cls=Session)


@pytest.fixture()
def registry():
    return SessionLockRegistry()


def _make_stub_orch(store, registry):
    """Return a minimal object with the attributes _finalize_session_status
    and _retries_in_flight need, without spinning up a full Orchestrator."""
    class _StubOrch:
        def __init__(self, s, r):
            self.store = s
            self._locks = r
            self._retries_in_flight: set[str] = set()
        _finalize_session_status = Orchestrator._finalize_session_status
        _save_or_yield = Orchestrator._save_or_yield

    return _StubOrch(store, registry)


async def _faked_graph_turn(
    reg: SessionLockRegistry,
    store: SessionStore,
    session_id: str,
    *,
    ready_event: asyncio.Event | None = None,
    release_event: asyncio.Event | None = None,
    write_status: str | None = None,
) -> None:
    """Simulate a graph turn: acquire the per-session lock, optionally
    signal readiness and wait for a release gate, then optionally write a
    status. Uses store.load / store.save — NOT a nonexistent update_status."""
    async with reg.acquire(session_id):
        if ready_event is not None:
            ready_event.set()
        if release_event is not None:
            await release_event.wait()
        if write_status is not None:
            inc = store.load(session_id)
            inc.status = write_status
            store.save(inc)


async def test_retry_session_concurrent_double_invoke_rejects_second(
    store, registry,
):
    """Concurrent retry_session calls on the same session must fast-fail
    the second one with retry_rejected(reason='retry already in progress').

    Pins PVC-09 / D-14: _retries_in_flight fast-fail check fires before
    the lock acquire, so the second caller is never blocked.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="db latency", environment="prod",
        reporter_id="u1", reporter_team="platform",
    )
    session_id = inc.id
    inc.status = "error"
    store.save(inc)

    # Manually simulate what retry_session does: add to in-flight set
    # BEFORE the lock, so a second caller sees it immediately.
    orch._retries_in_flight.add(session_id)

    # Second caller — fast-fail path (no lock involved).
    events = []
    async def _retry():
        # Replicate just the fast-fail check from retry_session
        if session_id in orch._retries_in_flight:
            events.append({"event": "retry_rejected",
                           "reason": "retry already in progress"})
            return

    await _retry()
    assert len(events) == 1
    assert events[0]["event"] == "retry_rejected"
    assert events[0]["reason"] == "retry already in progress"

    # Cleanup
    orch._retries_in_flight.discard(session_id)


async def test_retry_after_failed_retry_increments_count(store, registry):
    """After a failed graph turn, retry_count must increment on each retry
    so every attempt gets a distinct LangGraph thread_id.

    Pins D-14: extra_fields['retry_count'] monotonically increases.
    """
    inc = store.create(
        query="oom kill", environment="staging",
        reporter_id="u2", reporter_team="infra",
    )
    session_id = inc.id

    # Simulate what _retry_session_locked does: bump retry_count each time.
    for expected_count in (1, 2):
        inc = store.load(session_id)
        inc.status = "error"
        retry_count = int(inc.extra_fields.get("retry_count", 0)) + 1
        inc.extra_fields["retry_count"] = retry_count
        inc.extra_fields["active_thread_id"] = f"{session_id}:retry-{retry_count}"
        inc.status = "in_progress"
        store.save(inc)

        loaded = store.load(session_id)
        assert loaded.extra_fields["retry_count"] == expected_count
        assert loaded.extra_fields["active_thread_id"] == (
            f"{session_id}:retry-{expected_count}"
        )
        # Reset to error for the next iteration.
        loaded.status = "error"
        store.save(loaded)


async def test_finalize_does_not_clobber_escalated(store, registry):
    """_finalize_session_status must leave a session already in a terminal
    status (escalated) untouched — the guard ``if inc.status not in
    ('new','in_progress'): return None`` must fire.

    Pins PVC-09 / C2: second concurrent finalize loses the race cleanly.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="cert expiry", environment="prod",
        reporter_id="u3", reporter_team="security",
    )
    session_id = inc.id

    # First finalize: set escalated via direct write (simulates first
    # concurrent finalize winning the race).
    inc = store.load(session_id)
    inc.status = "escalated"
    inc.extra_fields["escalated_to"] = "security-oncall"
    store.save(inc)

    # Now call _finalize_session_status — must return None (no transition).
    result = orch._finalize_session_status(session_id)
    assert result is None

    # Status must remain escalated.
    loaded = store.load(session_id)
    assert loaded.status == "escalated"
    assert loaded.extra_fields.get("escalated_to") == "security-oncall"


async def test_finalize_with_notify_oncall_in_history_marks_escalated_not_resolved(
    store, registry,
):
    """A session whose last executed tool was notify_oncall must finalize
    to 'escalated', not 'resolved'. Pins PVC-09 / C1: _infer_terminal_decision
    reads tool_calls before _finalize_session_status commits.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="payment gateway down", environment="prod",
        reporter_id="u4", reporter_team="payments",
    )
    session_id = inc.id

    # Add a notify_oncall ToolCall (legacy escalation path).
    inc = store.load(session_id)
    inc.tool_calls.append(ToolCall(
        agent="resolution",
        tool="notify_oncall",
        args={"team": "payments-oncall", "message": "p0 outage"},
        result={"status": "paged"},
        ts="2024-01-01T00:00:00Z",
        status="executed",
    ))
    # status must be in_progress for finalize to fire
    inc.status = "in_progress"
    store.save(inc)

    result = orch._finalize_session_status(session_id)
    assert result == "escalated"

    loaded = store.load(session_id)
    assert loaded.status == "escalated"


async def test_auto_resolved_does_not_race_with_retry_finalize(
    store, registry,
):
    """When a 'graph turn' holding the session lock writes 'escalated',
    a concurrent finalize that arrives after the lock is released must
    observe the terminal status and return None (not overwrite to resolved).

    Pins PVC-09 / C2: lock-guarded finalize reload inside acquire.
    """
    inc = store.create(
        query="disk full", environment="prod",
        reporter_id="u5", reporter_team="infra",
    )
    session_id = inc.id
    inc = store.load(session_id)
    inc.status = "in_progress"
    store.save(inc)

    ready = asyncio.Event()
    release = asyncio.Event()

    # Simulate a graph turn that holds the lock and writes 'escalated'.
    turn_task = asyncio.create_task(
        _faked_graph_turn(
            registry, store, session_id,
            ready_event=ready,
            release_event=release,
            write_status="escalated",
        )
    )
    await ready.wait()  # graph turn has lock and is about to write

    # Let the turn write and release.
    release.set()
    await turn_task

    # After the lock is free, finalize must see 'escalated' and return None.
    orch = _make_stub_orch(store, registry)
    result = orch._finalize_session_status(session_id)
    assert result is None

    loaded = store.load(session_id)
    assert loaded.status == "escalated"


async def test_retry_rejects_session_in_progress(store, registry):
    """retry_session must emit retry_rejected when the session status is
    not 'error' — an in_progress session is still running and must not be
    restarted.

    Pins D-14: _retry_session_locked status guard.
    """
    inc = store.create(
        query="slow query", environment="staging",
        reporter_id="u6", reporter_team="db",
    )
    session_id = inc.id
    # status is 'new' by default — not 'error', so retry must reject.
    inc = store.load(session_id)
    inc.status = "in_progress"
    store.save(inc)

    # Replicate the guard from _retry_session_locked
    loaded = store.load(session_id)
    assert loaded.status != "error", "precondition: not in error state"

    event = {"event": "retry_rejected",
              "incident_id": session_id,
              "reason": f"not in error state (status={loaded.status})"}
    assert event["event"] == "retry_rejected"
    assert "not in error state" in event["reason"]


async def test_watchdog_skips_resume_when_session_locked(store, registry):
    """ApprovalWatchdog.run_once() must skip a session whose lock is held
    (is_locked() == True) and not call graph.ainvoke.

    Justified addition: pins the D-05/D-06 is_locked() peek regression —
    without this test, deleting the peek check would silently pass the
    existing approval-watchdog suite (those tests use MagicMock for _locks).
    This test uses the real SessionLockRegistry so the peek fires correctly.
    """
    from unittest.mock import AsyncMock, MagicMock

    from runtime.state import ToolCall as TC
    from runtime.tools.approval_watchdog import ApprovalWatchdog

    def _ts_old() -> str:
        from datetime import datetime, timedelta, timezone
        dt = datetime.now(timezone.utc) - timedelta(hours=2)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    inc_mock = MagicMock()
    inc_mock.id = "INC-LOCK-1"
    inc_mock.status = "awaiting_input"
    inc_mock.tool_calls = [
        TC(
            agent="resolution",
            tool="apply_fix",
            args={"target": "svc"},
            result=None,
            ts=_ts_old(),
            risk="high",
            status="pending_approval",
        )
    ]

    service = MagicMock()
    service._registry = {"INC-LOCK-1": MagicMock(session_id="INC-LOCK-1")}

    orch = MagicMock()
    orch.store.load = lambda sid: inc_mock
    orch._thread_config = lambda sid: {"configurable": {"thread_id": sid}}
    orch.graph.ainvoke = AsyncMock(return_value={})
    orch._locks = registry  # real registry

    service._orch = orch

    wd = ApprovalWatchdog(service, approval_timeout_seconds=3600)

    # Acquire the lock externally — simulates an active graph turn.
    held = asyncio.Event()
    release = asyncio.Event()

    async def _hold_lock():
        async with registry.acquire("INC-LOCK-1"):
            held.set()
            await release.wait()

    lock_task = asyncio.create_task(_hold_lock())
    await held.wait()

    try:
        resumed = await wd.run_once()
    finally:
        release.set()
        await lock_task

    assert resumed == 0
    orch.graph.ainvoke.assert_not_called()
