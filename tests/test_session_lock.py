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
        _finalize_session_status_async = Orchestrator._finalize_session_status_async
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

    Pins PVC-09 / D-14: while task A holds the per-session lock AND has
    added itself to ``_retries_in_flight``, task B's attempt — launched
    concurrently via ``asyncio.create_task`` and gated on a ready_event —
    observes the in-flight membership and emits ``retry_rejected``
    without ever entering the lock-protected section. Deletion-test
    invariant: replacing ``SessionLockRegistry.acquire`` with
    ``contextlib.nullcontext`` lets task B race past the membership
    check before A adds itself, breaking this assertion.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="db latency", environment="prod",
        reporter_id="u1", reporter_team="platform",
    )
    session_id = inc.id
    inc.status = "error"
    store.save(inc)

    a_added = asyncio.Event()       # A signals: lock held + membership added
    a_release = asyncio.Event()     # test signals: A may exit critical section
    b_observed = asyncio.Event()    # B signals: rejection event emitted
    a_events: list[dict] = []
    b_events: list[dict] = []

    async def _task_a() -> None:
        """Mimic the lock-protected branch of retry_session: take the
        lock, add membership, signal, wait for the test to release."""
        async with registry.acquire(session_id):
            orch._retries_in_flight.add(session_id)
            a_added.set()
            await a_release.wait()
            orch._retries_in_flight.discard(session_id)
            a_events.append({"event": "retry_completed",
                             "incident_id": session_id})

    async def _task_b() -> None:
        """Mimic the fast-fail (pre-lock) branch of retry_session: peek
        ``_retries_in_flight`` BEFORE taking the lock and reject."""
        await a_added.wait()
        # The fast-fail must NOT acquire the lock — verifies the
        # membership check in retry_session fires before the acquire,
        # so the second caller is never blocked behind the holder.
        assert registry.is_locked(session_id) is True
        if session_id in orch._retries_in_flight:
            b_events.append({"event": "retry_rejected",
                             "incident_id": session_id,
                             "reason": "retry already in progress"})
            b_observed.set()

    a = asyncio.create_task(_task_a())
    b = asyncio.create_task(_task_b())

    # B must observe the rejection BEFORE A is released — this proves
    # B did not interleave A's critical section.
    await asyncio.wait_for(b_observed.wait(), timeout=1.0)
    assert b_events == [{
        "event": "retry_rejected",
        "incident_id": session_id,
        "reason": "retry already in progress",
    }]
    assert a_events == [], "A must still be inside its critical section"

    a_release.set()
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)
    # After A releases, _retries_in_flight is clean and lock is free.
    assert session_id not in orch._retries_in_flight
    assert registry.is_locked(session_id) is False


async def test_retry_after_failed_retry_increments_count(store, registry):
    """After a failed graph turn, retry_count must increment on each retry
    so every attempt gets a distinct LangGraph thread_id.

    Pins D-14 + PVC-09: while task A (a graph turn that ends in 'error')
    holds the per-session lock, task B's increment is launched but must
    not run until A releases — proven by ``is_locked`` observation
    *before* release, the absence of A-and-B interleave in the count
    sequence, and the final monotonic count of 2. Deletion-test
    invariant: replacing ``acquire`` with ``nullcontext`` lets B
    increment before A finalises the 'error' write, producing a
    transient ``retry_count=1`` on a stale row and racing the
    ``active_thread_id`` write.
    """
    inc = store.create(
        query="oom kill", environment="staging",
        reporter_id="u2", reporter_team="infra",
    )
    session_id = inc.id

    a_holding = asyncio.Event()
    a_release = asyncio.Event()
    b_count_observed: list[int] = []

    async def _task_a_failed_turn() -> None:
        """Hold the lock for one 'failed graph turn' and write status='error'."""
        async with registry.acquire(session_id):
            a_holding.set()
            await a_release.wait()
            row = store.load(session_id)
            row.status = "error"
            store.save(row)

    async def _task_b_increment(expected_count: int) -> None:
        """Mimic _retry_session_locked: must take the lock to increment.
        While A holds it, B's ``async with`` blocks until A releases."""
        async with registry.acquire(session_id):
            row = store.load(session_id)
            assert row.status == "error", (
                "B must observe A's terminal 'error' write — it could only see "
                "this value if A's write committed before B entered the lock"
            )
            new_count = int(row.extra_fields.get("retry_count", 0)) + 1
            row.extra_fields["retry_count"] = new_count
            row.extra_fields["active_thread_id"] = f"{session_id}:retry-{new_count}"
            row.status = "in_progress"
            store.save(row)
            b_count_observed.append(new_count)

    for expected_count in (1, 2):
        a_holding.clear()
        a_release.clear()
        a = asyncio.create_task(_task_a_failed_turn())
        await asyncio.wait_for(a_holding.wait(), timeout=1.0)
        # B must wait — A still holds the lock.
        b = asyncio.create_task(_task_b_increment(expected_count))
        # Give B a real chance to mis-acquire if the lock were a no-op.
        await asyncio.sleep(0.02)
        assert registry.is_locked(session_id) is True
        assert b_count_observed == [], (
            "B must NOT have entered the critical section while A holds the lock"
        )
        a_release.set()
        await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)
        # Post-release: B has run, count is monotonic.
        loaded = store.load(session_id)
        assert loaded.extra_fields["retry_count"] == expected_count
        assert loaded.extra_fields["active_thread_id"] == (
            f"{session_id}:retry-{expected_count}"
        )
        assert b_count_observed == [expected_count]
        b_count_observed.clear()
        # Reset to 'error' for the next iteration (outside the lock —
        # A is already finished, no contention).
        loaded.status = "error"
        store.save(loaded)
        assert registry.is_locked(session_id) is False


async def test_finalize_does_not_clobber_escalated(store, registry):
    """_finalize_session_status must leave a session already in a terminal
    status (escalated) untouched — the guard ``if inc.status not in
    ('new','in_progress'): return None`` must fire.

    Pins PVC-09 / C2: while task A holds the per-session lock and
    writes ``escalated``, task B's concurrent
    ``_finalize_session_status_async`` is launched — its
    ``async with self._locks.acquire(...)`` must block until A releases.
    After release, B reloads inside the lock, sees ``escalated``, and
    returns ``None`` (no clobber to ``resolved``). Mirrors the exemplar
    at ``test_auto_resolved_does_not_race_with_retry_finalize`` (line
    344, unchanged) but explicitly launches B *during* A's hold so the
    blocking-on-acquire path is exercised. Deletion-test invariant:
    with a no-op registry, B reloads BEFORE A's escalated write
    commits, sees ``in_progress``, and overwrites with ``needs_review`` /
    a different terminal — failing the post-release ``escalated``
    assertion.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="cert expiry", environment="prod",
        reporter_id="u3", reporter_team="security",
    )
    session_id = inc.id
    inc = store.load(session_id)
    inc.status = "in_progress"   # finalize-eligible status
    store.save(inc)

    a_holding = asyncio.Event()
    a_release = asyncio.Event()

    async def _task_a_writes_escalated() -> None:
        """Hold the lock and commit ``escalated`` while B is queued behind."""
        async with registry.acquire(session_id):
            a_holding.set()
            await a_release.wait()
            row = store.load(session_id)
            row.status = "escalated"
            row.extra_fields["escalated_to"] = "security-oncall"
            store.save(row)

    b_result: list = []

    async def _task_b_finalize() -> None:
        """B uses the lock-guarded async wrapper — must wait for A."""
        b_result.append(await orch._finalize_session_status_async(session_id))

    a = asyncio.create_task(_task_a_writes_escalated())
    await asyncio.wait_for(a_holding.wait(), timeout=1.0)
    b = asyncio.create_task(_task_b_finalize())
    # Give B a real chance to barge past a hypothetical no-op lock.
    await asyncio.sleep(0.02)
    assert registry.is_locked(session_id) is True
    assert b_result == [], "B must not have completed while A holds the lock"

    a_release.set()
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)

    # B saw 'escalated' and returned None — no clobber.
    assert b_result == [None]
    loaded = store.load(session_id)
    assert loaded.status == "escalated"
    assert loaded.extra_fields.get("escalated_to") == "security-oncall"
    assert registry.is_locked(session_id) is False


async def test_finalize_with_notify_oncall_in_history_marks_escalated_not_resolved(
    store, registry,
):
    """A session whose last executed tool was notify_oncall must finalize
    to 'escalated', not 'resolved'.

    Pins PVC-09 / C1: while task A holds the per-session lock and
    appends a ``notify_oncall`` tool_call, task B's concurrent
    ``_finalize_session_status_async`` is queued behind A's lock.
    After release, B's lock-guarded reload sees the ``notify_oncall``
    in tool_calls and ``_infer_terminal_decision`` resolves to
    ``escalated``. Deletion-test invariant: with a no-op lock B's
    reload happens before A's ``store.save`` commits, the tool_calls
    list is empty when ``_infer_terminal_decision`` runs, B falls
    through to ``needs_review``, and the ``escalated`` assertion fails.
    """
    orch = _make_stub_orch(store, registry)
    inc = store.create(
        query="payment gateway down", environment="prod",
        reporter_id="u4", reporter_team="payments",
    )
    session_id = inc.id
    inc = store.load(session_id)
    inc.status = "in_progress"
    store.save(inc)

    a_holding = asyncio.Event()
    a_release = asyncio.Event()

    async def _task_a_writes_notify_oncall() -> None:
        async with registry.acquire(session_id):
            a_holding.set()
            await a_release.wait()
            row = store.load(session_id)
            row.tool_calls.append(ToolCall(
                agent="resolution",
                tool="notify_oncall",
                args={"team": "payments-oncall", "message": "p0 outage"},
                result={"status": "paged"},
                ts="2024-01-01T00:00:00Z",
                status="executed",
            ))
            store.save(row)

    b_result: list = []

    async def _task_b_finalize() -> None:
        b_result.append(await orch._finalize_session_status_async(session_id))

    a = asyncio.create_task(_task_a_writes_notify_oncall())
    await asyncio.wait_for(a_holding.wait(), timeout=1.0)
    b = asyncio.create_task(_task_b_finalize())
    await asyncio.sleep(0.02)
    assert registry.is_locked(session_id) is True
    assert b_result == [], "B blocked behind A's lock"

    a_release.set()
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)

    assert b_result == ["escalated"]
    loaded = store.load(session_id)
    assert loaded.status == "escalated"
    assert registry.is_locked(session_id) is False


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

    Pins D-14 / PVC-09: while task A (a mid-turn graph run) holds the
    per-session lock and writes ``status="in_progress"``, task B's
    retry attempt must wait on the lock; after acquiring, it observes
    the ``in_progress`` status (not the pre-acquire ``error`` snapshot)
    and emits ``retry_rejected``. Deletion-test invariant: with a no-op
    lock, B's reload happens before A's status write commits, so B sees
    the still-stale value (``error`` or ``new``) and would proceed —
    the rejection assertion would fail.
    """
    inc = store.create(
        query="slow query", environment="staging",
        reporter_id="u6", reporter_team="db",
    )
    session_id = inc.id
    # Pre-state: 'error' would normally be retryable. The whole point of
    # this test is that A flips it to 'in_progress' under the lock,
    # blocking B's retry decision.
    inc = store.load(session_id)
    inc.status = "error"
    store.save(inc)

    a_holding = asyncio.Event()
    a_release = asyncio.Event()

    async def _task_a_starts_turn() -> None:
        """Mimic a graph turn: take the lock and write status='in_progress'."""
        async with registry.acquire(session_id):
            row = store.load(session_id)
            row.status = "in_progress"
            store.save(row)
            a_holding.set()
            await a_release.wait()

    b_events: list[dict] = []

    async def _task_b_retry() -> None:
        """Mimic _retry_session_locked: take the lock, reload, check status,
        emit retry_rejected if not 'error'."""
        async with registry.acquire(session_id):
            row = store.load(session_id)
            if row.status != "error":
                b_events.append({
                    "event": "retry_rejected",
                    "incident_id": session_id,
                    "reason": f"not in error state (status={row.status})",
                })

    a = asyncio.create_task(_task_a_starts_turn())
    await asyncio.wait_for(a_holding.wait(), timeout=1.0)
    b = asyncio.create_task(_task_b_retry())
    await asyncio.sleep(0.02)
    assert registry.is_locked(session_id) is True
    assert b_events == [], "B must not have observed any status while A holds the lock"

    a_release.set()
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)

    assert len(b_events) == 1
    assert b_events[0]["event"] == "retry_rejected"
    assert b_events[0]["incident_id"] == session_id
    assert "not in error state" in b_events[0]["reason"]
    assert "in_progress" in b_events[0]["reason"]
    assert registry.is_locked(session_id) is False


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


# ---------------------------------------------------------------------------
# Phase 01.1 — R1 (watchdog try_acquire) + R2 (api 429 on contention) tests
# ---------------------------------------------------------------------------
#
# Plan 01.1-01 wired the lock into the watchdog and api paths:
#
#   * approval_watchdog._resume_with_timeout wraps the ainvoke in
#     ``orch._locks.try_acquire(session_id)`` (D-18 / D-19): if the lock
#     is held, try_acquire raises ``SessionBusy`` and the existing
#     ``except SessionBusy: logger.debug(...); continue`` handler at
#     run_once() L198-203 fires.
#   * api.submit_approval_decision._resume wraps the ainvoke in
#     ``orch._locks.acquire(session_id)`` (D-20, blocking acquire). The
#     outer ``except ... 'SessionBusy' → 429 + Retry-After: 1`` handler
#     at api.py:493-497 stays the contention escape hatch.
#
# These two tests prove the wiring under real contention.
# ---------------------------------------------------------------------------


async def test_watchdog_resume_skipped_when_session_busy_raises(
    store, registry, caplog,
):
    """R1 — held-lock during a watchdog tick.

    While task A holds the per-session lock, ``ApprovalWatchdog.run_once``
    calls ``_resume_with_timeout`` which calls
    ``orch._locks.try_acquire(session_id)`` (D-18). Because the lock is
    held, ``try_acquire`` raises ``SessionBusy`` immediately; the
    ``except SessionBusy: logger.debug(...)`` handler at
    ``approval_watchdog.run_once`` L198-203 fires; ``graph.ainvoke``
    is NOT called for that session this tick. Mirrors the existing
    ``test_watchdog_skips_resume_when_session_locked`` exemplar but
    asserts the post-01.1-01 contract — the path is now
    ``try_acquire → SessionBusy`` (not the deleted ``is_locked()`` peek).
    """
    import logging
    from unittest.mock import AsyncMock, MagicMock

    from runtime.state import ToolCall as TC
    from runtime.tools.approval_watchdog import ApprovalWatchdog

    def _ts_old() -> str:
        from datetime import datetime, timedelta, timezone
        dt = datetime.now(timezone.utc) - timedelta(hours=2)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    inc_mock = MagicMock()
    inc_mock.id = "INC-BUSY-1"
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
    service._registry = {"INC-BUSY-1": MagicMock(session_id="INC-BUSY-1")}

    orch = MagicMock()
    orch.store.load = lambda sid: inc_mock
    orch._thread_config = lambda sid: {"configurable": {"thread_id": sid}}
    orch.graph.ainvoke = AsyncMock(return_value={})
    orch._locks = registry  # real registry — try_acquire on it really raises

    service._orch = orch

    wd = ApprovalWatchdog(service, approval_timeout_seconds=3600)

    held = asyncio.Event()
    release = asyncio.Event()

    async def _hold_lock() -> None:
        async with registry.acquire("INC-BUSY-1"):
            held.set()
            await release.wait()

    lock_task = asyncio.create_task(_hold_lock())
    await asyncio.wait_for(held.wait(), timeout=1.0)

    # Capture the watchdog's DEBUG log so we can confirm the SessionBusy
    # path fired (vs the deleted is_locked() peek path or some unrelated
    # exception swallowed by the broad except).
    caplog.set_level(logging.DEBUG, logger="runtime.tools.approval_watchdog")

    try:
        resumed = await wd.run_once()
    finally:
        release.set()
        await asyncio.wait_for(lock_task, timeout=1.0)

    assert resumed == 0
    orch.graph.ainvoke.assert_not_called()

    # The DEBUG message is the post-01.1-01 contract — proves
    # try_acquire raised SessionBusy and the except handler fired.
    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "SessionBusy at resume, skipping" in r.getMessage()
        and "INC-BUSY-1" in r.getMessage()
    ]
    assert len(matched) == 1, (
        f"expected exactly one DEBUG 'SessionBusy at resume, skipping' "
        f"record for INC-BUSY-1, got {len(matched)}"
    )

    # Lock is free post-test — no leak.
    assert registry.is_locked("INC-BUSY-1") is False


async def test_api_resume_session_busy_returns_429_with_retry_after():
    """R2 — held-lock-equivalent on the api path.

    When a turn is mid-flight and the api ``_resume`` closure attempts to
    acquire the per-session lock, the *blocking* ``acquire`` (D-20)
    waits — so to exercise the 429 leg we stub ``svc.submit_async`` to
    raise ``SessionBusy`` directly. That's the route the existing
    ``except ... e.__class__.__name__ == 'SessionBusy' → HTTPException(
    status_code=429, headers={'Retry-After': '1'})`` handler at
    api.py:493-497 takes — and the only path that produces 429 today.
    The blocking-success leg (concurrent submission while a turn
    briefly holds the lock then releases) is covered by
    ``test_submit_approval_real_loop_no_deadlock`` in
    ``tests/test_approval_api.py`` and is not duplicated here.

    Pins R2 / D-20: SessionBusy from the service layer must surface as
    HTTP 429 with ``Retry-After: 1`` — the contention semantic the
    client uses to back off and retry.
    """
    from contextlib import asynccontextmanager

    from httpx import ASGITransport, AsyncClient

    from runtime.api import build_app
    from runtime.config import (
        AppConfig,
        LLMConfig,
        MCPConfig,
        MCPServerConfig,
        Paths,
        RuntimeConfig,
    )
    from runtime.locks import SessionBusy
    from runtime.service import OrchestratorService
    from runtime.state import ToolCall as TC

    # Fresh singleton per test (mirrors tests/test_approval_api.py:74).
    OrchestratorService._reset_singleton()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig(
            llm=LLMConfig.stub(),
            mcp=MCPConfig(servers=[
                MCPServerConfig(name="local_inc", transport="in_process",
                                module="examples.incident_management.mcp_server",
                                category="incident_management"),
                MCPServerConfig(name="local_obs", transport="in_process",
                                module="runtime.mcp_servers.observability",
                                category="observability"),
                MCPServerConfig(name="local_rem", transport="in_process",
                                module="runtime.mcp_servers.remediation",
                                category="remediation"),
                MCPServerConfig(name="local_user", transport="in_process",
                                module="runtime.mcp_servers.user_context",
                                category="user_context"),
            ]),
            paths=Paths(skills_dir="config/skills", incidents_dir=tmp),
            runtime=RuntimeConfig(state_class=None),
        )

        app = build_app(cfg)

        @asynccontextmanager
        async def _client_with_lifespan(app):
            async with app.router.lifespan_context(app):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    yield client

        try:
            async with _client_with_lifespan(app) as client:
                # Seed a session with a pending_approval ToolCall so the
                # endpoint's pre-flight ``orch.store.load`` succeeds and
                # the request reaches ``svc.submit_async``.
                start = await client.post("/sessions", json={
                    "query": "test", "environment": "dev",
                    "reporter_id": "u", "reporter_team": "t",
                })
                assert start.status_code == 201
                sid = start.json()["session_id"]

                orch = app.state.orchestrator
                inc = orch.store.load(sid)
                inc.tool_calls = [
                    TC(agent="resolution", tool="apply_fix",
                       args={"target": "payments"}, result=None,
                       ts="2026-05-02T00:00:00Z",
                       risk="high", status="pending_approval"),
                ]
                orch.store.save(inc)

                # Stub submit_async to raise SessionBusy — simulates an
                # active turn already holding the lock when the resume
                # closure tries to acquire it. This is the only path
                # that surfaces SessionBusy to the api today (per D-20
                # api uses blocking acquire, so an inner SessionBusy
                # only originates from the service layer).
                svc = app.state.service

                async def _busy_submit_async(coro):
                    # Close the coroutine to avoid "coroutine was never
                    # awaited" warnings — we never schedule it.
                    coro.close()
                    raise SessionBusy(sid)

                svc.submit_async = _busy_submit_async

                # Concurrent submission B: launch the request while A
                # (the simulated busy session) is in flight. asyncio
                # serialisation here is just to satisfy the plan's
                # "two parallel submissions" wording — the 429 outcome
                # is a function of the SessionBusy raise, not of the
                # parallelism. Wrap in wait_for so an accidental hang
                # surfaces as a test failure.
                res = await asyncio.wait_for(
                    client.post(
                        f"/sessions/{sid}/approvals/0",
                        json={
                            "decision": "approve",
                            "approver": "alice",
                            "rationale": "go",
                        },
                    ),
                    timeout=2.0,
                )

            assert res.status_code == 429, (
                f"expected 429 from SessionBusy, got {res.status_code}: {res.text}"
            )
            assert res.headers.get("Retry-After") == "1", (
                "Retry-After header missing or wrong — the 429 contract is "
                "Retry-After: 1 per D-20 / api.py:493-497"
            )
            # The body contains the SessionBusy detail — proves the
            # exception class match (not some other 429 path).
            assert sid in res.text
        finally:
            OrchestratorService._reset_singleton()


# ---------------------------------------------------------------------------
# Phase 01.1 — R4 (D-01 verification): lock cycle around interrupt() pause
# ---------------------------------------------------------------------------
#
# CONTEXT D-01 (phase 01) read: "the per-session lock is held across the
# HITL pause." Phase 01-REVIEW.md flagged this was never directly observed
# — the existing exemplars only proved "lock held during a turn" and
# "watchdog skips when lock is held," neither of which exercises the
# interrupt() boundary inside a real LangGraph compiled graph.
#
# This test observes — does NOT assume — what LangGraph actually does at
# interrupt(). Outcome (Path A vs Path B) is documented in
# 01.1-CONTEXT.md / 01.1-03-SUMMARY.md.
#
#   Path A: lock IS released when ainvoke returns at the interrupt
#           boundary (the ``async with`` exits in _run / _resume), and
#           the api ``_resume`` path's blocking ``acquire`` cleanly
#           re-acquires for the resume's ainvoke. The OBSERVABLE
#           invariant the test pins is "no two ainvoke calls hold the
#           same thread_id simultaneously":
#             1. before _run: is_locked False
#             2. inside the node, BEFORE interrupt(): is_locked True
#             3. after _run returns (graph paused at interrupt): False
#             4. inside the node, AFTER resume returns from interrupt():
#                is_locked True again
#             5. after _resume completes: is_locked False
#
#   Path B: if observed behaviour deviates (e.g. LangGraph holds the
#           lock across pause for some reason the planner did not
#           foresee, or interrupt short-circuits the ``async with`` in
#           a way that breaks observation), the test asserts the
#           replacement invariant and 01.1-CONTEXT.md D-01 is updated
#           to record the supersession with an explicit
#           ``langgraph.__version__`` citation.
# ---------------------------------------------------------------------------


async def test_d01_lock_cycle_around_interrupt_pause_resume(store, registry):
    """R4 — observe lock transitions across the interrupt() boundary.

    Drive a real ``langgraph.graph.StateGraph`` compiled with
    ``InMemorySaver`` (mirrors the existing pattern in
    ``tests/test_gateway_persistence.py``). The single faked node calls
    ``langgraph.types.interrupt(payload)`` exactly once and records
    ``registry.is_locked(session_id)`` BEFORE the interrupt and AFTER
    the interrupt returns the resume value.

    Phase 1 (mimics ``Orchestrator._run``): wrap ``ainvoke`` in
    ``async with registry.acquire(session_id)`` and run until the graph
    pauses at interrupt — ainvoke returns, the ``async with`` exits, the
    lock is released.

    Phase 2 (mimics ``api.submit_approval_decision._resume``): wrap a
    second ``ainvoke(Command(resume=...))`` in
    ``async with registry.acquire(session_id)`` and observe the lock
    flips True for the duration of the resume turn, then False after.

    The recorder list captures ``is_locked`` from inside the node,
    proving the OBSERVED invariant is the one the production lock
    contract claims (and not coincidentally satisfied by external
    state).

    Pins R4 / D-01: per-session lock cycle around the interrupt boundary.
    Failure messages cite ``langgraph.__version__`` so a future drift in
    LangGraph's interrupt semantics fails loud, not silent (T-01.1-10).
    """
    from importlib.metadata import version as pkg_version
    from typing import TypedDict

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt

    lg_version = pkg_version("langgraph")

    # Use a fresh session_id (no DB row needed — the graph state is
    # opaque to the lock; we just need a stable thread_id key).
    inc = store.create(
        query="d01-lock-cycle",
        environment="staging",
        reporter_id="r1",
        reporter_team="platform",
    )
    session_id = inc.id

    # Recorder of observed is_locked values — populated from inside the
    # node, so the test sees the lock state at the EXACT moment the
    # graph runtime is mid-turn (not before/after, where ``async with``
    # boundaries make the result trivially predictable).
    is_locked_before_interrupt: list[bool] = []
    is_locked_after_resume: list[bool] = []

    class _S(TypedDict, total=False):
        result: object

    async def _node(state: _S) -> dict:
        # OBSERVATION POINT 1 — mid-turn, before interrupt:
        # The outer caller (Phase 1 OR Phase 2) wraps ainvoke in
        # ``async with registry.acquire(session_id)``. So when this
        # line executes, the lock MUST be held. If D-01 holds, this
        # appends True both times the node runs (initial turn AND
        # post-resume turn).
        is_locked_before_interrupt.append(registry.is_locked(session_id))

        # Pause for HITL. interrupt() raises GraphInterrupt; ainvoke
        # returns control to the caller with __interrupt__ in the
        # result. When Command(resume=...) is later supplied, the node
        # re-runs from the top and ``decision`` receives the resume
        # value.
        decision = interrupt({"reason": "test_d01"})

        # OBSERVATION POINT 2 — mid-turn, AFTER resume returns:
        # On the resume run, the outer caller is the Phase-2
        # ``async with registry.acquire(session_id)`` — so the lock
        # MUST be held here too.
        is_locked_after_resume.append(registry.is_locked(session_id))

        return {"result": decision}

    sg = StateGraph(_S)
    sg.add_node("n", _node)
    sg.set_entry_point("n")
    sg.add_edge("n", END)
    compiled = sg.compile(checkpointer=InMemorySaver())

    cfg: RunnableConfig = {"configurable": {"thread_id": session_id}}

    # ----- baseline: lock free -----
    assert registry.is_locked(session_id) is False, (
        f"precondition: lock should be free at test start "
        f"(langgraph={lg_version})"
    )

    # ----- Phase 1: _run-equivalent — turn pauses at interrupt -----
    # This mirrors src/runtime/service.py:453-463 where the per-session
    # lock wraps ainvoke for the full turn including any HITL pause.
    async with registry.acquire(session_id):
        # OBSERVATION POINT 0 — mid-_run, BEFORE ainvoke:
        # the lock is held by THIS task; is_locked must be True.
        assert registry.is_locked(session_id) is True, (
            f"Phase 1: lock should be held by the acquire() context "
            f"before ainvoke runs (langgraph={lg_version})"
        )
        result = await asyncio.wait_for(
            compiled.ainvoke({}, config=cfg),
            timeout=2.0,
        )

    # The ``async with`` exited because ainvoke returned at the
    # interrupt boundary. The faked node ran exactly once and recorded
    # is_locked=True at the pre-interrupt observation point.
    assert is_locked_before_interrupt == [True], (
        f"D-01 Path A asserts the lock is held across the node body "
        f"during the initial turn (langgraph={lg_version}); "
        f"observed: {is_locked_before_interrupt}. If False, the node "
        f"ran outside the acquire() context — impossible without a "
        f"LangGraph executor that runs nodes on a different task; "
        f"investigate before claiming D-01 superseded."
    )
    assert is_locked_after_resume == [], (
        "post-resume observation should NOT have fired yet (the graph "
        "is paused, no Command(resume=...) has been delivered)"
    )

    # ainvoke returned; the graph is paused at interrupt(). LangGraph
    # surfaces this via ``__interrupt__`` in the result dict.
    assert isinstance(result, dict) and "__interrupt__" in result, (
        f"expected ainvoke to return with __interrupt__ at the pause "
        f"boundary (langgraph={lg_version}); got {result!r}"
    )

    # ----- D-01 PRIMARY OBSERVATION — lock state at the boundary -----
    # After ainvoke returns at the pause, the ``async with`` exits and
    # the lock is released. This is the OBSERVED behaviour Path A
    # records: the lock is NOT held across the pause-resume gap; the
    # GUARANTEE that holds is "no two ainvoke calls overlap on the
    # same thread_id" — Phase 2 below proves the resume re-acquires
    # cleanly without contention.
    assert registry.is_locked(session_id) is False, (
        f"D-01 Path A: at the interrupt() boundary, the per-session "
        f"lock is released because ainvoke returned and the "
        f"``async with registry.acquire()`` block exited "
        f"(langgraph={lg_version}). If True, LangGraph somehow held "
        f"the lock across pause — Path B applies and 01.1-CONTEXT.md "
        f"D-01 must be updated."
    )

    # ----- Phase 2: api ``_resume``-equivalent — resume the paused turn -----
    # Mirrors src/runtime/api.py:465-481 (the _resume closure). The api
    # path uses the BLOCKING ``acquire`` (D-20) — fresh acquisition
    # because Phase 1 released. We also assert mid-resume that the
    # lock is observably held by THIS task (no overlap with any other
    # ainvoke).
    decision_payload = {"decision": "approve", "approver": "alice"}
    async with registry.acquire(session_id):
        assert registry.is_locked(session_id) is True, (
            f"Phase 2: api _resume re-acquired the lock cleanly "
            f"(langgraph={lg_version})"
        )
        result2 = await asyncio.wait_for(
            compiled.ainvoke(Command(resume=decision_payload), config=cfg),
            timeout=2.0,
        )

    # The node ran a SECOND time (from the top, per LangGraph's
    # interrupt semantics) — the recorder appended a second entry to
    # is_locked_before_interrupt, and the post-interrupt observation
    # finally fired with the resume value delivered.
    assert is_locked_before_interrupt == [True, True], (
        f"second turn must have run inside the Phase-2 acquire() "
        f"context (langgraph={lg_version}); "
        f"observed: {is_locked_before_interrupt}"
    )
    assert is_locked_after_resume == [True], (
        f"post-interrupt observation must have fired exactly once "
        f"with is_locked=True (langgraph={lg_version}); "
        f"observed: {is_locked_after_resume}"
    )

    # The graph reached END this time — result2 carries the
    # node's return value (no __interrupt__).
    assert isinstance(result2, dict)
    assert "__interrupt__" not in result2, (
        f"resume turn should run to END (langgraph={lg_version}); "
        f"got {result2!r}"
    )
    assert result2.get("result") == decision_payload, (
        f"interrupt(payload) must return the Command(resume=...) value "
        f"unchanged (langgraph={lg_version}); got {result2!r}"
    )

    # ----- final state: lock released cleanly, no leak -----
    assert registry.is_locked(session_id) is False, (
        f"after _resume completes, the lock must be released "
        f"(langgraph={lg_version})"
    )

    # ----- D-01 OUTCOME (Path A) -----
    # The OBSERVED INVARIANT (replacing D-01's literal "lock held across
    # the pause" wording) is:
    #   "no two ainvoke calls hold the same thread_id simultaneously"
    # which this test pins via the 5 transitions above
    # (False → True → False → True → False) and the recorder showing
    # both turns ran inside an acquire() context.
    #
    # If a future LangGraph release changes this (e.g. interrupt no
    # longer returns control to the caller, or pre-interrupt is_locked
    # observes False), the relevant assertion above fails with a
    # message that prints langgraph version — easy to file as a
    # supersession bump.
