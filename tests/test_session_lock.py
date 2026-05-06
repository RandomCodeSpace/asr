import asyncio
import pytest
from runtime.locks import SessionBusy, SessionLockRegistry


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
