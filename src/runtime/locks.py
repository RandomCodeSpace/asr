"""Per-session asyncio locks.

Status mutations on the same session must serialise. The registry hands
out one ``asyncio.Lock`` per session id; callers acquire it for the
duration of any read-modify-write block on that session's row.

The ``acquire`` context manager is **task-reentrant**: a coroutine that
already holds the lock for a given session id can re-enter it without
deadlocking. This matters when nested helpers (e.g. retry → finalize)
both want to take the lock — without re-entry, the inner ``acquire``
would wait forever for the outer to release.

Locks live in-process. Multi-process deployments must layer SQLite
``BEGIN IMMEDIATE`` (already configured) or move to row-level locking.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SessionBusy(RuntimeError):
    """Raised when a session is already executing and cannot accept a new turn.

    Callers should surface this as HTTP 429 with a ``Retry-After: 1`` header
    so that clients know the session will become available shortly.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id!r} is already executing")
        self.session_id = session_id


class _Slot:
    """Per-session lock state: the lock plus reentrancy tracking."""

    __slots__ = ("lock", "owner", "depth")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task | None = None
        self.depth = 0


class SessionLockRegistry:
    """In-process registry of per-session task-reentrant asyncio locks.

    TODO(v2): evict idle slots to cap memory usage for long-running servers.
    """

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}  # TODO(v2): add eviction for idle sessions

    def _slot(self, session_id: str) -> _Slot:
        slot = self._slots.get(session_id)
        if slot is None:
            slot = _Slot()
            self._slots[session_id] = slot
        return slot

    def get(self, session_id: str) -> asyncio.Lock:
        """Return the underlying lock for ``session_id``.

        Direct ``async with reg.get(sid):`` does NOT honour reentrancy.
        Prefer ``async with reg.acquire(sid):`` for nested-safe entry.
        """
        return self._slot(session_id).lock

    def is_locked(self, session_id: str) -> bool:
        """Return ``True`` iff ``session_id`` currently holds the lock.

        Non-blocking. Returns ``False`` for unknown / never-seen session ids
        (no slot is created as a side-effect of this call).
        """
        slot = self._slots.get(session_id)
        return slot is not None and slot.lock.locked()

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Acquire the per-session lock for the duration of the block.

        Reentrant on the current ``asyncio.Task``: if this task already
        holds the lock, the call is a no-op (depth is bumped and yields
        immediately). The actual ``Lock.release`` only happens when the
        outermost ``acquire`` exits.
        """
        slot = self._slot(session_id)
        current = asyncio.current_task()
        if slot.owner is current and current is not None:
            slot.depth += 1
            try:
                yield
            finally:
                slot.depth -= 1
            return
        await slot.lock.acquire()
        slot.owner = current
        slot.depth = 1
        try:
            yield
        finally:
            slot.depth -= 1
            if slot.depth == 0:
                slot.owner = None
                slot.lock.release()
