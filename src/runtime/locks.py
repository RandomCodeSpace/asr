"""Per-session asyncio locks.

Status mutations on the same session must serialise. The registry hands
out one ``asyncio.Lock`` per session id; callers acquire it for the
duration of any read-modify-write block on that session's row.

Locks live in-process. Multi-process deployments must layer SQLite
``BEGIN IMMEDIATE`` (already configured) or move to row-level locking.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SessionLockRegistry:
    """In-process registry of per-session asyncio locks."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, session_id: str) -> asyncio.Lock:
        """Return the lock for ``session_id``, creating it if absent."""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Acquire the per-session lock for the duration of the block."""
        lock = self.get(session_id)
        async with lock:
            yield
