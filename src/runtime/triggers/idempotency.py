"""Idempotency-Key dedup store: in-memory LRU + SQLite write-through.

Same DB as session metadata (``storage.metadata.url``); one connection
pool, one filesystem path, one backup story. SQLite WAL mode (already
enabled by ``runtime.storage.engine.build_engine``) handles concurrent
reads from the LRU and the orchestrator.

Cold-restart survival: the LRU is rebuilt on demand; ``get`` falls
through to SQLite when the LRU misses, so a fresh process still
returns the cached ``session_id`` for an unexpired ``Idempotency-Key``.

Schema (registered against ``runtime.storage.models.Base`` so
``Base.metadata.create_all(engine)`` picks it up — no Alembic change
required, matching the existing P3 pattern):

    trigger_idempotency_keys
        trigger_name TEXT NOT NULL
        key          TEXT NOT NULL
        session_id   TEXT NOT NULL
        created_at   TIMESTAMP NOT NULL
        expires_at   TIMESTAMP NOT NULL
        PRIMARY KEY (trigger_name, key)
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timezone, timedelta

from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Mapped, Session as SqlaSession, mapped_column
from sqlalchemy.pool import NullPool

from runtime.storage.models import Base

_LRU_MAX_PER_TRIGGER = 1024


class IdempotencyRow(Base):
    """SQLite-backed dedup record. One row per (trigger_name, key)."""

    __tablename__ = "trigger_idempotency_keys"

    trigger_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IdempotencyStore:
    """Thread-safe LRU with SQLite write-through.

    The LRU bounds memory; SQLite handles cold-restart survival and
    cross-process sharing (e.g. multi-worker uvicorn). Mutations are
    serialised via a single threading lock — the critical section is
    a few dict ops plus one SQL round-trip, which dwarfs lock overhead.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # Ensure the table exists even if the orchestrator hasn't run
        # ``Base.metadata.create_all`` yet (early lifespan path).
        # ``IdempotencyRow.__table__`` is a ``Table`` at runtime; the
        # SQLAlchemy stub types it as the wider ``FromClause``.
        Base.metadata.create_all(engine, tables=[IdempotencyRow.__table__])  # pyright: ignore[reportArgumentType]
        self._lru: dict[str, OrderedDict[str, str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls, db_url: str) -> "IdempotencyStore":
        """Build a store backed by a fresh SQLAlchemy engine.

        Convenience for tests + standalone tooling. Production paths
        should reuse the orchestrator's engine via the constructor so a
        single SQLite file is opened once.
        """
        engine = create_engine(db_url, poolclass=NullPool, future=True)
        return cls(engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, trigger_name: str, key: str) -> str | None:
        """Return the cached ``session_id`` for the key, or ``None``.

        LRU first, SQLite second; a SQLite hit refills the LRU so
        subsequent reads stay in memory.
        """
        with self._lock:
            cache = self._lru.get(trigger_name)
            if cache is not None and key in cache:
                # Bump recency
                cache.move_to_end(key)
                return cache[key] or None
        # SQLite fall-through (outside the threading lock — sqlite3 has
        # its own locking, and this path is rare).
        with SqlaSession(self._engine) as s:
            row = s.execute(
                select(IdempotencyRow).where(
                    IdempotencyRow.trigger_name == trigger_name,
                    IdempotencyRow.key == key,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                # SQLite drops tz on round-trip; assume UTC (we always
                # write UTC).
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= _utc_now():
                # Stale row — opportunistic delete.
                s.execute(
                    delete(IdempotencyRow).where(
                        IdempotencyRow.trigger_name == trigger_name,
                        IdempotencyRow.key == key,
                    )
                )
                s.commit()
                return None
            session_id = row.session_id
            if not session_id:
                return None
        # Refill LRU.
        with self._lock:
            cache = self._lru.setdefault(trigger_name, OrderedDict())
            cache[key] = session_id
            cache.move_to_end(key)
            self._evict_if_needed(cache)
        return session_id

    def put(
        self,
        trigger_name: str,
        key: str,
        session_id: str,
        *,
        ttl_hours: int = 24,
    ) -> None:
        """Cache a fresh (key -> session_id) binding.

        Writes through to SQLite with ``expires_at = now + ttl``. The
        LRU receives the same record. Calling ``put`` for an existing
        key overwrites both layers.
        """
        now = _utc_now()
        expires_at = now + timedelta(hours=ttl_hours)
        with SqlaSession(self._engine) as s:
            existing = s.get(IdempotencyRow, (trigger_name, key))
            if existing is None:
                s.add(IdempotencyRow(
                    trigger_name=trigger_name,
                    key=key,
                    session_id=session_id,
                    created_at=now,
                    expires_at=expires_at,
                ))
            else:
                existing.session_id = session_id
                existing.created_at = now
                existing.expires_at = expires_at
            s.commit()
        with self._lock:
            cache = self._lru.setdefault(trigger_name, OrderedDict())
            cache[key] = session_id
            cache.move_to_end(key)
            self._evict_if_needed(cache)
        # Opportunistic purge of expired rows so a long-running process
        # doesn't accumulate dead records. Cheap (range-bounded delete).
        self.purge_expired()

    def reserve(
        self,
        trigger_name: str,
        key: str,
        *,
        ttl_hours: int = 24,
    ) -> bool:
        """Atomically reserve a fresh idempotency key.

        Returns ``True`` only for the caller that inserted the row. A
        ``False`` result means another request has either already
        completed and stored ``session_id`` or is still in flight.
        """
        now = _utc_now()
        expires_at = now + timedelta(hours=ttl_hours)
        with SqlaSession(self._engine) as s:
            existing = s.get(IdempotencyRow, (trigger_name, key))
            if existing is not None:
                existing_expires = existing.expires_at
                if existing_expires.tzinfo is None:
                    existing_expires = existing_expires.replace(tzinfo=timezone.utc)
                if existing_expires <= now:
                    s.delete(existing)
                    s.commit()
                else:
                    return False
            s.add(IdempotencyRow(
                trigger_name=trigger_name,
                key=key,
                session_id="",
                created_at=now,
                expires_at=expires_at,
            ))
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                return False
        with self._lock:
            cache = self._lru.setdefault(trigger_name, OrderedDict())
            cache[key] = ""
            cache.move_to_end(key)
            self._evict_if_needed(cache)
        return True

    def purge_expired(self) -> int:
        """Delete all rows whose ``expires_at`` is in the past. Returns
        the number of rows removed."""
        with SqlaSession(self._engine) as s:
            result = s.execute(
                delete(IdempotencyRow).where(
                    IdempotencyRow.expires_at <= _utc_now()
                )
            )
            s.commit()
            # ``rowcount`` is exposed on ``CursorResult`` (the concrete
            # return of DML execute); the abstract ``Result`` stub does
            # not declare it.
            return result.rowcount or 0  # pyright: ignore[reportAttributeAccessIssue]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _evict_if_needed(cache: "OrderedDict[str, str]") -> None:
        while len(cache) > _LRU_MAX_PER_TRIGGER:
            cache.popitem(last=False)
