"""IdempotencyStore tests — LRU + SQLite write-through + cold-restart survival."""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from runtime.triggers.idempotency import (
    IdempotencyRow,
    IdempotencyStore,
    _LRU_MAX_PER_TRIGGER,
)


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'idem.db'}"


def test_get_returns_none_for_unknown_key(db_url):
    store = IdempotencyStore.connect(db_url)
    assert store.get("trigger-x", "k1") is None


def test_put_then_get_returns_session_id(db_url):
    store = IdempotencyStore.connect(db_url)
    store.put("pd", "key-1", "INC-1")
    assert store.get("pd", "key-1") == "INC-1"


def test_put_overwrites_existing_key(db_url):
    store = IdempotencyStore.connect(db_url)
    store.put("pd", "key-1", "INC-1")
    store.put("pd", "key-1", "INC-2")
    assert store.get("pd", "key-1") == "INC-2"


def test_get_falls_through_to_sqlite_on_lru_miss(db_url):
    """Cold-restart survival: a fresh process reads the disk row."""
    store = IdempotencyStore.connect(db_url)
    store.put("pd", "key-1", "INC-1")
    # Simulate cold restart: build a brand new store on the same DB.
    store2 = IdempotencyStore.connect(db_url)
    assert store2.get("pd", "key-1") == "INC-1"


def test_purge_expired_drops_stale_rows(db_url):
    store = IdempotencyStore.connect(db_url)
    store.put("pd", "k1", "INC-1", ttl_hours=24)
    # Hand-craft an expired row.
    from sqlalchemy.orm import Session as SqlaSession
    with SqlaSession(store._engine) as s:
        s.add(IdempotencyRow(
            trigger_name="pd",
            key="k-stale",
            session_id="INC-stale",
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ))
        s.commit()
    removed = store.purge_expired()
    assert removed >= 1
    assert store.get("pd", "k-stale") is None
    assert store.get("pd", "k1") == "INC-1"


def test_get_filters_expired_row(db_url):
    """A row past its TTL must read as a miss even before purge runs."""
    store = IdempotencyStore.connect(db_url)
    from sqlalchemy.orm import Session as SqlaSession
    with SqlaSession(store._engine) as s:
        s.add(IdempotencyRow(
            trigger_name="pd",
            key="k-expired",
            session_id="INC-exp",
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ))
        s.commit()
    assert store.get("pd", "k-expired") is None


def test_lru_eviction_does_not_leak_disk(db_url):
    """Evicted LRU entries must still be readable from SQLite."""
    store = IdempotencyStore.connect(db_url)
    # Fill past the LRU bound for a single trigger.
    overflow = _LRU_MAX_PER_TRIGGER + 5
    for i in range(overflow):
        store.put("pd", f"k-{i}", f"INC-{i}")
    # The earliest keys are LRU-evicted; SQLite still has them.
    early = store.get("pd", "k-0")
    assert early == "INC-0"
