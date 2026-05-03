"""Tests for the "duplicate" status + parent_session_id linkage.

Covers:
  * ``Session`` accepts and round-trips the ``parent_session_id`` and
    ``dedup_rationale`` fields.
  * ``SessionStore.list_recent`` filters out duplicates by default and
    surfaces them when ``include_duplicates=True``.
  * ``SessionStore.list_children`` returns the linked-list of duplicates
    that point at a parent.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    e = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def store(engine):
    return SessionStore(engine=engine)


def test_session_round_trip_with_dedup_fields(store):
    inc = store.create(query="payments latency", environment="production",
                       reporter_id="u-1", reporter_team="platform")
    inc.status = "duplicate"
    inc.parent_session_id = "INC-19000101-001"
    inc.dedup_rationale = "Same error signature in payments service."
    store.save(inc)
    loaded = store.load(inc.id)
    assert loaded.status == "duplicate"
    assert loaded.parent_session_id == "INC-19000101-001"
    assert loaded.dedup_rationale == "Same error signature in payments service."


def test_list_recent_excludes_duplicates_by_default(store):
    parent = store.create(query="api latency", environment="production",
                          reporter_id="u", reporter_team="t")
    parent.status = "resolved"
    store.save(parent)

    dup = store.create(query="api latency", environment="production",
                       reporter_id="u", reporter_team="t")
    dup.status = "duplicate"
    dup.parent_session_id = parent.id
    store.save(dup)

    visible = [i.id for i in store.list_recent(50)]
    assert parent.id in visible
    assert dup.id not in visible


def test_list_recent_includes_duplicates_when_opted_in(store):
    parent = store.create(query="x", environment="dev",
                          reporter_id="u", reporter_team="t")
    parent.status = "resolved"
    store.save(parent)

    dup = store.create(query="x", environment="dev",
                       reporter_id="u", reporter_team="t")
    dup.status = "duplicate"
    dup.parent_session_id = parent.id
    store.save(dup)

    all_ids = [i.id for i in store.list_recent(50, include_duplicates=True)]
    assert parent.id in all_ids
    assert dup.id in all_ids


def test_list_children_returns_only_linked_duplicates(store):
    parent = store.create(query="cache miss", environment="production",
                          reporter_id="u", reporter_team="t")
    parent.status = "resolved"
    store.save(parent)

    dup1 = store.create(query="cache miss", environment="production",
                        reporter_id="u", reporter_team="t")
    dup1.status = "duplicate"
    dup1.parent_session_id = parent.id
    store.save(dup1)

    dup2 = store.create(query="cache miss again", environment="production",
                        reporter_id="u", reporter_team="t")
    dup2.status = "duplicate"
    dup2.parent_session_id = parent.id
    store.save(dup2)

    # An unrelated session that points nowhere should not show up.
    other = store.create(query="unrelated", environment="dev",
                         reporter_id="u", reporter_team="t")
    other.status = "resolved"
    store.save(other)

    children = store.list_children(parent.id)
    child_ids = {c.id for c in children}
    assert child_ids == {dup1.id, dup2.id}


def test_list_children_empty_for_unmatched_parent(store):
    inc = store.create(query="standalone", environment="production",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    assert store.list_children(inc.id) == []


def test_default_session_has_null_dedup_fields(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    assert inc.parent_session_id is None
    assert inc.dedup_rationale is None
