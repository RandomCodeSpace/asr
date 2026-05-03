"""Tests — retraction API + dedup_retractions audit table.

Covers:
  * :class:`SessionStore.un_duplicate` happy path + 409 / 404 conditions.
  * :class:`DedupRetractionRow` audit insertion.
  * The HTTP router exposed by :mod:`runtime.api_dedup` against a
    pristine FastAPI app — no full lifespan / orchestrator service.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as SqlSession

from examples.incident_management.state import IncidentState
from runtime.api_dedup import register_dedup_routes
from runtime.storage.models import Base, DedupRetractionRow
from runtime.storage.session_store import SessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    e = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def store(engine):
    return SessionStore(engine=engine, state_cls=IncidentState)


@pytest.fixture()
def app_and_client(store):
    app = FastAPI()
    register_dedup_routes(app, store_provider=lambda: store)
    return app, TestClient(app)


def _make_duplicate(store: SessionStore) -> tuple[str, str]:
    """Seed a parent + a duplicate child; return their ids."""
    parent = store.create(query="payments outage", environment="production",
                          reporter_id="u", reporter_team="t")
    parent.status = "resolved"
    store.save(parent)
    dup = store.create(query="payments outage repeated",
                       environment="production",
                       reporter_id="u", reporter_team="t")
    dup.status = "duplicate"
    dup.parent_session_id = parent.id
    dup.dedup_rationale = "stage 2 confirmed"
    store.save(dup)
    return parent.id, dup.id


# ---------------------------------------------------------------------------
# SessionStore.un_duplicate
# ---------------------------------------------------------------------------


def test_un_duplicate_flips_status_and_clears_parent(store, engine):
    parent_id, dup_id = _make_duplicate(store)
    updated = store.un_duplicate(dup_id, retracted_by="alice@example.com",
                                 note="false positive")
    assert updated.status == "new"
    assert updated.parent_session_id is None
    assert updated.dedup_rationale is None

    # Audit row inserted with the captured original match id.
    with SqlSession(engine) as s:
        rows = s.execute(select(DedupRetractionRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].session_id == dup_id
    assert rows[0].original_match_id == parent_id
    assert rows[0].retracted_by == "alice@example.com"
    assert rows[0].note == "false positive"


def test_un_duplicate_rejects_non_duplicate(store):
    inc = store.create(query="x", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    with pytest.raises(ValueError, match="not a duplicate"):
        store.un_duplicate(inc.id)


def test_un_duplicate_missing_id_raises(store):
    with pytest.raises(FileNotFoundError):
        store.un_duplicate("INC-19000101-999")


def test_un_duplicate_idempotent_second_call_raises(store, engine):
    _, dup_id = _make_duplicate(store)
    store.un_duplicate(dup_id)  # OK
    # Second call: row no longer in duplicate state -> ValueError.
    with pytest.raises(ValueError):
        store.un_duplicate(dup_id)
    # Audit table reflects exactly one retraction.
    with SqlSession(engine) as s:
        rows = s.execute(select(DedupRetractionRow)).scalars().all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# HTTP route
# ---------------------------------------------------------------------------


def test_post_un_duplicate_returns_200(store, app_and_client):
    parent_id, dup_id = _make_duplicate(store)
    _, client = app_and_client
    resp = client.post(
        f"/sessions/{dup_id}/un-duplicate",
        json={"retracted_by": "ops@x", "note": "miscategorized"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == dup_id
    assert body["status"] == "new"
    assert body["parent_session_id"] is None
    assert body["original_match_id"] == parent_id
    assert body["retracted_by"] == "ops@x"
    assert body["note"] == "miscategorized"


def test_post_un_duplicate_returns_409_for_non_duplicate(store, app_and_client):
    inc = store.create(query="x", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    _, client = app_and_client
    resp = client.post(f"/sessions/{inc.id}/un-duplicate", json={})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    # FastAPI nests the structured detail unchanged.
    assert detail["error"] == "not a duplicate"
    assert detail["status"] == "resolved"


def test_post_un_duplicate_returns_404_for_unknown(app_and_client):
    _, client = app_and_client
    resp = client.post(
        "/sessions/INC-19000101-999/un-duplicate", json={},
    )
    assert resp.status_code == 404


def test_post_un_duplicate_returns_404_for_malformed_id(app_and_client):
    _, client = app_and_client
    resp = client.post("/sessions/not-an-id/un-duplicate", json={})
    assert resp.status_code == 404


def test_post_un_duplicate_accepts_empty_body(store, app_and_client):
    _, dup_id = _make_duplicate(store)
    _, client = app_and_client
    resp = client.post(f"/sessions/{dup_id}/un-duplicate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "new"
    assert body["retracted_by"] is None
    assert body["note"] is None
