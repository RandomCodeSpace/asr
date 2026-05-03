"""``SessionStore`` round-trips ``Session.extra_fields`` losslessly.

Post the typed-subclass migration, app-specific session data lives in
``Session.extra_fields: dict[str, Any]``. The schema's ``extra_fields``
JSON column persists this verbatim and merges it back on load.
"""
from __future__ import annotations

import pytest

from runtime.config import MetadataConfig
from runtime.state import Session
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def engine(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# Code-review-shaped extra_fields round-trip
# ---------------------------------------------------------------------------
def test_code_review_extra_fields_round_trips(engine):
    store = SessionStore(engine=engine)
    sess = store.create(
        query="fix payments retry",
        environment="acme/api",
        reporter_id="alice",
        reporter_team="payments",
    )
    sess.extra_fields = {
        "pr": {
            "repo": "acme/api", "number": 101, "title": "fix payments retry",
            "author": "alice", "base_sha": "aaaaaaa", "head_sha": "bbbbbbb",
            "additions": 12, "deletions": 4, "files_changed": 2,
        },
        "review_findings": [
            {"severity": "warning", "file": "src/api.py", "line": 42,
             "category": "style", "message": "prefer f-string",
             "suggestion": "use f'...'"},
        ],
        "overall_recommendation": "comment",
        "review_summary": "LGTM modulo style.",
        "review_token_budget": 1234,
    }
    store.save(sess)
    loaded = store.load(sess.id)

    assert isinstance(loaded, Session)
    pr = loaded.extra_fields["pr"]
    assert pr["repo"] == "acme/api" and pr["number"] == 101
    assert loaded.extra_fields["overall_recommendation"] == "comment"
    assert loaded.extra_fields["review_summary"] == "LGTM modulo style."
    assert loaded.extra_fields["review_token_budget"] == 1234
    assert loaded.extra_fields["review_findings"][0]["file"] == "src/api.py"


def test_code_review_minimal_extra_fields_round_trips(engine):
    store = SessionStore(engine=engine)
    sess = store.create(
        query="t", environment="acme/api", reporter_id="a", reporter_team="t",
    )
    sess.extra_fields = {
        "pr": {"repo": "acme/api", "number": 99, "title": "t",
               "author": "a", "base_sha": "x", "head_sha": "y"},
    }
    store.save(sess)
    loaded = store.load(sess.id)
    assert loaded.extra_fields["pr"]["repo"] == "acme/api"
    assert "review_findings" not in loaded.extra_fields
    assert "overall_recommendation" not in loaded.extra_fields


# ---------------------------------------------------------------------------
# Incident-shaped extra_fields round-trip
# ---------------------------------------------------------------------------
def test_incident_extra_fields_round_trip_preserved(engine):
    store = SessionStore(engine=engine)
    inc = store.create(
        query="payments outage",
        environment="production",
        reporter_id="alice",
        reporter_team="payments",
    )
    inc.extra_fields = {
        "severity": "high",
        "summary": "summary text",
        "tags": ["payments", "redis"],
    }
    store.save(inc)

    loaded = store.load(inc.id)
    assert isinstance(loaded, Session)
    assert loaded.extra_fields["severity"] == "high"
    assert loaded.extra_fields["summary"] == "summary text"
    assert sorted(loaded.extra_fields["tags"]) == ["payments", "redis"]


def test_extra_fields_column_is_additive_for_legacy_rows(engine):
    """A row written by an older binary (no ``extra_fields``) still loads.

    Simulates the back-compat case by writing an ORM row directly with
    ``extra_fields=None`` and confirming ``_row_to_incident`` produces a
    valid ``Session``.
    """
    from datetime import datetime, timezone
    from sqlalchemy.orm import Session as SqlSession

    from runtime.storage.models import IncidentRow

    store = SessionStore(engine=engine)
    now = datetime.now(timezone.utc)
    with SqlSession(engine) as sess:
        sess.add(IncidentRow(
            id="INC-20260503-001",
            status="resolved",
            created_at=now,
            updated_at=now,
            query="legacy",
            environment="production",
            reporter_id="bob",
            reporter_team="ops",
            summary="legacy",
            tags=[],
            agents_run=[],
            tool_calls=[],
            findings={},
            user_inputs=[],
            extra_fields=None,
        ))
        sess.commit()

    loaded = store.load("INC-20260503-001")
    assert isinstance(loaded, Session)
    assert loaded.id == "INC-20260503-001"
    assert loaded.status == "resolved"
    # Bare ``Session`` surfaces typed-column data via ``extra_fields``
    # so apps that drop their typed Session subclass still see the
    # legacy row's domain fields.
    assert loaded.extra_fields["query"] == "legacy"
    assert loaded.extra_fields["environment"] == "production"
    assert loaded.extra_fields["reporter"] == {"id": "bob", "team": "ops"}
    assert loaded.extra_fields["summary"] == "legacy"
