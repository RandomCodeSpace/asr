"""P8-J: ``SessionStore`` round-trips any ``Session`` subclass.

The framework was originally written against an incident-shaped row schema
(``query / environment / reporter / severity / ...``). A second example
app (``code_review``) introduced a ``CodeReviewState`` whose typed fields
(``pr: PullRequest``, ``review_findings: list[ReviewFinding]``,
``overall_recommendation``) had no place in the row dict — they silently
dropped on round-trip.

P8-J adds an ``extra_fields: JSON`` column. Round-trip is now driven by
``state_cls.model_fields``: known typed-column fields go to their own
columns; everything else lands in ``extra_fields`` and hydrates back
through pydantic on load.

Tests cover both apps to prevent regression of either.
"""
from __future__ import annotations

import pytest

from examples.code_review.state import (
    CodeReviewState,
    PullRequest,
    ReviewFinding,
)
from examples.incident_management.state import IncidentState, Reporter
from orchestrator.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def engine(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# Code-review round-trip — the canonical failing case before P8-J.
# ---------------------------------------------------------------------------
def test_code_review_state_round_trips_typed_fields(engine):
    store = SessionStore[CodeReviewState](
        engine=engine, state_cls=CodeReviewState
    )
    pr = PullRequest(
        repo="acme/api",
        number=101,
        title="fix payments retry",
        author="alice",
        base_sha="aaaaaaa",
        head_sha="bbbbbbb",
        additions=12,
        deletions=4,
        files_changed=2,
    )
    state = CodeReviewState(
        id=CodeReviewState.id_format(seq=1),
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        pr=pr,
        review_findings=[
            ReviewFinding(
                severity="warning",
                file="src/api.py",
                line=42,
                category="style",
                message="prefer f-string",
                suggestion="use f'...'",
            ),
        ],
        overall_recommendation="comment",
        review_summary="LGTM modulo style.",
        review_token_budget=1234,
    )
    store.save(state)
    loaded = store.load(state.id)

    assert isinstance(loaded, CodeReviewState)
    assert loaded.pr == pr, "PullRequest must round-trip"
    assert loaded.review_findings == state.review_findings
    assert loaded.overall_recommendation == "comment"
    assert loaded.review_summary == "LGTM modulo style."
    assert loaded.review_token_budget == 1234


def test_code_review_minimal_state_round_trips(engine):
    store = SessionStore[CodeReviewState](
        engine=engine, state_cls=CodeReviewState
    )
    state = CodeReviewState(
        id=CodeReviewState.id_format(seq=1),
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        pr=PullRequest(
            repo="acme/api", number=99, title="t", author="a",
            base_sha="x", head_sha="y",
        ),
    )
    store.save(state)
    loaded = store.load(state.id)
    assert loaded.pr.repo == "acme/api"
    assert loaded.review_findings == []
    assert loaded.overall_recommendation is None


# ---------------------------------------------------------------------------
# Incident round-trip regression — must not break.
# ---------------------------------------------------------------------------
def test_incident_state_round_trip_preserved(engine):
    store = SessionStore[IncidentState](
        engine=engine, state_cls=IncidentState
    )
    inc = store.create(
        query="payments outage",
        environment="production",
        reporter_id="alice",
        reporter_team="payments",
    )
    inc.severity = "high"
    inc.summary = "summary text"
    inc.tags = ["payments", "redis"]
    store.save(inc)

    loaded = store.load(inc.id)
    assert isinstance(loaded, IncidentState)
    assert loaded.query == "payments outage"
    assert loaded.environment == "production"
    assert loaded.reporter == Reporter(id="alice", team="payments")
    assert loaded.severity == "high"
    assert loaded.summary == "summary text"
    assert sorted(loaded.tags) == ["payments", "redis"]


def test_extra_fields_column_is_additive_for_legacy_rows(engine):
    """A row written by an older binary (no ``extra_fields``) still loads.

    Simulates the back-compat case by writing an ORM row directly with
    ``extra_fields=None`` and confirming ``_row_to_incident`` produces a
    valid ``IncidentState``.
    """
    from datetime import datetime, timezone
    from sqlalchemy.orm import Session as SqlSession

    from runtime.storage.models import IncidentRow

    store = SessionStore[IncidentState](
        engine=engine, state_cls=IncidentState
    )
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
    assert isinstance(loaded, IncidentState)
    assert loaded.query == "legacy"
    assert loaded.reporter == Reporter(id="bob", team="ops")
