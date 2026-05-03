"""P8-L: incident-management + code-review run side-by-side.

The decisive proof that the framework is generic: two ``SessionStore``
instances, each parametrised on a different ``Session`` subclass,
coexist without:

* colliding on the session id space (P8-C ``id_format`` hook),
* losing typed fields on round-trip (P8-J ``extra_fields`` JSON column),
* leaking domain shapes into each other (e.g. an incident's ``reporter``
  showing up on a code-review state).

The test stays at the storage + state-resolver layer: spinning up a
real ``Orchestrator`` would require LLM, MCP, and skill-loader
fixturing for both apps, which is out of scope here. The genericity
proof lives in the storage round-trip; orchestrator wiring is
covered by Phase-2/3 tests.
"""
from __future__ import annotations

import pytest

from examples.code_review.state import (
    CodeReviewState,
    PullRequest,
    ReviewFinding,
)
from examples.incident_management.state import IncidentState, Reporter
from runtime.config import MetadataConfig
from runtime.state_resolver import resolve_state_class
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


# ---------------------------------------------------------------------------
# Fixtures: two isolated stores against two isolated SQLite files.
# ---------------------------------------------------------------------------
@pytest.fixture
def incident_store(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/incident.db"))
    Base.metadata.create_all(eng)
    return SessionStore[IncidentState](engine=eng, state_cls=IncidentState)


@pytest.fixture
def code_review_store(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/code_review.db"))
    Base.metadata.create_all(eng)
    return SessionStore[CodeReviewState](engine=eng, state_cls=CodeReviewState)


# ---------------------------------------------------------------------------
# Sanity: dotted-path resolution returns the right class for each app.
# ---------------------------------------------------------------------------
def test_state_resolver_handles_both_apps():
    inc_cls = resolve_state_class("examples.incident_management.state.IncidentState")
    cr_cls = resolve_state_class("examples.code_review.state.CodeReviewState")
    assert inc_cls is IncidentState
    assert cr_cls is CodeReviewState
    # Distinct classes — no accidental shared base above ``Session``.
    assert inc_cls is not cr_cls


# ---------------------------------------------------------------------------
# P8-C: each app mints its own id format, so the two id spaces cannot
# collide even if both stores share a metadata DB.
# ---------------------------------------------------------------------------
def test_each_app_mints_its_own_id_format(incident_store, code_review_store):
    inc = incident_store.create(
        query="api latency",
        environment="production",
        reporter_id="alice",
        reporter_team="payments",
    )
    assert inc.id.startswith("INC-")

    # Code-review state has no ``query``/``environment`` — build it
    # explicitly via ``save`` so we exercise the id_format hook on a
    # state class that *cannot* go through the incident-shaped
    # ``create()`` helper.
    sid = CodeReviewState.id_format(seq=1)
    assert sid.startswith("CR-")

    cr = CodeReviewState(
        id=sid,
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        pr=PullRequest(
            repo="acme/api", number=101, title="t", author="alice",
            base_sha="x", head_sha="y",
        ),
    )
    code_review_store.save(cr)

    # The two id namespaces are disjoint.
    assert not inc.id.startswith("CR-")
    assert not cr.id.startswith("INC-")


# ---------------------------------------------------------------------------
# P8-J: each store round-trips its own typed fields without leaking the
# other app's domain shape.
# ---------------------------------------------------------------------------
def test_two_stores_round_trip_independently(incident_store, code_review_store):
    # Seed an incident with all the incident-shaped fields populated.
    inc = incident_store.create(
        query="redis OOM",
        environment="production",
        reporter_id="bob",
        reporter_team="ops",
    )
    inc.severity = "high"
    inc.tags = ["redis", "payments"]
    inc.summary = "redis OOMKill"
    incident_store.save(inc)

    # Seed a code-review session with PR + findings.
    pr = PullRequest(
        repo="acme/api",
        number=42,
        title="harden retry",
        author="alice",
        base_sha="aaa",
        head_sha="bbb",
        additions=10,
        deletions=2,
        files_changed=1,
    )
    cr = CodeReviewState(
        id=CodeReviewState.id_format(seq=1),
        status="analyzing",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        pr=pr,
        review_findings=[
            ReviewFinding(
                severity="error",
                file="src/api.py",
                line=12,
                category="bug",
                message="retry loop unbounded",
            ),
        ],
        overall_recommendation="request_changes",
        review_summary="needs guardrail",
    )
    code_review_store.save(cr)

    # Reload through the typed store and confirm full identity.
    inc_loaded = incident_store.load(inc.id)
    cr_loaded = code_review_store.load(cr.id)

    # Incident keeps its incident-shaped fields.
    assert isinstance(inc_loaded, IncidentState)
    assert inc_loaded.query == "redis OOM"
    assert inc_loaded.environment == "production"
    assert inc_loaded.reporter == Reporter(id="bob", team="ops")
    assert inc_loaded.severity == "high"
    assert sorted(inc_loaded.tags) == ["payments", "redis"]

    # Code-review keeps its PR + findings.
    assert isinstance(cr_loaded, CodeReviewState)
    assert cr_loaded.pr == pr
    assert len(cr_loaded.review_findings) == 1
    assert cr_loaded.review_findings[0].file == "src/api.py"
    assert cr_loaded.overall_recommendation == "request_changes"
    assert cr_loaded.review_summary == "needs guardrail"

    # Crucial cross-check: a code-review session has *no* reporter
    # attribute (it's not declared on its state class) and an
    # incident has *no* PR attribute. The framework must not have
    # silently bolted either domain shape onto the other.
    assert not hasattr(cr_loaded, "reporter")
    assert not hasattr(cr_loaded, "environment")
    assert not hasattr(inc_loaded, "pr")
    assert not hasattr(inc_loaded, "review_findings")


# ---------------------------------------------------------------------------
# P8-J negative: a code-review session id is rejected by an incident
# store (and vice-versa) only when the row genuinely doesn't exist —
# both stores happily *accept* either id shape via the relaxed
# ``_SESSION_ID_RE`` validator. This guards against accidentally
# tightening that regex back to ``INC-only``.
# ---------------------------------------------------------------------------
def test_id_format_validator_accepts_both_prefixes(
    incident_store, code_review_store
):
    # incident_store.load on a CR-shaped id raises FileNotFoundError,
    # not ValueError — i.e. the id passed validation, the row was
    # simply absent. If the regex were back to ``INC-only`` we'd see
    # ValueError here instead.
    with pytest.raises(FileNotFoundError):
        incident_store.load("CR-20260503-001")
    with pytest.raises(FileNotFoundError):
        code_review_store.load("INC-20260503-001")
