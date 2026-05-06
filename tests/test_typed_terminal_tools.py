import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from examples.incident_management.mcp_server import IncidentMCPServer
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def server_and_store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    s = SessionStore(engine=engine)
    srv = IncidentMCPServer()
    srv.configure(store=s, history=None,
                  escalation_teams=["platform-oncall", "data-oncall"])
    return srv, s


# ========== mark_resolved ==========

@pytest.mark.asyncio
async def test_mark_resolved_sets_status_and_resolution(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    out = await srv._tool_mark_resolved(
        incident_id=inc.id,
        resolution_summary="rolled back v1.117",
        confidence=0.9,
        confidence_rationale="strong evidence",
    )
    assert out["status"] == "resolved"
    assert out["confidence"] == 0.9
    fresh = store.load(inc.id)
    assert fresh.status == "resolved"
    assert fresh.extra_fields["resolution"] == "rolled back v1.117"


@pytest.mark.asyncio
async def test_mark_resolved_rejects_out_of_range_confidence(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValidationError):
        await srv._tool_mark_resolved(
            incident_id=inc.id,
            resolution_summary="ok",
            confidence=1.5,
            confidence_rationale="r",
        )


# ========== mark_escalated ==========

@pytest.mark.asyncio
async def test_mark_escalated_sets_status_team_and_reason(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    out = await srv._tool_mark_escalated(
        incident_id=inc.id,
        team="platform-oncall",
        reason="approval rejected",
        confidence=0.5,
        confidence_rationale="hedged",
    )
    assert out["status"] == "escalated"
    assert out["team"] == "platform-oncall"
    fresh = store.load(inc.id)
    assert fresh.status == "escalated"
    assert fresh.extra_fields["escalated_to"] == "platform-oncall"
    assert fresh.extra_fields["escalation_reason"] == "approval rejected"


@pytest.mark.asyncio
async def test_mark_escalated_rejects_unknown_team(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="not in escalation_teams"):
        await srv._tool_mark_escalated(
            incident_id=inc.id,
            team="nope-team",
            reason="r",
            confidence=0.5,
            confidence_rationale="r",
        )


@pytest.mark.asyncio
async def test_mark_escalated_accepts_when_no_roster_configured(tmp_path):
    """If escalation_teams is empty (e.g. legacy/test config), the
    runtime accepts any non-empty team string. The schema's min_length=1
    still fires for empty strings."""
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    s = SessionStore(engine=engine)
    srv = IncidentMCPServer()
    srv.configure(store=s, history=None)  # no escalation_teams
    inc = s.create(query="q", environment="dev",
                   reporter_id="u", reporter_team="t")
    out = await srv._tool_mark_escalated(
        incident_id=inc.id,
        team="any-team",
        reason="r",
        confidence=0.5,
        confidence_rationale="r",
    )
    assert out["team"] == "any-team"


# ========== submit_hypothesis ==========

@pytest.mark.asyncio
async def test_submit_hypothesis_writes_findings_and_returns_confidence(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    out = await srv._tool_submit_hypothesis(
        incident_id=inc.id,
        hypotheses="1. upstream timeout 2. memory pressure",
        confidence=0.78,
        confidence_rationale="multiple plausible causes",
    )
    assert out["confidence"] == 0.78
    assert out["confidence_rationale"] == "multiple plausible causes"
    assert out["findings_for"] == "deep_investigator"
    fresh = store.load(inc.id)
    assert "deep_investigator" in fresh.findings
    assert "upstream timeout" in fresh.findings["deep_investigator"]


@pytest.mark.asyncio
async def test_submit_hypothesis_custom_findings_for(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    await srv._tool_submit_hypothesis(
        incident_id=inc.id,
        hypotheses="ranked list",
        confidence=0.6,
        confidence_rationale="r",
        findings_for="triage",
    )
    fresh = store.load(inc.id)
    assert "triage" in fresh.findings


@pytest.mark.asyncio
async def test_submit_hypothesis_rejects_missing_confidence_rationale(server_and_store):
    srv, store = server_and_store
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValidationError):
        await srv._tool_submit_hypothesis(
            incident_id=inc.id,
            hypotheses="h",
            confidence=0.5,
            confidence_rationale="",  # min_length=1 → reject
        )
