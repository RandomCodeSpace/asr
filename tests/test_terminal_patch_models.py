import pytest
from pydantic import ValidationError

from examples.incident_management.mcp_server import (
    EscalateRequest,
    HypothesisSubmission,
    ResolveRequest,
    UpdateIncidentPatch,
)


def test_resolve_request_requires_summary_and_confidence():
    with pytest.raises(ValidationError):
        ResolveRequest(incident_id="INC-1")  # missing required fields


def test_resolve_request_accepts_full_payload():
    req = ResolveRequest(
        incident_id="INC-1",
        resolution_summary="rolled back v1.117",
        confidence=0.85,
        confidence_rationale="strong evidence",
    )
    assert req.confidence == 0.85
    assert req.resolution_summary == "rolled back v1.117"


def test_resolve_request_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        ResolveRequest(
            incident_id="INC-1",
            resolution_summary="ok",
            confidence=0.8,
            confidence_rationale="r",
            statuss="resolved",  # typo — extra=forbid rejects
        )


def test_resolve_request_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        ResolveRequest(
            incident_id="INC-1",
            resolution_summary="ok",
            confidence=1.5,  # > 1.0
            confidence_rationale="r",
        )


def test_escalate_request_requires_team_and_reason():
    with pytest.raises(ValidationError):
        EscalateRequest(
            incident_id="INC-1",
            confidence=0.5,
            confidence_rationale="r",
        )


def test_escalate_request_accepts_full_payload():
    req = EscalateRequest(
        incident_id="INC-1",
        team="platform-oncall",
        reason="approval rejected",
        confidence=0.5,
        confidence_rationale="hedged",
    )
    assert req.team == "platform-oncall"


def test_escalate_request_rejects_empty_team():
    with pytest.raises(ValidationError):
        EscalateRequest(
            incident_id="INC-1",
            team="",
            reason="r",
            confidence=0.5,
            confidence_rationale="r",
        )


def test_hypothesis_submission_requires_hypotheses_and_confidence():
    with pytest.raises(ValidationError):
        HypothesisSubmission(incident_id="INC-1", confidence=0.5,
                             confidence_rationale="r")  # no hypotheses


def test_hypothesis_submission_defaults_findings_for_to_deep_investigator():
    req = HypothesisSubmission(
        incident_id="INC-1",
        hypotheses="1. upstream timeout",
        confidence=0.78,
        confidence_rationale="r",
    )
    assert req.findings_for == "deep_investigator"


def test_update_incident_patch_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        UpdateIncidentPatch(confidance=0.8)  # typo


def test_update_incident_patch_accepts_partial_payload():
    p = UpdateIncidentPatch(severity="high", category="availability")
    assert p.severity == "high"
    assert p.category == "availability"
    # Other fields default to None / empty
    assert p.summary is None


def test_update_incident_patch_rejects_status_field():
    """Terminal status is set via mark_resolved / mark_escalated, NOT
    via update_incident. The schema enforces this by omitting status
    from the allowed fields and using extra=forbid."""
    with pytest.raises(ValidationError):
        UpdateIncidentPatch(status="resolved")


def test_update_incident_patch_rejects_resolution_field():
    """resolution is set by mark_resolved, not update_incident."""
    with pytest.raises(ValidationError):
        UpdateIncidentPatch(resolution="rolled back")
