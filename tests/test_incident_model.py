from datetime import datetime, timezone
from orchestrator.incident import (
    Incident, Reporter, ToolCall, AgentRun, Findings, IncidentStatus,
)


def test_incident_minimal_construction():
    inc = Incident(
        id="INC-20260430-001",
        status="new",
        created_at="2026-04-30T15:51:00Z",
        updated_at="2026-04-30T15:51:00Z",
        query="API latency spike",
        environment="production",
        reporter=Reporter(id="user-mock", team="platform"),
    )
    assert inc.id == "INC-20260430-001"
    assert inc.tool_calls == []
    assert inc.findings.triage is None
    assert inc.matched_prior_inc is None


def test_status_must_be_valid_enum():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        Incident(
            id="INC-1", status="invalid", created_at="x", updated_at="y",
            query="q", environment="dev", reporter=Reporter(id="u", team="t"),
        )


import pytest  # noqa: E402  (kept here for self-contained file)
