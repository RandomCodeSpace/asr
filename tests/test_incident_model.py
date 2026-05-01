import pytest
from orchestrator.incident import (
    Incident, Reporter,
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
    # New intervention fields default cleanly.
    assert inc.pending_intervention is None
    assert inc.user_inputs == []


def test_agent_run_confidence_defaults_none():
    from orchestrator.incident import AgentRun
    run = AgentRun(
        agent="intake", started_at="t1", ended_at="t2", summary="ok",
    )
    assert run.confidence is None
    assert run.confidence_rationale is None


def test_agent_run_with_confidence_round_trips():
    from orchestrator.incident import AgentRun
    run = AgentRun(
        agent="deep_investigator", started_at="t1", ended_at="t2",
        summary="ok", confidence=0.42, confidence_rationale="weak signal",
    )
    again = AgentRun.model_validate_json(run.model_dump_json())
    assert again.confidence == pytest.approx(0.42)
    assert again.confidence_rationale == "weak signal"


def test_awaiting_input_and_stopped_are_valid_statuses():
    Incident(
        id="INC-1", status="awaiting_input", created_at="x", updated_at="y",
        query="q", environment="dev", reporter=Reporter(id="u", team="t"),
    )
    Incident(
        id="INC-2", status="stopped", created_at="x", updated_at="y",
        query="q", environment="dev", reporter=Reporter(id="u", team="t"),
    )


def test_status_must_be_valid_enum():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        Incident(
            id="INC-1", status="invalid", created_at="x", updated_at="y",
            query="q", environment="dev", reporter=Reporter(id="u", team="t"),
        )




def test_agent_run_signal_defaults_to_none():
    from orchestrator.incident import AgentRun
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1", summary="ok")
    assert run.signal is None


def test_agent_run_signal_explicit():
    from orchestrator.incident import AgentRun
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1",
                   summary="ok", signal="success")
    assert run.signal == "success"
