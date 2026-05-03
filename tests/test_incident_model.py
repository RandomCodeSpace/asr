"""Domain-state tests for the example ``IncidentState`` model.

P2-J: rewritten off the deleted ``runtime.incident.Incident``. All
incident-shaped fields now live on ``examples.incident_management.state.IncidentState``;
``AgentRun`` lives on ``runtime.state``.
"""
import pytest

from examples.incident_management.state import IncidentState, Reporter
from runtime.state import AgentRun


def test_incident_minimal_construction():
    inc = IncidentState(
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
    assert inc.findings == {}
    assert inc.matched_prior_inc is None
    # New intervention fields default cleanly.
    assert inc.pending_intervention is None
    assert inc.user_inputs == []


def test_agent_run_confidence_defaults_none():
    run = AgentRun(
        agent="intake", started_at="t1", ended_at="t2", summary="ok",
    )
    assert run.confidence is None
    assert run.confidence_rationale is None


def test_agent_run_with_confidence_round_trips():
    run = AgentRun(
        agent="deep_investigator", started_at="t1", ended_at="t2",
        summary="ok", confidence=0.42, confidence_rationale="weak signal",
    )
    again = AgentRun.model_validate_json(run.model_dump_json())
    assert again.confidence == pytest.approx(0.42)
    assert again.confidence_rationale == "weak signal"


def test_awaiting_input_and_stopped_are_valid_statuses():
    IncidentState(
        id="INC-1", status="awaiting_input", created_at="x", updated_at="y",
        query="q", environment="dev", reporter=Reporter(id="u", team="t"),
    )
    IncidentState(
        id="INC-2", status="stopped", created_at="x", updated_at="y",
        query="q", environment="dev", reporter=Reporter(id="u", team="t"),
    )


def test_status_string_accepted():
    """``Session.status`` is a free-form string (the framework no longer
    enforces an incident-shaped ``Literal`` enum). The example app applies
    its own ``IncidentStatus`` ``Literal`` only at the UI layer.
    """
    inc = IncidentState(
        id="INC-1", status="invalid", created_at="x", updated_at="y",
        query="q", environment="dev", reporter=Reporter(id="u", team="t"),
    )
    assert inc.status == "invalid"


def test_agent_run_signal_defaults_to_none():
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1", summary="ok")
    assert run.signal is None


def test_agent_run_signal_explicit():
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1",
                   summary="ok", signal="success")
    assert run.signal == "success"
