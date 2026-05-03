"""Tests for the ``Session.to_agent_input`` hook.

Only the framework-default ``Session.to_agent_input`` is exercised here.
The ``IncidentState`` / ``CodeReviewState`` per-app overrides went away
with the move to ``Session.extra_fields``; their preamble shape is now
the responsibility of the per-app supervisor / runner, not of a
typed-subclass override on the state model.
"""
from __future__ import annotations

from runtime.state import Session


def _base_session(**overrides) -> Session:
    fields = dict(
        id="S-1",
        status="in_progress",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
    )
    fields.update(overrides)
    return Session(**fields)


# ---------------------------------------------------------------------------
# Default Session impl — id + status only, plus findings/user_inputs.
# ---------------------------------------------------------------------------


def test_default_session_to_agent_input_is_framework_neutral():
    s = _base_session()
    text = s.to_agent_input()
    assert "Session S-1" in text
    assert "Status: in_progress" in text
    # Must NOT mention domain-specific concepts.
    assert "Incident" not in text
    assert "Environment" not in text
    assert "Pull Request" not in text


def test_default_session_appends_findings_and_user_inputs():
    s = _base_session(
        findings={"triage": "looks like a flake"},
        user_inputs=["operator note: rerun"],
    )
    text = s.to_agent_input()
    assert "Findings (triage): looks like a flake" in text
    assert "operator note: rerun" in text
