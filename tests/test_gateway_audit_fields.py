"""Tests for the ``ToolCall`` audit-field extension.

The JSON column on ``incidents.tool_calls`` already accepts arbitrary
structure, so no DDL is needed. The schema must:

  * default ``status="executed"`` so legacy rows hydrate cleanly
  * accept all gateway statuses (executed, executed_with_notify,
    pending_approval, approved, rejected, timeout)
  * accept all risk levels (low, medium, high) and ``None``
  * reject unknown statuses / risks at construction time
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.state import ToolCall


def test_legacy_payload_hydrates_with_executed_default():
    """Pre-Phase-4 row in DB has no ``status`` key — must hydrate as executed."""
    legacy = {
        "agent": "intake",
        "tool": "create_incident",
        "args": {"q": "x"},
        "result": {"id": "INC-1"},
        "ts": "2026-05-02T00:00:00Z",
    }
    tc = ToolCall(**legacy)
    assert tc.status == "executed"
    assert tc.risk is None
    assert tc.approver is None
    assert tc.approved_at is None
    assert tc.approval_rationale is None


def test_all_status_literals_accepted():
    for s in [
        "executed",
        "executed_with_notify",
        "pending_approval",
        "approved",
        "rejected",
        "timeout",
    ]:
        tc = ToolCall(
            agent="x", tool="y", args={}, result=None,
            ts="2026-05-02T00:00:00Z", status=s,
        )
        assert tc.status == s


def test_all_risk_literals_accepted():
    for r in ["low", "medium", "high", None]:
        tc = ToolCall(
            agent="x", tool="y", args={}, result=None,
            ts="2026-05-02T00:00:00Z", risk=r,
        )
        assert tc.risk == r


def test_full_audit_record_round_trips_via_json():
    """Pydantic round-trip should preserve every audit field bit-for-bit."""
    tc = ToolCall(
        agent="resolution",
        tool="apply_fix",
        args={"target": "payments-svc"},
        result={"ok": True},
        ts="2026-05-02T00:00:00Z",
        risk="high",
        status="approved",
        approver="alice@ops",
        approved_at="2026-05-02T00:01:23Z",
        approval_rationale="rollback plan verified, low blast radius",
    )
    payload = tc.model_dump()
    rehydrated = ToolCall(**payload)
    assert rehydrated == tc


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        ToolCall(
            agent="x", tool="y", args={}, result=None,
            ts="2026-05-02T00:00:00Z", status="not_a_real_status",
        )


def test_invalid_risk_rejected():
    with pytest.raises(ValidationError):
        ToolCall(
            agent="x", tool="y", args={}, result=None,
            ts="2026-05-02T00:00:00Z", risk="EXTREME",
        )


def test_back_compat_existing_session_load():
    """A Session with ``tool_calls`` produced before the audit-field extension must still load."""
    from runtime.state import Session

    raw = {
        "id": "S-1",
        "status": "in_progress",
        "created_at": "2026-05-02T00:00:00Z",
        "updated_at": "2026-05-02T00:00:00Z",
        "tool_calls": [
            {
                "agent": "intake",
                "tool": "lookup",
                "args": {},
                "result": None,
                "ts": "2026-05-02T00:00:00Z",
            },
        ],
    }
    s = Session(**raw)
    assert len(s.tool_calls) == 1
    assert s.tool_calls[0].status == "executed"
    assert s.tool_calls[0].risk is None
