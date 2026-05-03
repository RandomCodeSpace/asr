"""Session.extra_fields — bag for app-specific session data."""
from __future__ import annotations

from runtime.state import Session


def _base_kwargs(**over):
    base = {
        "id": "S-1",
        "status": "running",
        "created_at": "2026-05-03T00:00:00Z",
        "updated_at": "2026-05-03T00:00:00Z",
    }
    base.update(over)
    return base


def test_session_extra_fields_default_empty() -> None:
    s = Session(**_base_kwargs())
    assert s.extra_fields == {}


def test_session_extra_fields_round_trip() -> None:
    s = Session(**_base_kwargs(extra_fields={
        "severity": "sev1",
        "reporter": {"id": "u1", "team": "infra"},
    }))
    assert s.extra_fields["severity"] == "sev1"
    assert s.extra_fields["reporter"]["id"] == "u1"


def test_session_extra_fields_serialise() -> None:
    s = Session(**_base_kwargs(extra_fields={"severity": "sev2"}))
    dumped = s.model_dump()
    assert dumped["extra_fields"] == {"severity": "sev2"}


def test_session_extra_fields_independent_per_instance() -> None:
    """default_factory must produce a new dict per instance, not a shared one."""
    a = Session(**_base_kwargs(id="S-A"))
    b = Session(**_base_kwargs(id="S-B"))
    a.extra_fields["x"] = 1
    assert "x" not in b.extra_fields
