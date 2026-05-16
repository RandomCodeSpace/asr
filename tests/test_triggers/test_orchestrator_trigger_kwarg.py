"""Orchestrator + Service start_session(trigger=...) plumbing test."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from runtime.triggers.base import TriggerInfo


async def _always_paused() -> bool:
    return True


@pytest.mark.asyncio
async def test_orchestrator_start_session_records_trigger(tmp_path, monkeypatch):
    """``Orchestrator.start_session(trigger=...)`` stamps provenance on
    ``inc.findings['trigger']`` before invoking the graph."""
    # Build a stub orchestrator with just enough surface to test the
    # trigger-stamp branch: a fake store + a fake graph whose ``ainvoke``
    # is a no-op.
    from runtime.orchestrator import Orchestrator

    captured = {}

    class _FakeStore:
        def __init__(self):
            self.saved = None

        def create(self, **kwargs):
            from types import SimpleNamespace
            inc = SimpleNamespace(
                id="INC-001",
                findings={},
                **kwargs,
            )
            captured["created"] = inc
            return inc

        def save(self, inc):
            captured["saved"] = inc

    class _FakeGraph:
        async def ainvoke(self, state, config):
            captured["invoked"] = (state, config)

    # Bypass __init__ by constructing via __new__ (Orchestrator.__init__
    # wires real subsystems we don't need for this unit-test).
    orch = Orchestrator.__new__(Orchestrator)
    orch.store = _FakeStore()
    orch.graph = _FakeGraph()
    orch._thread_config = lambda sid: {"configurable": {"thread_id": sid}}
    orch._is_graph_paused = lambda sid: _always_paused()
    # Tests that bypass __init__ must set the dedup pipeline
    # attribute to ``None`` so the dedup-check shortcut returns False
    # without touching the (uninitialised) attribute.
    orch.dedup_pipeline = None

    info = TriggerInfo(
        name="pd",
        transport="webhook",
        target_app="incident_management",
        received_at=datetime.now(timezone.utc),
    )
    sid = await orch.start_session(
        query="boom", environment="prod", trigger=info
    )
    assert sid == "INC-001"
    inc = captured["saved"]
    assert inc.findings["trigger"]["name"] == "pd"
    assert inc.findings["trigger"]["transport"] == "webhook"
    assert inc.findings["trigger"]["target_app"] == "incident_management"


@pytest.mark.asyncio
async def test_orchestrator_start_session_without_trigger_skips_stamp():
    """Back-compat: trigger=None must not call store.save()."""
    from runtime.orchestrator import Orchestrator

    saved = []

    class _FakeStore:
        def create(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(id="INC-002", findings={}, **kwargs)

        def save(self, inc):
            saved.append(inc)

    class _FakeGraph:
        async def ainvoke(self, state, config):
            return None

    orch = Orchestrator.__new__(Orchestrator)
    orch.store = _FakeStore()
    orch.graph = _FakeGraph()
    orch._thread_config = lambda sid: {"configurable": {"thread_id": sid}}
    orch._is_graph_paused = lambda sid: _always_paused()
    orch.dedup_pipeline = None

    sid = await orch.start_session(query="q", environment="dev")
    assert sid == "INC-002"
    assert saved == []
