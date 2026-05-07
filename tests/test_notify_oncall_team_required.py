"""``notify_oncall`` team-roster guards.

Phase 7 (DECOUPLE-04 / D-07-03) replaced the module-level
``set_escalation_teams`` setter with a generic ``register(mcp_app, cfg)``
adapter. The orchestrator no longer mutates per-module globals
directly; it imports each module listed under
``cfg.orchestrator.mcp_servers`` and calls ``register(None, cfg)``.

These tests bind the roster the same way: by constructing a minimal
``cfg`` namespace whose ``framework.escalation_teams`` carries the
allowed list, then calling ``register(None, cfg)``. The tools' guards
read the snapshotted module-level tuple at call time, exactly as
production does.
"""
from types import SimpleNamespace

import pytest

from examples.incident_management.mcp_servers.remediation import (
    notify_oncall,
    register,
)


def _bind(teams: list[str]) -> None:
    """Install ``teams`` as the module's escalation roster via the
    Phase-7 ``register(mcp_app, cfg)`` contract.
    """
    cfg = SimpleNamespace(framework=SimpleNamespace(escalation_teams=list(teams)))
    register(None, cfg)


@pytest.mark.asyncio
async def test_notify_oncall_team_required():
    _bind(["platform-oncall", "data-oncall"])
    with pytest.raises(ValueError, match="team"):
        await notify_oncall(incident_id="INC-1", message="m", team="")


@pytest.mark.asyncio
async def test_notify_oncall_rejects_team_not_in_roster():
    _bind(["platform-oncall"])
    with pytest.raises(ValueError, match="not in escalation_teams"):
        await notify_oncall(incident_id="INC-1", message="m", team="random-team")


@pytest.mark.asyncio
async def test_notify_oncall_accepts_configured_team():
    _bind(["platform-oncall"])
    out = await notify_oncall(incident_id="INC-1", message="m", team="platform-oncall")
    assert out["team"] == "platform-oncall"
