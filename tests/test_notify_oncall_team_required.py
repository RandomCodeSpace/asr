import pytest

from runtime.mcp_servers.remediation import (
    notify_oncall, set_escalation_teams,
)


@pytest.mark.asyncio
async def test_notify_oncall_team_required():
    set_escalation_teams(["platform-oncall", "data-oncall"])
    with pytest.raises(ValueError, match="team"):
        await notify_oncall(incident_id="INC-1", message="m", team="")


@pytest.mark.asyncio
async def test_notify_oncall_rejects_team_not_in_roster():
    set_escalation_teams(["platform-oncall"])
    with pytest.raises(ValueError, match="not in escalation_teams"):
        await notify_oncall(incident_id="INC-1", message="m", team="random-team")


@pytest.mark.asyncio
async def test_notify_oncall_accepts_configured_team():
    set_escalation_teams(["platform-oncall"])
    out = await notify_oncall(incident_id="INC-1", message="m", team="platform-oncall")
    assert out["team"] == "platform-oncall"
