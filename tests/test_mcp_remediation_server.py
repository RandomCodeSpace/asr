import pytest
from examples.incident_management.mcp_servers.remediation import propose_fix, apply_fix, notify_oncall


@pytest.mark.asyncio
async def test_propose_fix_returns_proposal_with_safe_flag():
    out = await propose_fix(hypothesis="memory leak in worker", environment="production")
    assert "proposal" in out
    assert "auto_apply_safe" in out
    assert isinstance(out["auto_apply_safe"], bool)


@pytest.mark.asyncio
async def test_apply_fix_returns_status():
    out = await apply_fix(proposal_id="prop-001", environment="production")
    assert out["status"] in {"applied", "failed"}


@pytest.mark.asyncio
async def test_notify_oncall_returns_page_id():
    out = await notify_oncall(incident_id="INC-1", message="escalating",
                              team="platform-oncall")
    assert "page_id" in out
