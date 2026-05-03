import pytest
from runtime.mcp_servers.user_context import get_user_context


@pytest.mark.asyncio
async def test_get_user_context_returns_team_and_role():
    out = await get_user_context(user_id="user-mock")
    assert out["user_id"] == "user-mock"
    assert "team" in out
    assert "role" in out
