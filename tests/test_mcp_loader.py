from contextlib import AsyncExitStack
import pytest
from orchestrator.config import MCPConfig, MCPServerConfig
from orchestrator.incident import IncidentStore
from orchestrator.mcp_loader import load_tools, ToolRegistry
from orchestrator.mcp_servers.incident import set_state as set_inc_state


@pytest.fixture
def cfg(tmp_path):
    set_inc_state(store=IncidentStore(tmp_path), similarity_threshold=0.5)
    return MCPConfig(servers=[
        MCPServerConfig(
            name="local_inc", transport="in_process",
            module="orchestrator.mcp_servers.incident",
            category="incident_management",
        ),
        MCPServerConfig(
            name="local_observability", transport="in_process",
            module="orchestrator.mcp_servers.observability",
            category="observability",
        ),
        MCPServerConfig(
            name="local_user", transport="in_process",
            module="orchestrator.mcp_servers.user_context",
            category="user_context",
        ),
        MCPServerConfig(
            name="external_off", transport="http", url="http://x.example/mcp",
            category="ticketing", enabled=False,
        ),
    ])


@pytest.mark.asyncio
async def test_loader_skips_disabled_servers(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        server_names = {entry.server for entry in registry.entries.values()}
        assert "external_off" not in server_names


@pytest.mark.asyncio
async def test_loader_builds_categorized_registry(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        assert "lookup_similar_incidents" in registry.entries
        assert registry.entries["lookup_similar_incidents"].category == "incident_management"
        assert registry.entries["get_logs"].category == "observability"


@pytest.mark.asyncio
async def test_registry_get_tools_for_subset(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        tools = registry.get(["lookup_similar_incidents", "get_logs"])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"lookup_similar_incidents", "get_logs"}


@pytest.mark.asyncio
async def test_registry_get_unknown_tool_raises(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        with pytest.raises(KeyError, match="does_not_exist"):
            registry.get(["does_not_exist"])
