from contextlib import AsyncExitStack
import pytest

from runtime.config import EmbeddingConfig, MCPConfig, MCPServerConfig, MetadataConfig, ProviderConfig
from runtime.mcp_loader import load_tools, ToolRegistry
from examples.incident_management.mcp_server import set_state as set_inc_state
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


def _make_repo(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    store = SessionStore(engine=eng, embedder=embedder)
    history = HistoryStore(engine=eng, embedder=embedder,
                           similarity_threshold=0.5)
    return store, history


@pytest.fixture
def cfg(tmp_path):
    store, history = _make_repo(tmp_path)
    set_inc_state(store=store, history=history)
    return MCPConfig(servers=[
        MCPServerConfig(
            name="local_inc", transport="in_process",
            module="examples.incident_management.mcp_server",
            category="incident_management",
        ),
        MCPServerConfig(
            name="local_observability", transport="in_process",
            module="runtime.mcp_servers.observability",
            category="observability",
        ),
        MCPServerConfig(
            name="local_user", transport="in_process",
            module="runtime.mcp_servers.user_context",
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
        # Entries are now keyed by (server, original_name)
        key = ("local_inc", "lookup_similar_incidents")
        assert key in registry.entries
        assert registry.entries[key].category == "incident_management"
        obs_key = ("local_observability", "get_logs")
        assert registry.entries[obs_key].category == "observability"


@pytest.mark.asyncio
async def test_registry_tools_are_name_prefixed(cfg):
    """Each tool's LangChain .name must be '<server>:<original>'."""
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        entry = registry.entries[("local_inc", "lookup_similar_incidents")]
        assert entry.tool.name == "local_inc:lookup_similar_incidents"
        assert entry.name == "lookup_similar_incidents"


@pytest.mark.asyncio
async def test_registry_resolve_wildcard_local(cfg):
    """resolve with local:['*'] returns all in-process tools."""
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        tools = registry.resolve({"local": ["*"]}, cfg)
        names = {t.name for t in tools}
        assert "local_inc:lookup_similar_incidents" in names
        assert "local_observability:get_logs" in names
        assert "local_user:get_user_context" in names
        # disabled server must not appear
        assert not any("external_off" in n for n in names)


@pytest.mark.asyncio
async def test_registry_resolve_selective_local(cfg):
    """resolve with specific local names returns only those tools."""
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        tools = registry.resolve(
            {"local": ["lookup_similar_incidents", "get_logs"]}, cfg
        )
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"local_inc:lookup_similar_incidents",
                         "local_observability:get_logs"}


@pytest.mark.asyncio
async def test_registry_resolve_wildcard_named_server(cfg):
    """resolve with <server>:['*'] returns all tools from that server."""
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        tools = registry.resolve({"local_inc": ["*"]}, cfg)
        names = {t.name for t in tools}
        assert all(n.startswith("local_inc:") for n in names)
        assert "local_inc:lookup_similar_incidents" in names


@pytest.mark.asyncio
async def test_registry_resolve_selective_named_server(cfg):
    """resolve with specific named-server tool returns exactly that tool."""
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        tools = registry.resolve({"local_observability": ["get_logs"]}, cfg)
        assert len(tools) == 1
        assert tools[0].name == "local_observability:get_logs"


@pytest.mark.asyncio
async def test_registry_resolve_unknown_server_raises(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        with pytest.raises(ValueError, match="unknown server"):
            registry.resolve({"ghost_server": ["*"]}, cfg)


@pytest.mark.asyncio
async def test_registry_resolve_unknown_tool_raises(cfg):
    async with AsyncExitStack() as stack:
        registry: ToolRegistry = await load_tools(cfg, stack)
        with pytest.raises(ValueError, match="does_not_exist"):
            registry.resolve({"local": ["does_not_exist"]}, cfg)


@pytest.mark.asyncio
async def test_loaded_tool_is_invocable_after_load_returns(cfg):
    """Pin the closed-transport bug: tools must work after load_tools returns.

    Before the fix, ``load_tools`` opened a FastMCP ``Client`` inside an
    ``async with`` block and returned the LangChain wrappers AFTER the client
    closed. The first ainvoke would raise:
        unable to perform operation on <TCPTransport closed=True ...>
    With the AsyncExitStack-based contract the client stays alive until the
    stack is closed, so the tool can be invoked normally.
    """
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg, stack)
        entry = registry.entries[("local_user", "get_user_context")]
        # invoke the tool — must not raise "TCPTransport closed"
        result = await entry.tool.ainvoke({"user_id": "test-user"})
        assert "team" in str(result) or "user_id" in str(result)
