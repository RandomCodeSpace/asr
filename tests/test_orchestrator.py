import pytest
from runtime.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths, RuntimeConfig
from runtime.orchestrator import Orchestrator


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="runtime.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="runtime.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="runtime.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class=None,
        ),
    )


@pytest.mark.asyncio
async def test_list_agents_returns_4_with_metadata(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        agents = orch.list_agents()
        names = {a["name"] for a in agents}
        assert names == {"intake", "triage", "deep_investigator", "resolution"}
        intake = next(a for a in agents if a["name"] == "intake")
        # Tools are now prefixed: "<server>:<original>"
        assert "local_inc:lookup_similar_incidents" in intake["tools"]
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_list_tools_returns_grouped_by_category(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        tools = orch.list_tools()
        cats = {t["category"] for t in tools}
        assert {"incident_management", "observability", "remediation", "user_context"} <= cats
        # Each tool reports which agents use it; name is now prefixed
        lookup = next(t for t in tools
                      if t["name"] == "local_inc:lookup_similar_incidents")
        assert "intake" in lookup["bound_agents"]
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_start_investigation_creates_incident_and_runs_graph(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = await orch.start_investigation(query="api latency", environment="production")
        # Test fixture builds AppConfig without overriding
        # ``session_id_prefix``, so the framework default ``SES``
        # applies. Real deployments configure this in YAML.
        assert inc_id.startswith("SES-")
        inc = orch.get_incident(inc_id)
        # Stub LLM emits no confidence → gate halts and marks awaiting_input.
        assert inc["status"] in {
            "in_progress", "matched", "resolved", "escalated",
            "new", "awaiting_input", "stopped",
        }
        assert inc["agents_run"]
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_stream_events_yields_at_least_one(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        events = []
        async for ev in orch.stream_investigation(query="api latency", environment="production"):
            events.append(ev)
        # Expect at least: start, an agent enter, an agent exit, end
        assert any(e["event"] == "investigation_started" for e in events)
        assert any(e["event"] == "investigation_completed" for e in events)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent(cfg):
    """aclose() must be safe to call multiple times (e.g., async-with + finally)."""
    orch = await Orchestrator.create(cfg)
    await orch.aclose()
    # Second close must not raise
    await orch.aclose()


@pytest.mark.asyncio
async def test_async_context_manager(cfg):
    """Orchestrator supports `async with` for clean teardown."""
    async with await Orchestrator.create(cfg) as orch:
        assert orch.list_agents()
