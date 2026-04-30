import pytest
from pathlib import Path
from orchestrator.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths
from orchestrator.orchestrator import Orchestrator


@pytest.fixture
def cfg(tmp_path):
    skills_dir = Path("config/skills")
    return AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="orchestrator.mcp_servers.incident",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="orchestrator.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="orchestrator.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="orchestrator.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir=str(skills_dir), incidents_dir=str(tmp_path)),
    )


@pytest.mark.asyncio
async def test_list_agents_returns_4_with_metadata(cfg):
    orch = await Orchestrator.create(cfg)
    agents = orch.list_agents()
    names = {a["name"] for a in agents}
    assert names == {"intake", "triage", "deep_investigator", "resolution"}
    intake = next(a for a in agents if a["name"] == "intake")
    assert "lookup_similar_incidents" in intake["tools"]


@pytest.mark.asyncio
async def test_list_tools_returns_grouped_by_category(cfg):
    orch = await Orchestrator.create(cfg)
    tools = orch.list_tools()
    cats = {t["category"] for t in tools}
    assert {"incident_management", "observability", "remediation", "user_context"} <= cats
    # Each tool reports which agents use it
    lookup = next(t for t in tools if t["name"] == "lookup_similar_incidents")
    assert "intake" in lookup["bound_agents"]


@pytest.mark.asyncio
async def test_start_investigation_creates_incident_and_runs_graph(cfg):
    orch = await Orchestrator.create(cfg)
    inc_id = await orch.start_investigation(query="api latency", environment="production")
    assert inc_id.startswith("INC-")
    inc = orch.get_incident(inc_id)
    assert inc["status"] in {"in_progress", "matched", "resolved", "escalated", "new"}
    assert inc["agents_run"]


@pytest.mark.asyncio
async def test_stream_events_yields_at_least_one(cfg):
    orch = await Orchestrator.create(cfg)
    events = []
    async for ev in orch.stream_investigation(query="api latency", environment="production"):
        events.append(ev)
    # Expect at least: start, an agent enter, an agent exit, end
    assert any(e["event"] == "investigation_started" for e in events)
    assert any(e["event"] == "investigation_completed" for e in events)
