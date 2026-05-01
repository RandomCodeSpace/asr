from contextlib import AsyncExitStack
import pytest
from orchestrator.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig
from orchestrator.incident import IncidentStore
from orchestrator.mcp_loader import load_tools
from orchestrator.mcp_servers.incident import set_state as set_inc_state
from orchestrator.graph import build_graph, GraphState
from orchestrator.skill import load_all_skills


@pytest.fixture
def cfg(tmp_path):
    set_inc_state(store=IncidentStore(tmp_path), similarity_threshold=0.5)
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
    )


@pytest.mark.asyncio
async def test_build_graph_compiles_with_4_agents(cfg, tmp_path):
    skills = load_all_skills("config/skills.yaml")
    store = IncidentStore(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry)
        expected = {"intake", "triage", "deep_investigator", "resolution"}
        actual = set(graph.get_graph().nodes.keys())
        assert expected.issubset(actual)


@pytest.mark.asyncio
async def test_full_graph_runs_to_terminal_with_stub_llm(cfg, tmp_path):
    skills = load_all_skills("config/skills.yaml")
    store = IncidentStore(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry)
        inc = store.create(query="api latency in production", environment="production",
                           reporter_id="user-mock", reporter_team="platform")
        final_state = await graph.ainvoke(
            GraphState(incident=inc, next_route=None, last_agent=None, error=None)
        )
        # Stub DI never emits a confidence value, so the gate halts the graph
        # before resolution. last_agent is then "gate" (or "intake" on the
        # known-issue short-circuit path).
        assert final_state["last_agent"] in {"resolution", "intake", "gate"}
        reloaded = store.load(inc.id)
        assert reloaded.agents_run, "expected at least one agent to run"
