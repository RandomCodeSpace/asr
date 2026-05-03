from contextlib import AsyncExitStack
import pytest

from examples.incident_management.state import IncidentState
from runtime.config import AppConfig, EmbeddingConfig, LLMConfig, MCPConfig, MCPServerConfig, MetadataConfig, ProviderConfig
from runtime.mcp_loader import load_tools
from examples.incident_management.mcp_server import set_state as set_inc_state
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.graph import build_graph, GraphState
from runtime.skill import load_all_skills


def _make_repo(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, state_cls=IncidentState, embedder=embedder)


def _make_history(store: SessionStore) -> HistoryStore:
    return HistoryStore(engine=store.engine, state_cls=IncidentState,
                        embedder=store.embedder, similarity_threshold=0.5)


@pytest.fixture
def cfg(tmp_path):
    store = _make_repo(tmp_path)
    set_inc_state(store=store, history=_make_history(store))
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
    )


@pytest.mark.asyncio
async def test_build_graph_compiles_with_4_agents(cfg, tmp_path):
    skills = load_all_skills("config/skills")
    store = _make_repo(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry)
        expected = {"intake", "triage", "deep_investigator", "resolution"}
        actual = set(graph.get_graph().nodes.keys())
        assert expected.issubset(actual)


@pytest.mark.asyncio
async def test_full_graph_runs_to_terminal_with_stub_llm(cfg, tmp_path):
    """End-to-end: graph runs through agents, gate interrupts on low
    confidence (P2-H), and the row carries pending_intervention. Stub DI
    never emits a confidence value, so the gate's interrupt() pauses the
    graph mid-flight rather than reaching resolution.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    skills = load_all_skills("config/skills")
    store = _make_repo(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        # P2-H: the gate calls langgraph.types.interrupt(), which requires a
        # Pregel checkpointer to capture paused state. Compile with an
        # InMemorySaver so the gate can interrupt cleanly.
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry,
                                  checkpointer=InMemorySaver())
        inc = store.create(query="api latency in production", environment="production",
                           reporter_id="user-mock", reporter_team="platform")
        final_state = await graph.ainvoke(
            GraphState(session=inc, next_route=None, last_agent=None, error=None),
            config={"configurable": {"thread_id": inc.id}},
        )
        # Either the graph reached the gate and interrupted (low/missing
        # confidence) → __interrupt__ key surfaces; or in the legacy short-
        # circuit path it ended at intake/resolution.
        if "__interrupt__" in final_state:
            # Gate paused: row should reflect awaiting_input + dual-write.
            reloaded = store.load(inc.id)
            assert reloaded.status == "awaiting_input"
            assert reloaded.pending_intervention is not None
        else:
            assert final_state["last_agent"] in {"resolution", "intake", "gate"}
        reloaded = store.load(inc.id)
        assert reloaded.agents_run, "expected at least one agent to run"


@pytest.mark.asyncio
async def test_build_graph_honours_entry_agent_from_config(cfg, tmp_path):
    """Entry node is whichever agent cfg.orchestrator.entry_agent names."""
    from runtime.config import OrchestratorConfig
    skills = load_all_skills("config/skills")
    store = _make_repo(tmp_path)
    # Override entry to triage.
    cfg2 = cfg.model_copy(update={
        "orchestrator": OrchestratorConfig(entry_agent="triage"),
    })
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg2.mcp, stack)
        graph = await build_graph(cfg=cfg2, skills=skills, store=store,
                                  registry=registry)
        compiled = graph.get_graph()
        # Find edges leaving __start__; the head should now be triage.
        edges_from_start = [
            e for e in compiled.edges if e.source == "__start__"
        ]
        targets = {e.target for e in edges_from_start}
        assert "triage" in targets


@pytest.mark.asyncio
async def test_build_graph_inserts_gate_for_gated_route(cfg, tmp_path):
    """A skill with a gate-marked route edge should result in the gate node
    being inserted between that agent and its target — there must be NO
    direct edge from deep_investigator to resolution; the path must go
    through the gate."""
    skills = load_all_skills("config/skills")
    store = _make_repo(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry)
        compiled = graph.get_graph()
        nodes = set(compiled.nodes.keys())
        assert "gate" in nodes
        # The gated edge (deep_investigator -> resolution, gate: confidence)
        # must redirect through the gate. There should be NO direct edge
        # from deep_investigator straight to resolution.
        di_to_resolution = [
            e for e in compiled.edges
            if e.source == "deep_investigator" and e.target == "resolution"
        ]
        assert not di_to_resolution, (
            f"expected no direct deep_investigator -> resolution edge "
            f"(must go through gate); got {di_to_resolution}"
        )
        # And the gate must connect to resolution.
        gate_to_resolution = [
            e for e in compiled.edges
            if e.source == "gate" and e.target == "resolution"
        ]
        assert gate_to_resolution, (
            f"expected gate -> resolution edge; got edges from gate: "
            f"{[e for e in compiled.edges if e.source == 'gate']}"
        )


@pytest.mark.asyncio
async def test_build_graph_raises_on_unknown_entry_agent(cfg, tmp_path):
    """Misconfigured entry_agent must raise loudly at build time, not silently
    produce a broken graph."""
    from runtime.config import OrchestratorConfig
    skills = load_all_skills("config/skills")
    store = _make_repo(tmp_path)
    cfg2 = cfg.model_copy(update={
        "orchestrator": OrchestratorConfig(entry_agent="nonexistent"),
    })
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg2.mcp, stack)
        with pytest.raises(ValueError, match="not a known skill"):
            await build_graph(cfg=cfg2, skills=skills, store=store,
                              registry=registry)
