import pytest
from orchestrator.graph import GraphState, make_agent_node
from orchestrator.incident import Incident, IncidentStore, Reporter
from orchestrator.skill import Skill, RouteRule
from orchestrator.llm import StubChatModel


@pytest.fixture
def incident(tmp_path):
    store = IncidentStore(tmp_path)
    return store.create(query="api latency", environment="dev",
                        reporter_id="u", reporter_team="t"), store


@pytest.mark.asyncio
async def test_agent_node_runs_llm_records_agent_run_and_routes(incident):
    inc, store = incident
    skill = Skill(
        name="intake", description="d",
        routes=[RouteRule(when="default", next="triage")],
        system_prompt="You are intake.",
    )
    llm = StubChatModel(role="intake", canned_responses={"intake": "ok"})
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    out = await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    assert out["next_route"] == "triage"
    assert out["last_agent"] == "intake"
    reloaded = store.load(inc.id)
    assert any(r.agent == "intake" for r in reloaded.agents_run)
