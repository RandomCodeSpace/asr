import pytest
from orchestrator.graph import GraphState, make_agent_node
from orchestrator.incident import IncidentStore, TokenUsage
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
    intake_runs = [r for r in reloaded.agents_run if r.agent == "intake"]
    assert intake_runs
    # Token usage field must exist on every AgentRun and on the incident.
    # StubChatModel does not emit usage_metadata, so we assert structure
    # (the fields are present and are TokenUsage instances) rather than values.
    assert isinstance(intake_runs[0].token_usage, TokenUsage)
    assert intake_runs[0].token_usage.total_tokens == 0
    assert isinstance(reloaded.token_usage, TokenUsage)
    assert reloaded.token_usage.total_tokens == 0
