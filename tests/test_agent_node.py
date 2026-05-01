import logging

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
    # Stub does not emit a confidence patch, so AgentRun.confidence stays None.
    assert intake_runs[0].confidence is None
    assert intake_runs[0].confidence_rationale is None


@pytest.mark.asyncio
async def test_agent_node_captures_confidence_from_update_incident(incident):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    # The stub emits a single update_incident tool call carrying confidence.
    # The graph node should capture it and stamp it on the AgentRun.
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {
                "incident_id": inc.id,
                "patch": {
                    "severity": "sev3",
                    "confidence": 0.83,
                    "confidence_rationale": "deploy correlates with timing",
                },
            },
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == 0.83
    assert triage_runs[0].confidence_rationale == "deploy correlates with timing"


def _build_node_with_confidence_patch(inc, store, *, conf_value):
    """Helper that wires a stub LLM emitting a single update_incident with the
    given confidence value (any type — bool, str, float, junk).
    """
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {
                "incident_id": inc.id,
                "patch": {"confidence": conf_value},
            },
        }],
    )
    return make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )


@pytest.mark.asyncio
async def test_confidence_rejects_bool(incident, caplog):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value=True)
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    # bool must be rejected — confidence stays None
    assert triage_runs[0].confidence is None
    assert any("bool" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("label,expected", [
    ("high", 0.9), ("HIGH", 0.9), ("High", 0.9),
    ("medium", 0.6), ("Medium", 0.6),
    ("low", 0.3), ("LOW", 0.3),
])
async def test_confidence_coerces_string_labels(incident, label, expected):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value=label)
    await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [(5.0, 1.0), (-1.0, 0.0), (1.5, 1.0), (-0.2, 0.0)])
async def test_confidence_clamps_out_of_range(incident, caplog, raw, expected):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value=raw)
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == expected
    # A clamp warning must be emitted.
    assert any("clamp" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_confidence_unknown_string_is_none(incident, caplog):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value="meh")
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence is None
    assert any("meh" in rec.getMessage() for rec in caplog.records)
