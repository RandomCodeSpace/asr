import logging

import pytest
from pytest import approx
from runtime.config import EmbeddingConfig, MetadataConfig, ProviderConfig
from runtime.graph import GraphState, _decide_from_signal, make_agent_node
from runtime.state import TokenUsage
from runtime.skill import Skill, RouteRule
from runtime.llm import StubChatModel
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.event_log import EventLog
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore

# Phase 6 (DECOUPLE-02): the harvester recognition surface comes from
# OrchestratorConfig.terminal_tools / patch_tools / harvest_terminal_tools.
# Tests pass them explicitly so they don't depend on a global registry.
_TEST_TERMINAL_NAMES = frozenset({
    "mark_resolved", "mark_escalated", "submit_hypothesis",
})
_TEST_PATCH_NAMES = frozenset({"update_incident"})


def _make_repo(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, embedder=embedder)


@pytest.fixture
def incident(tmp_path):
    store = _make_repo(tmp_path)
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
        event_log=EventLog(engine=store.engine),
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    out = await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
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
    # Phase 10 (FOC-03): the runner now wraps every turn in an
    # AgentTurnOutput envelope; StubChatModel.with_structured_output
    # populates result["structured_response"] with the configured
    # default envelope (0.85 confidence, "stub envelope rationale").
    # The runner stamps these onto the AgentRun.
    assert intake_runs[0].confidence == approx(0.85)
    assert intake_runs[0].confidence_rationale == "stub envelope rationale"
    events = list(EventLog(engine=store.engine).iter_for(inc.id))
    running = [e for e in events if e.kind == "session.agent_running"]
    assert running
    assert running[0].payload == {"id": inc.id, "agent": "intake"}


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
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == approx(0.83)
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
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )


@pytest.mark.asyncio
async def test_confidence_rejects_bool(incident, caplog):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value=True)
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    # The bool patch-tool-arg confidence must be rejected (harvested → None).
    # Phase 10 (FOC-03): when the harvest yields None, the envelope's
    # confidence becomes the recorded value (reconcile_confidence falls
    # through to the envelope when tool_arg_value is None). The bool
    # rejection itself is still asserted via the WARN log.
    assert triage_runs[0].confidence == approx(0.85)
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
    await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [(5.0, 1.0), (-1.0, 0.0), (1.5, 1.0), (-0.2, 0.0)])
async def test_confidence_clamps_out_of_range(incident, caplog, raw, expected):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value=raw)
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    assert triage_runs[0].confidence == approx(expected)
    # A clamp warning must be emitted.
    assert any("clamp" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_confidence_unknown_string_is_none(incident, caplog):
    inc, store = incident
    node = _build_node_with_confidence_patch(inc, store, conf_value="meh")
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs
    # Unknown-string patch-tool-arg confidence is rejected (harvested → None).
    # Phase 10 (FOC-03): the envelope's confidence becomes the recorded value
    # via reconcile_confidence's tool_arg_value=None fallthrough. The
    # WARN log still names the offending value.
    assert triage_runs[0].confidence == approx(0.85)
    assert any("meh" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_agent_node_captures_signal_from_update_incident(incident):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="success", next="deep_investigator"),
                RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "success"}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs and triage_runs[-1].signal == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    ("success", "success"), ("SUCCESS", "success"),
    ("failed", "failed"), ("Failed", "failed"),
    ("needs_input", "needs_input"),
])
async def test_agent_node_signal_normalises_case(incident, raw, expected):
    inc, store = incident
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
            "args": {"incident_id": inc.id,
                     "patch": {"signal": raw}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal == expected


@pytest.mark.asyncio
async def test_agent_node_signal_unknown_string_is_none(incident, caplog):
    inc, store = incident
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
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "halfway"}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal is None
    assert any("halfway" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_agent_node_signal_rejects_bool(incident, caplog):
    inc, store = incident
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
            "args": {"incident_id": inc.id,
                     "patch": {"signal": True}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(session=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal is None
    assert any("bool" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_agent_node_routes_on_emitted_signal(incident):
    """When the agent emits signal=failed, the node should pick the route
    rule with when=failed even though when=default also matches."""
    inc, store = incident
    skill = Skill(
        name="intake", description="d",
        routes=[
            RouteRule(when="success", next="triage"),
            RouteRule(when="failed", next="__end__"),
            RouteRule(when="default", next="triage"),
        ],
        system_prompt="You are intake.",
    )
    llm = StubChatModel(
        role="intake",
        canned_responses={"intake": "no luck"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "failed"}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=_decide_from_signal,
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    out = await node(GraphState(session=inc, next_route=None,
                                last_agent=None, error=None))
    assert out["next_route"] == "__end__"


@pytest.mark.asyncio
async def test_agent_node_falls_back_to_default_when_no_signal(incident):
    """When the agent emits no signal, the decider returns "default" and
    the node picks the route rule with when=default."""
    inc, store = incident
    skill = Skill(
        name="intake", description="d",
        routes=[
            RouteRule(when="success", next="resolution"),
            RouteRule(when="default", next="triage"),
        ],
        system_prompt="You are intake.",
    )
    # Stub emits no signal at all.
    llm = StubChatModel(role="intake", canned_responses={"intake": "ok"})
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=_decide_from_signal,
        store=store,
        terminal_tool_names=_TEST_TERMINAL_NAMES,
        patch_tool_names=_TEST_PATCH_NAMES,
    )
    out = await node(GraphState(session=inc, next_route=None,
                                last_agent=None, error=None))
    assert out["next_route"] == "triage"
