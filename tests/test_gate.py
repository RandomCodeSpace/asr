"""Tests for the intervention gate node between deep_investigator and resolution."""
import pytest
from pytest import approx

from langgraph.errors import GraphInterrupt

from examples.incident_management.state import IncidentState
from orchestrator.config import AppConfig, EmbeddingConfig, LLMConfig, MCPConfig, MetadataConfig, ProviderConfig
from runtime.config import FrameworkAppConfig
from orchestrator.graph import (
    GraphState, make_agent_node, make_gate_node, _coerce_confidence,
)
from runtime.state import AgentRun
from orchestrator.llm import StubChatModel
from orchestrator.skill import RouteRule, Skill
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.session_store import SessionStore


def _make_repo(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, state_cls=IncidentState, embedder=embedder)


def _cfg(threshold: float = 0.75) -> AppConfig:
    # NOTE: confidence_threshold + escalation_teams now flow through
    # FrameworkAppConfig as explicit make_gate_node parameters (post-merge
    # FrameworkAppConfig refactor). Tests pass them in directly so the gate
    # is no longer reaching for examples.incident_management.config at all.
    assert threshold == 0.75, (
        "test_gate._cfg: only the default 0.75 threshold is exercised; "
        "pass threshold explicitly to make_gate_node to override."
    )
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(),
    )


# Stand-in for the cross-cutting framework knobs the gate reads. Mirrors
# the incident-management defaults so the existing assertions on
# pending_intervention.escalation_teams keep working.
def _incident_framework_cfg() -> FrameworkAppConfig:
    return FrameworkAppConfig(
        confidence_threshold=0.75,
        escalation_teams=["platform-oncall", "data-oncall", "security-oncall"],
    )


def _seed(store: SessionStore, *, di_confidence: float | None):
    inc = store.create(query="api latency", environment="production",
                       reporter_id="u", reporter_team="t")
    # Pretend deep_investigator already ran and stamped a confidence value.
    inc.agents_run.append(AgentRun(
        agent="deep_investigator",
        started_at="2026-04-30T10:00:00Z",
        ended_at="2026-04-30T10:00:30Z",
        summary="hypothesis emitted",
        confidence=di_confidence,
    ))
    store.save(inc)
    return inc


async def _drive_gate_via_graph(store, inc, *, last_agent="deep_investigator",
                                 next_route=None, resume_value=None):
    """Compile a tiny single-node graph around ``gate`` (with checkpointer)
    and invoke it so ``interrupt()`` runs inside a Pregel context.

    LangGraph's modern ``interrupt()`` does not propagate as an
    exception out of ``ainvoke``; instead the result dict carries an
    ``__interrupt__`` key listing the captured ``Interrupt`` objects.
    This helper returns ``(result, interrupted)`` — ``interrupted`` is
    the list of ``Interrupt`` objects (or ``None`` when the gate
    completed without interrupting). ``resume_value`` (when given)
    drives a second invocation via ``Command(resume=resume_value)``
    and the second invocation's result is what's returned.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END
    from langgraph.types import Command

    sg = StateGraph(GraphState)
    fw = _incident_framework_cfg()
    sg.add_node("gate", make_gate_node(
        cfg=_cfg(), store=store,
        threshold=fw.confidence_threshold,
        teams=list(fw.escalation_teams),
    ))
    sg.set_entry_point("gate")
    sg.add_edge("gate", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": inc.id}}

    result = await compiled.ainvoke(
        GraphState(session=inc, next_route=next_route,
                   last_agent=last_agent, error=None),
        config=config,
    )
    interrupted = result.get("__interrupt__") if isinstance(result, dict) else None

    if resume_value is not None:
        result = await compiled.ainvoke(
            Command(resume=resume_value), config=config,
        )

    return result, interrupted


@pytest.mark.asyncio
async def test_gate_pauses_when_no_di_confidence(tmp_path):
    """P2-H: missing confidence → gate dual-writes Session row + interrupts."""
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=None)
    _, interrupted = await _drive_gate_via_graph(store, inc)
    assert interrupted, "gate must surface an Interrupt on low confidence"
    reloaded = store.load(inc.id)
    assert reloaded.status == "awaiting_input"
    assert reloaded.pending_intervention["reason"] == "low_confidence"
    assert reloaded.pending_intervention["threshold"] == approx(0.75)
    assert reloaded.pending_intervention["confidence"] is None
    assert set(reloaded.pending_intervention["options"]) == {
        "resume_with_input", "escalate", "stop",
    }
    assert "platform-oncall" in reloaded.pending_intervention["escalation_teams"]


@pytest.mark.asyncio
async def test_gate_pauses_when_below_threshold(tmp_path):
    """P2-H: low confidence below threshold → row dual-write + interrupt."""
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.42)
    _, interrupted = await _drive_gate_via_graph(store, inc)
    assert interrupted
    reloaded = store.load(inc.id)
    assert reloaded.status == "awaiting_input"
    assert reloaded.pending_intervention["confidence"] == approx(0.42)


@pytest.mark.asyncio
async def test_gate_passes_when_at_or_above_threshold(tmp_path):
    """High confidence → no interrupt; gate falls through to the gated target."""
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.9)
    result, interrupted = await _drive_gate_via_graph(store, inc)
    assert not interrupted, "high-confidence gate must not interrupt"
    assert result is not None
    reloaded = store.load(inc.id)
    # Status stays whatever DI left it (e.g. in_progress); no intervention payload.
    assert reloaded.pending_intervention is None
    assert reloaded.status != "awaiting_input"


@pytest.mark.asyncio
async def test_gate_pauses_when_string_label_confidence_below_threshold(tmp_path):
    """End-to-end: an LLM emits a "low" string label, the agent node coerces
    it via _coerce_confidence (fix 1), the gate then pauses on the resulting
    0.3 because it's below 0.75. This pins that fix-1 plays nicely with the
    gate's threshold logic.
    """
    store = _make_repo(tmp_path)
    inc = store.create(query="api latency", environment="production",
                       reporter_id="u", reporter_team="t")
    skill = Skill(
        name="deep_investigator", description="d",
        routes=[RouteRule(when="default", next="resolution")],
        system_prompt="You are DI.",
    )
    llm = StubChatModel(
        role="deep_investigator",
        canned_responses={"deep_investigator": "weak"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"confidence": "low"}},
        }],
    )
    di_node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda i: "default", store=store,
    )
    await di_node(GraphState(session=inc, next_route=None,
                             last_agent=None, error=None))
    # DI run recorded with the coerced 0.3 value.
    assert _coerce_confidence("low") == approx(0.3)
    reloaded = store.load(inc.id)
    di_runs = [r for r in reloaded.agents_run if r.agent == "deep_investigator"]
    assert di_runs and di_runs[-1].confidence == approx(0.3)
    # Gate sees 0.3 < 0.75 → interrupts (now goes through a real graph).
    _, interrupted = await _drive_gate_via_graph(store, reloaded)
    assert interrupted
    assert store.load(inc.id).status == "awaiting_input"


@pytest.mark.asyncio
async def test_gate_clears_stale_intervention_on_pass(tmp_path):
    """If a prior run set pending_intervention but the new DI is confident,
    the gate should clear the old payload before forwarding to resolution."""
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.92)
    inc.pending_intervention = {"reason": "low_confidence", "confidence": 0.3,
                                "threshold": 0.75,
                                "options": ["stop"], "escalation_teams": []}
    store.save(inc)
    # High confidence path doesn't call interrupt(), so direct invocation works.
    fw = _incident_framework_cfg()
    gate = make_gate_node(
        cfg=_cfg(threshold=0.75), store=store,
        threshold=fw.confidence_threshold,
        teams=list(fw.escalation_teams),
    )
    out = await gate(GraphState(session=inc, next_route=None,
                                last_agent="deep_investigator", error=None))
    assert out["next_route"] == "default"
    reloaded = store.load(inc.id)
    assert reloaded.pending_intervention is None


@pytest.mark.asyncio
async def test_gate_dual_writes_pending_intervention_and_interrupts(tmp_path):
    """P2-H: when confidence is low, gate must:
       1. Persist Session.pending_intervention to disk (dual-write).
       2. Surface a LangGraph Interrupt so the checkpointer pauses.

    The row save MUST happen before interrupt() — otherwise a UI that
    polls Session.pending_intervention would never see the pending state.
    """
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.42)
    _, interrupted = await _drive_gate_via_graph(
        store, inc, last_agent="deep_investigator", next_route="resolution",
    )

    # 1. interrupt() captured → checkpointer paused the graph.
    assert interrupted, "expected an Interrupt entry to surface from the gate"
    first = interrupted[0]
    payload_value = getattr(first, "value", first)
    assert isinstance(payload_value, dict)
    assert payload_value["reason"] == "low_confidence"
    assert payload_value["confidence"] == approx(0.42)

    # 2. Dual-write: same payload landed on the Session row before the
    #    interrupt suspended execution.
    reloaded = store.load(inc.id)
    assert reloaded.status == "awaiting_input"
    assert reloaded.pending_intervention is not None
    assert reloaded.pending_intervention["reason"] == "low_confidence"
    assert reloaded.pending_intervention["confidence"] == approx(0.42)
    assert reloaded.pending_intervention["intended_target"] == "resolution"


# ---------------------------------------------------------------------------
# Post-merge FrameworkAppConfig refactor: gate must honour whichever
# threshold/teams the caller (build_graph) supplied — not bake in
# incident-management defaults. Pins #1 from the code-review fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_uses_caller_supplied_threshold_and_teams(tmp_path):
    """A non-incident FrameworkAppConfig must drive the gate's behaviour.

    Use a code-review-shaped framework cfg (different threshold, empty
    roster) and confirm the resulting pending_intervention payload
    surfaces those values. Catches a regression where the gate was
    pulling values from examples.incident_management.config at build time.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END

    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.55)  # below 0.7 cutoff

    code_review_fw = FrameworkAppConfig(
        confidence_threshold=0.7,
        escalation_teams=["code-owners-payments"],
    )

    sg = StateGraph(GraphState)
    sg.add_node("gate", make_gate_node(
        cfg=_cfg(), store=store,
        threshold=code_review_fw.confidence_threshold,
        teams=list(code_review_fw.escalation_teams),
    ))
    sg.set_entry_point("gate")
    sg.add_edge("gate", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": inc.id}}

    result = await compiled.ainvoke(
        GraphState(session=inc, next_route=None,
                   last_agent="deep_investigator", error=None),
        config=config,
    )
    interrupted = result.get("__interrupt__") if isinstance(result, dict) else None
    assert interrupted, "0.55 < 0.7 must interrupt"

    reloaded = store.load(inc.id)
    pi = reloaded.pending_intervention
    assert pi["threshold"] == approx(0.7)
    # Critical: NOT the incident default ("platform-oncall", ...).
    assert pi["escalation_teams"] == ["code-owners-payments"]
    assert "platform-oncall" not in pi["escalation_teams"]


@pytest.mark.asyncio
async def test_gate_default_framework_threshold_is_0_75(tmp_path):
    """Calling make_gate_node without threshold/teams falls back to the
    bare FrameworkAppConfig() defaults (0.75 / empty roster) — preserves
    back-compat for tests that build the gate directly without an
    orchestrator."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END

    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.5)

    sg = StateGraph(GraphState)
    sg.add_node("gate", make_gate_node(cfg=_cfg(), store=store))
    sg.set_entry_point("gate")
    sg.add_edge("gate", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": inc.id}}

    result = await compiled.ainvoke(
        GraphState(session=inc, next_route=None,
                   last_agent="deep_investigator", error=None),
        config=config,
    )
    interrupted = result.get("__interrupt__") if isinstance(result, dict) else None
    assert interrupted, "0.5 < 0.75 default must interrupt"
    reloaded = store.load(inc.id)
    assert reloaded.pending_intervention["threshold"] == approx(0.75)
    # Default escalation_teams is empty for the bare framework cfg.
    assert reloaded.pending_intervention["escalation_teams"] == []


@pytest.mark.asyncio
async def test_gate_resume_appends_user_input_and_clears_intervention(tmp_path):
    """P2-H: resuming the gate via Command(resume=user_input) must:
       - Append the input to session.user_inputs.
       - Clear pending_intervention.
       - Move status off awaiting_input.
       - Fall through to the gated target (next_route == "default").
    """
    store = _make_repo(tmp_path)
    inc = _seed(store, di_confidence=0.42)
    result, interrupted = await _drive_gate_via_graph(
        store, inc, last_agent="deep_investigator",
        next_route="resolution",
        resume_value="operator note: db pool exhausted",
    )
    assert interrupted, "first invocation must interrupt"
    assert result is not None, "second invocation (resume) must return a result"

    reloaded = store.load(inc.id)
    assert "operator note: db pool exhausted" in reloaded.user_inputs
    assert reloaded.pending_intervention is None
    assert reloaded.status == "in_progress"
