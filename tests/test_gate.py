"""Tests for the intervention gate node between deep_investigator and resolution."""
import pytest

from orchestrator.config import AppConfig, InterventionConfig, LLMConfig, MCPConfig
from orchestrator.graph import GraphState, make_gate_node
from orchestrator.incident import AgentRun, IncidentStore


def _cfg(threshold: float = 0.75) -> AppConfig:
    return AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
        mcp=MCPConfig(),
        intervention=InterventionConfig(confidence_threshold=threshold),
    )


def _seed(store: IncidentStore, *, di_confidence: float | None):
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


@pytest.mark.asyncio
async def test_gate_pauses_when_no_di_confidence(tmp_path):
    store = IncidentStore(tmp_path)
    inc = _seed(store, di_confidence=None)
    gate = make_gate_node(cfg=_cfg(), store=store)
    out = await gate(GraphState(incident=inc, next_route=None,
                                last_agent="deep_investigator", error=None))
    assert out["next_route"] == "__end__"
    reloaded = store.load(inc.id)
    assert reloaded.status == "awaiting_input"
    assert reloaded.pending_intervention["reason"] == "low_confidence"
    assert reloaded.pending_intervention["threshold"] == 0.75
    assert reloaded.pending_intervention["confidence"] is None
    assert set(reloaded.pending_intervention["options"]) == {
        "resume_with_input", "escalate", "stop",
    }
    assert "platform-oncall" in reloaded.pending_intervention["escalation_teams"]


@pytest.mark.asyncio
async def test_gate_pauses_when_below_threshold(tmp_path):
    store = IncidentStore(tmp_path)
    inc = _seed(store, di_confidence=0.42)
    gate = make_gate_node(cfg=_cfg(threshold=0.75), store=store)
    out = await gate(GraphState(incident=inc, next_route=None,
                                last_agent="deep_investigator", error=None))
    assert out["next_route"] == "__end__"
    reloaded = store.load(inc.id)
    assert reloaded.status == "awaiting_input"
    assert reloaded.pending_intervention["confidence"] == 0.42


@pytest.mark.asyncio
async def test_gate_passes_when_at_or_above_threshold(tmp_path):
    store = IncidentStore(tmp_path)
    inc = _seed(store, di_confidence=0.9)
    gate = make_gate_node(cfg=_cfg(threshold=0.75), store=store)
    out = await gate(GraphState(incident=inc, next_route=None,
                                last_agent="deep_investigator", error=None))
    assert out["next_route"] == "default"
    reloaded = store.load(inc.id)
    # Status stays whatever DI left it (e.g. in_progress); no intervention payload.
    assert reloaded.pending_intervention is None
    assert reloaded.status != "awaiting_input"


@pytest.mark.asyncio
async def test_gate_clears_stale_intervention_on_pass(tmp_path):
    """If a prior run set pending_intervention but the new DI is confident,
    the gate should clear the old payload before forwarding to resolution."""
    store = IncidentStore(tmp_path)
    inc = _seed(store, di_confidence=0.92)
    inc.pending_intervention = {"reason": "low_confidence", "confidence": 0.3,
                                "threshold": 0.75,
                                "options": ["stop"], "escalation_teams": []}
    store.save(inc)
    gate = make_gate_node(cfg=_cfg(threshold=0.75), store=store)
    out = await gate(GraphState(incident=inc, next_route=None,
                                last_agent="deep_investigator", error=None))
    assert out["next_route"] == "default"
    reloaded = store.load(inc.id)
    assert reloaded.pending_intervention is None
