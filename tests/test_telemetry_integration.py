"""M3 integration — per-step telemetry emitted at the tool-call boundary.

Two layers:

1. **End-to-end** (orchestrator boot → stub session): asserts the agent
   boundary kinds fire for every responsive agent —
   ``agent_started → confidence_emitted → route_decided → agent_finished``.
   The default stub flow does not execute domain tools (it terminates on
   the envelope tool-call), so ``tool_invoked`` is exercised separately
   in the focused tests below.

2. **Focused gateway tests** (``wrap_tool`` + stub ``EventLog``): assert
   that every tool invocation routed through the gateway emits a
   ``tool_invoked`` event with ``latency_ms`` populated, and that
   ``gate_fired`` is emitted BEFORE the interrupt when the gateway
   rates the tool ``high``.
"""
from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy import create_engine

from runtime.config import (
    AppConfig,
    GatewayConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    Paths,
    RuntimeConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.state import Session
from runtime.storage.event_log import EventLog
from runtime.storage.models import Base
from runtime.tools.gateway import wrap_tool


def _cfg_for(tmp_path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="examples.incident_management.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="examples.incident_management.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="examples.incident_management.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(state_class=None),
    )


@pytest.fixture
def event_log(tmp_path):
    """A standalone EventLog over an in-memory schema for focused tests."""
    engine = create_engine(f"sqlite:///{tmp_path/'tele.db'}")
    Base.metadata.create_all(engine)
    return EventLog(engine=engine)


# ===================================================================
# End-to-end: agent-boundary kinds fire for every responsive agent.
# ===================================================================

@pytest.mark.asyncio
async def test_stub_session_emits_ordered_agent_boundary_events(tmp_path):
    cfg = _cfg_for(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_investigation(
            query="latency spike on payments-api",
            environment="staging",
        )
        assert orch.event_log is not None
        events = list(orch.event_log.iter_for(sid))
        kinds = [e.kind for e in events]

        # Every responsive agent emits the full quartet.
        assert "agent_started" in kinds, kinds
        assert "confidence_emitted" in kinds, kinds
        assert "route_decided" in kinds, kinds
        assert "agent_finished" in kinds, kinds

        # agent_started precedes the agent's own confidence_emitted /
        # route_decided / agent_finished for the first agent.
        i_started = kinds.index("agent_started")
        i_conf = kinds.index("confidence_emitted")
        i_route = kinds.index("route_decided")
        i_finish = kinds.index("agent_finished")
        assert i_started < i_conf < i_route < i_finish

        # confidence_emitted values in [0,1].
        for e in events:
            if e.kind == "confidence_emitted":
                v = e.payload.get("value")
                assert isinstance(v, float) and 0.0 <= v <= 1.0, e.payload

        # agent_finished carries the three token_usage counts.
        finish_events = [e for e in events if e.kind == "agent_finished"]
        for e in finish_events:
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                assert k in e.payload
                assert e.payload[k] >= 0
    finally:
        await orch.aclose()


# ===================================================================
# Focused: wrap_tool emits tool_invoked with latency_ms on every call.
# ===================================================================

class _PingArgs(BaseModel):
    msg: str = ""


def _make_session(*, sid: str = "INC-1", environment: str = "staging") -> Session:
    s = Session(
        id=sid,
        status="open",
        created_at="2026-05-12T00:00:00Z",
        updated_at="2026-05-12T00:00:00Z",
    )
    object.__setattr__(s, "environment", environment)
    return s


def _make_ping_tool() -> StructuredTool:
    def _impl(msg: str = "") -> dict:
        return {"echo": msg or "default"}
    return StructuredTool.from_function(
        func=_impl,
        name="ping",
        description="echo the input",
        args_schema=_PingArgs,
    )


def test_wrap_tool_auto_path_emits_tool_invoked(event_log):
    """No gateway config => action="auto"; one tool_invoked with
    status=executed and risk=low. latency_ms is populated and >= 0."""
    sess = _make_session()
    tool = _make_ping_tool()
    wrapped = wrap_tool(
        tool, session=sess, gateway_cfg=None,
        agent_name="triage", event_log=event_log,
    )
    out = wrapped.invoke({"msg": "hello"})
    assert out == {"echo": "hello"}

    events = list(event_log.iter_for(sess.id))
    assert [e.kind for e in events] == ["tool_invoked"]
    payload = events[0].payload
    assert payload["tool"] == "ping"
    assert payload["agent"] == "triage"
    assert payload["status"] == "executed"
    assert payload["risk"] == "low"
    assert payload["result_kind"] == "dict"
    assert payload["latency_ms"] >= 0
    assert payload["args"] == {"msg": "hello"}


def test_wrap_tool_notify_path_emits_tool_invoked_with_notify_status(event_log):
    """gateway policy `ping: medium` => action="notify"; tool_invoked is
    emitted with status=executed_with_notify and risk=medium."""
    sess = _make_session()
    tool = _make_ping_tool()
    cfg = GatewayConfig(policy={"ping": "medium"})
    wrapped = wrap_tool(
        tool, session=sess, gateway_cfg=cfg,
        agent_name="triage", event_log=event_log,
    )
    wrapped.invoke({"msg": "soft"})

    events = list(event_log.iter_for(sess.id))
    tool_events = [e for e in events if e.kind == "tool_invoked"]
    assert len(tool_events) == 1
    payload = tool_events[0].payload
    assert payload["status"] == "executed_with_notify"
    assert payload["risk"] == "medium"
    assert payload["latency_ms"] >= 0


def test_wrap_tool_high_risk_emits_gate_fired_then_approved(event_log, monkeypatch):
    """gateway policy `ping: high` in production => decision.gate=True;
    a `gate_fired` event is emitted BEFORE the tool actually runs. The
    real interrupt path needs a LangGraph scratchpad, so we patch
    ``interrupt`` to return a synthetic ``approve`` verdict — the
    resulting flow exercises the gate_fired + approved tool_invoked
    pair in order."""
    import langgraph.types as lg_types

    monkeypatch.setattr(lg_types, "interrupt", lambda _payload: "approve")

    sess = _make_session(environment="production")
    tool = _make_ping_tool()
    cfg = GatewayConfig(policy={"ping": "high"})
    wrapped = wrap_tool(
        tool, session=sess, gateway_cfg=cfg,
        agent_name="resolution", event_log=event_log,
    )
    out = wrapped.invoke({"msg": "danger"})
    assert out == {"echo": "danger"}

    events = list(event_log.iter_for(sess.id))
    kinds = [e.kind for e in events]

    # Causality: gate_fired must be recorded BEFORE the tool runs and
    # therefore before the tool_invoked event for the approved call.
    assert "gate_fired" in kinds, kinds
    assert "tool_invoked" in kinds, kinds
    gate_idx = kinds.index("gate_fired")
    tool_idx = kinds.index("tool_invoked")
    assert gate_idx < tool_idx, kinds

    gate_event = events[gate_idx]
    assert gate_event.payload.get("reason") in {
        "high_risk_tool", "gated_env", "low_confidence",
    }
    assert gate_event.payload["tool"] == "ping"
    assert gate_event.payload["agent"] == "resolution"

    tool_event = events[tool_idx]
    assert tool_event.payload["status"] == "approved"
    assert tool_event.payload["risk"] == "high"
    assert tool_event.payload["latency_ms"] >= 0


def test_wrap_tool_no_event_log_is_noop():
    """event_log=None must not break the wrapper; tool still runs."""
    sess = _make_session()
    tool = _make_ping_tool()
    wrapped = wrap_tool(
        tool, session=sess, gateway_cfg=None,
        agent_name="triage", event_log=None,
    )
    out = wrapped.invoke({"msg": "noevent"})
    assert out == {"echo": "noevent"}
