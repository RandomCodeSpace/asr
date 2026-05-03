"""Integration tests for the risk-rated tool gateway plumbed through
``make_agent_node`` (P4-F).

Each test drives a tiny single-node LangGraph with one wrapped MCP-style
``BaseTool`` so we can assert the gateway-wrapped tool surface that the
ReAct agent sees in production. The wrapping happens *inside the node
body* so the closure captures the live ``Session`` for that run — these
tests verify the wrap fires and audits onto the right session.

Layers exercised:

  * low risk    -> auto: tool runs, ``session.tool_calls`` empty.
  * medium risk -> notify: tool runs, ``ToolCall(status="executed_with_notify")``.
  * high risk   -> approve: ``interrupt()`` pauses the graph; the
    pending tool-approval payload is observable in graph state.
"""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langchain_core.tools import BaseTool

from runtime.config import GatewayConfig
from runtime.state import Session


# ---------------------------------------------------------------------------
# Test doubles — a tiny BaseTool that records each call so we can assert
# the inner tool ran (or didn't), and a Session fixture for the wrap to
# audit into.
# ---------------------------------------------------------------------------


class _RecordingTool(BaseTool):
    name: str = "recorder"
    description: str = "Records each invocation; returns the args back."
    calls: list = []

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("sync", args, dict(kwargs)))
        return {"echoed": dict(kwargs) or list(args)}

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("async", args, dict(kwargs)))
        return {"echoed": dict(kwargs) or list(args)}


def _make_recorder(name: str) -> _RecordingTool:
    t = _RecordingTool()
    object.__setattr__(t, "calls", [])
    object.__setattr__(t, "name", name)
    return t


def _new_session() -> Session:
    return Session(
        id="S-int-1",
        status="in_progress",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Helper — drive the graph that ``make_agent_node`` would build, but
# minus the LLM ReAct loop. We exercise the *wrap* itself: a single
# node body that calls ``wrapped.ainvoke``. This mirrors what
# ``create_react_agent`` does once the model selects the tool.
# ---------------------------------------------------------------------------


def _run_wrap_in_graph(wrapped, args: dict, *, resume_value=None):
    """Execute a wrapped tool inside a one-node graph + InMemorySaver.

    Returns ``(final_state, interrupts)``. ``interrupts`` is the list of
    interrupt records captured on the result dict's ``__interrupt__``
    key — empty / ``None`` when the tool ran without pausing.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END
    from langgraph.types import Command

    class _S(TypedDict, total=False):
        result: object

    async def node(_state: _S) -> dict:
        out = await wrapped.ainvoke(args)
        return {"result": out}

    sg = StateGraph(_S)
    sg.add_node("n", node)
    sg.set_entry_point("n")
    sg.add_edge("n", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-int"}}

    async def run():
        first = await compiled.ainvoke({}, config=cfg)
        ints = first.get("__interrupt__") if isinstance(first, dict) else None
        if resume_value is not None:
            second = await compiled.ainvoke(Command(resume=resume_value), config=cfg)
            return second, ints
        return first, ints

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# P4-F integration tests — verify the gateway is correctly *integrated*
# (i.e. callers get the right tool surface). The wrap factory is unit-
# tested separately in ``test_gateway_wrap.py``.
# ---------------------------------------------------------------------------


def test_low_risk_tool_runs_through_wrapped_agent_without_audit():
    """A tool flagged ``low`` in the gateway policy must run through the
    wrap and leave ``session.tool_calls`` empty (auto bypass)."""
    from runtime.tools.gateway import wrap_tool

    session = _new_session()
    inner = _make_recorder("create_incident")
    cfg = GatewayConfig(policy={"create_incident": "low"})

    wrapped = wrap_tool(
        inner, session=session, gateway_cfg=cfg, agent_name="intake",
    )
    final, interrupts = _run_wrap_in_graph(wrapped, {"q": "x"})

    assert not interrupts, "low-risk tool must not pause the graph"
    assert final.get("result") == {"echoed": {"q": "x"}}
    assert len(inner.calls) == 1
    assert session.tool_calls == [], "auto path must not append a ToolCall"


def test_high_risk_tool_pauses_graph_with_tool_approval_payload():
    """A tool flagged ``high`` in the policy must trigger an
    ``interrupt()`` whose payload is visible in the graph state. The
    inner tool must NOT run before approval resolves."""
    from runtime.tools.gateway import wrap_tool

    session = _new_session()
    inner = _make_recorder("apply_fix")
    cfg = GatewayConfig(policy={"apply_fix": "high"})

    wrapped = wrap_tool(
        inner, session=session, gateway_cfg=cfg, agent_name="resolution",
    )
    _final, interrupts = _run_wrap_in_graph(wrapped, {"target": "payments-svc"})

    assert interrupts, "high-risk wrap must surface an Interrupt"
    payload = interrupts[0].value
    assert payload["kind"] == "tool_approval"
    assert payload["tool"] == "apply_fix"
    assert inner.calls == [], "high-risk tool must NOT run before approval"
    # P4-I: a ``pending_approval`` audit row is persisted BEFORE the
    # GraphInterrupt fires so the watchdog can observe stale approvals.
    # The row carries the tool + open-ts but no result/approver yet.
    assert len(session.tool_calls) == 1
    pending = session.tool_calls[0]
    assert pending.status == "pending_approval"
    assert pending.tool == "apply_fix"
    assert pending.risk == "high"
    assert pending.result is None


def test_medium_risk_tool_runs_and_persists_executed_with_notify():
    """A tool flagged ``medium`` in the policy must run AND a
    ``ToolCall(status="executed_with_notify")`` must be appended to
    ``session.tool_calls`` for the audit log."""
    from runtime.tools.gateway import wrap_tool

    session = _new_session()
    inner = _make_recorder("update_incident")
    cfg = GatewayConfig(policy={"update_incident": "medium"})

    wrapped = wrap_tool(
        inner, session=session, gateway_cfg=cfg, agent_name="resolution",
    )
    final, interrupts = _run_wrap_in_graph(
        wrapped, {"id": "INC-1", "patch": {"summary": "x"}},
    )

    assert not interrupts, "medium-risk tool runs without pausing the graph"
    assert final.get("result") == {
        "echoed": {"id": "INC-1", "patch": {"summary": "x"}}
    }
    assert len(inner.calls) == 1
    assert len(session.tool_calls) == 1
    rec = session.tool_calls[0]
    assert rec.tool == "update_incident"
    assert rec.risk == "medium"
    assert rec.status == "executed_with_notify"
    assert rec.agent == "resolution"


def test_make_agent_node_accepts_gateway_cfg_kwarg():
    """``make_agent_node`` must accept the ``gateway_cfg`` kwarg added
    in P4-F. Calling without it (back-compat) keeps the legacy
    ``None`` default. Both forms must return a callable node.
    """
    import inspect

    from runtime.graph import make_agent_node

    sig = inspect.signature(make_agent_node)
    assert "gateway_cfg" in sig.parameters, (
        "P4-F regression: make_agent_node missing gateway_cfg kwarg"
    )
    # Default must be None so legacy callers (no gateway) keep working.
    assert sig.parameters["gateway_cfg"].default is None
