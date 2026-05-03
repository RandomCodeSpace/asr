"""Tests for ``wrap_tool`` (P4-C): hybrid HITL dispatch around BaseTool.

The wrapper is a *closure factory*: it captures the live ``Session`` and
``GatewayConfig`` per agent invocation and returns a new ``BaseTool``.
Per locked decision P4.1, the dispatch is:

  ``auto``     -> invoke inner tool, no audit overhead
  ``notify``   -> invoke inner tool, append ToolCall(risk="medium",
                                                    status="executed_with_notify")
  ``approve``  -> raise ``langgraph.types.interrupt(...)`` BEFORE the
                  inner tool runs; on resume, run + audit per the
                  resume payload.

These tests use a stub ``BaseTool`` that records its invocations so we
can assert the inner tool ran (or didn't) and verify the audit row.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool

from runtime.config import GatewayConfig, ProdOverrides
from runtime.state import Session, ToolCall
from runtime.tools.gateway import wrap_tool


class _RecordingTool(BaseTool):
    """Tiny BaseTool that records every call so tests can assert behaviour."""

    name: str = "recorder"
    description: str = "Records each invocation; returns the args back."
    calls: list = []

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("sync", args, dict(kwargs)))
        return {"echoed": dict(kwargs) or list(args)}

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("async", args, dict(kwargs)))
        return {"echoed": dict(kwargs) or list(args)}


def _new_session(env: str | None = None) -> Session:
    """Build a minimal generic ``Session`` (no domain fields)."""
    s = Session(
        id="S-test-1",
        status="in_progress",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
    )
    # Phase 4 prod-override path reads ``getattr(session, "environment", None)``;
    # patch the attribute dynamically for env-based tests so we don't need
    # to import IncidentState here (gateway is framework-side and must work
    # for non-incident apps too).
    if env is not None:
        object.__setattr__(s, "environment", env)
    return s


def _make_recorder(name: str = "recorder") -> _RecordingTool:
    t = _RecordingTool()
    object.__setattr__(t, "calls", [])
    object.__setattr__(t, "name", name)
    return t


# ----------------------------- AUTO PATH -----------------------------


def test_auto_path_invokes_tool_without_audit():
    """``low`` risk => no audit row appended (auto bypass)."""
    session = _new_session()
    inner = _make_recorder("create_incident")
    cfg = GatewayConfig(policy={"create_incident": "low"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="intake")
    result = wrapped.invoke({"q": "x"})

    assert result == {"echoed": {"q": "x"}}
    assert len(inner.calls) == 1
    assert session.tool_calls == []  # auto: no gateway audit


def test_auto_path_when_no_gateway_cfg():
    session = _new_session()
    inner = _make_recorder("anything")

    wrapped = wrap_tool(inner, session=session, gateway_cfg=None, agent_name="any")
    result = wrapped.invoke({"foo": 1})

    assert result == {"echoed": {"foo": 1}}
    assert session.tool_calls == []


# ---------------------------- NOTIFY PATH ----------------------------


def test_notify_path_runs_tool_and_appends_audit():
    """``medium`` risk => tool runs AND a ToolCall(status='executed_with_notify')
    is appended to ``session.tool_calls``."""
    session = _new_session()
    inner = _make_recorder("update_incident")
    cfg = GatewayConfig(policy={"update_incident": "medium"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="resolution")
    result = wrapped.invoke({"id": "INC-1", "patch": {"summary": "x"}})

    assert result == {"echoed": {"id": "INC-1", "patch": {"summary": "x"}}}
    assert len(inner.calls) == 1, "inner tool must still execute on notify path"
    assert len(session.tool_calls) == 1
    rec = session.tool_calls[0]
    assert isinstance(rec, ToolCall)
    assert rec.tool == "update_incident"
    assert rec.risk == "medium"
    assert rec.status == "executed_with_notify"
    assert rec.agent == "resolution"


# ---------------------------- APPROVE PATH ---------------------------
#
# ``interrupt()`` requires a Pregel runtime context (it reads scratchpad
# from the graph config). We drive the wrap through a tiny single-node
# graph so the interrupt is captured by the checkpointer the same way
# the production resolution flow does.


def _drive_wrap_through_graph(wrapped, args: dict, *, resume_value=None):
    """Run a wrapped tool inside a tiny single-node graph + InMemorySaver.

    Returns ``(result, interrupts)``. ``interrupts`` is the list captured
    on the result dict's ``__interrupt__`` key (None if the tool did not
    pause). When ``resume_value`` is provided, a second invocation drives
    the resume path.
    """
    import asyncio
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END
    from langgraph.types import Command

    class _S(TypedDict, total=False):
        result: object

    async def node(state: _S) -> dict:
        out = await wrapped.ainvoke(args)
        return {"result": out}

    sg = StateGraph(_S)
    sg.add_node("n", node)
    sg.set_entry_point("n")
    sg.add_edge("n", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-wrap"}}

    async def run():
        first = await compiled.ainvoke({}, config=cfg)
        ints = first.get("__interrupt__") if isinstance(first, dict) else None
        if resume_value is not None:
            second = await compiled.ainvoke(Command(resume=resume_value), config=cfg)
            return second, ints
        return first, ints

    return asyncio.run(run())


def test_approve_path_pauses_via_interrupt_before_invoking_tool():
    """``high`` risk => the inner tool MUST NOT run before approval —
    ``interrupt()`` pauses the graph and the captured payload describes
    the pending tool call.

    P4-I: an audit row with ``status="pending_approval"`` is persisted
    BEFORE the GraphInterrupt fires so the timeout watchdog has a
    record to scan. The pending row carries the tool name + open-ts
    but no ``result`` / ``approver`` yet.
    """
    session = _new_session()
    inner = _make_recorder("apply_fix")
    cfg = GatewayConfig(policy={"apply_fix": "high"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="resolution")
    _result, interrupts = _drive_wrap_through_graph(wrapped, {"target": "payments-svc"})

    assert interrupts, "high-risk wrap must surface an Interrupt"
    assert inner.calls == [], "high-risk tool must NOT run before approval"
    # P4-I: a pending_approval audit row is persisted before interrupt.
    assert len(session.tool_calls) == 1
    pending = session.tool_calls[0]
    assert pending.status == "pending_approval"
    assert pending.tool == "apply_fix"
    assert pending.risk == "high"
    assert pending.result is None
    assert pending.approver is None

    # Interrupt payload should describe the tool approval handshake.
    payload = interrupts[0].value
    assert payload["kind"] == "tool_approval"
    assert payload["tool"] == "apply_fix"


def test_approve_path_resume_with_approve_runs_tool_and_audits():
    """Resume with decision='approve' => inner tool runs, ToolCall has
    status='approved' with approver/rationale captured."""
    session = _new_session()
    inner = _make_recorder("apply_fix")
    cfg = GatewayConfig(policy={"apply_fix": "high"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="resolution")
    final, interrupts = _drive_wrap_through_graph(
        wrapped,
        {"target": "payments-svc"},
        resume_value={"decision": "approve", "approver": "alice", "rationale": "ok"},
    )

    assert interrupts, "first invocation must have interrupted"
    # On resume, inner tool ran exactly once.
    assert len(inner.calls) == 1
    # Audit row for the approved call is on the live session.
    assert len(session.tool_calls) == 1
    rec = session.tool_calls[0]
    assert rec.status == "approved"
    assert rec.risk == "high"
    assert rec.approver == "alice"
    assert rec.approval_rationale == "ok"
    assert final is not None


def test_approve_path_resume_with_reject_skips_tool_and_audits_rejection():
    """Resume with decision='reject' => inner tool DOES NOT run, ToolCall
    has status='rejected'."""
    session = _new_session()
    inner = _make_recorder("apply_fix")
    cfg = GatewayConfig(policy={"apply_fix": "high"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="resolution")
    _final, interrupts = _drive_wrap_through_graph(
        wrapped,
        {"target": "payments-svc"},
        resume_value={"decision": "reject", "approver": "bob", "rationale": "blast radius too high"},
    )

    assert interrupts
    assert inner.calls == [], "rejected tool must NOT run"
    assert len(session.tool_calls) == 1
    rec = session.tool_calls[0]
    assert rec.status == "rejected"
    assert rec.approver == "bob"
    assert rec.approval_rationale == "blast radius too high"


# ----------------------- PROD OVERRIDE WIRING ------------------------


def test_prod_override_forces_approve_through_wrap():
    """End-to-end: a tool that's normally ``low`` but matched by prod-override
    glob must trigger interrupt, not auto-execute."""
    session = _new_session(env="production")
    inner = _make_recorder("update_incident")
    cfg = GatewayConfig(
        policy={"update_incident": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],
        ),
    )

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="resolution")
    _result, interrupts = _drive_wrap_through_graph(wrapped, {"id": "INC-1"})

    assert interrupts, "prod-override must force a pause via interrupt()"
    assert inner.calls == []


# ------------------ CLOSURE CAPTURES LIVE SESSION --------------------


def test_wrap_is_a_closure_per_session():
    """Two separate ``wrap_tool`` calls must capture two separate Sessions —
    auditing into the wrong one would corrupt cross-session history."""
    session_a = _new_session()
    session_b = _new_session()
    inner = _make_recorder("update_incident")
    cfg = GatewayConfig(policy={"update_incident": "medium"})

    wrapped_a = wrap_tool(inner, session=session_a, gateway_cfg=cfg, agent_name="a")
    wrapped_b = wrap_tool(inner, session=session_b, gateway_cfg=cfg, agent_name="b")

    wrapped_a.invoke({"x": 1})
    wrapped_b.invoke({"x": 2})

    assert len(session_a.tool_calls) == 1
    assert len(session_b.tool_calls) == 1
    assert session_a.tool_calls[0].agent == "a"
    assert session_b.tool_calls[0].agent == "b"
    # The two sessions must NOT share the same ToolCall object.
    assert session_a.tool_calls[0] is not session_b.tool_calls[0]


# ------------------------ NAME / DESC PROPAGATION --------------------


def test_wrapped_tool_preserves_name_and_description():
    """LangGraph's ReAct prompt builder reads ``name`` + ``description`` —
    the wrapper must mirror them so the agent sees the same tool surface."""
    session = _new_session()
    inner = _make_recorder("create_incident")
    object.__setattr__(inner, "description", "Create a new incident.")
    cfg = GatewayConfig()

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="intake")

    assert wrapped.name == "create_incident"
    assert wrapped.description == "Create a new incident."


# ---------- DOUBLE-WRAP IDEMPOTENCE & ASYNC-ONLY GUARD (Codex finding) -----


def test_double_wrap_is_idempotent():
    """``wrap_tool(wrap_tool(t))`` must return the SAME wrapped object —
    nesting wrappers would recurse forever because the outer ``_run``
    calls ``inner.invoke`` which re-enters the inner ``_run``."""
    session = _new_session()
    inner = _make_recorder("update_incident")
    cfg = GatewayConfig(policy={"update_incident": "medium"})

    once = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="r")
    twice = wrap_tool(once, session=session, gateway_cfg=cfg, agent_name="r")

    assert once is twice, "second wrap must short-circuit and return the same object"

    # Sanity: the (singly) wrapped tool still works and audits exactly once.
    result = twice.invoke({"x": 1})
    assert result == {"echoed": {"x": 1}}
    assert len(inner.calls) == 1
    assert len(session.tool_calls) == 1


class _AsyncOnlyTool(BaseTool):
    """Native-async-only tool: ``_run`` raises so only ``ainvoke`` works.

    This is the real-world shape — ``BaseTool._run`` is abstract, so an
    async-only tool must provide a ``_run`` stub (typically raising
    ``NotImplementedError``). The wrapper must translate that into a
    clearer error rather than letting the bare langchain message bubble
    up — and must NOT silently produce a 'coroutine was never awaited'.
    """

    name: str = "async_only"
    description: str = "Implements only _arun."

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("async_only tool does not support sync invoke")

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return {"echoed": dict(kwargs)}


def test_native_async_only_tool_sync_invoke_raises_clear_error():
    """Sync-invoking a native-async-only wrapped tool must raise a
    clear ``NotImplementedError`` whose message points the caller at
    ``ainvoke`` / ``_arun`` — not a bare langchain message and not a
    silent 'coroutine was never awaited' warning."""
    session = _new_session()
    inner = _AsyncOnlyTool()
    cfg = GatewayConfig(policy={"async_only": "low"})  # auto path => simplest

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="r")

    with pytest.raises(NotImplementedError) as excinfo:
        wrapped.invoke({"x": 1})

    msg = str(excinfo.value)
    assert "async_only" in msg
    assert "ainvoke" in msg or "_arun" in msg


@pytest.mark.asyncio
async def test_native_async_only_tool_async_invoke_works():
    """The same async-only tool MUST still work through ``ainvoke`` —
    the guard is only for the sync path."""
    session = _new_session()
    inner = _AsyncOnlyTool()
    object.__setattr__(inner, "name", "async_only_ok")
    cfg = GatewayConfig(policy={"async_only_ok": "low"})

    wrapped = wrap_tool(inner, session=session, gateway_cfg=cfg, agent_name="r")
    result = await wrapped.ainvoke({"x": 2})
    assert result == {"echoed": {"x": 2}}
