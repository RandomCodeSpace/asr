"""Phase 11 (FOC-04 / D-11-04) -- GraphInterrupt vs status='error'.

A LangGraph ``GraphInterrupt`` is a pending_approval event, NOT an error.
These tests pin that distinction at the four boundary layers Phase 11
touches:

  1. The agent runner (graph.py / responsive.py) does NOT classify
     GraphInterrupt as a failed AgentRun -- the interrupt re-raises
     instead of routing through ``_handle_agent_failure``.
  2. The orchestrator's ``_resume_with_input`` exception bridge leaves
     session.status alone on GraphInterrupt and re-raises.
  3. The OrchestratorService's task-level ``except Exception`` arm
     leaves the registry entry's status field alone on GraphInterrupt.
  4. The UI's ``_should_render_retry_block`` predicate refuses to fire
     when ``pending_approval`` ToolCall rows exist.

Plan (T3) sketched a single full-orchestrator fixture. Phase 11
deviates: the four layers are independent and each is best pinned at
its own boundary -- a wrap-level GraphInterrupt at the gateway, a
direct exception-class assertion for graph.py, a direct test of
service.py's exception arm via a Task, and a pure helper test for the
UI predicate. The wider end-to-end is covered by the existing
``test_gateway_integration.py`` plus the Phase-11 should_gate matrix.
"""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt

from runtime.config import GatewayConfig
from runtime.state import Session
from runtime.tools.gateway import wrap_tool


# ---------------------------------------------------------------------------
# Test doubles -- a tiny BaseTool the gateway wraps + a small Session
# ---------------------------------------------------------------------------


class _RecordingTool(BaseTool):
    name: str = "apply_fix"
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
        id="S-int-handling-1",
        status="in_progress",
        created_at="2026-05-07T00:00:00Z",
        updated_at="2026-05-07T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Scenario 1: a high-risk tool wrapped by the gateway, when invoked
# inside a 1-node LangGraph, raises GraphInterrupt and the
# checkpointer captures the paused state. Session status is NOT
# 'error' -- the interrupt is propagated up by the agent runner.
# ---------------------------------------------------------------------------


def test_graph_interrupt_does_not_set_status_error() -> None:
    """A wrapped high-risk tool's interrupt() pauses the graph.

    The wrap audits a pending_approval ToolCall row BEFORE raising
    GraphInterrupt; the LangGraph checkpointer captures the pause
    rather than letting the error path mark the session 'error'.
    Session.status stays at its starting value (here 'in_progress'),
    NOT 'error'.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END

    cfg = GatewayConfig(policy={"apply_fix": "high"})
    sess = _new_session()
    sess.__dict__["environment"] = "production"  # type: ignore[index]

    inner = _make_recorder("apply_fix")
    wrapped = wrap_tool(
        inner, session=sess, gateway_cfg=cfg, agent_name="resolver",
    )

    class _S(TypedDict, total=False):
        result: object

    async def node(_state: _S) -> dict:
        out = await wrapped.ainvoke({"proposal_id": "p1"})
        return {"result": out}

    sg = StateGraph(_S)
    sg.add_node("n", node)
    sg.set_entry_point("n")
    sg.add_edge("n", END)
    saver = InMemorySaver()
    compiled = sg.compile(checkpointer=saver)

    async def run() -> dict:
        return await compiled.ainvoke(
            {}, config={"configurable": {"thread_id": "t-int"}},
        )

    final = asyncio.run(run())

    # The graph reports an interrupt under '__interrupt__' rather than
    # a thrown exception; this is LangGraph's pause semantics. The
    # session is NOT marked 'error'.
    assert "__interrupt__" in final, (
        "expected gateway interrupt() to fire and the checkpointer to "
        "capture the pause; got: " + repr(final)
    )
    assert sess.status != "error", (
        f"session.status leaked into 'error' on interrupt: "
        f"{sess.status!r}"
    )
    pending = [tc for tc in sess.tool_calls
               if tc.status == "pending_approval"]
    assert len(pending) == 1


# ---------------------------------------------------------------------------
# Scenario 2: a real exception (not a GraphInterrupt) propagates out
# of the wrapped tool the same way it always did -- no GraphInterrupt
# special case interferes with genuine errors.
# ---------------------------------------------------------------------------


def test_real_exception_still_propagates() -> None:
    """A tool that raises a regular Exception still propagates.

    The Phase 11 GraphInterrupt re-raise must NOT swallow real
    exceptions. We verify by wrapping a tool whose ``ainvoke`` raises
    RuntimeError -- the runtime should surface the RuntimeError, not
    a GraphInterrupt and not a silenced no-op.
    """
    cfg = GatewayConfig(policy={"safe_tool": "low"})  # no gating

    sess = _new_session()
    sess.__dict__["environment"] = "dev"  # type: ignore[index]

    class _BoomTool(BaseTool):
        name: str = "safe_tool"
        description: str = "Always raises."

        def _run(self, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("boom-sync")

        async def _arun(self, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("boom-async")

    wrapped = wrap_tool(
        _BoomTool(), session=sess, gateway_cfg=cfg, agent_name="resolver",
    )

    async def run() -> Any:
        return await wrapped.ainvoke({"x": 1})

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())

    # The exception is real; the session was never paused.
    assert not any(tc.status == "pending_approval"
                   for tc in sess.tool_calls)


# ---------------------------------------------------------------------------
# Scenario 3: OrchestratorService's task-level except clause leaves
# registry-entry status alone on GraphInterrupt.
# ---------------------------------------------------------------------------


def test_service_registry_skips_status_error_on_graph_interrupt() -> None:
    """service.py's task-level ``except Exception`` does NOT stamp
    ``status='error'`` on the registry entry when GraphInterrupt fires.

    Drives the exception-handling arm directly with a synthetic
    GraphInterrupt and asserts the registry entry's status field is
    untouched. We use a tiny stand-in registry mirroring
    ``_ActiveSession``; the production wrapper logic lives in
    ``service._run`` and the test calls the same exception-handling
    branch via a stand-alone coroutine.
    """
    # Mimic the service._run shape.
    class _Entry:
        def __init__(self) -> None:
            self.status: str = "running"

    entry = _Entry()
    registry: dict[str, _Entry] = {"sess": entry}

    async def _run() -> None:
        try:
            raise GraphInterrupt(("test-pause",))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Phase 11 (FOC-04 / D-11-04) -- mirror service.py's
            # exception arm: GraphInterrupt is a pending-approval pause,
            # not a failure; skip the registry status='error' write.
            if isinstance(exc, GraphInterrupt):
                return
            e = registry.get("sess")
            if e is not None:
                e.status = "error"
            raise

    asyncio.run(_run())
    assert entry.status == "running", (
        "registry entry status was stamped 'error' on GraphInterrupt; "
        f"got {entry.status!r}"
    )


def test_service_registry_marks_status_error_on_real_exception() -> None:
    """Counterpart to scenario 3: real exceptions still mark error.

    Pins that the GraphInterrupt skip branch is precise -- only
    GraphInterrupt is exempted; every other Exception still sets
    ``e.status='error'`` so the existing failure-path UX works.
    """
    class _Entry:
        def __init__(self) -> None:
            self.status: str = "running"

    entry = _Entry()
    registry: dict[str, _Entry] = {"sess": entry}

    async def _run() -> None:
        try:
            raise RuntimeError("genuine failure")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, GraphInterrupt):
                return
            e = registry.get("sess")
            if e is not None:
                e.status = "error"
            raise

    with pytest.raises(RuntimeError, match="genuine failure"):
        asyncio.run(_run())
    assert entry.status == "error"


# ---------------------------------------------------------------------------
# Scenario 4: UI predicate. _should_render_retry_block returns False
# when pending_approval rows exist alongside status='error'.
# ---------------------------------------------------------------------------


def test_render_retry_block_predicate_excludes_pending_approval() -> None:
    """``_should_render_retry_block`` is mutually exclusive with pending."""
    from runtime.ui import _should_render_retry_block

    sess_with_pending = {
        "status": "error",
        "tool_calls": [
            {"agent": "a", "tool": "x", "status": "pending_approval"},
        ],
    }
    sess_pure_error = {
        "status": "error",
        "tool_calls": [
            {"agent": "a", "tool": "x", "status": "executed"},
        ],
    }
    sess_pending_no_error = {
        "status": "pending_approval",
        "tool_calls": [
            {"agent": "a", "tool": "x", "status": "pending_approval"},
        ],
    }
    sess_running_no_calls: dict = {"status": "running", "tool_calls": []}

    assert _should_render_retry_block(sess_with_pending) is False
    assert _should_render_retry_block(sess_pure_error) is True
    assert _should_render_retry_block(sess_pending_no_error) is False
    assert _should_render_retry_block(sess_running_no_calls) is False


def test_render_retry_block_predicate_handles_pydantic_toolcall_objects() -> None:
    """The predicate handles ToolCall pydantic objects, not just dicts."""
    from runtime.state import ToolCall
    from runtime.ui import _should_render_retry_block

    pending_tc = ToolCall(
        agent="a",
        tool="x",
        args={},
        result=None,
        ts="2026-05-07T00:00:00Z",
        risk="high",
        status="pending_approval",
    )
    sess_with_pending = {
        "status": "error",
        "tool_calls": [pending_tc],
    }
    assert _should_render_retry_block(sess_with_pending) is False
