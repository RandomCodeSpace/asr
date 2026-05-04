"""When the gateway pauses for HITL, the pending_approval ToolCall row
must be visible to a concurrent ``store.load`` (the watchdog reads from
the DB, not from the in-memory session). This test exercises that
contract."""
import asyncio
from typing import Any, TypedDict

import pytest
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from sqlalchemy import create_engine

from runtime.config import GatewayConfig
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.tools.gateway import wrap_tool


class _UpdateIncidentTool(BaseTool):
    """Stub update_incident for persistence tests."""

    name: str = "update_incident"
    description: str = "Apply a patch to the incident."

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    return SessionStore(engine=engine)


def test_pending_approval_row_persists_before_interrupt(store):
    """Mirrors the production code path: the wrap_tool wrapper saves
    the in-memory mutation to the DB before raising GraphInterrupt so
    the watchdog and /approvals UI can see the pending row."""
    inc = store.create(query="q", environment="production")
    gw = GatewayConfig(policy={"update_incident": "high"})
    fresh = store.load(inc.id)

    wrapped = wrap_tool(
        _UpdateIncidentTool(), agent_name="resolution", session=fresh,
        gateway_cfg=gw, store=store,
    )

    # interrupt() requires a Pregel runtime context — drive through a
    # minimal single-node graph with a checkpointer (same pattern as
    # test_gateway_wrap.py).
    class _S(TypedDict, total=False):
        result: object

    async def node(state: _S) -> dict:
        out = await wrapped.ainvoke({"incident_id": inc.id, "patch": {"status": "resolved"}})
        return {"result": out}

    sg = StateGraph(_S)
    sg.add_node("n", node)
    sg.set_entry_point("n")
    sg.add_edge("n", END)
    compiled = sg.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-persist"}}

    result = asyncio.run(compiled.ainvoke({}, config=cfg))
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    assert interrupts, "high-risk wrap must surface an Interrupt"

    # A fresh load (mimicking the watchdog) sees the pending row.
    reloaded = store.load(inc.id)
    pending = [tc for tc in reloaded.tool_calls if tc.status == "pending_approval"]
    assert len(pending) == 1
    assert pending[0].tool == "update_incident"
