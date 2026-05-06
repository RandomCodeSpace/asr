import asyncio
import pytest
from sqlalchemy import create_engine

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    OrchestratorConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.locks import SessionLockRegistry
from runtime.state import ToolCall
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.terminal_tools import StatusDef, TerminalToolRule


@pytest.mark.asyncio
async def test_concurrent_finalize_only_one_transition(tmp_path):
    """Two concurrent finalize calls — exactly one should transition.
    The second sees status already terminal post-load and returns None.
    """
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)

    cfg = AppConfig(
        llm=LLMConfig(),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(
            statuses={
                "open":         StatusDef(name="open",         terminal=False, kind="pending"),
                "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
                "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
            },
            terminal_tools=[
                TerminalToolRule(tool_name="mark_resolved", status="resolved"),
            ],
            default_terminal_status="needs_review",
        ),
    )

    class _O:
        def __init__(self, s, c):
            self.store = s
            self.cfg = c
            self._locks = SessionLockRegistry()
        _finalize_session_status = Orchestrator._finalize_session_status
        _finalize_session_status_async = Orchestrator._finalize_session_status_async
        _save_or_yield = Orchestrator._save_or_yield
        _infer_terminal_decision = Orchestrator._infer_terminal_decision
        _extract_field = Orchestrator._extract_field

    orch = _O(store, cfg)
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_resolved", args={}, result={},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)

    results = await asyncio.gather(
        orch._finalize_session_status_async(inc.id),
        orch._finalize_session_status_async(inc.id),
    )
    transitioned = [r for r in results if r is not None]
    assert len(transitioned) == 1, "exactly one of the calls should transition"
    assert transitioned[0] == "resolved"
    assert store.load(inc.id).status == "resolved"
