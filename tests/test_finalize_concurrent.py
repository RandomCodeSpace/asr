import asyncio
import pytest
from sqlalchemy import create_engine

from runtime.orchestrator import Orchestrator
from runtime.locks import SessionLockRegistry
from runtime.state import ToolCall
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.mark.asyncio
async def test_concurrent_finalize_only_one_transition(tmp_path):
    """Two concurrent finalize calls — exactly one should transition.
    The second sees status already terminal post-load and returns None.
    """
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)

    class _O:
        def __init__(self, s):
            self.store = s
            self._locks = SessionLockRegistry()
        _finalize_session_status = Orchestrator._finalize_session_status
        _finalize_session_status_async = Orchestrator._finalize_session_status_async
        _save_or_yield = Orchestrator._save_or_yield

    orch = _O(store)
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
