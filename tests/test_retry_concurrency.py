import asyncio
import pytest
from sqlalchemy import create_engine

from runtime.orchestrator import Orchestrator
from runtime.locks import SessionLockRegistry
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.mark.asyncio
async def test_concurrent_retry_rejects_second_call(tmp_path, monkeypatch):
    """Two retry_session calls in parallel — only one runs the graph,
    the other yields retry_rejected with reason 'in progress'.
    """
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)

    # Stub orchestrator: only the bits retry_session needs.
    class _O:
        def __init__(self, s):
            self.store = s
            self._locks = SessionLockRegistry()
            self._retries_in_flight: set[str] = set()
        retry_session = Orchestrator.retry_session
        _retry_session_locked = Orchestrator._retry_session_locked

        async def _drain_existing_thread(self, sid):
            return  # no-op for the test stub

        async def _finalize_session_status_async(self, sid):
            return None

    orch = _O(store)
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "error"
    store.save(inc)

    # Stub _retry_session_locked to a slow generator that yields
    # retry_started then sleeps long enough for the second caller to
    # observe the in-flight flag.
    async def _slow_locked(self, sid):
        yield {"event": "retry_started", "incident_id": sid,
               "ts": "t"}
        await asyncio.sleep(0.05)

    monkeypatch.setattr(_O, "_retry_session_locked", _slow_locked)

    events_a, events_b = [], []

    async def _drain(it, out):
        async for ev in it:
            out.append(ev)

    await asyncio.gather(
        _drain(orch.retry_session(inc.id), events_a),
        _drain(orch.retry_session(inc.id), events_b),
    )
    rejected = [ev for ev in events_a + events_b
                if ev["event"] == "retry_rejected"]
    started = [ev for ev in events_a + events_b
               if ev["event"] == "retry_started"]
    assert len(started) == 1, f"expected 1 retry_started, got {len(started)}"
    assert len(rejected) == 1, f"expected 1 retry_rejected, got {len(rejected)}"
    assert "in progress" in rejected[0]["reason"]
