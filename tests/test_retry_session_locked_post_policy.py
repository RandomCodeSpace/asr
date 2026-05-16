"""Coverage tests for ``Orchestrator._retry_session_locked`` post-policy
execution path (orchestrator.py:1552-1587).

The retry method has three early-exit branches (session-not-found,
not-in-error-state, policy-rejected) that other tests already cover.
This file exercises what happens *after* the policy accepts: the
failed-AgentRun filter, retry_count + active_thread_id pinning, the
graph re-stream, and the pause-vs-finalize fork at the tail.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from runtime.config import OrchestratorConfig
from runtime.locks import SessionLockRegistry
from runtime.orchestrator import Orchestrator
from runtime.state import AgentRun
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


# ---------------------------------------------------------------------------
# Stub orchestrator that pulls in the real `_retry_session_locked` body but
# substitutes the surrounding integration points (graph, finalize, pause).
# ---------------------------------------------------------------------------
class _StubOrch:
    """Minimum surface needed by ``_retry_session_locked`` lines 1515-1588."""

    def __init__(self, store: SessionStore, *, paused: bool, finalized: str | None,
                 graph_events: list[dict] | None = None) -> None:
        self.store = store
        self.cfg = type("_Cfg", (), {"orchestrator": OrchestratorConfig()})()
        self._locks = SessionLockRegistry()
        self._retries_in_flight: set[str] = set()
        self._paused = paused
        self._finalized = finalized
        self._graph_events = graph_events or [
            {"event": "on_chain_start", "name": "intake"},
            {"event": "on_chain_end", "name": "intake"},
        ]
        self._streamed_state = None  # captures the GraphState handed to astream
        # Stub graph object
        outer = self

        class _Graph:
            async def astream_events(self, state, *, version, config):
                outer._streamed_state = state
                outer._streamed_config = config
                for ev in outer._graph_events:
                    yield ev

        self.graph = _Graph()

    # Pull real method bodies in directly. The static helpers must be
    # re-wrapped because reading them off the class strips the
    # staticmethod descriptor and they would bind as instance methods.
    retry_session = Orchestrator.retry_session
    _retry_session_locked = Orchestrator._retry_session_locked
    _to_ui_event = staticmethod(Orchestrator._to_ui_event)
    _extract_last_error = staticmethod(Orchestrator._extract_last_error)
    _extract_last_confidence = staticmethod(Orchestrator._extract_last_confidence)

    def _thread_config(self, sid: str) -> dict:
        return {"configurable": {"thread_id": sid}}

    async def _is_graph_paused(self, sid: str) -> bool:
        return self._paused

    async def _finalize_session_status_async(self, sid: str) -> str | None:
        return self._finalized

    async def _mark_session_paused_async(self, sid: str) -> str | None:
        # Issue #42: retry path now writes 'awaiting_input' on the
        # paused branch in addition to the existing finalize call.
        # The stub records that it was called (paused branch flag).
        self._marked_paused_calls = getattr(self, "_marked_paused_calls", 0) + 1
        return "awaiting_input" if self._paused else None


@pytest.fixture
def store(tmp_path) -> SessionStore:
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(eng)
    return SessionStore(engine=eng)


def _seed_failed_session(store: SessionStore) -> str:
    inc = store.create(query="probe", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "error"
    inc.agents_run = [
        AgentRun(
            agent="intake",
            started_at="2026-05-15T00:00:00Z",
            ended_at="2026-05-15T00:00:01Z",
            summary="completed: routed",
        ),
        AgentRun(
            agent="triage",
            started_at="2026-05-15T00:00:02Z",
            ended_at="2026-05-15T00:00:03Z",
            summary="agent failed: TimeoutError: provider hung",
        ),
    ]
    store.save(inc)
    return inc.id


@pytest.mark.asyncio
async def test_retry_completes_drops_failed_runs_bumps_thread_finalizes(store):
    sid = _seed_failed_session(store)
    orch = _StubOrch(store, paused=False, finalized="resolved")

    events = []
    async for ev in orch.retry_session(sid):
        events.append(ev)

    kinds = [e["event"] for e in events]
    # Started + at least one streamed event + status_auto_finalized + completed.
    assert "retry_started" in kinds
    assert "status_auto_finalized" in kinds
    assert kinds[-1] == "retry_completed"
    assert "session_paused" not in kinds  # finalize branch took it

    # Persisted state reflects all post-policy mutations.
    inc = store.load(sid)
    # Failed AgentRun was filtered; only the successful intake run remains
    # in the pre-streamed timeline (the stub graph doesn't add new runs).
    assert all(not (r.summary or "").startswith("agent failed:") for r in inc.agents_run)
    assert [r.agent for r in inc.agents_run] == ["intake"]
    # retry_count bumped from 0 to 1.
    assert inc.extra_fields.get("retry_count") == 1
    # active_thread_id pinned to retry-1.
    assert inc.extra_fields.get("active_thread_id") == f"{sid}:retry-1"
    # Status flipped back to in_progress when entering the retry stream.
    # (Subsequent _finalize would normally update it, but the stub returns
    # "resolved" without writing back to the DB — that's the orchestrator's
    # _finalize_session_status_async responsibility, which we stubbed out.)
    assert inc.status == "in_progress"

    # The status_auto_finalized event carries the stub's "resolved" status.
    finalized_event = next(e for e in events if e["event"] == "status_auto_finalized")
    assert finalized_event["status"] == "resolved"


@pytest.mark.asyncio
async def test_retry_pause_branch_yields_session_paused_not_finalized(store):
    sid = _seed_failed_session(store)
    orch = _StubOrch(store, paused=True, finalized="should-not-be-used")

    events = []
    async for ev in orch.retry_session(sid):
        events.append(ev)

    kinds = [e["event"] for e in events]
    assert "retry_started" in kinds
    assert "session_paused" in kinds
    assert "status_auto_finalized" not in kinds  # pause branch took it
    assert kinds[-1] == "retry_completed"


@pytest.mark.asyncio
async def test_retry_increments_retry_count_across_calls(store):
    sid = _seed_failed_session(store)
    orch = _StubOrch(store, paused=False, finalized=None)

    # First retry: 0 -> 1
    async for _ in orch.retry_session(sid):
        pass
    assert store.load(sid).extra_fields["retry_count"] == 1

    # Reset the session back to error so a second retry is allowed.
    inc = store.load(sid)
    inc.status = "error"
    inc.agents_run.append(AgentRun(
        agent="triage",
        started_at="2026-05-15T00:00:10Z",
        ended_at="2026-05-15T00:00:11Z",
        summary="agent failed: TimeoutError: still broken",
    ))
    store.save(inc)

    # Second retry: 1 -> 2; thread id pin reflects the new count.
    async for _ in orch.retry_session(sid):
        pass
    inc = store.load(sid)
    assert inc.extra_fields["retry_count"] == 2
    assert inc.extra_fields["active_thread_id"] == f"{sid}:retry-2"
