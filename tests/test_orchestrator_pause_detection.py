"""Regression: ``Orchestrator._is_graph_paused`` reports True iff the
compiled graph has a pending step waiting for resume.

This is the single source of truth that lets ``stream_session`` /
``retry_session`` / the API approval handler skip
``_finalize_session_status_async`` on a HITL pause. Without it, a
paused session would be coerced to ``default_terminal_status``,
orphaning the gateway's ``pending_approval`` row and turning the UI's
Approve / Reject buttons into no-ops.

The helper is a thin wrapper over ``self.graph.aget_state(...).next``;
the tests cover both branches and the defensive ``except`` arm
that returns False when the checkpointer has no entry for the thread
(e.g. unknown session id).
"""
from __future__ import annotations

import pytest

from runtime.orchestrator import Orchestrator


class _FakeStateSnapshot:
    def __init__(self, next_=()):
        self.next = next_


class _FakeGraph:
    def __init__(self, snapshot=None, raises=False):
        self._snapshot = snapshot
        self._raises = raises

    async def aget_state(self, _config):  # noqa: D401
        if self._raises:
            raise RuntimeError("no checkpoint for this thread")
        return self._snapshot


def _orch_with_graph(graph) -> Orchestrator:
    """Build a bare Orchestrator instance just sufficient for the helper.

    Bypasses ``__init__`` so we don't have to spin up the full MCP +
    LLM + checkpointer stack just to test a four-line helper. The
    helper only touches ``self.graph`` and ``self._thread_config``.
    """
    orch: Orchestrator = object.__new__(Orchestrator)
    orch.graph = graph

    class _FakeStore:
        def load(self_, _id):  # noqa: D401
            raise FileNotFoundError(_id)

    orch.store = _FakeStore()  # type: ignore[assignment]
    return orch


@pytest.mark.asyncio
async def test_is_graph_paused_returns_true_when_next_is_non_empty():
    """A non-empty ``next`` tuple = the graph has steps queued waiting
    for ``Command(resume=...)`` — i.e. paused on an interrupt."""
    orch = _orch_with_graph(_FakeGraph(snapshot=_FakeStateSnapshot(next_=("tools",))))
    assert await orch._is_graph_paused("INC-1") is True


@pytest.mark.asyncio
async def test_is_graph_paused_returns_false_when_next_is_empty():
    """An empty ``next`` tuple = the graph completed (or never ran).
    Finalize is safe to fire."""
    orch = _orch_with_graph(_FakeGraph(snapshot=_FakeStateSnapshot(next_=())))
    assert await orch._is_graph_paused("INC-1") is False


@pytest.mark.asyncio
async def test_is_graph_paused_returns_false_when_aget_state_raises():
    """Defensive: if the checkpointer has no entry for the thread (or
    raises for any other reason), treat as not-paused. Anything else
    risks blocking finalize on a transient checkpointer hiccup."""
    orch = _orch_with_graph(_FakeGraph(raises=True))
    assert await orch._is_graph_paused("INC-1") is False
