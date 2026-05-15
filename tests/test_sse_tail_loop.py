"""Coverage tests for the SSE ``_stream`` tail loop in
``runtime.api.build_app``.

The existing ``test_sse_events_replays_backlog`` covers the backlog
drain by forcing ``is_disconnected`` to return True before the tail
loop runs. These tests target the tail-poll branch:

  * one tail iteration delivers a newly-recorded event, then a
    second is_disconnected check exits the loop;
  * task cancellation surfaces as ``asyncio.CancelledError``, which
    the loop must propagate (the bug previously swallowed it via
    ``except CancelledError: return``; PR #13 removed the suppressing
    handler so cancellation propagates by Python default).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from runtime.api import build_app
from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    Paths,
    RuntimeConfig,
)


@pytest.fixture
def cfg(tmp_path):
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
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(state_class=None),
    )


def _build_request(app, sid: str, *, is_disconnected) -> StarletteRequest:
    scope = {
        "type": "http", "method": "GET",
        "path": f"/api/v1/sessions/{sid}/events",
        "query_string": b"since=0",
        "headers": [],
        "app": app,
    }
    request = StarletteRequest(scope)
    request.is_disconnected = is_disconnected  # type: ignore[method-assign]
    return request


def _sse_route(app):
    return next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/api/v1/sessions/{session_id}/events"
    )


@pytest.mark.asyncio
async def test_tail_loop_delivers_post_drain_event(cfg, monkeypatch):
    """One backlog event → drain. Then is_disconnected returns False
    once (tail iterates), record a new event, second is_disconnected
    returns True (exit). Verifies the tail-loop body (lines 880-888)
    runs and yields the new frame.
    """
    # Cut sleep duration so the test stays fast. Capture the original
    # before patching so the replacement doesn't recurse into itself.
    real_sleep = asyncio.sleep

    async def _instant_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    app = build_app(cfg)
    with TestClient(app):
        orch = app.state.orchestrator
        orch.event_log.record("SES-TAIL", "agent_started", agent="triage")

        disconnect_results = iter([False, True])

        async def _is_disconnected() -> bool:
            try:
                v = next(disconnect_results)
            except StopIteration:
                return True
            # First "False" return: record a NEW event so the next
            # iter_for(since=last_seq) call inside the tail body
            # yields something. This proves the tail loop ran.
            if v is False:
                orch.event_log.record(
                    "SES-TAIL", "tool_invoked",
                    tool="post-drain", agent="triage",
                )
            return v

        request = _build_request(app, "SES-TAIL", is_disconnected=_is_disconnected)
        response = await _sse_route(app).endpoint(
            session_id="SES-TAIL", request=request, since=0,
        )

        frames: list[dict] = []
        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            for line in text.splitlines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[len("data: "):]))

    kinds = [f["kind"] for f in frames]
    # The backlog drain delivered the first event; the tail loop
    # delivered the post-drain event added inside _is_disconnected.
    assert "agent_started" in kinds
    assert "tool_invoked" in kinds, (
        f"tail loop never yielded a post-drain frame; got kinds={kinds}"
    )


@pytest.mark.asyncio
async def test_tail_loop_propagates_cancellation(cfg, monkeypatch):
    """Cancelling the SSE generator must raise ``CancelledError`` out
    of the ``_stream`` body — not be swallowed.

    This pins the PR #13 bug fix: the original code swallowed
    ``CancelledError`` via ``except: return``; the post-fix loop has
    no ``except`` wrapper at all, so cancellation propagates by
    Python's default. Previously this test would observe
    ``StopAsyncIteration`` instead of ``CancelledError``.
    """
    # Force the sleep inside the tail loop to raise CancelledError on
    # first call so we can observe propagation without racing on real
    # cancellation timing.
    async def _cancel_immediately(*_a, **_k):
        raise asyncio.CancelledError()

    monkeypatch.setattr("asyncio.sleep", _cancel_immediately)

    app = build_app(cfg)
    with TestClient(app):
        orch = app.state.orchestrator
        orch.event_log.record("SES-CANCEL", "agent_started", agent="triage")

        async def _never_disconnect() -> bool:
            return False

        request = _build_request(app, "SES-CANCEL", is_disconnected=_never_disconnect)
        response = await _sse_route(app).endpoint(
            session_id="SES-CANCEL", request=request, since=0,
        )

        frames: list[dict] = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else chunk
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(json.loads(line[len("data: "):]))

        # The backlog drain delivered the seeded event before the tail
        # loop's first sleep raised — proves CancelledError propagated
        # naturally out of the generator (not swallowed).
        assert any(f["kind"] == "agent_started" for f in frames)
