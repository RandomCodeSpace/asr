"""React-readiness API surface tests.

Covers every endpoint added under T2-T4 of the api-react-readiness
branch:

  GET  /sessions/recent
  GET  /sessions/{sid}
  POST /sessions/{sid}/resume          (SSE)
  POST /sessions/{sid}/retry           (SSE)
  GET  /sessions/{sid}/retry/preview
  GET  /sessions/{sid}/lessons
  GET  /sessions/{sid}/events          (SSE)
  WS   /ws/sessions/{session_id}/events

Plus CORS middleware + structured error envelope for HTTPException.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from runtime.api import build_app
from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    OrchestratorConfig,
    Paths,
    RuntimeConfig,
)
from runtime.state import AgentRun, ToolCall
from runtime.terminal_tools import StatusDef, TerminalToolRule


_STATUSES = {
    "open":         StatusDef(name="open",         terminal=False, kind="pending"),
    "in_progress":  StatusDef(name="in_progress",  terminal=False, kind="pending"),
    "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
    "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
}
_RULES = [TerminalToolRule(tool_name="mark_resolved", status="resolved")]


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
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        orchestrator=OrchestratorConfig(
            statuses=_STATUSES,
            terminal_tools=_RULES,
            default_terminal_status="needs_review",
        ),
        environments=["production", "staging", "dev"],
        runtime=RuntimeConfig(state_class=None),
    )


@pytest.fixture(autouse=True)
def _reset_orchestrator_service_singleton():
    yield
    try:
        from runtime.service import OrchestratorService
        OrchestratorService._reset_singleton()
    except Exception:
        pass


@asynccontextmanager
async def _client_with_lifespan(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


def _seed_resolved_session(orch, *, query: str) -> str:
    """Create a session via store, append a mark_resolved tool call +
    agent_run, save. Returns the session id."""
    inc = orch.store.create(
        query=query, environment="staging",
        reporter_id="u", reporter_team="t",
    )
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_resolved",
        args={}, result={"status": "resolved"},
        ts="2026-05-13T00:00:00Z", status="executed",
    ))
    inc.agents_run.append(AgentRun(
        agent="resolution",
        started_at="2026-05-13T00:00:00Z",
        ended_at="2026-05-13T00:00:05Z",
        summary="resolved", confidence=0.9, signal="success",
    ))
    inc.status = "in_progress"
    orch.store.save(inc)
    return inc.id


# ===================================================================
# T2: generic /sessions/* endpoints
# ===================================================================

@pytest.mark.asyncio
async def test_get_sessions_recent_returns_list(cfg):
    """``GET /sessions/recent`` lists recent sessions of any status."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        orch = app.state.orchestrator
        _seed_resolved_session(orch, query="latency spike")
        res = await client.get("/sessions/recent?limit=5")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) == 1


@pytest.mark.asyncio
async def test_get_session_detail_404_for_unknown_id(cfg):
    """``GET /sessions/{sid}`` returns the structured error envelope
    on a 404."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions/SES-DOES-NOT-EXIST")
    assert res.status_code == 404
    body = res.json()
    assert "error" in body
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "session not found"


@pytest.mark.asyncio
async def test_get_session_detail_returns_row(cfg):
    """``GET /sessions/{sid}`` returns the session dump on the new
    non-legacy URL."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        orch = app.state.orchestrator
        sid = _seed_resolved_session(orch, query="api latency")
        res = await client.get(f"/sessions/{sid}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == sid
    assert body["status"] == "in_progress"


@pytest.mark.asyncio
async def test_get_retry_preview_404_for_unknown(cfg):
    """An id that fails the SessionStore format check raises
    ValueError ahead of ``preview_retry_decision``'s FileNotFoundError
    branch, which the endpoint maps to 404."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions/UNKNOWN/retry/preview")
    assert res.status_code == 404
    body = res.json()
    assert body["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_retry_preview_happy_path_returns_decision(cfg):
    """Seed a session in ``status=error`` and assert preview returns
    a typed RetryDecisionPreview."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        orch = app.state.orchestrator
        inc = orch.store.create(
            query="latency", environment="staging",
            reporter_id="u", reporter_team="t",
        )
        inc.status = "error"
        inc.extra_fields["retry_count"] = 0
        orch.store.save(inc)
        res = await client.get(f"/sessions/{inc.id}/retry/preview")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["retry"], bool)
    assert isinstance(body["reason"], str)
    # The framework's default policy with retry_count=0 + no error
    # signal yields a recognised reason value.
    assert body["reason"] in {
        "auto_retry", "max_retries_exceeded", "permanent_error",
        "low_confidence_no_retry", "transient_disabled",
    }


@pytest.mark.asyncio
async def test_get_session_lessons_returns_extracted_rows(cfg):
    """A session that hit a terminal status produces a lesson row;
    ``GET /sessions/{sid}/lessons`` surfaces it."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        orch = app.state.orchestrator
        sid = _seed_resolved_session(orch, query="payments-svc")
        # Drive the finalize hook so the M5 lesson row lands.
        orch._finalize_session_status(sid)
        res = await client.get(f"/sessions/{sid}/lessons")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    lesson = body[0]
    assert lesson["source_session_id"] == sid
    assert lesson["outcome_status"] == "resolved"
    assert "id" in lesson and "tools" in lesson


@pytest.mark.asyncio
async def test_get_session_lessons_empty_when_no_corpus(cfg):
    """Sessions that never resolved produce no lessons."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions/SES-EMPTY/lessons")
    # No matching row -> empty list, not 404.
    assert res.status_code == 200
    assert res.json() == []


# ===================================================================
# T3: SSE event stream
# ===================================================================

@pytest.mark.asyncio
async def test_sse_events_replays_backlog(cfg):
    """SSE stream's backlog-drain phase yields one ``data: <env>\\n\\n``
    frame per event. Calls the handler coroutine directly (rather than
    through a real HTTP round-trip) because httpx ASGITransport +
    TestClient both block in their stream-close paths when the server
    generator is in a long-poll loop. The HTTP wire format is the same
    pre-poll; the WebSocket test exercises the full wire round-trip
    (see ``test_websocket_event_stream_replays_backlog``).
    """
    from runtime.api import EventEnvelope

    app = build_app(cfg)
    with TestClient(app):
        orch = app.state.orchestrator
        orch.event_log.record("SES-SSE", "agent_started", agent="triage")
        orch.event_log.record(
            "SES-SSE", "tool_invoked", tool="ping", agent="triage",
            latency_ms=12,
        )
        orch.event_log.record("SES-SSE", "agent_finished", agent="triage")

        # Find the SSE route + invoke its handler directly.
        sse_route = next(
            r for r in app.router.routes
            if getattr(r, "path", "") == "/sessions/{session_id}/events"
        )
        # The handler is the async function under .endpoint.
        # Fake a Request with the orchestrator wired + a disconnect-False
        # callable so the generator drains the backlog and then exits
        # cleanly the moment we stop iterating.
        from starlette.requests import Request as StarletteRequest

        async def _disconnected() -> bool:
            return True  # forces the loop to exit after the backlog drain

        scope = {
            "type": "http", "method": "GET",
            "path": "/sessions/SES-SSE/events",
            "query_string": b"since=0",
            "headers": [],
            "app": app,
        }
        request = StarletteRequest(scope)
        # Inject a fake is_disconnected so the tail loop exits.
        request.is_disconnected = _disconnected  # type: ignore[method-assign]
        response = await sse_route.endpoint(
            session_id="SES-SSE", request=request, since=0,
        )

        # Drain the generator manually.
        frames: list[dict] = []
        body_iter = response.body_iterator
        async for chunk in body_iter:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            for line in text.splitlines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[len("data: "):]))

    assert len(frames) == 3
    assert frames[0]["kind"] == "agent_started"
    assert frames[1]["kind"] == "tool_invoked"
    assert frames[1]["payload"]["latency_ms"] == 12
    assert frames[2]["kind"] == "agent_finished"
    seqs = [f["seq"] for f in frames]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3
    # Each frame validates against EventEnvelope (wire-shape lock).
    for f in frames:
        EventEnvelope.model_validate(f)


def test_event_log_iter_for_since_filters_backlog(cfg):
    """``EventLog.iter_for(sid, since=N)`` returns only rows whose
    ``seq > N``. The SSE endpoint just plumbs this primitive through;
    testing the primitive is faster + more reliable than trying to
    cleanly disconnect from a streaming SSE response.
    """
    app = build_app(cfg)
    with TestClient(app):
        orch = app.state.orchestrator
        orch.event_log.record("SES-SKIP", "agent_started", agent="a")
        orch.event_log.record("SES-SKIP", "agent_finished", agent="a")
        latest = list(orch.event_log.iter_for("SES-SKIP"))
        assert len(latest) == 2
        max_seq = latest[-1].seq

        # Backlog past max_seq is empty — exactly what the SSE drain
        # loop iterates over before entering the tail-poll branch.
        after = list(orch.event_log.iter_for("SES-SKIP", since=max_seq))
        assert after == []

        # Adding a new event shows up.
        orch.event_log.record("SES-SKIP", "tool_invoked", tool="x")
        after = list(orch.event_log.iter_for("SES-SKIP", since=max_seq))
        assert [e.kind for e in after] == ["tool_invoked"]


# ===================================================================
# Resume + retry SSE happy paths — these exercise the full HTTP
# round-trip on SSE endpoints that yield a finite event stream and
# close naturally (resume / retry orchestrator generators terminate
# when the underlying coroutine completes, unlike the open-ended
# events stream whose poll loop is bounded by client-disconnect).
# The events SSE wire format is covered by:
#   * test_sse_events_replays_backlog (direct generator),
#   * test_websocket_event_stream_replays_backlog (same envelope
#     shape, real transport).
# ===================================================================

def test_post_resume_sse_returns_event_stream(cfg):
    """POST /sessions/{sid}/resume returns text/event-stream and
    produces at least one frame (event or structured error envelope —
    the orchestrator may produce an error on an unresumable session,
    which the handler maps to the structured envelope)."""
    app = build_app(cfg)
    with TestClient(app) as client:
        orch = app.state.orchestrator
        sid = _seed_resolved_session(orch, query="resume-target")
        with client.stream(
            "POST", f"/sessions/{sid}/resume",
            json={"decision": "resume_with_input", "user_input": "go"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            frame_payloads: list[dict] = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    frame_payloads.append(json.loads(line[len("data: "):]))
                # Resume on an already-resolved session emits a small
                # number of frames or a single error envelope; stop
                # after first frame so the test never hangs on a tail
                # poll.
                if frame_payloads:
                    break

    assert len(frame_payloads) >= 1
    f = frame_payloads[0]
    # Either an orchestrator event ({event: ...}) or the structured
    # error envelope when the session can't be resumed in this state.
    assert isinstance(f, dict)


def test_post_retry_sse_returns_event_stream(cfg):
    """POST /sessions/{sid}/retry returns text/event-stream.
    Mirrors the resume contract; the orchestrator's retry path
    emits framed events the React client renders."""
    app = build_app(cfg)
    with TestClient(app) as client:
        orch = app.state.orchestrator
        # A session in error state is the realistic retry target;
        # seed one so the handler exercises the orchestrator path.
        inc = orch.store.create(
            query="retry-target", environment="staging",
            reporter_id="u", reporter_team="t",
        )
        inc.status = "error"
        orch.store.save(inc)

        with client.stream(
            "POST", f"/sessions/{inc.id}/retry",
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            frame_payloads: list[dict] = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    frame_payloads.append(json.loads(line[len("data: "):]))
                if frame_payloads:
                    break

    assert len(frame_payloads) >= 1
    assert isinstance(frame_payloads[0], dict)


# ===================================================================
# T4: WebSocket fallback
# ===================================================================

def test_websocket_event_stream_replays_backlog(cfg):
    """The WS endpoint mirrors the SSE payload shape."""
    app = build_app(cfg)
    with TestClient(app) as client:
        # TestClient triggers the FastAPI lifespan so app.state.orchestrator
        # is wired.
        orch = app.state.orchestrator
        orch.event_log.record("SES-WS", "agent_started", agent="triage")
        orch.event_log.record(
            "SES-WS", "tool_invoked", tool="ping", agent="triage",
            latency_ms=5,
        )
        orch.event_log.record("SES-WS", "agent_finished", agent="triage")

        with client.websocket_connect(
            "/ws/sessions/SES-WS/events?since=0",
        ) as ws:
            frames = [ws.receive_json() for _ in range(3)]

    assert [f["kind"] for f in frames] == [
        "agent_started", "tool_invoked", "agent_finished",
    ]
    assert frames[1]["payload"]["latency_ms"] == 5


# ===================================================================
# T5: CORS middleware
# ===================================================================

@pytest.mark.asyncio
async def test_cors_allows_react_dev_origins(cfg):
    """The Vite + CRA dev origins must be in the CORS allow-list by
    default so React can call the API without preflight rejections."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.options(
            "/sessions",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ===================================================================
# Structured error envelope
# ===================================================================

@pytest.mark.asyncio
async def test_404_renders_structured_error_envelope(cfg):
    """HTTPException 404 -> {"error":{"code":"not_found", ...}}."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions/NOPE")
    assert res.status_code == 404
    body = res.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "not_found"
    assert isinstance(body["error"]["message"], str)
    assert isinstance(body["error"]["details"], dict)


@pytest.mark.asyncio
async def test_unknown_endpoint_returns_404_envelope(cfg):
    """A 404 from FastAPI's router is also wrapped in the envelope
    (the global handler runs on any HTTPException, including the
    automatic 404 raised by Starlette for unknown routes)."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/this-route-does-not-exist")
    assert res.status_code == 404
    body = res.json()
    # Starlette's default 404 body is ``{"detail": "Not Found"}``; the
    # global handler normalises it to the envelope shape.
    assert "error" in body
    assert body["error"]["code"] == "not_found"


# ===================================================================
# T6 e2e: full React-shaped flow against a stub session
# ===================================================================

@pytest.mark.asyncio
async def test_react_surface_e2e_terminal_session(cfg):
    """End-to-end: seed -> finalize -> assert via every endpoint a
    React UI would call. SSE leg uses the WebSocket transport (same
    wire shape, doesn't hang on close like the HTTP SSE round-trip)."""
    app = build_app(cfg)
    with TestClient(app) as client:
        orch = app.state.orchestrator
        sid = _seed_resolved_session(orch, query="e2e demo")
        orch._finalize_session_status(sid)

        # 1. GET /sessions/recent — session A is in the list.
        recent = client.get("/sessions/recent").json()
        assert any(r["id"] == sid for r in recent)

        # 2. GET /sessions/{sid} — terminal status.
        detail = client.get(f"/sessions/{sid}").json()
        assert detail["status"] == "resolved"

        # 3. GET /sessions/{sid}/lessons — at least one lesson row.
        lessons = client.get(f"/sessions/{sid}/lessons").json()
        assert len(lessons) >= 1
        assert lessons[0]["outcome_status"] == "resolved"

        # 4. WS /ws/sessions/{sid}/events — events present
        #    (status_changed + lesson_extracted at minimum). WS gives
        #    the same wire shape as the SSE endpoint and supports a
        #    clean test-time disconnect.
        frames: list[dict] = []
        with client.websocket_connect(
            f"/ws/sessions/{sid}/events?since=0",
        ) as ws:
            # Pull all backlog frames quickly. We seeded the session
            # so the finalize emitted at least status_changed +
            # lesson_extracted; the corpus add path emits those.
            for _ in range(10):
                try:
                    frames.append(ws.receive_json())
                except Exception:
                    break
                if {"status_changed", "lesson_extracted"}.issubset(
                    {f["kind"] for f in frames}
                ):
                    break
        kinds = {f["kind"] for f in frames}
        assert "status_changed" in kinds
        assert "lesson_extracted" in kinds
