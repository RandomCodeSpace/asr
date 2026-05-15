"""Cross-session SSE — emits session-level events across all sessions."""
import asyncio
import json
from fastapi.testclient import TestClient
from runtime.api import build_app
from tests.test_api_v1_url_move import _cfg


def test_recent_events_replays_session_creates(tmp_path):
    """Two POST /sessions calls each emit a session.created event;
    the recent SSE stream replays them on connect."""
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        # Create two sessions
        for q in ["alpha", "beta"]:
            r = client.post("/api/v1/sessions", json={
                "query": q, "environment": "dev",
                "submitter": {"id": "u1", "team": "p"},
            })
            assert r.status_code == 201

        # Direct-call the SSE handler with a forced-disconnect to drain
        # the backlog (avoids long-poll deadlock in TestClient).
        from starlette.requests import Request as StarletteRequest

        async def _disc():
            return True  # exit tail loop after backlog drain

        sse_route = next(
            r for r in app.router.routes
            if getattr(r, "path", "") == "/api/v1/sessions/recent/events"
        )
        scope = {
            "type": "http", "method": "GET",
            "path": "/api/v1/sessions/recent/events",
            "query_string": b"since=0", "headers": [], "app": app,
        }
        request = StarletteRequest(scope)
        request.is_disconnected = _disc  # type: ignore[method-assign]
        response = asyncio.run(
            sse_route.endpoint(request=request, since=0)  # type: ignore[attr-defined]
        )

        async def _drain():
            frames = []
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else chunk
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(json.loads(line[len("data: "):]))
            return frames

        frames = asyncio.run(_drain())
        kinds = [f["kind"] for f in frames]
        assert kinds.count("session.created") == 2
