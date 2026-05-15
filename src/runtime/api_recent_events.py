"""Cross-session SSE — used by the React UI's Other Sessions monitor.

Emits session-level events (not per-session detail events): session.created,
session.status_changed, session.agent_running. Lower frequency than the
per-session stream.

Registered only via :func:`runtime.api.build_app` (requires
``app.state.orchestrator``); not suitable for lightweight test fixtures
that construct a bare ``FastAPI()`` app — use ``build_app(cfg)`` for tests.
"""
from __future__ import annotations
import asyncio as _asyncio
import json
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

_SSE_MEDIA_TYPE = "text/event-stream"
_SESSION_KINDS = frozenset({
    "session.created",
    "session.status_changed",
    "session.agent_running",
})


def add_recent_events_routes(api_v1: APIRouter) -> None:
    """Mount the /sessions/recent/events SSE handler on the api_v1 router.

    Module-qualified name so the bundler can flatten alongside sibling
    ``api_*`` side-cars without ``add_routes`` collisions. See
    ``runtime.api_session_full.add_session_full_routes``.
    """

    @api_v1.get("/sessions/recent/events")
    async def stream_recent_events(request: Request, since: int = 0):
        orch = request.app.state.orchestrator
        event_log = getattr(orch, "event_log", None)
        if event_log is None:
            raise HTTPException(
                status_code=503, detail="event_log not configured",
            )

        async def _stream() -> AsyncIterator[str]:
            last_seq = since
            # Backlog: emit session-level events past `since`
            for ev in event_log.iter_recent(since=last_seq):
                if ev.kind in _SESSION_KINDS:
                    payload = {"seq": ev.seq, "kind": ev.kind,
                               "session_id": ev.session_id,
                               "payload": ev.payload, "ts": ev.ts}
                    last_seq = ev.seq
                    yield f"data: {json.dumps(payload)}\n\n"
            # Tail: poll for new rows; exit on client disconnect
            while not await request.is_disconnected():
                await _asyncio.sleep(0.5)
                for ev in event_log.iter_recent(since=last_seq):
                    if ev.kind in _SESSION_KINDS:
                        payload = {"seq": ev.seq, "kind": ev.kind,
                                   "session_id": ev.session_id,
                                   "payload": ev.payload, "ts": ev.ts}
                        last_seq = ev.seq
                        yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_stream(), media_type=_SSE_MEDIA_TYPE)
