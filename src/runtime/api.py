"""FastAPI app — health, listings, incident, and multi-session endpoints.

``build_app(cfg)`` is sync and constructs the FastAPI instance. The long-lived
:class:`runtime.service.OrchestratorService` is created during the app's
startup lifespan and stored on ``app.state.service``; its underlying
:class:`runtime.orchestrator.Orchestrator` is exposed as
``app.state.orchestrator`` so legacy routes keep working without
double-building the FastMCP transports / SQLite engines.

The shutdown hook calls ``service.shutdown()`` which cancels in-flight
session tasks, closes MCP clients, joins the background loop thread, and
resets the process-singleton.

``POST /sessions``, ``GET /sessions``, ``DELETE /sessions/{id}`` delegate
to ``OrchestratorService``. The legacy ``POST /investigate`` is preserved
as a deprecated alias and delegates to the same long-lived service so
old clients keep working.

The module-level ``get_app()`` is a no-arg factory suitable for
``uvicorn --factory``: it reads ``ASR_CONFIG`` (default
``config/config.yaml``) and returns a fresh app.
"""
from __future__ import annotations
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from runtime import api_apps_overlay, api_session_full, api_ui_hints
from runtime.config import AppConfig, load_config

_log = logging.getLogger("runtime.api")


# Wire-format constants (extracted to keep S1192 — duplicated literal
# strings — in check; every SSE endpoint uses _SSE_MEDIA_TYPE, every
# session-not-found path raises with _SESSION_NOT_FOUND_DETAIL).
_SSE_MEDIA_TYPE = "text/event-stream"
_SESSION_NOT_FOUND_DETAIL = "session not found"


# HTTP status -> structured error code. Used by the global exception
# handler to keep React's error UI from having to switch on every
# integer status code.
_STATUS_TO_CODE: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "unprocessable_entity",
    429: "rate_limited",
    500: "internal_error",
    501: "not_implemented",
    503: "service_unavailable",
}


def _resolve_environments(dotted: str | None) -> list[str]:
    """Resolve ``RuntimeConfig.environments_provider_path`` to a list.

    Returns an empty list when ``dotted`` is unset (apps that don't
    expose an environments roster). Provider callables must return a
    sequence of strings; anything else raises ``TypeError``.
    """
    if dotted is None:
        return []
    if ":" not in dotted:
        raise ValueError(
            f"environments_provider_path={dotted!r} must be in "
            "'module.path:callable' form"
        )
    import importlib
    module_name, _, attr = dotted.partition(":")
    mod = importlib.import_module(module_name)
    provider = getattr(mod, attr)
    envs = provider()
    if not isinstance(envs, (list, tuple)):
        raise TypeError(
            f"environments provider {dotted!r} returned "
            f"{type(envs).__name__}; expected list[str]"
        )
    return [str(e) for e in envs]


class InvestigateRequest(BaseModel):
    query: str
    environment: str
    reporter_id: str = "user-mock"
    reporter_team: str = "platform"


class InvestigateResponse(BaseModel):
    incident_id: str


class ResumeRequest(BaseModel):
    decision: Literal["resume_with_input", "escalate", "stop"]
    user_input: str | None = None


# ---------------------------------------------------------------------------
# Multi-session schemas
# ---------------------------------------------------------------------------


class SessionStartBody(BaseModel):
    query: str
    environment: str
    # Generic submitter dict — the framework projects ``id``/``team``
    # onto the row's reporter columns; apps interpret the rest. The
    # legacy ``reporter_id`` / ``reporter_team`` fields were removed
    # from this body because the deprecation path on the runtime
    # would emit a warning at every request — production-log noise.
    submitter: dict | None = None


class SessionStartResponse(BaseModel):
    session_id: str


class SessionStatus(BaseModel):
    session_id: str
    status: str
    started_at: str
    current_agent: str | None = None


# ---------------------------------------------------------------------------
# HITL approval schemas (risk-rated tool gateway)
# ---------------------------------------------------------------------------


class ApprovalDecisionBody(BaseModel):
    """Request body for ``POST /sessions/{sid}/approvals/{tool_call_id}``.

    The wrap_tool closure interprets ``decision`` as either ``approve``
    (run the tool, audit) or ``reject`` (skip the tool, audit rejection).
    ``approver`` is the operator id; ``rationale`` is optional free text.
    """

    decision: Literal["approve", "reject"]
    approver: str
    rationale: str | None = None


class RetryDecisionPreview(BaseModel):
    """Response from ``GET /sessions/{sid}/retry/preview``."""
    retry: bool
    reason: str


class LessonResponse(BaseModel):
    """Response item for ``GET /sessions/{sid}/lessons``."""
    id: str
    source_session_id: str
    outcome_status: str
    outcome_summary: str
    confidence_final: float | None = None
    tools: list[str] = Field(default_factory=list)
    created_at: str


class EventEnvelope(BaseModel):
    """Single SSE/WS event payload. Wraps M1 :class:`SessionEvent`."""
    seq: int
    session_id: str
    kind: str
    payload: dict
    ts: str


class ErrorDetail(BaseModel):
    """Body of the structured JSON error envelope."""
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """Wire shape for every 4xx/5xx body the API returns. React calls
    can rely on a stable ``{"error": {"code", "message", "details"}}``
    shape regardless of which handler raised."""
    error: ErrorDetail


def _error_envelope(
    *,
    code: str,
    message: str,
    details: dict | None = None,
    status: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a structured JSON error response. ``headers`` preserves
    the original :class:`HTTPException.headers` (e.g. ``Retry-After``
    on a 429) so contract tests + clients see them through the
    global exception handler.
    """
    return JSONResponse(
        status_code=status,
        content=ErrorEnvelope(
            error=ErrorDetail(
                code=code, message=message, details=details or {},
            ),
        ).model_dump(),
        headers=headers,
    )


class PendingApproval(BaseModel):
    """Snapshot of one pending tool approval read from session.tool_calls."""

    tool_call_id: str
    agent: str
    tool: str
    args: dict
    ts: str


def _make_lifespan(cfg: AppConfig):
    """Build the lifespan context manager for an app constructed with ``cfg``.

    Constructs the :class:`runtime.service.OrchestratorService` singleton,
    starts its background loop, eagerly builds the underlying
    :class:`runtime.orchestrator.Orchestrator` (so legacy routes that
    expect ``app.state.orchestrator`` keep working), and builds the
    :class:`runtime.triggers.TriggerRegistry` from ``cfg.triggers``. The
    webhook router is mounted on the FastAPI app here; APScheduler is
    started by the schedule transport's ``start``.

    On shutdown, the registry's ``stop_all`` runs first (drains
    APScheduler), then ``service.shutdown()`` tears the orchestrator down.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Lazy import: ``runtime.service`` transitively pulls a lot of
        # heavyweight modules (FastMCP, SQLAlchemy). Importing at function
        # scope keeps ``import runtime.api`` cheap for tests/tools that
        # only need ``build_app``.
        from runtime.service import OrchestratorService
        from runtime.triggers import IdempotencyStore, TriggerRegistry
        from runtime.triggers.transports.webhook import WebhookTransport

        svc = OrchestratorService.get_or_create(cfg)
        svc.start()
        # Eagerly build the shared Orchestrator so legacy routes can read
        # it via ``app.state.orchestrator`` without racing on the
        # lazy-build path. ``_ensure_orchestrator`` is on the loop thread,
        # so we hop through the sync bridge.
        orch = svc.submit_and_wait(svc._ensure_orchestrator(), timeout=30.0)
        app.state.service = svc
        app.state.orchestrator = orch
        # Surface the validated AppConfig so app.state.cfg-readers
        # (e.g. /api/v1/config/ui-hints) don't have to re-load YAML.
        app.state.cfg = cfg
        # Environments roster is app-specific (incident-management has
        # production/staging/dev/local; code-review doesn't expose one).
        # Read it from the YAML's top-level ``environments:`` block;
        # fall back to the legacy ``environments_provider_path`` callable
        # for deployments that still wire it.
        if cfg.environments:
            app.state.environments = list(cfg.environments)
        else:
            app.state.environments = _resolve_environments(
                getattr(cfg.runtime, "environments_provider_path", None),
            )

        # ------------------------------------------------------------
        # Build & start the trigger registry
        # ------------------------------------------------------------
        plugin_transports = getattr(app.state, "plugin_transports", None)

        async def _start_session_fn(**kwargs):
            # The registry's dispatch sink. Bridges through the
            # OrchestratorService so we share the one MCP pool / one
            # orchestrator / one DB engine that the rest of the app uses.
            return svc.submit_and_wait(
                _trigger_dispatch(svc, kwargs), timeout=60.0
            )

        async def _trigger_dispatch(service, kwargs):
            # ``svc.start_session`` is sync (returns the session id); the
            # registry awaits us. Trampoline through the loop's
            # default executor.
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: service.start_session(**kwargs)
            )

        idempotency: IdempotencyStore | None = None
        if cfg.triggers:
            try:
                idempotency = IdempotencyStore(orch.store.engine)
            except AttributeError:
                # Older test stubs don't expose ``store.engine``; the
                # registry tolerates ``idempotency=None`` (no caching).
                idempotency = None
        registry = TriggerRegistry.create(
            list(cfg.triggers),
            start_session_fn=_start_session_fn,
            idempotency=idempotency,
            plugin_transports=plugin_transports,
        )
        app.state.trigger_registry = registry
        await registry.start_all()
        # Mount any webhook routers onto the FastAPI app so the routes
        # become live.
        for t in registry.transports:
            if isinstance(t, WebhookTransport):
                app.include_router(t.router)
        try:
            yield
        finally:
            try:
                await registry.stop_all()
            except Exception:  # noqa: BLE001
                # Best-effort: a misbehaving trigger transport must not
                # block ``svc.shutdown()`` below. Surface for observability.
                _log.warning(
                    "trigger registry stop_all failed during lifespan teardown",
                    exc_info=True,
                )
            # ``shutdown()`` cancels in-flight session tasks, closes the
            # underlying Orchestrator + MCP pool, joins the loop thread,
            # and resets the process-singleton.
            svc.shutdown()
    return _lifespan


def build_app(cfg: AppConfig) -> FastAPI:
    """Construct the FastAPI app. Synchronous.

    The :class:`OrchestratorService` and its underlying
    :class:`Orchestrator` are created during the app's startup lifespan
    and are reachable as ``app.state.service`` / ``app.state.orchestrator``
    from any route handler.
    """
    fastapi_app = FastAPI(
        title="ASR — Agent Orchestrator",
        lifespan=_make_lifespan(cfg),
    )
    # All framework routes (except /health) live under /api/v1 so the
    # React client can stably target a versioned surface; /health stays
    # at root for monitor / load-balancer health-check conventions.
    api_v1 = APIRouter(prefix="/api/v1")

    # CORS: configure once with the AppConfig-supplied origins so the
    # React dev server (Vite at :5173, CRA/Next at :3000 by default) can
    # call every endpoint, SSE included. Production deployments lock
    # the origin list down via YAML — same shape, narrower allow-list.
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=cfg.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global handler: HTTPException → structured error envelope. React
    # clients can assume every 4xx/5xx body matches the
    # ``{"error":{"code","message","details"}}`` shape regardless of
    # which handler raised. Per-handler ``raise HTTPException(...,
    # detail=...)`` still works; the handler below normalises the body.
    @fastapi_app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _request: Request, exc: StarletteHTTPException,
    ):
        detail = exc.detail
        # Preserve per-exception headers (e.g. Retry-After on 429).
        passthrough_headers = getattr(exc, "headers", None) or None
        if isinstance(detail, dict) and "error" in detail:
            # Caller already structured it; pass through unchanged.
            return JSONResponse(
                status_code=exc.status_code,
                content=detail,
                headers=passthrough_headers,
            )
        code = _STATUS_TO_CODE.get(exc.status_code, "http_error")
        message = detail if isinstance(detail, str) else str(detail)
        return _error_envelope(
            code=code, message=message,
            status=exc.status_code,
            headers=passthrough_headers,
        )

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    @api_v1.get("/agents")
    async def agents():
        return fastapi_app.state.orchestrator.list_agents()

    @api_v1.get("/tools")
    async def tools():
        return fastapi_app.state.orchestrator.list_tools()

    @api_v1.get("/incidents")
    async def incidents(limit: int = 20):
        return fastapi_app.state.orchestrator.list_recent_incidents(limit=limit)

    @api_v1.get("/incidents/{incident_id}")
    async def incident(incident_id: str):
        return fastapi_app.state.orchestrator.get_incident(incident_id)

    @api_v1.delete("/incidents/{incident_id}")
    async def delete_incident(incident_id: str):
        return fastapi_app.state.orchestrator.delete_incident(incident_id)

    @api_v1.post("/investigate")
    async def investigate(req: InvestigateRequest, request: Request) -> InvestigateResponse:
        """Legacy alias for ``POST /sessions`` — kept for back-compat.

        .. deprecated::
            Prefer ``POST /sessions``. This route now delegates to
            :meth:`OrchestratorService.start_session` so old clients keep
            working with the long-lived service backing.
        """
        svc = request.app.state.service
        # Coerce the legacy HTTP body into the generic runtime kwargs
        # BEFORE delegating, so the runtime's deprecation path never
        # fires on a hot HTTP route. Production logs stay quiet.
        try:
            sid = svc.start_session(
                query=req.query,
                state_overrides={"environment": req.environment},
                submitter={
                    "id": req.reporter_id,
                    "team": req.reporter_team,
                },
            )
        except Exception as e:  # noqa: BLE001
            # ``SessionCapExceeded`` and ``SessionBusy`` are matched by class
            # name to avoid a hard import dependency at module-load time.
            if e.__class__.__name__ in ("SessionCapExceeded", "SessionBusy"):
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
            raise
        return InvestigateResponse(incident_id=sid)

    @api_v1.get("/environments")
    async def environments():
        return fastapi_app.state.environments

    @api_v1.post("/investigate/stream")
    async def investigate_stream(req: InvestigateRequest) -> StreamingResponse:
        orch = fastapi_app.state.orchestrator

        async def _events():
            async for ev in orch.stream_investigation(
                query=req.query, environment=req.environment,
                reporter_id=req.reporter_id, reporter_team=req.reporter_team,
            ):
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(_events(), media_type=_SSE_MEDIA_TYPE)

    @api_v1.post("/incidents/{incident_id}/resume")
    async def resume_incident(incident_id: str, req: ResumeRequest) -> StreamingResponse:
        orch = fastapi_app.state.orchestrator
        decision: dict = {"action": req.decision}
        if req.user_input is not None:
            decision["input"] = req.user_input

        async def _events():
            try:
                async for ev in orch.resume_investigation(incident_id, decision):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                # CodeQL py/stack-trace-exposure: never serialise raw
                # str(exc) into a client-bound stream — exception text
                # can carry stack-trace-equivalent details (file paths,
                # internal IDs). Use the exception class name + the
                # structured envelope shape the rest of the API uses.
                err = {
                    "error": {
                        "code": "resume_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

        return StreamingResponse(_events(), media_type=_SSE_MEDIA_TYPE)

    # ------------------------------------------------------------------
    # Multi-session endpoints
    # ------------------------------------------------------------------

    @api_v1.post(
        "/sessions",
        status_code=201,
    )
    async def start_session_endpoint(
        body: SessionStartBody, request: Request
    ) -> SessionStartResponse:
        """Start a new long-running session. Returns ``201 {session_id}``.

        Returns ``429`` if the configured concurrent-session cap is hit
        (raised by ``OrchestratorService.start_session``). The exception
        class is matched by name so this handler does not depend on a
        hard import.
        """
        svc = request.app.state.service
        try:
            sid = svc.start_session(
                query=body.query,
                state_overrides={"environment": body.environment},
                submitter=body.submitter,
            )
        except Exception as e:  # noqa: BLE001
            if e.__class__.__name__ in ("SessionCapExceeded", "SessionBusy"):
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
            raise
        return SessionStartResponse(session_id=sid)

    @api_v1.get("/sessions")
    async def list_sessions_endpoint(request: Request) -> list[SessionStatus]:
        """Snapshot of in-flight sessions (running / awaiting_input / error)."""
        svc = request.app.state.service
        return [SessionStatus(**row) for row in svc.list_active_sessions()]

    # ------------------------------------------------------------------
    # HITL approval endpoints (risk-rated tool gateway)
    # ------------------------------------------------------------------

    @api_v1.get("/sessions/{session_id}/approvals")
    async def list_pending_approvals(
        session_id: str, request: Request
    ) -> list[PendingApproval]:
        """Return the list of pending tool approvals for a session.

        Filters ``session.tool_calls`` to entries with
        ``status="pending_approval"``. Returns an empty list when the
        session has no pending approvals; ``404`` when the session id
        is unknown.
        """
        svc = request.app.state.service
        orch = request.app.state.orchestrator
        try:
            inc = orch.store.load(session_id)
        except (FileNotFoundError, ValueError, LookupError) as e:  # KeyError is a LookupError subclass
            # ``ValueError`` covers the SessionStore id-format guard
            # (``Invalid incident id ...``) which we treat as a 404
            # at the API boundary — the client passed an id that
            # cannot exist, semantically equivalent to "not found".
            raise HTTPException(
                status_code=404, detail=_SESSION_NOT_FOUND_DETAIL
            ) from e
        # Defensive: ``svc`` is unused here today — the read goes through
        # the orchestrator's store. We keep the reference so a future
        # observability hook (per-call metrics) lives next to the read.
        _ = svc
        out: list[PendingApproval] = []
        for idx, tc in enumerate(inc.tool_calls):
            if tc.status == "pending_approval":
                # tool_call_id is the index in the audit list; stable
                # within the lifetime of the session because tool calls
                # are append-only.
                out.append(PendingApproval(
                    tool_call_id=str(idx),
                    agent=tc.agent,
                    tool=tc.tool,
                    args=tc.args,
                    ts=tc.ts,
                ))
        return out

    @api_v1.post(
        "/sessions/{session_id}/approvals/{tool_call_id}",
        status_code=200,
    )
    async def submit_approval_decision(
        session_id: str,
        tool_call_id: str,
        body: ApprovalDecisionBody,
        request: Request,
    ) -> dict:
        """Resolve a pending tool approval by resuming the paused graph.

        Resumes via ``Command(resume={decision, approver, rationale})``
        against the session's thread_id. The wrap_tool closure reads the
        resume value and either runs the tool (``approve``) or short-
        circuits with ``status="rejected"`` (``reject``).
        """
        svc = request.app.state.service
        orch = request.app.state.orchestrator
        try:
            orch.store.load(session_id)
        except (FileNotFoundError, ValueError, LookupError) as e:  # KeyError is a LookupError subclass
            raise HTTPException(
                status_code=404, detail=_SESSION_NOT_FOUND_DETAIL
            ) from e

        decision_payload = {
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

        async def _resume() -> None:
            from langgraph.types import Command

            # Per D-20: wrap the ainvoke in the per-session lock so an
            # approval submission cannot interleave checkpoint writes
            # against any other turn on the same thread_id. Uses the
            # blocking ``acquire`` (not ``try_acquire``) — if a turn is
            # mid-flight the approval waits for it to release; the
            # service loop's overall request deadline bounds wait.
            # Future fail-fast switch is a one-line change to
            # try_acquire (the existing 429 handler at L484-489 already
            # routes ``SessionBusy`` to HTTP 429).
            async with orch._locks.acquire(session_id):
                await orch.graph.ainvoke(
                    Command(resume=decision_payload),
                    config=orch._thread_config(session_id),
                )
                # Finalize after the verdict-driven run completes so
                # the session row reflects the terminal status (or
                # falls through to ``default_terminal_status``). Skip
                # if a fresh interrupt re-paused the graph — the
                # session is again awaiting operator input. Mirrors
                # the same guard in the UI's
                # ``_submit_approval_via_service`` and the
                # ``stream_session`` finalize call site.
                if not await orch._is_graph_paused(session_id):
                    await orch._finalize_session_status_async(session_id)

        # Submit the resume onto the long-lived service loop so we
        # don't fight the lifespan thread for the same FastMCP/SQLite
        # transports. We use the async bridge (``submit_async``) rather
        # than ``submit_and_wait`` because this handler may run on the
        # very loop the service is hosting (FastAPI under
        # ``httpx.AsyncClient + ASGITransport``, or any single-loop
        # deployment): blocking that loop while waiting for work
        # scheduled onto it would deadlock.
        try:
            await svc.submit_async(_resume())
        except Exception as e:  # noqa: BLE001
            if e.__class__.__name__ == "SessionBusy":
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
            raise
        return {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

    @api_v1.delete("/sessions/{session_id}", status_code=204)
    async def stop_session_endpoint(
        session_id: str, request: Request
    ) -> Response:
        """Cancel an in-flight session and evict its registry entry.

        Returns ``501 Not Implemented`` when the service does not expose
        ``stop_session`` rather than crashing.
        """
        svc = request.app.state.service
        if not hasattr(svc, "stop_session"):
            raise HTTPException(
                status_code=501,
                detail="stop_session not available",
            )
        try:
            svc.stop_session(session_id)
        except Exception as e:  # noqa: BLE001
            # Translate a "session not found" condition into 404 when
            # the underlying error class is recognisable. Otherwise
            # re-raise.
            name = e.__class__.__name__
            if name in {"KeyError", "SessionNotFound"}:
                raise HTTPException(status_code=404, detail=str(e)) from e
            raise
        return Response(status_code=204)

    # ==================================================================
    # T2: generic /sessions/* endpoints (React-ready, non-legacy).
    # ==================================================================

    @api_v1.get("/sessions/recent")
    async def recent_sessions(request: Request, limit: int = 20) -> list[dict]:
        """List recent sessions of ANY status — closed + active.

        Replaces the legacy session-list route which used a domain-
        flavoured noun. React's history panel calls this.
        """
        orch = request.app.state.orchestrator
        return orch.list_recent_sessions(limit=limit)

    @api_v1.get("/sessions/{session_id}")
    async def get_session_detail(session_id: str, request: Request) -> dict:
        """Full session detail. Generic equivalent of the legacy
        domain-flavoured detail route. 404 when the id is unknown."""
        orch = request.app.state.orchestrator
        try:
            return orch.get_session(session_id)
        except (FileNotFoundError, ValueError, LookupError) as e:  # KeyError is a LookupError subclass
            raise HTTPException(
                status_code=404, detail=_SESSION_NOT_FOUND_DETAIL,
            ) from e

    @api_v1.post("/sessions/{session_id}/resume")
    async def resume_session_sse(
        session_id: str, req: ResumeRequest, request: Request,
    ) -> StreamingResponse:
        """Generic resume — SSE stream of orchestrator events.

        Mirrors the legacy domain-flavoured resume route but on the
        non-legacy URL the React client will use. Error frames map to
        the structured error envelope; raw exception text never reaches
        the wire.
        """
        orch = request.app.state.orchestrator
        decision: dict = {"action": req.decision}
        if req.user_input is not None:
            decision["input"] = req.user_input

        async def _events():
            try:
                async for ev in orch.resume_investigation(
                    session_id, decision,
                ):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                err = {
                    "error": {
                        "code": "resume_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

        return StreamingResponse(_events(), media_type=_SSE_MEDIA_TYPE)

    @api_v1.post("/sessions/{session_id}/retry")
    async def retry_session_sse(
        session_id: str, request: Request,
    ) -> StreamingResponse:
        """Retry a failed session. SSE stream of orchestrator events."""
        orch = request.app.state.orchestrator

        async def _events():
            try:
                async for ev in orch.retry_session(session_id):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                err = {
                    "error": {
                        "code": "retry_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

        return StreamingResponse(_events(), media_type=_SSE_MEDIA_TYPE)

    @api_v1.get("/sessions/{session_id}/retry/preview")
    async def preview_retry(
        session_id: str, request: Request,
    ) -> RetryDecisionPreview:
        """Preview whether a retry would proceed without actually
        running it. Used by the UI to render the retry button's
        enabled/disabled state."""
        orch = request.app.state.orchestrator
        try:
            decision = orch.preview_retry_decision(session_id)
        except (FileNotFoundError, ValueError, LookupError) as e:  # KeyError is a LookupError subclass
            raise HTTPException(
                status_code=404, detail=_SESSION_NOT_FOUND_DETAIL,
            ) from e
        return RetryDecisionPreview(
            retry=bool(decision.retry),
            reason=str(decision.reason),
        )

    @api_v1.get("/sessions/{session_id}/lessons")
    async def list_session_lessons(
        session_id: str, request: Request,
    ) -> list[LessonResponse]:
        """List M5 SessionLessonRows whose source_session_id matches
        this session — i.e. the lessons this session contributed to
        the corpus. Empty list when the session never reached a
        terminal status."""
        orch = request.app.state.orchestrator
        lesson_store = getattr(orch, "lesson_store", None)
        if lesson_store is None:
            return []
        from sqlalchemy import select as _select
        from sqlalchemy.orm import Session as _SqlaSession

        from runtime.storage.models import SessionLessonRow

        with _SqlaSession(lesson_store.engine) as s:
            stmt = (
                _select(SessionLessonRow)
                .where(SessionLessonRow.source_session_id == session_id)
                .order_by(SessionLessonRow.created_at.desc())
            )
            rows = list(s.execute(stmt).scalars())
            out: list[LessonResponse] = []
            for row in rows:
                tools = [
                    t.get("tool") for t in row.tool_sequence
                    if t.get("tool")
                ]
                out.append(LessonResponse(
                    id=row.id,
                    source_session_id=row.source_session_id,
                    outcome_status=row.outcome_status,
                    outcome_summary=row.outcome_summary,
                    confidence_final=row.confidence_final,
                    tools=tools,
                    created_at=row.created_at.isoformat(),
                ))
            return out

    # ==================================================================
    # T3: SSE event stream + T4: WebSocket fallback.
    # ==================================================================

    @api_v1.get("/sessions/{session_id}/events")
    async def sse_events(
        session_id: str, request: Request, since: int = 0,
    ) -> StreamingResponse:
        """Server-Sent Events stream of the M1 EventLog for a session.

        Pushes every row whose ``seq > since`` as a JSON
        :class:`EventEnvelope` frame. Polls the EventLog at 250ms
        intervals — simple and reliable; an asyncio-Queue pub/sub layer
        can replace this when perf demands it.

        Disconnect-aware: each iteration checks
        ``request.is_disconnected()`` so the poll loop terminates
        promptly when the client closes the connection. Closes within
        one poll interval (~250ms) of disconnect.
        """
        import asyncio as _asyncio
        orch = request.app.state.orchestrator
        event_log = getattr(orch, "event_log", None)
        if event_log is None:
            raise HTTPException(
                status_code=503, detail="event_log not configured",
            )

        async def _stream():
            last_seq = since
            # Initial drain: replay any backlog past `since` —
            # unconditionally; the disconnect check belongs on the
            # tail-poll loop, not mid-backlog (otherwise an eager
            # disconnect-check ASGI client drops some events).
            for ev in event_log.iter_for(session_id, since=last_seq):
                envelope = EventEnvelope(
                    seq=ev.seq, session_id=ev.session_id,
                    kind=ev.kind, payload=ev.payload, ts=ev.ts,
                )
                last_seq = ev.seq
                yield f"data: {envelope.model_dump_json()}\n\n"
            # Tail: poll for new rows. Bounded by client-disconnect.
            # CancelledError (from task cancellation, e.g. when the
            # client closes the connection) propagates naturally — no
            # try/except needed; suppressing it would break asyncio's
            # cancellation contract (Sonar python:S7497).
            while not await request.is_disconnected():
                await _asyncio.sleep(0.25)
                for ev in event_log.iter_for(session_id, since=last_seq):
                    envelope = EventEnvelope(
                        seq=ev.seq, session_id=ev.session_id,
                        kind=ev.kind, payload=ev.payload, ts=ev.ts,
                    )
                    last_seq = ev.seq
                    yield f"data: {envelope.model_dump_json()}\n\n"

        return StreamingResponse(_stream(), media_type=_SSE_MEDIA_TYPE)

    @api_v1.websocket("/ws/sessions/{session_id}/events")
    async def ws_events(websocket: WebSocket, session_id: str) -> None:
        """WebSocket fallback for the SSE event stream. Same payload
        shape (:class:`EventEnvelope`); clients that prefer WS over
        SSE call this instead. ``since`` is read from the
        ``?since=N`` query string."""
        import asyncio as _asyncio
        await websocket.accept()
        orch = websocket.app.state.orchestrator
        event_log = getattr(orch, "event_log", None)
        if event_log is None:
            await websocket.close(code=1011, reason="event_log not configured")
            return
        since_raw = websocket.query_params.get("since", "0")
        try:
            last_seq = int(since_raw)
        except ValueError:
            last_seq = 0
        try:
            # Initial backlog drain.
            for ev in event_log.iter_for(session_id, since=last_seq):
                last_seq = ev.seq
                await websocket.send_json(
                    EventEnvelope(
                        seq=ev.seq, session_id=ev.session_id,
                        kind=ev.kind, payload=ev.payload, ts=ev.ts,
                    ).model_dump()
                )
            # Tail loop.
            while True:
                await _asyncio.sleep(0.25)
                for ev in event_log.iter_for(session_id, since=last_seq):
                    last_seq = ev.seq
                    await websocket.send_json(
                        EventEnvelope(
                            seq=ev.seq, session_id=ev.session_id,
                            kind=ev.kind, payload=ev.payload, ts=ev.ts,
                        ).model_dump()
                    )
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 — close cleanly on any sink error
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass

    # ==================================================================
    # Bootstrap bundle: GET /api/v1/sessions/{id}/full
    # Single round-trip the React UI calls on session open. Module
    # lives next door so this file stays focused on routing wiring.
    # ==================================================================
    api_session_full.add_routes(api_v1)

    # ==================================================================
    # UI hints: GET /api/v1/config/ui-hints
    # Drives the React shell's brand block, environment switcher list,
    # and approval-rationale dropdown. Read once at app boot.
    # ==================================================================
    api_ui_hints.add_routes(api_v1)

    # ==================================================================
    # App-overlay UI views: GET /api/v1/apps/{app}/ui-views
    # Approach C extensibility — apps register bespoke deep-dive pages
    # (e.g. "Deploy diff") that the framework UI's Selected-detail
    # panel lists as "App-specific views →" links.
    # ==================================================================
    api_apps_overlay.add_routes(api_v1)

    # Legacy /incidents/* and /investigate redirects to /api/v1/* equivalents.
    # 308 preserves method + body so legacy POSTs (e.g. /incidents/{id}/resume)
    # keep working transparently. Removed in v2.1.
    @fastapi_app.api_route(
        "/incidents", methods=["GET", "POST"], include_in_schema=False,
    )
    async def _legacy_incidents_collection() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/sessions", status_code=308)

    @fastapi_app.api_route(
        "/incidents/{path:path}",
        methods=["GET", "POST", "DELETE", "PUT"],
        include_in_schema=False,
    )
    async def _legacy_incidents_detail(path: str) -> RedirectResponse:
        return RedirectResponse(url=f"/api/v1/sessions/{path}", status_code=308)

    @fastapi_app.api_route(
        "/investigate", methods=["POST"], include_in_schema=False,
    )
    async def _legacy_investigate() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/investigate", status_code=308)

    @fastapi_app.api_route(
        "/investigate/{path:path}",
        methods=["POST"],
        include_in_schema=False,
    )
    async def _legacy_investigate_subpath(path: str) -> RedirectResponse:
        return RedirectResponse(
            url=f"/api/v1/investigate/{path}", status_code=308,
        )

    # Mount the versioned router. /health stays at root (registered
    # directly on ``fastapi_app`` above); everything else lives under
    # /api/v1.
    fastapi_app.include_router(api_v1)
    return fastapi_app


def get_app() -> FastAPI:
    """No-arg factory for ``uvicorn --factory``.

    Reads config from the ``ASR_CONFIG`` env var (default
    ``config/config.yaml``) and returns a fresh FastAPI app. The
    orchestrator is created lazily during the app's startup lifespan,
    not eagerly here, so this factory is safe to call from inside
    uvicorn's running event loop.

    Launch::

        python -m uvicorn --app-dir dist app:get_app --factory --port 37776
    """
    cfg_path = Path(os.environ.get("ASR_CONFIG", "config/config.yaml"))
    return build_app(load_config(cfg_path))
