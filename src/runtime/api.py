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
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runtime.config import AppConfig, load_config


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
                pass
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

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    @fastapi_app.get("/agents")
    async def agents():
        return fastapi_app.state.orchestrator.list_agents()

    @fastapi_app.get("/tools")
    async def tools():
        return fastapi_app.state.orchestrator.list_tools()

    @fastapi_app.get("/incidents")
    async def incidents(limit: int = 20):
        return fastapi_app.state.orchestrator.list_recent_incidents(limit=limit)

    @fastapi_app.get("/incidents/{incident_id}")
    async def incident(incident_id: str):
        return fastapi_app.state.orchestrator.get_incident(incident_id)

    @fastapi_app.delete("/incidents/{incident_id}")
    async def delete_incident(incident_id: str):
        return fastapi_app.state.orchestrator.delete_incident(incident_id)

    @fastapi_app.post("/investigate")
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
            # ``SessionCapExceeded`` is matched by class name to avoid a
            # hard import dependency at module-load time.
            if e.__class__.__name__ == "SessionCapExceeded":
                raise HTTPException(status_code=429, detail=str(e)) from e
            raise
        return InvestigateResponse(incident_id=sid)

    @fastapi_app.get("/environments")
    async def environments():
        return fastapi_app.state.environments

    @fastapi_app.post("/investigate/stream")
    async def investigate_stream(req: InvestigateRequest) -> StreamingResponse:
        orch = fastapi_app.state.orchestrator

        async def _events():
            async for ev in orch.stream_investigation(
                query=req.query, environment=req.environment,
                reporter_id=req.reporter_id, reporter_team=req.reporter_team,
            ):
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    @fastapi_app.post("/incidents/{incident_id}/resume")
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
                yield f"data: {json.dumps({'event': 'error', 'error': str(exc)}, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Multi-session endpoints
    # ------------------------------------------------------------------

    @fastapi_app.post(
        "/sessions",
        response_model=SessionStartResponse,
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
            if e.__class__.__name__ == "SessionCapExceeded":
                raise HTTPException(status_code=429, detail=str(e)) from e
            raise
        return SessionStartResponse(session_id=sid)

    @fastapi_app.get("/sessions", response_model=list[SessionStatus])
    async def list_sessions_endpoint(request: Request) -> list[SessionStatus]:
        """Snapshot of in-flight sessions (running / awaiting_input / error)."""
        svc = request.app.state.service
        return [SessionStatus(**row) for row in svc.list_active_sessions()]

    # ------------------------------------------------------------------
    # HITL approval endpoints (risk-rated tool gateway)
    # ------------------------------------------------------------------

    @fastapi_app.get(
        "/sessions/{session_id}/approvals",
        response_model=list[PendingApproval],
    )
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
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            # ``ValueError`` covers the SessionStore id-format guard
            # (``Invalid incident id ...``) which we treat as a 404
            # at the API boundary — the client passed an id that
            # cannot exist, semantically equivalent to "not found".
            raise HTTPException(
                status_code=404, detail="session not found"
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

    @fastapi_app.post(
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
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            raise HTTPException(
                status_code=404, detail="session not found"
            ) from e

        decision_payload = {
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

        async def _resume() -> None:
            from langgraph.types import Command

            await orch.graph.ainvoke(
                Command(resume=decision_payload),
                config=orch._thread_config(session_id),
            )

        # Submit the resume onto the long-lived service loop so we
        # don't fight the lifespan thread for the same FastMCP/SQLite
        # transports. We use the async bridge (``submit_async``) rather
        # than ``submit_and_wait`` because this handler may run on the
        # very loop the service is hosting (FastAPI under
        # ``httpx.AsyncClient + ASGITransport``, or any single-loop
        # deployment): blocking that loop while waiting for work
        # scheduled onto it would deadlock.
        await svc.submit_async(_resume())
        return {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

    @fastapi_app.delete("/sessions/{session_id}", status_code=204)
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
