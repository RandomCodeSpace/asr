"""FastAPI app — health, listings, and incident endpoints.

``build_app(cfg)`` is sync and constructs the FastAPI instance. The
orchestrator (which holds long-lived FastMCP transports) is created during
the app's startup lifespan and stored on ``app.state.orchestrator``; the
shutdown hook closes it cleanly. Routes read the orchestrator via
``app.state``.

The module-level ``get_app()`` is a no-arg factory suitable for
``uvicorn --factory``: it reads ``ASR_CONFIG`` (default
``config/config.yaml``) and returns a fresh app.
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.config import AppConfig, load_config
from orchestrator.orchestrator import Orchestrator


class InvestigateRequest(BaseModel):
    query: str
    environment: str
    reporter_id: str = "user-mock"
    reporter_team: str = "platform"


class InvestigateResponse(BaseModel):
    incident_id: str


def _make_lifespan(cfg: AppConfig):
    """Build the lifespan context manager for an app constructed with ``cfg``.

    The orchestrator owns FastMCP transports tied to an asyncio event loop;
    creating it inside the lifespan ensures it lives on uvicorn's loop and
    is closed cleanly on shutdown.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        orch = await Orchestrator.create(cfg)
        app.state.orchestrator = orch
        try:
            yield
        finally:
            await orch.aclose()
    return _lifespan


def build_app(cfg: AppConfig) -> FastAPI:
    """Construct the FastAPI app. Synchronous.

    The ``orchestrator`` is created during the app's startup lifespan and
    is reachable as ``app.state.orchestrator`` from any route handler.
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

    @fastapi_app.post("/investigate")
    async def investigate(req: InvestigateRequest) -> InvestigateResponse:
        inc_id = await fastapi_app.state.orchestrator.start_investigation(
            query=req.query, environment=req.environment,
            reporter_id=req.reporter_id, reporter_team=req.reporter_team,
        )
        return InvestigateResponse(incident_id=inc_id)

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
