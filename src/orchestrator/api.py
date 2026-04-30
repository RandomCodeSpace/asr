"""FastAPI app — health, listings, and (future) external orchestrator endpoints."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.config import AppConfig
from orchestrator.orchestrator import Orchestrator


class InvestigateRequest(BaseModel):
    query: str
    environment: str
    reporter_id: str = "user-mock"
    reporter_team: str = "platform"


class InvestigateResponse(BaseModel):
    incident_id: str


async def build_app(cfg: AppConfig) -> FastAPI:
    orch = await Orchestrator.create(cfg)
    app = FastAPI(title="ASR — Agent Orchestrator")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/agents")
    async def agents():
        return orch.list_agents()

    @app.get("/tools")
    async def tools():
        return orch.list_tools()

    @app.get("/incidents")
    async def incidents(limit: int = 20):
        return orch.list_recent_incidents(limit=limit)

    @app.get("/incidents/{incident_id}")
    async def incident(incident_id: str):
        return orch.get_incident(incident_id)

    @app.post("/investigate", response_model=InvestigateResponse)
    async def investigate(req: InvestigateRequest) -> InvestigateResponse:
        inc_id = await orch.start_investigation(
            query=req.query, environment=req.environment,
            reporter_id=req.reporter_id, reporter_team=req.reporter_team,
        )
        return InvestigateResponse(incident_id=inc_id)

    app.state.orchestrator = orch
    return app
