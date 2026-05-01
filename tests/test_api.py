import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport
from orchestrator.api import build_app
from orchestrator.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="orchestrator.mcp_servers.incident",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="orchestrator.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="orchestrator.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="orchestrator.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
    )


@asynccontextmanager
async def _client_with_lifespan(app):
    """Wrap an httpx client around an ASGI app, triggering lifespan
    startup/shutdown around the request scope.

    Required because :class:`httpx.ASGITransport` does NOT auto-trigger
    lifespan; ``app.state.orchestrator`` is set during the FastAPI
    lifespan and would be missing without this wrapper.
    """
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.mark.asyncio
async def test_health_returns_200(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_agents_endpoint_returns_4(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/agents")
    assert res.status_code == 200
    names = {a["name"] for a in res.json()}
    assert names == {"intake", "triage", "deep_investigator", "resolution"}


@pytest.mark.asyncio
async def test_investigate_endpoint_creates_incident(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.post(
            "/investigate",
            json={"query": "api latency", "environment": "production"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["incident_id"].startswith("INC-")
