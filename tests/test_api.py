import pytest
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
        paths=Paths(skills_file="config/skills.yaml", incidents_dir=str(tmp_path)),
    )


@pytest.mark.asyncio
async def test_health_returns_200(cfg):
    app = await build_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_agents_endpoint_returns_4(cfg):
    app = await build_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/agents")
    assert res.status_code == 200
    names = {a["name"] for a in res.json()}
    assert names == {"intake", "triage", "deep_investigator", "resolution"}


@pytest.mark.asyncio
async def test_investigate_endpoint_creates_incident(cfg):
    app = await build_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/investigate", json={"query": "api latency", "environment": "production"})
    assert res.status_code == 200
    body = res.json()
    assert body["incident_id"].startswith("INC-")
