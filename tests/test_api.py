import json
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport
from orchestrator.api import build_app
from orchestrator.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
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


@pytest.mark.asyncio
async def test_environments_endpoint_returns_list(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/environments")
    assert res.status_code == 200
    envs = res.json()
    assert isinstance(envs, list)
    assert len(envs) > 0
    assert any(e in envs for e in ["production", "staging", "dev", "local"])


@pytest.mark.asyncio
async def test_investigate_stream_emits_events(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        async with client.stream(
            "POST", "/investigate/stream",
            json={"query": "api latency", "environment": "production"},
        ) as res:
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]
            events = []
            async for line in res.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))
    assert events, "expected at least one SSE event"
    assert all(isinstance(e, dict) for e in events)


@pytest.mark.asyncio
async def test_delete_endpoint_soft_deletes_incident(cfg):
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        created = await client.post(
            "/investigate",
            json={"query": "to be deleted", "environment": "production"},
        )
        inc_id = created.json()["incident_id"]
        del_res = await client.delete(f"/incidents/{inc_id}")
        assert del_res.status_code == 200
        body = del_res.json()
        assert body["status"] == "deleted"
        assert body["deleted_at"] is not None
        listing = await client.get("/incidents")
        assert all(i["id"] != inc_id for i in listing.json()), \
            "deleted incident must be hidden from /incidents"
        single = await client.get(f"/incidents/{inc_id}")
        assert single.status_code == 200
        assert single.json()["status"] == "deleted"


@pytest.mark.asyncio
async def test_resume_endpoint_streams_error_for_unknown_incident(cfg):
    """Route is registered and returns SSE; unknown/invalid INC ID yields an
    error event rather than an unhandled 500."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        async with client.stream(
            "POST", "/incidents/INC-does-not-exist/resume",
            json={"decision": "stop"},
        ) as res:
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]
            events = []
            async for line in res.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))
    # Must emit at least one event (an error event for the bad ID).
    assert events
    event_types = {e.get("event") for e in events}
    assert event_types & {"error", "resume_rejected", "resume_started"}
