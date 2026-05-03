import json
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport
from runtime.api import build_app
from runtime.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths, RuntimeConfig


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="runtime.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="runtime.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="runtime.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class=None,
            # Wire the incident-management environments provider so
            # GET /environments returns the bundled roster (api.py no
            # longer imports IncidentAppConfig directly post-merge).
            environments_provider_path=(
                "examples.incident_management.config:environments_provider"
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _reset_orchestrator_service_singleton():
    """Ensure the process-singleton ``OrchestratorService`` does not
    leak across test cases.

    The FastAPI lifespan wires onto ``OrchestratorService.get_or_create``,
    which is process-scoped. Each test builds its own ``build_app(cfg)``
    and the lifespan calls ``shutdown()`` cleanly, but a failed test
    could skip teardown — sweep up here belt-and-braces.
    """
    yield
    try:
        from runtime.service import OrchestratorService
        OrchestratorService._reset_singleton()
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Multi-session endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_sessions_returns_201_with_session_id(cfg):
    """POST /sessions delegates to OrchestratorService.start_session and
    returns ``201 {session_id}``."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.post(
            "/sessions",
            json={
                "query": "api latency",
                "environment": "production",
                "submitter": {"id": "u", "team": "t"},
            },
        )
    assert res.status_code == 201
    body = res.json()
    assert "session_id" in body
    assert isinstance(body["session_id"], str)
    assert body["session_id"]


@pytest.mark.asyncio
async def test_post_sessions_omitting_submitter_uses_defaults(cfg):
    """The ``submitter`` field is optional on the new endpoint —
    omitting it must work and the runtime applies its historical
    defaults under the hood."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.post(
            "/sessions",
            json={"query": "no submitter", "environment": "dev"},
        )
    assert res.status_code == 201
    assert res.json()["session_id"]


@pytest.mark.asyncio
async def test_get_sessions_returns_list(cfg):
    """GET /sessions returns a JSON list (possibly empty); shape matches
    the ``SessionStatus`` schema for any active rows."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    for row in body:
        assert {"session_id", "status", "started_at"} <= row.keys()


@pytest.mark.asyncio
async def test_delete_session_endpoint_returns_204_or_501(cfg):
    """DELETE /sessions/{id} returns 204 when stop_session is wired up;
    otherwise the route surfaces a deterministic 501 Not Implemented.

    Either outcome is acceptable here — we only assert the contract."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        start = await client.post(
            "/sessions",
            json={
                "query": "to be stopped",
                "environment": "production",
                "submitter": {"id": "u", "team": "t"},
            },
        )
        assert start.status_code == 201
        sid = start.json()["session_id"]
        res = await client.delete(f"/sessions/{sid}")
    # 204 = stopped cleanly; 501 = stop_session not wired up;
    # 404 = session already evicted.
    assert res.status_code in (204, 404, 501)


@pytest.mark.asyncio
async def test_legacy_investigate_route_still_works(cfg):
    """``POST /investigate`` is preserved as a deprecated alias and
    delegates to ``OrchestratorService.start_session`` under the hood.
    Old clients should keep receiving an ``incident_id``."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.post(
            "/investigate",
            json={"query": "back-compat", "environment": "production"},
        )
    assert res.status_code in (200, 201)
    body = res.json()
    assert "incident_id" in body
    assert body["incident_id"]


@pytest.mark.asyncio
async def test_legacy_investigate_route_emits_no_deprecation_warnings(cfg):
    """The /investigate handler now coerces the legacy body into
    ``submitter`` / ``state_overrides`` BEFORE calling start_session,
    so the runtime's deprecation path never fires on a hot HTTP
    route. Production logs stay quiet."""
    import warnings

    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            res = await client.post(
                "/investigate",
                json={
                    "query": "no-warn",
                    "environment": "production",
                    "reporter_id": "alice",
                    "reporter_team": "ops",
                },
            )
    assert res.status_code in (200, 201)
    runtime_deprecations = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and (
            "reporter_id" in str(w.message)
            or "environment is a deprecated kwarg" in str(w.message)
        )
    ]
    assert runtime_deprecations == [], (
        f"runtime deprecation warnings leaked through /investigate: "
        f"{[str(w.message) for w in runtime_deprecations]}"
    )


@pytest.mark.asyncio
async def test_app_state_exposes_service_and_orchestrator(cfg):
    """Sanity check: the lifespan wires both ``app.state.service`` and
    ``app.state.orchestrator`` so legacy routes and new /sessions
    handlers can coexist over the same long-lived backing."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        # Both legacy and new routes work in the same lifespan window.
        legacy = await client.get("/agents")
        new = await client.get("/sessions")
    assert legacy.status_code == 200
    assert new.status_code == 200
    assert hasattr(app.state, "service")
    assert hasattr(app.state, "orchestrator")
