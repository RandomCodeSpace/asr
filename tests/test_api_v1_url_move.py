"""Verify all API endpoints have moved under /api/v1/*."""
from fastapi.testclient import TestClient
from runtime.api import build_app
from runtime.config import (
    AppConfig, LLMConfig, MCPConfig, MCPServerConfig,
    Paths, RuntimeConfig,
)


def _cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="examples.incident_management.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="examples.incident_management.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="examples.incident_management.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="examples/incident_management/skills",
                    incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(state_class=None),
    )


def test_health_remains_at_root(tmp_path):
    """Health stays at /health (monitor convention)."""
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_sessions_endpoints_under_api_v1(tmp_path):
    """GET /api/v1/sessions returns 200 (was at /sessions)."""
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/sessions")
        assert r.status_code == 200


def test_agents_under_api_v1(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/agents")
        assert r.status_code == 200


def test_tools_under_api_v1(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/tools")
        assert r.status_code == 200


def test_environments_under_api_v1(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/environments")
        assert r.status_code == 200
