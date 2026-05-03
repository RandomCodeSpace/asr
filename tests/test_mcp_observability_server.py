import pytest
from runtime.mcp_servers.observability import (
    get_logs, get_metrics, get_service_health, check_deployment_history,
)


@pytest.mark.asyncio
async def test_get_logs_returns_canned_lines():
    out = await get_logs(service="api", environment="production", minutes=15)
    assert isinstance(out["lines"], list)
    assert len(out["lines"]) >= 1
    assert any("ERROR" in line or "WARN" in line for line in out["lines"])


@pytest.mark.asyncio
async def test_get_metrics_returns_numeric_series():
    out = await get_metrics(service="api", environment="production", minutes=15)
    assert "p99_latency_ms" in out
    assert isinstance(out["p99_latency_ms"], (int, float))


@pytest.mark.asyncio
async def test_service_health_known_status_values():
    out = await get_service_health(environment="production")
    assert out["status"] in {"healthy", "degraded", "unhealthy"}


@pytest.mark.asyncio
async def test_deployment_history_lists_recent():
    out = await check_deployment_history(environment="production", hours=24)
    assert isinstance(out["deployments"], list)
    assert all("service" in d and "deployed_at" in d for d in out["deployments"])
