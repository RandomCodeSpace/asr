"""FastMCP server: observability mock tools."""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Annotated
from fastmcp import FastMCP
from pydantic import BeforeValidator

mcp = FastMCP("observability")


def _coerce_int(default: int):
    """Build a BeforeValidator that coerces LLM-supplied junk to ``default``.

    LLMs occasionally pass placeholder strings (``"??"``, ``""``,
    ``"unknown"``) into numeric tool args. Strict pydantic validation
    aborts the tool call and the agent often abandons the turn instead
    of retrying. Coercing to a sane default keeps the investigation
    moving with the documented lookback window.
    """
    def _coerce(v: object) -> int:
        if v is None or v == "":
            return default
        if isinstance(v, bool):
            return default
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
    return _coerce


_Minutes = Annotated[int, BeforeValidator(_coerce_int(15))]
_Hours = Annotated[int, BeforeValidator(_coerce_int(24))]


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


@mcp.tool()
async def get_logs(service: str, environment: str, minutes: _Minutes = 15) -> dict:
    """Return canned recent log lines for a service in an environment."""
    seed = _seed(service, environment, str(minutes))
    rng = (seed >> 4) % 4
    base = [
        f"{datetime.now(timezone.utc).isoformat()} INFO {service} request_id=abc123 path=/v1/items dur=42ms",
        f"{datetime.now(timezone.utc).isoformat()} WARN {service} slow_query duration=820ms table=orders",
        f"{datetime.now(timezone.utc).isoformat()} ERROR {service} upstream_timeout target=payments duration=5000ms",
        f"{datetime.now(timezone.utc).isoformat()} INFO {service} cache_miss key=user:42",
    ]
    return {"service": service, "environment": environment, "lines": base[rng:] + base[:rng]}


@mcp.tool()
async def get_metrics(service: str, environment: str, minutes: _Minutes = 15) -> dict:
    """Return canned metrics snapshot."""
    seed = _seed(service, environment)
    return {
        "service": service,
        "environment": environment,
        "window_minutes": minutes,
        "p50_latency_ms": 50 + (seed % 50),
        "p99_latency_ms": 800 + (seed % 1500),
        "error_rate": round(((seed % 100) / 100) * 0.05, 4),
        "rps": 120 + (seed % 300),
        "cpu_pct": 30 + (seed % 60),
        "mem_pct": 40 + (seed % 50),
    }


@mcp.tool()
async def get_service_health(environment: str) -> dict:
    """Return overall environment health summary."""
    seed = _seed(environment)
    statuses = ["healthy", "degraded", "unhealthy"]
    status = statuses[seed % 3]
    return {
        "environment": environment,
        "status": status,
        "services": {
            "api": "healthy" if status == "healthy" else status,
            "db": "healthy",
            "cache": "healthy",
            "queue": status,
        },
    }


@mcp.tool()
async def check_deployment_history(environment: str, hours: _Hours = 24) -> dict:
    """Return canned recent deployments."""
    now = datetime.now(timezone.utc)
    seed = _seed(environment, str(hours))
    deployments = [
        {"service": "api", "version": f"v1.{(seed % 50) + 100}", "deployed_at":
         (now - timedelta(hours=2)).isoformat(), "deployer": "deploy-bot"},
        {"service": "worker", "version": f"v2.{(seed % 30) + 50}", "deployed_at":
         (now - timedelta(hours=8)).isoformat(), "deployer": "deploy-bot"},
    ]
    return {"environment": environment, "window_hours": hours, "deployments": deployments}
