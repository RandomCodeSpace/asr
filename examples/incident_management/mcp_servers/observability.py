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


def build_environment_validator(allowed: list[str]):
    """Return an Annotated[str, BeforeValidator] that lowercases input
    and rejects values not in ``allowed``. Bound at server-init time
    from the framework env list. Tools using this type get a
    recoverable 422 from FastMCP when the LLM emits ``"prod"`` instead
    of ``"production"`` instead of silently passing through to a
    backend that has no policy entry for the typo.
    """
    allowed_lower = {a.lower() for a in allowed}

    def _validate(v: object) -> str:
        if not isinstance(v, str):
            raise ValueError(f"environment must be a string, got {type(v).__name__}")
        canonical = v.lower()
        if canonical not in allowed_lower:
            raise ValueError(
                f"environment {v!r} not in {sorted(allowed_lower)}"
            )
        return canonical

    return Annotated[str, BeforeValidator(_validate)]


def _make_environment_validator(environments: list[str]):
    """Build a per-registration env validator that closes over a snapshot
    of the configured environments. No module-level mutable state.
    """
    snapshot = list(environments)

    def _validate_environment(env: str) -> str:
        if not snapshot:
            return env
        canonical = env.lower() if isinstance(env, str) else env
        allowed_lower = {e.lower() for e in snapshot}
        if canonical not in allowed_lower:
            raise ValueError(
                f"environment {env!r} not in {sorted(allowed_lower)}"
            )
        return canonical

    return _validate_environment


# Default no-op validator used when tools are attached at module import
# time (`@mcp.tool()` decorators below). `register(mcp_app, cfg)` swaps
# the binding for one that closes over the configured roster.
_validate_environment = _make_environment_validator([])


def register(mcp_app, cfg) -> None:
    """App-MCP-server discovery contract: bind config-derived state.

    Called once by the orchestrator at create()-time. Reads the
    environments roster from ``cfg.environments`` and replaces the
    module-level ``_validate_environment`` reference with one that
    closes over the snapshot. Idempotent: re-binding overwrites the
    previous closure.

    ``mcp_app`` is the framework's primary FastMCP instance — accepted
    for contract uniformity but not used here (this module exposes its
    own ``mcp`` instance composed by the loader).
    """
    global _validate_environment
    envs = list(getattr(cfg, "environments", []) or [])
    _validate_environment = _make_environment_validator(envs)


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


@mcp.tool()
async def get_logs(service: str, environment: str, minutes: _Minutes = 15) -> dict:
    """Return canned recent log lines for a service in an environment."""
    environment = _validate_environment(environment)
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
    environment = _validate_environment(environment)
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
    environment = _validate_environment(environment)
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
    environment = _validate_environment(environment)
    now = datetime.now(timezone.utc)
    seed = _seed(environment, str(hours))
    deployments = [
        {"service": "api", "version": f"v1.{(seed % 50) + 100}", "deployed_at":
         (now - timedelta(hours=2)).isoformat(), "deployer": "deploy-bot"},
        {"service": "worker", "version": f"v2.{(seed % 30) + 50}", "deployed_at":
         (now - timedelta(hours=8)).isoformat(), "deployer": "deploy-bot"},
    ]
    return {"environment": environment, "window_hours": hours, "deployments": deployments}
