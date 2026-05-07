"""FastMCP server: remediation mock tools."""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from fastmcp import FastMCP

mcp = FastMCP("remediation")


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


@mcp.tool()
async def propose_fix(hypothesis: str, environment: str) -> dict:
    """Generate a remediation proposal for a hypothesis. Returns auto_apply_safe flag."""
    seed = _seed(hypothesis, environment)
    safe = (seed % 3) == 0  # ~33% of mock proposals are auto-safe
    return {
        "proposal_id": f"prop-{seed % 1000:03d}",
        "proposal": f"Restart the affected service in {environment}; investigate {hypothesis}.",
        "auto_apply_safe": safe,
        "estimated_impact": "low" if safe else "medium",
    }


@mcp.tool()
async def apply_fix(proposal_id: str, environment: str) -> dict:
    """Apply a previously proposed fix. Mock returns success/failure deterministically."""
    seed = _seed(proposal_id, environment)
    success = (seed % 4) != 0
    return {
        "proposal_id": proposal_id,
        "environment": environment,
        "status": "applied" if success else "failed",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "details": "Mock remediation completed." if success else "Mock remediation failed.",
    }


# Snapshot of the escalation-teams roster, replaced by `register(mcp_app, cfg)`
# at orchestrator boot. Tools below dereference it at call time (not at
# decoration time), so swapping the tuple binding is sufficient — no
# module-level mutable list, no setter API.
_escalation_teams: tuple[str, ...] = ()


def register(mcp_app, cfg) -> None:
    """App-MCP-server discovery contract: bind config-derived state.

    Reads the escalation-teams roster from
    ``cfg.framework.escalation_teams`` (or ``cfg.escalation_teams`` as a
    fallback for older configs) and snapshots it into the module-level
    ``_escalation_teams`` tuple. Idempotent.

    ``mcp_app`` is accepted for contract uniformity; this module exposes
    its own ``mcp`` FastMCP instance composed by the loader.
    """
    global _escalation_teams
    teams: list[str] = []
    framework_cfg = getattr(cfg, "framework", None)
    if framework_cfg is not None and getattr(framework_cfg, "escalation_teams", None):
        teams = list(framework_cfg.escalation_teams)
    elif getattr(cfg, "escalation_teams", None):
        teams = list(cfg.escalation_teams)
    _escalation_teams = tuple(teams)


@mcp.tool()
async def notify_oncall(incident_id: str, message: str, team: str) -> dict:
    """Page the oncall engineer for the named team. ``team`` is REQUIRED
    and must be in the configured escalation_teams roster.
    """
    if not team:
        raise ValueError("team is required (got empty string)")
    if _escalation_teams and team not in _escalation_teams:
        raise ValueError(
            f"team {team!r} not in escalation_teams ({list(_escalation_teams)})"
        )
    return {
        "incident_id": incident_id,
        "team": team,
        "page_id": f"page-{abs(hash(incident_id + team)) % 10000:04d}",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
