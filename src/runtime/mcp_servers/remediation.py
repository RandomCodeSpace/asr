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


_escalation_teams: list[str] = []


def set_escalation_teams(teams: list[str]) -> None:
    """Bind the allowed escalation_teams roster from app config."""
    global _escalation_teams
    _escalation_teams = list(teams)


@mcp.tool()
async def notify_oncall(incident_id: str, message: str, team: str) -> dict:
    """Page the oncall engineer for the named team. ``team`` is REQUIRED
    and must be in the configured escalation_teams roster.
    """
    if not team:
        raise ValueError("team is required (got empty string)")
    if _escalation_teams and team not in _escalation_teams:
        raise ValueError(
            f"team {team!r} not in escalation_teams ({_escalation_teams})"
        )
    return {
        "incident_id": incident_id,
        "team": team,
        "page_id": f"page-{abs(hash(incident_id + team)) % 10000:04d}",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
