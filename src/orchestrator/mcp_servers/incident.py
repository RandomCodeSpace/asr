"""FastMCP server: incident_management tools, backed by IncidentRepository.

State scoping
-------------
Each Orchestrator constructs its own :class:`IncidentMCPServer`, so two
orchestrators in the same process do not share repository state. The
module-level ``mcp`` and ``set_state`` symbols are kept as a back-compat
surface for the MCP loader (``getattr(mod, "mcp")``) and for tests that
import these names directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fastmcp import FastMCP

from orchestrator.storage.repository import IncidentRepository


_DEFAULT_SEVERITY_ALIASES: dict[str, str] = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low", "informational": "low",
    "low": "low",
}


def normalize_severity(
    value: str | None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentMCPServer:
    """FastMCP server bound to a single :class:`IncidentRepository`."""
    repository: IncidentRepository | None = None
    severity_aliases: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SEVERITY_ALIASES)
    )
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(
        self, *,
        repository: IncidentRepository,
        severity_aliases: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases

    def _require_repo(self) -> IncidentRepository:
        if self.repository is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.repository

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        repo = self._require_repo()
        hits = repo.find_similar(query=query, environment=environment, limit=5)
        return {"matches": [
            {"id": i.id, "summary": i.summary, "resolution": i.resolution,
             "score": round(s, 3)}
            for i, s in hits
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    reporter_id: str = "user-mock",
                                    reporter_team: str = "platform") -> dict:
        """Create a new INC ticket and persist it."""
        inc = self._require_repo().create(query=query, environment=environment,
                                          reporter_id=reporter_id,
                                          reporter_team=reporter_team)
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC.

        Allowed keys:
          - status, severity, category, summary, tags, matched_prior_inc, resolution
          - findings_<agent_name> — writes ``inc.findings[<agent_name>] = value``.
        """
        repo = self._require_repo()
        inc = repo.load(incident_id)
        if "status" in patch:
            inc.status = patch["status"]
        if "severity" in patch:
            inc.severity = normalize_severity(patch["severity"], self.severity_aliases)
        if "category" in patch:
            inc.category = patch["category"]
        if "summary" in patch:
            inc.summary = patch["summary"]
        if "tags" in patch:
            inc.tags = list(patch["tags"])
        if "matched_prior_inc" in patch:
            inc.matched_prior_inc = patch["matched_prior_inc"]
        if "resolution" in patch:
            inc.resolution = patch["resolution"]
        for key, value in patch.items():
            if key.startswith("findings_"):
                inc.findings[key[len("findings_"):]] = value
        repo.save(inc)
        return inc.model_dump()


# ---------------------------------------------------------------------------
# Module-level default server (back-compat for the MCP loader path).
# The MCP loader imports ``mcp`` from this module by name; this keeps that
# contract working unchanged.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, repository: IncidentRepository,
              severity_aliases: dict[str, str] | None = None) -> None:
    """Configure the default IncidentMCPServer instance."""
    _default_server.configure(
        repository=repository,
        severity_aliases=severity_aliases,
    )


# Direct-call shims kept for tests that import these names.
async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str,
                          reporter_id: str = "user-mock",
                          reporter_team: str = "platform") -> dict:
    return await _default_server._tool_create_incident(
        query, environment, reporter_id, reporter_team
    )


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)
