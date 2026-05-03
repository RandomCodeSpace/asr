"""FastMCP server: incident_management tools, backed by SessionStore + HistoryStore.

Part of the incident-management example application. Framework code does
not import this module.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from fastmcp import FastMCP

from runtime.storage.history_store import HistoryStore
from runtime.storage.session_store import SessionStore
from examples.incident_management.config import load_app_config


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
    """FastMCP server bound to a single :class:`SessionStore` (+ optional :class:`HistoryStore`).

    Holds the active ``SessionStore`` and an optional ``HistoryStore``;
    only the ``lookup_similar_incidents`` tool needs the latter.
    """
    store: SessionStore | None = None
    history: HistoryStore | None = None
    severity_aliases: dict[str, str] = field(
        default_factory=lambda: load_app_config().severity_aliases
    )
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(
        self, *,
        store: SessionStore,
        history: HistoryStore | None = None,
        severity_aliases: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.history = history
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases

    def _require_store(self) -> SessionStore:
        if self.store is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.store

    def _require_history(self) -> HistoryStore:
        if self.history is None:
            raise RuntimeError(
                "incident_management server has no HistoryStore configured — "
                "pass history=... to configure() before calling lookup_similar_incidents"
            )
        return self.history

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        history = self._require_history()
        hits = history.find_similar(
            query=query,
            filter_kwargs={"environment": environment},
            limit=5,
        )
        return {"matches": [
            {"id": i.id, "summary": i.summary, "resolution": i.resolution,
             "score": round(s, 3)}
            for i, s in hits
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    submitter: dict | None = None,
                                    reporter_id: str | None = None,
                                    reporter_team: str | None = None) -> dict:
        """Create a new INC ticket and persist it.

        ``submitter`` is the canonical generic-runtime kwarg — for
        incident-management it carries ``{"id": "...", "team": "..."}``.
        ``reporter_id`` / ``reporter_team`` are deprecated; when
        supplied they are coerced into ``submitter`` and a
        ``DeprecationWarning`` is emitted. Passing both raises
        ``TypeError``.
        """
        legacy_supplied = reporter_id is not None or reporter_team is not None
        if submitter is not None and legacy_supplied:
            raise TypeError(
                "create_incident() received both submitter and "
                "reporter_id/reporter_team; pass submitter only "
                "(reporter_id/reporter_team are deprecated)"
            )
        if legacy_supplied:
            warnings.warn(
                "reporter_id and reporter_team are deprecated kwargs on "
                "create_incident(); pass submitter={'id': ..., 'team': ...} "
                "instead. The legacy kwargs will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            submitter = {
                "id": reporter_id if reporter_id is not None else "user-mock",
                "team": reporter_team if reporter_team is not None else "platform",
            }
        sub = submitter or {}
        inc = self._require_store().create(
            query=query,
            environment=environment,
            reporter_id=sub.get("id", "user-mock"),
            reporter_team=sub.get("team", "platform"),
        )
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC.

        Allowed keys:
          - status, severity, category, summary, tags, matched_prior_inc, resolution
          - findings_<agent_name> — writes ``inc.findings[<agent_name>] = value``.
        """
        store = self._require_store()
        inc = store.load(incident_id)
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
        store.save(inc)
        return inc.model_dump()


# ---------------------------------------------------------------------------
# Module-level default server (back-compat for the MCP loader path).
# The MCP loader imports ``mcp`` from this module by name; this keeps that
# contract working unchanged.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, store: SessionStore,
              history: HistoryStore | None = None,
              severity_aliases: dict[str, str] | None = None) -> None:
    """Configure the default IncidentMCPServer instance."""
    _default_server.configure(
        store=store,
        history=history,
        severity_aliases=severity_aliases,
    )


# Direct-call shims kept for tests that import these names.
async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str,
                          submitter: dict | None = None,
                          reporter_id: str | None = None,
                          reporter_team: str | None = None) -> dict:
    return await _default_server._tool_create_incident(
        query, environment,
        submitter=submitter,
        reporter_id=reporter_id,
        reporter_team=reporter_team,
    )


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)
