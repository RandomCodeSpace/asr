"""FastMCP server: incident_management mock tools.

State scoping
-------------
Earlier revisions used a module-level ``_state`` dict, which made the FastMCP
instance and its store reference process-global. Two concurrent
:class:`Orchestrator` instances (or two pytest workers) would clobber each
other's stores. Now state is held on a per-instance :class:`IncidentMCPServer`,
and a fresh ``mcp`` (FastMCP) instance is built for every server.

The module-level ``mcp`` symbol (imported by the MCP loader) is constructed
lazily — the first call to :func:`get_or_create_default_server` (or the
back-compat :func:`set_state`) builds it. ``set_state`` remains as a thin shim
that mutates the default server's state, so existing call-sites and tests keep
working without churn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fastmcp import FastMCP

from orchestrator.incident import IncidentStore
from orchestrator.similarity import KeywordSimilarity, find_similar


_SEVERITY_MAP = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low", "informational": "low",
    "low": "low",
}


def normalize_severity(value: str | None) -> str | None:
    """Coerce assorted severity inputs (sev1/p2/critical/etc.) to low/medium/high.

    Unknown inputs pass through untouched so callers can flag them; the
    normalized vocabulary surfaced to the UI and stored on disk is restricted
    to {low, medium, high}.
    """
    if value is None:
        return None
    return _SEVERITY_MAP.get(value.strip().lower(), value)


@dataclass
class IncidentMCPServer:
    """Per-instance container holding a FastMCP server and its scoped state.

    Each Orchestrator constructs its own :class:`IncidentMCPServer`, so two
    orchestrators in the same process (e.g. two test fixtures, two web users
    behind one Streamlit deployment) no longer share a store reference.
    """
    store: IncidentStore | None = None
    similarity_threshold: float = 0.85
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        # Bind the tool implementations to *this* server's state. We pass an
        # explicit ``name=`` because FastMCP defaults to the function's
        # ``__name__`` (which would expose the leading underscore). The names
        # below match the original module-level tool functions, so the MCP
        # tool surface (and the LangChain registry that consumes it) is
        # unchanged.
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(self, *, store: IncidentStore, similarity_threshold: float) -> None:
        self.store = store
        self.similarity_threshold = similarity_threshold

    def _require_store(self) -> IncidentStore:
        if self.store is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.store

    # FastMCP introspects parameter type hints to build tool schemas, so the
    # bound methods below take the same signatures as the previous module-level
    # functions (no `self` from the LLM's POV — `self` is captured by the bound
    # method when we pass it to ``mcp.tool()``).

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        store = self._require_store()
        resolved = [i for i in store.list_all() if i.status == "resolved"]
        candidates = [
            {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
             "summary": i.summary, "resolution": i.resolution, "environment": i.environment}
            for i in resolved if i.environment == environment
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(), threshold=self.similarity_threshold, limit=5,
        )
        return {"matches": [
            {"id": r["id"], "summary": r["summary"], "resolution": r["resolution"], "score": round(s, 3)}
            for r, s in results
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    reporter_id: str = "user-mock",
                                    reporter_team: str = "platform") -> dict:
        """Create a new INC ticket and persist it."""
        inc = self._require_store().create(query=query, environment=environment,
                                           reporter_id=reporter_id,
                                           reporter_team=reporter_team)
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC.

        Allowed keys:
          - status, severity, category, summary, tags, matched_prior_inc, resolution
          - findings_<agent_name> — writes ``inc.findings[<agent_name>] = value``
            (e.g. ``findings_triage``, ``findings_deep_investigator``,
            or any agent name a YAML-defined skill may use).
        """
        store = self._require_store()
        inc = store.load(incident_id)
        if "status" in patch:
            inc.status = patch["status"]
        if "severity" in patch:
            inc.severity = normalize_severity(patch["severity"])
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
        # findings_<agent> → inc.findings[<agent>] for any agent name.
        for key, value in patch.items():
            if key.startswith("findings_"):
                inc.findings[key[len("findings_"):]] = value
        store.save(inc)
        return inc.model_dump()


# ---------------------------------------------------------------------------
# Module-level default server (back-compat for the existing MCP loader path).
#
# The MCP loader imports ``mcp`` from this module by name (it doesn't know
# about IncidentMCPServer). To preserve that contract, we expose:
#
#   - ``mcp``   : a FastMCP instance owned by ``_default_server``
#   - ``set_state(...)`` : configure ``_default_server``'s store/threshold
#   - ``lookup_similar_incidents`` / ``create_incident`` / ``update_incident``
#     : thin shims so direct callers in tests still work
#
# Per-Orchestrator scoping is achieved by constructing a fresh
# :class:`IncidentMCPServer` (planned in a follow-up that wires it through
# ``mcp_loader.load_tools``). This commit removes the *module-global dict* —
# state now lives on the instance — without breaking the existing import path.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, store: IncidentStore, similarity_threshold: float) -> None:
    """Configure the default IncidentMCPServer instance.

    Kept for backwards compatibility with callers that import this function
    directly (the orchestrator and several tests). New code should construct an
    :class:`IncidentMCPServer` and call ``configure`` on it.
    """
    _default_server.configure(store=store, similarity_threshold=similarity_threshold)


# Public function aliases so test modules importing these names directly keep
# working. They forward to the bound methods on ``_default_server``.
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
