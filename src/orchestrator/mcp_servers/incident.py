"""FastMCP server: incident_management mock tools."""
from __future__ import annotations
from typing import Any
from fastmcp import FastMCP

from orchestrator.incident import IncidentStore
from orchestrator.similarity import KeywordSimilarity, find_similar

mcp = FastMCP("incident_management")

_state: dict[str, Any] = {"store": None, "similarity_threshold": 0.85}


def set_state(*, store: IncidentStore, similarity_threshold: float) -> None:
    _state["store"] = store
    _state["similarity_threshold"] = similarity_threshold


def _store() -> IncidentStore:
    if _state["store"] is None:
        raise RuntimeError("incident_management server not initialized — call set_state first")
    return _state["store"]


@mcp.tool()
async def lookup_similar_incidents(query: str, environment: str) -> dict:
    """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
    store = _store()
    resolved = [i for i in store.list_all() if i.status == "resolved"]
    candidates = [
        {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
         "summary": i.summary, "resolution": i.resolution, "environment": i.environment}
        for i in resolved if i.environment == environment
    ]
    results = find_similar(
        query=query, candidates=candidates, text_field="text",
        scorer=KeywordSimilarity(), threshold=_state["similarity_threshold"], limit=5,
    )
    return {"matches": [
        {"id": r["id"], "summary": r["summary"], "resolution": r["resolution"], "score": round(s, 3)}
        for r, s in results
    ]}


@mcp.tool()
async def create_incident(query: str, environment: str,
                          reporter_id: str = "user-mock", reporter_team: str = "platform") -> dict:
    """Create a new INC ticket and persist it."""
    inc = _store().create(query=query, environment=environment,
                          reporter_id=reporter_id, reporter_team=reporter_team)
    return inc.model_dump()


@mcp.tool()
async def update_incident(incident_id: str, patch: dict) -> dict:
    """Apply a flat patch to an INC. Allowed keys: status, severity, category, summary, tags,
    matched_prior_inc, resolution, findings_triage, findings_deep_investigator."""
    store = _store()
    inc = store.load(incident_id)
    if "status" in patch:
        inc.status = patch["status"]
    if "severity" in patch:
        inc.severity = patch["severity"]
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
    if "findings_triage" in patch:
        inc.findings.triage = patch["findings_triage"]
    if "findings_deep_investigator" in patch:
        inc.findings.deep_investigator = patch["findings_deep_investigator"]
    store.save(inc)
    return inc.model_dump()
