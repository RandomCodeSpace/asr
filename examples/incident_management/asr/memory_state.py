"""ASR memory-layer slots that ride on ``IncidentState.memory``.

Each layer in the ASR.md §3 7-layer model that the MVP slice exercises
(L2 / L5 / L7) gets a small pydantic model here so an investigation
can attach the context it fetched from that layer to the session
state. The whole bundle round-trips losslessly through the framework's
``extra_fields`` mechanism — no row schema changes are needed.

Read-only by construction: agents *consume* these slots; mutation via
MCP tools is not exposed.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class L2KGContext(BaseModel):
    """L2 Knowledge Graph subgraph snapshot.

    Mirrors ASR.md §3 L2 / §6: a small projection over the affected
    components plus their immediate upstream / downstream neighbours.
    ``raw`` carries the full assembled subgraph so downstream agents
    can render or re-traverse without a second store call.
    """

    model_config = ConfigDict(extra="forbid")

    components: list[str] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)


class L5ReleaseContext(BaseModel):
    """L5 Release Context window relevant to an investigation.

    Each entry in ``recent_releases`` is a release record dict with at
    least ``service``, ``sha``, ``deployed_at``, ``author``.
    ``suspect_releases`` is the subset of release ids correlated to
    the incident's start time within the configured window.
    """

    model_config = ConfigDict(extra="forbid")

    recent_releases: list[dict] = Field(default_factory=list)
    suspect_releases: list[str] = Field(default_factory=list)


class L7PlaybookSuggestion(BaseModel):
    """A single L7 playbook the matcher proposes for this investigation."""

    model_config = ConfigDict(extra="forbid")

    playbook_id: str
    score: float = Field(ge=0.0, le=1.0)
    matched_signals: list[str] = Field(default_factory=list)


class MemoryLayerState(BaseModel):
    """Container for the memory-layer slots attached to ``IncidentState``.

    The whole object is optional / empty by default so legacy sessions
    written before this field existed round-trip cleanly: the field
    hydrates to a default ``MemoryLayerState`` even when
    ``extra_fields`` is missing the key entirely.
    """

    model_config = ConfigDict(extra="forbid")

    l2_kg: L2KGContext | None = None
    l5_release: L5ReleaseContext | None = None
    l7_playbooks: list[L7PlaybookSuggestion] = Field(default_factory=list)
