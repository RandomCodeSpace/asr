"""Incident-management domain state.

``IncidentState`` extends ``Session`` with all incident-specific fields.
The framework never imports this module; only example-app code does.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from runtime.memory.session_state import MemoryLayerState
from runtime.state import Session

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

IncidentStatus = Literal[
    "new",
    "in_progress",
    "matched",
    "resolved",
    "escalated",
    "awaiting_input",
    "stopped",
    "deleted",
    # Terminal status set by the dedup pipeline when stage-2 LLM
    # confirms this session is a duplicate of a prior closed session.
    # Non-destructive: the row stays queryable with ``parent_session_id``
    # pointing at the original. Retraction (``POST /sessions/{id}/un-duplicate``)
    # flips status back to ``"new"``.
    "duplicate",
]


class Reporter(BaseModel):
    id: str
    team: str


class IncidentState(Session):
    """Incident-specific session fields layered on the generic ``Session``."""

    query: str
    environment: str
    reporter: Reporter
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    severity: str | None = None
    category: str | None = None
    matched_prior_inc: str | None = None
    embedding: list[float] | None = None
    resolution: Any = None

    # ASR memory-layer slots (L2 KG, L5 Release, L7 Playbooks).
    # Round-tripped via ``extra_fields`` — no row schema change needed.
    memory: MemoryLayerState = Field(default_factory=MemoryLayerState)

    # Override the framework agent-input hook with the incident-shaped
    # preamble (id, environment, query, status, findings, user inputs).
    # The shape was previously hardcoded inside ``runtime.graph`` —
    # moved here so the framework no longer assumes incident-management
    # fields exist on every session.
    def to_agent_input(self) -> str:
        base = (
            f"Incident {self.id}\n"
            f"Environment: {self.environment}\n"
            f"Query: {self.query}\n"
            f"Status: {self.status}\n"
        )
        for agent_key, finding in self.findings.items():
            base += f"Findings ({agent_key}): {finding}\n"
        if self.user_inputs:
            bullets = "\n".join(f"- {ui}" for ui in self.user_inputs)
            base += (
                "\nUser-provided context (appended via intervention):\n"
                f"{bullets}\n"
            )
        return base

    # Explicit override of the framework id_format hook so the
    # incident-management app continues to mint ``INC-YYYYMMDD-NNN``
    # ids regardless of any future change to the framework default.
    @classmethod
    def id_format(cls, *, seq: int) -> str:
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"INC-{today}-{seq:03d}"
