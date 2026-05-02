"""Incident domain model."""
from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, Field

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


IncidentStatus = Literal[
    "new", "in_progress", "matched", "resolved",
    "escalated", "awaiting_input", "stopped", "deleted",
]


class Reporter(BaseModel):
    id: str
    team: str


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_rationale: str | None = None
    signal: str | None = None


class Incident(BaseModel):
    id: str
    status: IncidentStatus
    created_at: str
    updated_at: str
    query: str
    environment: str
    reporter: Reporter
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    severity: str | None = None
    category: str | None = None
    matched_prior_inc: str | None = None
    embedding: list[float] | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Findings is an open mapping keyed by agent name (or any agent-declared
    # output key). Old saves with {"triage": ..., "deep_investigator": ...}
    # load transparently because Pydantic accepts those as dict entries.
    findings: dict[str, Any] = Field(default_factory=dict)
    resolution: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
    deleted_at: str | None = None
