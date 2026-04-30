"""Incident domain model."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


IncidentStatus = Literal["new", "in_progress", "matched", "resolved", "escalated"]


class Reporter(BaseModel):
    id: str
    team: str


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str


class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str


class Findings(BaseModel):
    triage: dict | None = None
    deep_investigator: dict | None = None


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
    findings: Findings = Field(default_factory=Findings)
    resolution: dict | None = None
