"""Incident domain model."""
from __future__ import annotations
from typing import Any, Literal
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


class Findings(BaseModel):
    triage: Any = None
    deep_investigator: Any = None


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
    resolution: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


class IncidentStore:
    """JSON-file-backed incident store. One file per INC."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        today = _utc_today()
        prefix = f"INC-{today}-"
        existing = [p.stem for p in self.base_dir.glob(f"{prefix}*.json")]
        max_seq = 0
        for stem in existing:
            try:
                max_seq = max(max_seq, int(stem.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"{prefix}{max_seq + 1:03d}"

    def create(self, *, query: str, environment: str,
               reporter_id: str, reporter_team: str) -> Incident:
        inc_id = self._next_id()
        now = _utc_now_iso()
        inc = Incident(
            id=inc_id,
            status="new",
            created_at=now,
            updated_at=now,
            query=query,
            environment=environment,
            reporter=Reporter(id=reporter_id, team=reporter_team),
        )
        self.save(inc)
        return inc

    def save(self, incident: Incident) -> None:
        incident.updated_at = _utc_now_iso()
        path = self.base_dir / f"{incident.id}.json"
        path.write_text(incident.model_dump_json(indent=2))

    def load(self, incident_id: str) -> Incident:
        path = self.base_dir / f"{incident_id}.json"
        if not path.exists():
            raise FileNotFoundError(incident_id)
        return Incident.model_validate_json(path.read_text())

    def list_all(self) -> list[Incident]:
        return [self.load(p.stem) for p in self.base_dir.glob("INC-*.json")]

    def list_recent(self, limit: int = 20) -> list[Incident]:
        all_inc = self.list_all()
        all_inc.sort(key=lambda i: (i.created_at, i.id), reverse=True)
        return all_inc[:limit]
