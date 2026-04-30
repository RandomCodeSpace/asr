"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
from typing import TypedDict
from datetime import datetime, timezone

from orchestrator.incident import Incident, ToolCall, AgentRun
from orchestrator.skill import Skill


class GraphState(TypedDict, total=False):
    incident: Incident
    next_route: str | None
    last_agent: str | None
    error: str | None


def route_from_skill(skill: Skill, signal: str) -> str:
    if not skill.routes:
        raise ValueError(f"Skill '{skill.name}' has no routes defined")
    for rule in skill.routes:
        if rule.when == signal:
            return rule.next
    for rule in skill.routes:
        if rule.when == "default":
            return rule.next
    return skill.routes[0].next


class AgentRunRecorder:
    """Helper to capture an agent's run + tool calls into the incident."""

    def __init__(self, *, agent: str, incident: Incident):
        self.agent = agent
        self.incident = incident
        self._started_at: str | None = None

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record_tool_call(self, tool: str, args: dict, result) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.incident.tool_calls.append(
            ToolCall(agent=self.agent, tool=tool, args=args, result=result, ts=ts)
        )

    def finish(self, *, summary: str) -> None:
        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.incident.agents_run.append(AgentRun(
            agent=self.agent,
            started_at=self._started_at or ended_at,
            ended_at=ended_at,
            summary=summary,
        ))
