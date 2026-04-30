"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
from typing import TypedDict, Callable, Awaitable
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from orchestrator.incident import Incident, ToolCall, AgentRun, IncidentStore
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


def _format_intake_input(incident: Incident) -> str:
    return (
        f"Incident {incident.id}\n"
        f"Environment: {incident.environment}\n"
        f"Query: {incident.query}\n"
        f"Status: {incident.status}\n"
        f"Findings (triage): {incident.findings.triage}\n"
        f"Findings (deep_investigator): {incident.findings.deep_investigator}\n"
    )


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Incident], str],
    store: IncidentStore,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route."""
    agent_executor = create_react_agent(llm, tools, prompt=skill.system_prompt)

    async def node(state: GraphState) -> dict:
        incident = state["incident"]
        recorder = AgentRunRecorder(agent=skill.name, incident=incident)
        recorder.start()

        try:
            result = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=_format_intake_input(incident))]}
            )
        except Exception as exc:  # noqa: BLE001
            recorder.finish(summary=f"agent failed: {exc}")
            store.save(incident)
            return {"incident": incident, "next_route": None,
                    "last_agent": skill.name, "error": str(exc)}

        # Walk the ReAct trace; capture each tool call into the incident.
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                recorder.record_tool_call(
                    tool=tc.get("name", "unknown"),
                    args=tc.get("args", {}) or {},
                    result=None,
                )

        # Capture tool responses as separate entries
        for msg in result.get("messages", []):
            if msg.__class__.__name__ == "ToolMessage":
                for entry in reversed(incident.tool_calls):
                    if entry.tool == getattr(msg, "name", None) and entry.result is None:
                        entry.result = getattr(msg, "content", None)
                        break

        # Use the final AI message's text as the summary
        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and msg.content:
                final_text = str(msg.content)[:500]
                break

        recorder.finish(summary=final_text or f"{skill.name} completed")
        next_route_signal = decide_route(incident)
        store.save(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"incident": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node
