"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
from typing import TypedDict, Callable, Awaitable
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from orchestrator.incident import Incident, ToolCall, AgentRun, IncidentStore
from orchestrator.skill import Skill
from orchestrator.config import AppConfig
from orchestrator.llm import get_llm
from orchestrator.mcp_loader import ToolRegistry


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
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            result = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=_format_intake_input(incident))]}
            )
        except Exception as exc:  # noqa: BLE001
            # Reload to absorb any partial writes from tools that ran before the failure.
            try:
                incident = store.load(inc_id)
            except FileNotFoundError:
                pass
            ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            incident.agents_run.append(AgentRun(
                agent=skill.name, started_at=started_at, ended_at=ended_at,
                summary=f"agent failed: {exc}",
            ))
            store.save(incident)
            return {"incident": incident, "next_route": None,
                    "last_agent": skill.name, "error": str(exc)}

        # Tools (e.g. update_incident) write straight to disk. Reload so the
        # node's own append of agent_run + tool_calls happens against the
        # tool-mutated state — otherwise saving the stale in-memory object
        # clobbers the tools' writes.
        incident = store.load(inc_id)

        # Record tool calls from the agent's message trace.
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                incident.tool_calls.append(ToolCall(
                    agent=skill.name,
                    tool=tc.get("name", "unknown"),
                    args=tc.get("args", {}) or {},
                    result=None,
                    ts=ts,
                ))

        # Pair tool responses with their tool calls.
        for msg in result.get("messages", []):
            if msg.__class__.__name__ == "ToolMessage":
                for entry in reversed(incident.tool_calls):
                    if entry.tool == getattr(msg, "name", None) and entry.result is None:
                        entry.result = getattr(msg, "content", None)
                        break

        # Final summary text from the agent's last AIMessage.
        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and msg.content:
                final_text = str(msg.content)[:500]
                break

        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        incident.agents_run.append(AgentRun(
            agent=skill.name, started_at=started_at, ended_at=ended_at,
            summary=final_text or f"{skill.name} completed",
        ))

        next_route_signal = decide_route(incident)
        store.save(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"incident": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


# Per-agent route decision functions.
def _decide_intake(inc: Incident) -> str:
    return "matched_known_issue" if inc.matched_prior_inc else "default"


def _decide_triage(inc: Incident) -> str:
    return "default"


def _decide_deep_investigator(inc: Incident) -> str:
    return "default"


def _decide_resolution(inc: Incident) -> str:
    return "default"


_DECIDERS: dict[str, Callable[[Incident], str]] = {
    "intake": _decide_intake,
    "triage": _decide_triage,
    "deep_investigator": _decide_deep_investigator,
    "resolution": _decide_resolution,
}


_STUB_CANNED = {
    "intake": "Created INC, no prior matches. Routing to triage.",
    "triage": "Severity sev3, category latency. No recent deploys correlate.",
    "deep_investigator": "Hypothesis: upstream payments timeout. Evidence: log line 'upstream_timeout target=payments'.",
    "resolution": "Proposed fix: restart api service. Auto-applied. INC resolved.",
}


async def build_graph(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                      registry: ToolRegistry):
    """Compile the LangGraph StateGraph from skills + tool registry.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph. This avoids double-loading (which would also create a
    second set of clients, immediately closed).
    """
    sg = StateGraph(GraphState)
    for agent_name, skill in skills.items():
        llm = get_llm(
            cfg.llm,
            role=agent_name,
            model=skill.model,
            temperature=skill.temperature,
            stub_canned=_STUB_CANNED,
        )
        tools = registry.get(skill.tools)
        decide = _DECIDERS.get(agent_name, lambda inc: "default")
        node = make_agent_node(skill=skill, llm=llm, tools=tools,
                               decide_route=decide, store=store)
        sg.add_node(agent_name, node)

    # Set entry point to the agent named 'intake'.
    sg.set_entry_point("intake")

    # Conditional edges: each agent's `next_route` (a node name OR "__end__") drives routing.
    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        return nr

    for agent_name in skills.keys():
        possible_targets = {s.name for s in skills.values()} | {END}
        target_map = {name: name for name in possible_targets if name != END}
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)

    return sg.compile()
