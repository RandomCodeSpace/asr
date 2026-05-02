"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""
from __future__ import annotations
from contextlib import AsyncExitStack
from typing import AsyncIterator
from datetime import datetime, timezone

from orchestrator.config import AppConfig
from orchestrator.incident import IncidentStore, ToolCall
from orchestrator.skill import load_all_skills, Skill
from orchestrator.mcp_loader import load_tools, ToolRegistry
from orchestrator.mcp_servers.incident import set_state
from orchestrator.graph import build_graph, build_resume_graph, GraphState


class Orchestrator:
    """High-level facade. Construct via ``await Orchestrator.create(cfg)``.

    The Orchestrator owns the lifecycle of the FastMCP clients underpinning the
    tool registry. Always call :meth:`aclose` (or use ``async with``) when done.
    """

    def __init__(self, cfg: AppConfig, store: IncidentStore,
                 skills: dict[str, Skill], registry: ToolRegistry, graph,
                 resume_graph, exit_stack: AsyncExitStack):
        self.cfg = cfg
        self.store = store
        self.skills = skills
        self.registry = registry
        self.graph = graph
        self.resume_graph = resume_graph
        self._exit_stack = exit_stack

    @classmethod
    async def create(cls, cfg: AppConfig) -> "Orchestrator":
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            store = IncidentStore(cfg.paths.incidents_dir)
            set_state(store=store, similarity_threshold=cfg.incidents.similarity_threshold)
            skills = load_all_skills(cfg.paths.skills_dir)
            for s in skills.values():
                if s.model is not None and s.model not in cfg.llm.models:
                    raise ValueError(
                        f"skill {s.name!r} references llm model {s.model!r} "
                        f"which is not defined in llm.models "
                        f"(known: {sorted(cfg.llm.models)})"
                    )
            registry = await load_tools(cfg.mcp, stack)
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry)
            # Build the resume graph only when at least one skill declares
            # a gated route. Without gates an INC can never enter the
            # ``awaiting_input`` state, so the resume path is dead code —
            # and ``build_resume_graph`` raises by design when gated_edges
            # is empty. This unblocks intake-only YAML configurations.
            has_gates = any(
                r.gate for s in skills.values() for r in s.routes
            )
            resume_graph = (
                await build_resume_graph(
                    cfg=cfg, skills=skills, store=store, registry=registry,
                ) if has_gates else None
            )
            return cls(cfg, store, skills, registry, graph, resume_graph, stack)
        except BaseException:
            await stack.aclose()
            raise

    async def aclose(self) -> None:
        """Close all owned MCP clients/transports. Idempotent."""
        await self._exit_stack.aclose()

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                # The named model entry the agent will use (resolved against
                # cfg.llm.default when the skill leaves model unset).
                "model": s.model or self.cfg.llm.default,
                # Expose the flat list of prefixed tool names the LLM sees.
                # resolve() returns list[BaseTool], so .name is on the tool directly.
                "tools": [
                    t.name
                    for t in self.registry.resolve(s.tools, self.cfg.mcp)
                ],
                "routes": [r.model_dump() for r in s.routes],
            }
            for s in self.skills.values()
        ]

    def list_tools(self) -> list[dict]:
        # Build reverse map: prefixed tool name -> list of skill names that bind it.
        # resolve() returns list[BaseTool]; tool.name is the prefixed form.
        bindings: dict[str, list[str]] = {}
        for skill in self.skills.values():
            for t in self.registry.resolve(skill.tools, self.cfg.mcp):
                bindings.setdefault(t.name, []).append(skill.name)
        # Map server name -> transport so callers can tell local from remote.
        transport_by_server = {s.name: s.transport for s in self.cfg.mcp.servers}
        return [
            {
                "name": e.tool.name,          # prefixed: "<server>:<original>"
                "original_name": e.name,      # original tool name as exposed by server
                "description": e.description,
                "category": e.category,
                "server": e.server,
                "transport": transport_by_server.get(e.server, "unknown"),
                "bound_agents": bindings.get(e.tool.name, []),
            }
            for e in self.registry.entries.values()
        ]

    def get_incident(self, incident_id: str) -> dict:
        return self.store.load(incident_id).model_dump()

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        return [i.model_dump() for i in self.store.list_recent(limit)]

    def delete_incident(self, incident_id: str) -> dict:
        return self.store.delete(incident_id).model_dump()

    async def start_investigation(self, *, query: str, environment: str,
                                  reporter_id: str = "user-mock",
                                  reporter_team: str = "platform") -> str:
        inc = self.store.create(query=query, environment=environment,
                                reporter_id=reporter_id, reporter_team=reporter_team)
        await self.graph.ainvoke(GraphState(incident=inc, next_route=None,
                                            last_agent=None, error=None))
        return inc.id

    async def stream_investigation(self, *, query: str, environment: str,
                                   reporter_id: str = "user-mock",
                                   reporter_team: str = "platform"
                                   ) -> AsyncIterator[dict]:
        inc = self.store.create(query=query, environment=environment,
                                reporter_id=reporter_id, reporter_team=reporter_team)
        yield {"event": "investigation_started", "incident_id": inc.id,
               "ts": _now()}
        async for ev in self.graph.astream_events(
            GraphState(incident=inc, next_route=None, last_agent=None, error=None),
            version="v2",
        ):
            yield self._to_ui_event(ev, inc.id)
        yield {"event": "investigation_completed", "incident_id": inc.id, "ts": _now()}

    async def resume_investigation(self, incident_id: str,
                                   decision: dict) -> AsyncIterator[dict]:
        """Resume a paused INC. ``decision`` shapes:

        - ``{"action": "resume_with_input", "input": "<text>"}``
        - ``{"action": "escalate", "team": "<team-name>"}``
        - ``{"action": "stop"}``

        Yields a small set of UI events: a ``resume_started`` event up front,
        the underlying graph events for ``resume_with_input``, then a
        ``resume_completed`` event with ``status`` set to the final state.
        """
        action = decision.get("action")
        yield {"event": "resume_started", "incident_id": incident_id,
               "action": action, "ts": _now()}

        inc = self.store.load(incident_id)

        # Guard: only paused INCs are resumable. A resolved/stopped/escalated
        # INC must not be advanced again — that would silently corrupt state
        # (e.g. re-pinging on-call after the incident has already closed).
        if inc.status != "awaiting_input":
            yield {"event": "resume_rejected", "incident_id": incident_id,
                   "reason": f"not_awaiting_input (status={inc.status})",
                   "ts": _now()}
            return

        if action == "stop":
            inc.status = "stopped"
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "stopped", "ts": _now()}
            return

        if action == "escalate":
            team = decision.get("team") or "platform-oncall"
            allowed = list(self.cfg.intervention.escalation_teams)
            if team not in allowed:
                # Reject the request entirely. The INC stays awaiting_input
                # so the user can retry with a valid team. Logging the
                # allowed roster on the event makes it actionable in the UI.
                yield {"event": "resume_rejected", "incident_id": incident_id,
                       "reason": (
                           f"team '{team}' not in allowed escalation_teams "
                           f"({allowed})"
                       ),
                       "ts": _now()}
                return
            message = (
                f"INC {incident_id} escalated by user — team {team}. "
                "Confidence below threshold."
            )
            tool_args = {"incident_id": incident_id, "message": message}
            tool_result = await self._invoke_tool("notify_oncall", tool_args)
            inc = self.store.load(incident_id)
            inc.tool_calls.append(ToolCall(
                agent="orchestrator",
                tool="notify_oncall",
                args=tool_args,
                result=tool_result,
                ts=_now(),
            ))
            inc.status = "escalated"
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "escalated", "team": team, "ts": _now()}
            return

        if action == "resume_with_input":
            async for ev in self._resume_with_input(incident_id, inc, decision):
                yield ev
            return

        raise ValueError(f"Unknown resume action: {action!r}")

    async def _resume_with_input(self, incident_id: str, inc, decision: dict):
        """Handle the resume_with_input action: append user text, re-run sub-graph,
        restore state on failure. Yields UI events."""
        user_text = (decision.get("input") or "").strip()
        if not user_text:
            raise ValueError("resume_with_input requires a non-empty 'input'")
        # The resume sub-graph only exists when the YAML declares at least
        # one gated route. An intake-only configuration has nothing to
        # resume into — bail with a rejection event rather than crashing.
        if self.resume_graph is None:
            yield {"event": "resume_rejected", "incident_id": incident_id,
                   "reason": "resume_with_input not available: no gated route configured",
                   "ts": _now()}
            return
        # Snapshot the intervention payload BEFORE we mutate the INC, so
        # we can restore it if the sub-graph blows up. Without this an
        # apply_fix exception leaves the INC stuck at in_progress with a
        # cleared pending_intervention — the user can no longer resolve it
        # via the UI.
        saved_pi = inc.pending_intervention
        inc.user_inputs.append(user_text)
        inc.pending_intervention = None
        inc.status = "in_progress"
        self.store.save(inc)
        inc = self.store.load(incident_id)  # reload as canonical state
        try:
            async for ev in self.resume_graph.astream_events(
                GraphState(incident=inc, next_route=None, last_agent=None,
                           error=None),
                version="v2",
            ):
                yield self._to_ui_event(ev, incident_id)
        except Exception as exc:  # noqa: BLE001 — restore on any failure
            # Reload from disk to absorb any partial writes from tools
            # that ran before the failure, then restore intervention
            # state so the UI can reprompt the user.
            try:
                inc = self.store.load(incident_id)
            except FileNotFoundError:
                pass
            inc.pending_intervention = saved_pi
            inc.status = "awaiting_input"
            self.store.save(inc)
            yield {"event": "resume_failed", "incident_id": incident_id,
                   "error": str(exc), "ts": _now()}
            return
        final = self.store.load(incident_id)
        yield {"event": "resume_completed", "incident_id": incident_id,
               "status": final.status, "ts": _now()}

    async def _invoke_tool(self, name: str, args: dict):
        """Call an MCP tool by original name, going through the LangChain wrapper.

        Searches the registry for any entry whose original ``name`` matches.
        Used for orchestrator-driven tool calls (e.g. notify_oncall on
        escalate) that aren't initiated by an LLM.
        """
        entry = next(
            (e for e in self.registry.entries.values() if e.name == name),
            None,
        )
        if entry is None:
            raise KeyError(f"tool '{name}' not registered")
        return await entry.tool.ainvoke(args)

    @staticmethod
    def _to_ui_event(raw: dict, incident_id: str) -> dict:
        kind = raw.get("event", "unknown")
        node = raw.get("name") or raw.get("metadata", {}).get("langgraph_node")
        return {
            "event": kind,
            "node": node,
            "incident_id": incident_id,
            "ts": _now(),
            "data": raw.get("data"),
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
