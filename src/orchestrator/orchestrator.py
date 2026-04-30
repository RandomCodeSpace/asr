"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""
from __future__ import annotations
from contextlib import AsyncExitStack
from typing import AsyncIterator
from datetime import datetime, timezone

from orchestrator.config import AppConfig
from orchestrator.incident import IncidentStore
from orchestrator.skill import load_all_skills, Skill
from orchestrator.mcp_loader import load_tools, ToolRegistry
from orchestrator.mcp_servers.incident import set_state as _set_inc_state
from orchestrator.graph import build_graph, GraphState


class Orchestrator:
    """High-level facade. Construct via ``await Orchestrator.create(cfg)``.

    The Orchestrator owns the lifecycle of the FastMCP clients underpinning the
    tool registry. Always call :meth:`aclose` (or use ``async with``) when done.
    """

    def __init__(self, cfg: AppConfig, store: IncidentStore,
                 skills: dict[str, Skill], registry: ToolRegistry, graph,
                 exit_stack: AsyncExitStack):
        self.cfg = cfg
        self.store = store
        self.skills = skills
        self.registry = registry
        self.graph = graph
        self._exit_stack = exit_stack

    @classmethod
    async def create(cls, cfg: AppConfig) -> "Orchestrator":
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            store = IncidentStore(cfg.paths.incidents_dir)
            _set_inc_state(store=store, similarity_threshold=cfg.incidents.similarity_threshold)
            skills = load_all_skills(cfg.paths.skills_dir)
            registry = await load_tools(cfg.mcp, stack)
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry)
            return cls(cfg, store, skills, registry, graph, stack)
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
                "model": s.model or self.cfg.llm.default_model,
                "tools": list(s.tools),
                "routes": [r.model_dump() for r in s.routes],
            }
            for s in self.skills.values()
        ]

    def list_tools(self) -> list[dict]:
        bindings: dict[str, list[str]] = {}
        for skill in self.skills.values():
            for tool_name in skill.tools:
                bindings.setdefault(tool_name, []).append(skill.name)
        return [
            {
                "name": e.name,
                "description": e.description,
                "category": e.category,
                "server": e.server,
                "bound_agents": bindings.get(e.name, []),
            }
            for e in self.registry.entries.values()
        ]

    def get_incident(self, incident_id: str) -> dict:
        return self.store.load(incident_id).model_dump()

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        return [i.model_dump() for i in self.store.list_recent(limit)]

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
