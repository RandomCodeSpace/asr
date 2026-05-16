from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MetadataConfig,
    OrchestratorConfig,
    Paths,
    StorageConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.service import OrchestratorService
from runtime.state import ToolCall
from runtime.terminal_tools import StatusDef, TerminalToolRule


def _cfg(tmp_path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(
                name="local_inc",
                transport="in_process",
                module="examples.incident_management.mcp_server",
                category="incident_management",
            ),
            MCPServerConfig(
                name="local_obs",
                transport="in_process",
                module="examples.incident_management.mcp_servers.observability",
                category="observability",
            ),
            MCPServerConfig(
                name="local_rem",
                transport="in_process",
                module="examples.incident_management.mcp_servers.remediation",
                category="remediation",
            ),
            MCPServerConfig(
                name="local_user",
                transport="in_process",
                module="examples.incident_management.mcp_servers.user_context",
                category="user_context",
            ),
        ]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        orchestrator=OrchestratorConfig(
            statuses={
                "in_progress": StatusDef(
                    name="in_progress", terminal=False, kind="pending",
                ),
                "resolved": StatusDef(
                    name="resolved", terminal=True, kind="success",
                ),
                "needs_review": StatusDef(
                    name="needs_review", terminal=True, kind="needs_review",
                ),
            },
            terminal_tools=[
                TerminalToolRule(tool_name="mark_resolved", status="resolved"),
            ],
            default_terminal_status="needs_review",
        ),
    )


class _CompletingGraph:
    def __init__(self, store, *, service=None) -> None:
        self.store = store
        self.service = service
        self.captured_entries = []

    async def ainvoke(self, state, *, config):
        inc = state["session"]
        if self.service is not None:
            self.captured_entries.append(self.service._registry.get(inc.id))
        inc.status = "in_progress"
        inc.tool_calls.append(ToolCall(
            agent="resolution",
            tool="mark_resolved",
            args={},
            result={"status": "resolved"},
            ts="2026-01-01T00:00:00Z",
            status="executed",
        ))
        self.store.save(inc)
        return {}

    async def aget_state(self, config):
        return SimpleNamespace(next=())


@pytest.mark.asyncio
async def test_orchestrator_start_session_finalizes_completed_non_streaming_run(
    tmp_path,
):
    orch = await Orchestrator.create(_cfg(tmp_path))
    try:
        orch.graph = _CompletingGraph(orch.store)
        sid = await orch.start_session(query="db pool exhausted")
        assert orch.store.load(sid).status == "resolved"
    finally:
        await orch.aclose()


def test_service_background_start_session_finalizes_completed_run(tmp_path):
    service = OrchestratorService.get_or_create(_cfg(tmp_path))
    service.start()
    try:
        async def _install_graph():
            orch = await service._ensure_orchestrator()
            graph = _CompletingGraph(orch.store, service=service)
            orch.graph = graph
            return graph

        graph = service.submit_and_wait(_install_graph(), timeout=10.0)
        sid = service.start_session(query="db pool exhausted")

        async def _await_background_task():
            while not graph.captured_entries:
                await asyncio.sleep(0.01)
            entry = graph.captured_entries[0]
            if entry is not None and entry.task is not None:
                await entry.task

        service.submit_and_wait(_await_background_task(), timeout=10.0)
        orch = service.submit_and_wait(service._ensure_orchestrator(), timeout=10.0)
        assert orch.store.load(sid).status == "resolved"
    finally:
        service.shutdown()
