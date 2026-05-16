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


class _PausedAtGateGraph:
    """Fake graph that simulates a HITL gate pause: writes a
    ``pending_approval`` ToolCall, returns from ``ainvoke``, then
    reports a non-empty ``next`` tuple to ``aget_state`` so the
    orchestrator's ``_is_graph_paused`` returns True. Mirrors what
    langgraph 1.x does when an ``interrupt()`` boundary is hit."""

    def __init__(self, store, *, service=None) -> None:
        self.store = store
        self.service = service
        self.captured_entries = []

    async def ainvoke(self, state, *, config):
        inc = state["session"]
        if self.service is not None:
            self.captured_entries.append(self.service._registry.get(inc.id))
        # Mid-run state: an in-flight tool call that the gateway has
        # parked at a HITL gate. Status stays at the pre-pause value
        # ('new' / 'in_progress') until the orchestrator writes
        # 'awaiting_input' itself.
        inc.tool_calls.append(ToolCall(
            agent="resolution",
            tool="apply_fix",
            args={"target": "payments-svc"},
            result=None,
            ts="2026-01-01T00:00:00Z",
            status="pending_approval",
        ))
        self.store.save(inc)
        return {}

    async def aget_state(self, config):
        # A non-empty ``next`` tuple is langgraph's way of saying
        # "the graph has steps queued to run when resumed" — i.e.
        # paused at an ``interrupt()``.
        return SimpleNamespace(next=("resume_node",))


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


# ---------------------------------------------------------------------------
# Issue #42: paused-at-gate sessions must transition to 'awaiting_input'.
# Sibling of the C#1 finalizer-asymmetry fix above — the finalizer skips
# paused graphs (correct: a HITL pause must not be coerced into a terminal
# status), but the row needs a paused-side status write so UIs filtering by
# 'awaiting_input' (approvals queue, sessions rail Active group) pick it up.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_start_session_marks_paused_run_awaiting_input(tmp_path):
    """Direct (non-streaming) start_session: when the graph pauses at a
    HITL gate, the session row should flip to 'awaiting_input' AND a
    status_changed event should be emitted."""
    orch = await Orchestrator.create(_cfg(tmp_path))
    try:
        orch.graph = _PausedAtGateGraph(orch.store)
        sid = await orch.start_session(query="db pool exhausted")
        loaded = orch.store.load(sid)
        assert loaded.status == "awaiting_input"
        # The pending_approval ToolCall written by the fake gateway
        # must still be on the row — pause-write should not clobber it.
        assert any(
            tc.status == "pending_approval" for tc in loaded.tool_calls
        ), loaded.tool_calls
        # status_changed (per-session) + session.status_changed
        # (cross-session SSE) events must be on the event log so the
        # React UI's recent-events stream surfaces the transition.
        kinds = [ev.kind for ev in orch.event_log.iter_for(sid)]
        assert "status_changed" in kinds
        assert "session.status_changed" in kinds
    finally:
        await orch.aclose()


def test_service_background_start_session_marks_paused_run_awaiting_input(tmp_path):
    """OrchestratorService background _run() path: same guarantee as
    the direct path — the inner finalizer block now mirrors
    orchestrator.start_session's pause handling."""
    service = OrchestratorService.get_or_create(_cfg(tmp_path))
    service.start()
    try:
        async def _install_graph():
            orch = await service._ensure_orchestrator()
            graph = _PausedAtGateGraph(orch.store, service=service)
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
        loaded = orch.store.load(sid)
        assert loaded.status == "awaiting_input"
        kinds = [ev.kind for ev in orch.event_log.iter_for(sid)]
        assert "session.status_changed" in kinds
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_mark_session_paused_is_no_op_when_already_awaiting_input(tmp_path):
    """Guard: calling _mark_session_paused_async on a session that's
    already awaiting_input must NOT emit a spurious status_changed
    event (it would re-trigger the React UI's row update with an
    identical to-status)."""
    orch = await Orchestrator.create(_cfg(tmp_path))
    try:
        inc = orch.store.create(query="x", environment="dev")
        inc.status = "awaiting_input"
        orch.store.save(inc)
        before = list(orch.event_log.iter_for(inc.id))
        result = await orch._mark_session_paused_async(inc.id)
        after = list(orch.event_log.iter_for(inc.id))
        assert result is None
        assert len(after) == len(before)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_mark_session_paused_is_no_op_on_terminal_status(tmp_path):
    """Guard: a late paused-write must not unwind a finalize that
    landed in between (e.g. the gateway raised after the audit row
    landed but before pausing, and the finalizer ran first)."""
    orch = await Orchestrator.create(_cfg(tmp_path))
    try:
        inc = orch.store.create(query="x", environment="dev")
        inc.status = "resolved"  # terminal per _cfg
        orch.store.save(inc)
        result = await orch._mark_session_paused_async(inc.id)
        assert result is None
        assert orch.store.load(inc.id).status == "resolved"
    finally:
        await orch.aclose()
