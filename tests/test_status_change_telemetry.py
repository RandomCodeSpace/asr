"""M4 — status_changed event emission on finalize.

Drives the finalize path with a seeded ``mark_resolved`` tool call and
asserts the EventLog captures exactly one ``status_changed`` event
with ``to=resolved`` and ``cause`` referencing the terminal tool name.
"""
from __future__ import annotations

import pytest
from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    OrchestratorConfig,
    Paths,
    RuntimeConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.state import ToolCall
from runtime.terminal_tools import StatusDef, TerminalToolRule


_INCIDENT_STATUSES = {
    "new":          StatusDef(name="new",          terminal=False, kind="pending"),
    "in_progress":  StatusDef(name="in_progress",  terminal=False, kind="pending"),
    "open":         StatusDef(name="open",         terminal=False, kind="pending"),
    "escalated":    StatusDef(name="escalated",    terminal=True,  kind="escalation"),
    "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
    "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
}
_INCIDENT_RULES = [
    TerminalToolRule(tool_name="mark_resolved", status="resolved"),
]


def _cfg_with_terminal_rules(tmp_path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="examples.incident_management.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="examples.incident_management.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="examples.incident_management.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        orchestrator=OrchestratorConfig(
            statuses=_INCIDENT_STATUSES,
            terminal_tools=_INCIDENT_RULES,
            default_terminal_status="needs_review",
        ),
        runtime=RuntimeConfig(state_class=None),
    )


@pytest.mark.asyncio
async def test_finalize_with_mark_resolved_emits_status_changed(tmp_path):
    """``mark_resolved`` executed -> status_changed(to=resolved,
    cause=mark_resolved) and exactly one such event in the log."""
    cfg = _cfg_with_terminal_rules(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        # Seed a session whose tool_calls contain an executed mark_resolved.
        inc = orch.store.create(
            query="payments-svc 500s",
            environment="staging",
            reporter_id="u",
            reporter_team="t",
        )
        inc.tool_calls.append(ToolCall(
            agent="resolution",
            tool="mark_resolved",
            args={},
            result={"status": "resolved"},
            ts="2026-05-12T00:00:00Z",
            status="executed",
        ))
        inc.status = "in_progress"
        orch.store.save(inc)

        new_status = orch._finalize_session_status(inc.id)
        assert new_status == "resolved"

        assert orch.event_log is not None
        events = list(orch.event_log.iter_for(inc.id))
        status_events = [e for e in events if e.kind == "status_changed"]
        assert len(status_events) == 1, [e.payload for e in status_events]
        e = status_events[0]
        # ``from`` is a reserved keyword so it's stored in payload as-is.
        assert e.payload["from"] == "in_progress"
        assert e.payload["to"] == "resolved"
        assert e.payload["cause"] == "mark_resolved"
        clear_events = [
            e for e in events
            if e.kind == "session.agent_running"
        ]
        assert clear_events[-1].payload == {"id": inc.id, "agent": None}
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_finalize_default_terminal_status_emits_cause_default(tmp_path):
    """No terminal-tool rule fired -> status_changed(to=needs_review,
    cause=default_terminal_status)."""
    cfg = _cfg_with_terminal_rules(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        inc = orch.store.create(
            query="latency", environment="dev",
            reporter_id="u", reporter_team="t",
        )
        inc.status = "in_progress"
        orch.store.save(inc)

        new_status = orch._finalize_session_status(inc.id)
        assert new_status == "needs_review"

        assert orch.event_log is not None
        events = list(orch.event_log.iter_for(inc.id))
        status_events = [e for e in events if e.kind == "status_changed"]
        assert len(status_events) == 1
        e = status_events[0]
        assert e.payload["from"] == "in_progress"
        assert e.payload["to"] == "needs_review"
        assert e.payload["cause"] == "default_terminal_status"
    finally:
        await orch.aclose()
