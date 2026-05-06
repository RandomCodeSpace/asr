"""Tests for the pending-approval timeout watchdog.

The watchdog scans active sessions for ``pending_approval`` ToolCall
rows older than the configured ``approval_timeout_seconds`` and
resumes them with ``Command(resume={"decision": "timeout", ...})``.
On resume, the wrap_tool path updates the audit row to
``status="timeout"``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime.config import (
    AppConfig,
    GatewayConfig,
    LLMConfig,
    MCPConfig,
    MetadataConfig,
    Paths,
    RuntimeConfig,
    StorageConfig,
)
from runtime.locks import SessionLockRegistry
from runtime.service import OrchestratorService
from runtime.state import ToolCall
from runtime.tools.approval_watchdog import ApprovalWatchdog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ts(seconds_ago: int) -> str:
    """Build an ISO-8601 timestamp ``seconds_ago`` in the past."""
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stub_session(*, sid: str, status: str, tool_calls: list[ToolCall]):
    """Build a minimal stub matching the attributes the watchdog reads."""
    inc = MagicMock()
    inc.id = sid
    inc.status = status
    inc.tool_calls = tool_calls
    return inc


def _build_watchdog(*, timeout_seconds: int = 3600,
                    sessions: dict | None = None,
                    registry_ids: list[str] | None = None):
    """Construct an ApprovalWatchdog hooked to a stub service + orchestrator.

    ``sessions`` maps session_id -> stub session returned by
    ``orch.store.load(sid)``. ``registry_ids`` is the in-flight
    registry that the watchdog snapshots.
    """
    sessions = sessions or {}
    registry_ids = registry_ids or list(sessions.keys())

    service = MagicMock()
    service._registry = {sid: MagicMock(session_id=sid) for sid in registry_ids}

    orch = MagicMock()
    orch.store.load = lambda sid: sessions[sid]
    orch._thread_config = lambda sid: {"configurable": {"thread_id": sid}}
    orch.graph.ainvoke = AsyncMock(return_value={})
    orch._locks = SessionLockRegistry()  # real registry so is_locked() works correctly
    service._orch = orch

    wd = ApprovalWatchdog(
        service, approval_timeout_seconds=timeout_seconds,
    )
    return wd, service, orch


# ---------------------------------------------------------------------------
# Tests — scanning and resume behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_approval_younger_than_timeout_is_left_alone():
    """A 5-minute-old pending approval with a 1h timeout must NOT be resumed."""
    inc = _stub_session(
        sid="INC-1",
        status="awaiting_input",
        tool_calls=[
            ToolCall(
                agent="resolution",
                tool="apply_fix",
                args={"target": "payments-svc"},
                result=None,
                ts=_ts(seconds_ago=5 * 60),  # 5 minutes old
                risk="high",
                status="pending_approval",
            ),
        ],
    )
    wd, _service, orch = _build_watchdog(
        timeout_seconds=3600, sessions={"INC-1": inc},
    )

    resumed = await wd.run_once()

    assert resumed == 0
    orch.graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_pending_approval_older_than_timeout_resumes_with_timeout_decision():
    """A 2h-old pending approval with a 1h timeout MUST resume with
    ``decision='timeout'`` against the same thread_id."""
    from langgraph.types import Command

    inc = _stub_session(
        sid="INC-2",
        status="awaiting_input",
        tool_calls=[
            ToolCall(
                agent="resolution",
                tool="apply_fix",
                args={"target": "billing-svc"},
                result=None,
                ts=_ts(seconds_ago=2 * 3600),  # 2 hours old
                risk="high",
                status="pending_approval",
            ),
        ],
    )
    wd, _service, orch = _build_watchdog(
        timeout_seconds=3600, sessions={"INC-2": inc},
    )

    resumed = await wd.run_once()

    assert resumed == 1
    orch.graph.ainvoke.assert_called_once()
    call = orch.graph.ainvoke.call_args
    arg = call.args[0]
    assert isinstance(arg, Command)
    assert arg.resume == {
        "decision": "timeout",
        "approver": "system",
        "rationale": "approval window expired",
    }
    config = call.kwargs.get("config") or call.args[1]
    assert config["configurable"]["thread_id"] == "INC-2"


@pytest.mark.asyncio
async def test_watchdog_skips_sessions_in_terminal_status():
    """A session whose row has already moved to a terminal status
    (``resolved``, ``stopped``, ``escalated``, ``duplicate``,
    ``deleted``, ``error``) must NOT be resumed even when its
    ``tool_calls`` still carries a stale ``pending_approval`` row."""
    inc = _stub_session(
        sid="INC-3",
        status="resolved",
        tool_calls=[
            ToolCall(
                agent="resolution",
                tool="apply_fix",
                args={},
                result=None,
                ts=_ts(seconds_ago=24 * 3600),  # very old
                risk="high",
                status="pending_approval",
            ),
        ],
    )
    wd, _service, orch = _build_watchdog(
        timeout_seconds=3600, sessions={"INC-3": inc},
    )

    resumed = await wd.run_once()

    assert resumed == 0
    orch.graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_run_once_no_orchestrator_returns_zero():
    """Before any session has run, the orchestrator is None — the
    watchdog must safely return 0 rather than crashing."""
    service = MagicMock()
    service._registry = {}
    service._orch = None
    wd = ApprovalWatchdog(service, approval_timeout_seconds=3600)
    resumed = await wd.run_once()
    assert resumed == 0


@pytest.mark.asyncio
async def test_watchdog_skips_sessions_not_in_awaiting_input():
    """A session in ``in_progress`` status (graph still running) must
    not be resumed — only ``awaiting_input`` is a watchdog candidate."""
    inc = _stub_session(
        sid="INC-4",
        status="in_progress",
        tool_calls=[
            ToolCall(
                agent="resolution",
                tool="apply_fix",
                args={},
                result=None,
                ts=_ts(seconds_ago=2 * 3600),
                risk="high",
                status="pending_approval",
            ),
        ],
    )
    wd, _service, orch = _build_watchdog(
        timeout_seconds=3600, sessions={"INC-4": inc},
    )

    resumed = await wd.run_once()
    assert resumed == 0
    orch.graph.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — service lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_with_gateway(tmp_path):
    """AppConfig with a configured gateway so the watchdog auto-starts."""
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(
            gateway=GatewayConfig(approval_timeout_seconds=3600),
        ),
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    yield
    OrchestratorService._reset_singleton()


def test_watchdog_lifecycle_starts_and_stops_with_service(cfg_with_gateway):
    """Starting the service arms the watchdog; shutdown stops it."""
    svc = OrchestratorService.get_or_create(cfg_with_gateway)
    svc.start()
    try:
        wd = svc._approval_watchdog
        assert wd is not None
        assert wd.is_running
    finally:
        svc.shutdown()
    # Post-shutdown, the watchdog reference is cleared.
    assert svc._approval_watchdog is None


def test_gateway_config_default_approval_timeout():
    """``GatewayConfig.approval_timeout_seconds`` defaults to 1 hour."""
    cfg = GatewayConfig()
    assert cfg.approval_timeout_seconds == 3600


def test_watchdog_not_started_when_gateway_unconfigured(tmp_path):
    """Without a gateway configured, the watchdog stays None — apps
    that haven't opted into HITL pay no startup cost."""
    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
    )
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    try:
        assert svc._approval_watchdog is None
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Tests — HARD-06 cancellation hygiene
# ---------------------------------------------------------------------------


async def test_stop_drains_cancelled_task_no_pending_at_teardown():
    """HARD-06: ApprovalWatchdog.stop() must await the cancelled task.

    After stop() returns, asyncio.all_tasks() should not contain the
    watchdog task. Without the drain (await task) added in this fix,
    ``Task was destroyed but it is pending`` warnings escape to
    Python's warnings stream at event-loop teardown.
    """
    wd, _service, _orch = _build_watchdog(timeout_seconds=3600)
    # We are already inside an asyncio event loop (asyncio_mode = "auto"),
    # so arm the watchdog directly rather than via run_coroutine_threadsafe.
    wd._stop_event = asyncio.Event()
    wd._task = asyncio.create_task(wd._run(), name="approval_watchdog")
    # Yield to let the polling loop's first iteration start before we stop.
    await asyncio.sleep(0)
    await wd.stop()
    # After stop(), no task referencing the watchdog should remain.
    pending = [
        t for t in asyncio.all_tasks()
        if "approval_watchdog" in (t.get_name() or "")
    ]
    assert pending == [], f"watchdog leaked tasks: {pending!r}"
