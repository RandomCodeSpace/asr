"""Coverage tests for ``OrchestratorService.start_session._run`` exception
branches (service.py:541-568).

The inner ``_run`` task supervises a single graph turn. Three exception
classes get distinct treatment:

  1. ``asyncio.CancelledError`` — propagated as-is.
  2. ``GraphInterrupt`` — propagated WITHOUT marking
     ``registry.status='error'`` (HITL pause is not a failure).
  3. Anything else — registry entry is stamped ``status='error'`` so a
     concurrent snapshot observes the failure before the done-callback
     evicts the entry.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.errors import GraphInterrupt

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MetadataConfig,
    Paths,
    StorageConfig,
)
from runtime.service import OrchestratorService


@pytest.fixture
def cfg(tmp_path):
    """AppConfig wired to in-process MCP servers so the example skills
    (which reference ``get_logs`` etc.) pass validation."""
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
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db")
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
    )


@pytest.fixture
def service(cfg):
    """Started OrchestratorService; teardown calls shutdown()."""
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


def _trap_graph_with(service: OrchestratorService, exc_factory):
    """Replace ``orch.graph.ainvoke`` with a coroutine that captures the
    in-flight registry entry then raises the supplied exception.

    Returns the captured-entries list (populated by the time the task
    has run); the caller asserts on ``captured[0].status``.
    """
    captured: list = []

    async def _setup_trap():
        orch = await service._ensure_orchestrator()

        async def _trapped(state, *, config):
            sid = state["session"].id
            entry = service._registry.get(sid)
            captured.append(entry)
            raise exc_factory()

        orch.graph.ainvoke = _trapped
        return None

    service.submit_and_wait(_setup_trap(), timeout=10.0)
    return captured


class TestGraphInterruptBranch:
    """GraphInterrupt must NOT flip registry.status to 'error'."""

    def test_pause_keeps_status_running(self, service):
        captured = _trap_graph_with(
            service,
            lambda: GraphInterrupt(),
        )
        sid = service.start_session(query="probe", state_overrides={"environment": "dev"})
        # Wait for the background task to finish (it will raise
        # GraphInterrupt, which is caught by the Task and surfaces on
        # ``await``).
        async def _await_done():
            entry = service._registry.get(sid) or (captured[0] if captured else None)
            if entry is not None and entry.task is not None:
                with __import__("contextlib").suppress(BaseException):
                    await entry.task
        service.submit_and_wait(_await_done(), timeout=10.0)

        assert captured, "trapped ainvoke never ran"
        # Phase 11 / D-11-04: pause must NOT mark the entry as failed.
        assert captured[0].status == "running"


class TestGenericExceptionBranch:
    """Non-pause exceptions must flip registry.status to 'error'."""

    def test_generic_failure_marks_status_error(self, service):
        captured = _trap_graph_with(
            service,
            lambda: ValueError("boom"),
        )
        sid = service.start_session(query="probe", state_overrides={"environment": "dev"})

        async def _await_done():
            entry = service._registry.get(sid) or (captured[0] if captured else None)
            if entry is not None and entry.task is not None:
                with __import__("contextlib").suppress(BaseException):
                    await entry.task
        service.submit_and_wait(_await_done(), timeout=10.0)

        assert captured, "trapped ainvoke never ran"
        # Generic exception → status flipped before the re-raise.
        assert captured[0].status == "error"


class TestCancelledErrorBranch:
    """CancelledError must propagate without modifying registry.status."""

    def test_cancellation_does_not_mark_error(self, service):
        captured = _trap_graph_with(
            service,
            lambda: asyncio.CancelledError(),
        )
        sid = service.start_session(query="probe", state_overrides={"environment": "dev"})

        async def _await_done():
            entry = service._registry.get(sid) or (captured[0] if captured else None)
            if entry is not None and entry.task is not None:
                with __import__("contextlib").suppress(BaseException):
                    await entry.task
        service.submit_and_wait(_await_done(), timeout=10.0)

        assert captured, "trapped ainvoke never ran"
        # CancelledError takes the early-return branch (line 541-542) —
        # the generic-exception status flip never runs.
        assert captured[0].status == "running"
