"""Multi-session integration tests.

Exercises ``OrchestratorService`` end-to-end: parallel session starts via
the public API, registry visibility while running, persisted-state
isolation between concurrent sessions, the resource cap, and thread-safe
read-while-write access to ``list_active_sessions()``.

The stub LLM completes very quickly, so individual sessions may transition
through the registry faster than a single snapshot can observe. Tests
treat "active or already completed" as success for liveness checks; the
non-interference and persistence assertions are what we really care about.
"""
from __future__ import annotations

import threading
import time

import pytest

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MetadataConfig,
    Paths,
    RuntimeConfig,
    StorageConfig,
)
from runtime.service import OrchestratorService, SessionCapExceeded


# ---------------------------------------------------------------------------
# Fixtures — same shape as ``cfg_full`` / ``service_full`` in
# tests/test_orchestrator_service.py. Duplicated here on purpose: the
# integration test file is self-contained and reads top-to-bottom.
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    """AppConfig wired to all four in-process MCP servers from the
    incident-management example. Required for ``Orchestrator.create``
    to load every skill (each ``local:`` ref resolves to the union of
    in-process servers)."""
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(
            servers=[
                MCPServerConfig(
                    name="local_inc",
                    transport="in_process",
                    module="examples.incident_management.mcp_server",
                    category="incident_management",
                ),
                MCPServerConfig(
                    name="local_obs",
                    transport="in_process",
                    module="runtime.mcp_servers.observability",
                    category="observability",
                ),
                MCPServerConfig(
                    name="local_rem",
                    transport="in_process",
                    module="runtime.mcp_servers.remediation",
                    category="remediation",
                ),
                MCPServerConfig(
                    name="local_user",
                    transport="in_process",
                    module="runtime.mcp_servers.user_context",
                    category="user_context",
                ),
            ]
        ),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/multi.db")
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )


@pytest.fixture
def service(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


@pytest.fixture(autouse=True)
def _sweep_singleton():
    """Belt-and-braces: ensure no service instance leaks between tests
    even if a test fails before its own teardown ran."""
    yield
    OrchestratorService._reset_singleton()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_all_complete(service, sids, timeout: float = 30.0) -> bool:
    """Poll ``list_active_sessions`` until none of ``sids`` remain."""
    deadline = time.monotonic() + timeout
    target = set(sids)
    while time.monotonic() < deadline:
        active = {s["session_id"] for s in service.list_active_sessions()}
        if not (active & target):
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Three concurrent sessions
# ---------------------------------------------------------------------------


def test_three_sessions_run_concurrently(service):
    """Three sessions kicked off back-to-back all complete cleanly.

    The stub LLM may finish each run in well under 100 ms, so we don't
    assert a snapshot containing all three simultaneously — that is a
    flaky property. We assert (a) all three ids are distinct, and (b)
    every session terminates (drops out of the active registry) within
    a generous timeout. Either is impossible if sessions interfere.
    """
    sids = [
        service.start_session(
            query=f"investigation #{i}",
            environment="dev",
            reporter_id=f"u{i}",
            reporter_team="platform",
        )
        for i in range(3)
    ]
    assert len(set(sids)) == 3, f"duplicate session ids: {sids}"

    assert _wait_for_all_complete(service, sids, timeout=30.0), (
        "sessions did not all complete in time; "
        f"still active = {service.list_active_sessions()}"
    )


def test_sessions_dont_cross_contaminate(service):
    """Each session's persisted state belongs to that session only.

    If two sessions raced into the same row (e.g. shared mutable orch
    state, broken store, registry collision) the persisted ``query``
    field would mismatch the input. This is the strongest single check
    that the per-session task isolation actually works.
    """
    sids = [
        service.start_session(
            query=f"unique-payload-{i}",
            environment="dev",
            reporter_id=f"u{i}",
            reporter_team="platform",
        )
        for i in range(3)
    ]
    assert _wait_for_all_complete(service, sids, timeout=30.0), (
        "sessions did not all complete in time; "
        f"still active = {service.list_active_sessions()}"
    )

    # Read each row back through the orchestrator's store. We don't
    # assert on tool_calls / agents_run length (the stub graph's exact
    # trace is implementation-detail), only that no row was overwritten
    # by another session and that each carries its own session_id +
    # reporter — proof of zero cross-contamination.
    orch = service._orch
    assert orch is not None, "orchestrator was never built"
    for i, sid in enumerate(sids):
        inc = orch.store.load(sid)
        assert inc.id == sid, f"row {sid} loaded with wrong id {inc.id}"
        assert inc.query == f"unique-payload-{i}", (
            f"session {sid} has wrong query: {inc.query!r}"
        )
        assert inc.reporter.id == f"u{i}", (
            f"session {sid} has wrong reporter: {inc.reporter.id!r}"
        )
        # Each session must own its own agents_run / tool_calls lists —
        # the lists are per-row, never shared. Type check only; the
        # exact counts depend on the stub graph trajectory.
        assert isinstance(inc.agents_run, list)
        assert isinstance(inc.tool_calls, list)


def test_cap_blocks_when_full(cfg):
    """When ``max_concurrent_sessions`` is exceeded, ``start_session``
    raises ``SessionCapExceeded``. We hold the cap full deterministically
    by injecting fake registry entries on the loop thread — racing the
    stub LLM (which completes in <100 ms) is not reliable.
    """
    from runtime.service import _ActiveSession  # type: ignore[attr-defined]

    cfg.runtime.max_concurrent_sessions = 2
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    try:
        # Seed the registry on the loop thread so size == cap before
        # we attempt a real start. This makes the test deterministic
        # against stub-LLM speed.
        async def _seed_registry():
            for i in range(2):
                fake_id = f"FAKE-CAP-{i}"
                svc._registry[fake_id] = _ActiveSession(
                    session_id=fake_id,
                    started_at="x",
                )

        svc.submit_and_wait(_seed_registry(), timeout=5)

        with pytest.raises(SessionCapExceeded):
            svc.start_session(
                query="should-be-rejected",
                environment="dev",
                reporter_id="u",
                reporter_team="t",
            )

        # Drain the fake entries so shutdown isn't fighting them.
        async def _drain():
            for k in list(svc._registry.keys()):
                if k.startswith("FAKE-CAP-"):
                    svc._registry.pop(k, None)

        svc.submit_and_wait(_drain(), timeout=5)
    finally:
        svc.shutdown()


def test_list_active_sessions_thread_safe_under_load(service):
    """``list_active_sessions`` is callable from many threads while
    sessions are being started from another. The snapshot must never
    raise (it's a copy of the registry; no half-built rows leak)."""
    errors: list[BaseException] = []

    def reader() -> None:
        for _ in range(40):
            try:
                snap = service.list_active_sessions()
                # Shape contract: list of dicts.
                assert isinstance(snap, list)
                for entry in snap:
                    assert isinstance(entry, dict)
                    assert "session_id" in entry
                time.sleep(0.005)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

    def starter() -> None:
        try:
            service.start_session(
                query="loadtest",
                environment="dev",
                reporter_id="u",
                reporter_team="t",
            )
        except SessionCapExceeded:
            pass  # acceptable under load
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=starter) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"race or unexpected error: {errors}"
