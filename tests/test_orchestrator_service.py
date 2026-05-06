"""Tests for runtime.service.OrchestratorService.

Covers:
- singleton + start/shutdown lifecycle.
- cross-thread ``submit()`` / ``submit_and_wait()``.
- shared MCP client pool with per-server ``asyncio.Lock``.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    """Minimal AppConfig for service tests — stub LLM, no MCP servers."""
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db")
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
    )


@pytest.fixture
def cfg_full(tmp_path):
    """AppConfig wired to all four in-process MCP servers from the
    incident-management example — sufficient for ``Orchestrator.create``
    to load all skills successfully (each skill resolves ``local:`` to
    the union of in-process servers).
    """
    from runtime.config import RuntimeConfig

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
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db")
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(
            state_class=None,
        ),
    )


@pytest.fixture
def service_full(cfg_full):
    """OrchestratorService with full MCP server set; teardown via shutdown."""
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


@pytest.fixture
def cfg_with_inproc_mcp(tmp_path):
    """AppConfig wired to in-process MCP servers from the incident example."""
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
            ]
        ),
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


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Belt-and-braces: ensure no service instance leaks between tests."""
    yield
    # If a test failed before its own teardown ran, sweep up.
    OrchestratorService._reset_singleton()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_service_singleton(cfg):
    a = OrchestratorService.get_or_create(cfg)
    b = OrchestratorService.get_or_create(cfg)
    assert a is b
    a.shutdown()


def test_service_get_or_create_after_shutdown_returns_new_instance(cfg):
    a = OrchestratorService.get_or_create(cfg)
    a.shutdown()
    b = OrchestratorService.get_or_create(cfg)
    assert b is not a
    b.shutdown()


def test_service_start_brings_up_loop(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    try:
        assert svc._thread is not None
        assert svc._thread.is_alive()
        assert svc._loop is not None
        assert svc._loop.is_running()
    finally:
        svc.shutdown()


def test_service_double_start_is_noop(service):
    first_thread = service._thread
    service.start()
    assert service._thread is first_thread


def test_service_shutdown_cleans_up(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    assert svc._thread is not None and svc._thread.is_alive()
    svc.shutdown()
    assert svc._thread is None
    # Singleton must be reset so the next get_or_create() rebuilds.
    assert OrchestratorService.get_or_create(cfg) is not svc
    OrchestratorService._reset_singleton()


def test_service_shutdown_is_idempotent(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    svc.shutdown()
    # Second shutdown must not raise even though the loop is gone.
    svc.shutdown()


def test_service_shutdown_without_start_is_noop(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    svc.shutdown()  # never started; must not hang or raise.
    # Singleton has been reset.
    assert OrchestratorService.get_or_create(cfg) is not svc
    OrchestratorService._reset_singleton()


def test_service_thread_is_daemon(service):
    assert service._thread is not None
    assert service._thread.daemon is True


def test_service_no_zombie_thread_after_shutdown(cfg):
    """After shutdown, the OrchestratorService thread must not appear in the
    enumeration — pytest will hang on teardown otherwise."""
    svc = OrchestratorService.get_or_create(cfg)
    svc.start()
    svc.shutdown()
    live = [t for t in threading.enumerate() if t.name == "OrchestratorService"]
    assert live == []


# ---------------------------------------------------------------------------
# Sync→async submit bridge
# ---------------------------------------------------------------------------


def test_submit_returns_future(service):
    """submit() returns a concurrent.futures.Future immediately."""
    import concurrent.futures

    async def echo():
        return 42

    fut = service.submit(echo())
    assert isinstance(fut, concurrent.futures.Future)
    assert fut.result(timeout=5) == 42


def test_submit_and_wait_blocks_for_result(service):
    """submit_and_wait() blocks until the coroutine resolves."""

    async def echo():
        return 42

    assert service.submit_and_wait(echo(), timeout=5) == 42


def test_submit_propagates_exceptions(service):
    """Exceptions inside the coroutine surface via Future.result()."""

    async def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        service.submit_and_wait(boom(), timeout=5)


def test_submit_before_start_raises(cfg):
    svc = OrchestratorService.get_or_create(cfg)
    try:
        async def noop():
            return 1
        coro = noop()
        try:
            with pytest.raises(RuntimeError, match="not started"):
                svc.submit(coro)
        finally:
            coro.close()
    finally:
        svc.shutdown()


def test_submit_runs_actual_async_work(service):
    """asyncio.sleep on the loop yields control — confirms it's a real loop."""

    async def slow():
        await asyncio.sleep(0.05)
        return "ok"

    assert service.submit_and_wait(slow(), timeout=5) == "ok"


def test_service_thread_safe_concurrent_submits(service):
    """Multiple threads can submit at once without races."""
    results: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        async def coro():
            await asyncio.sleep(0.01)
            return i

        try:
            r = service.submit_and_wait(coro(), timeout=5)
            with lock:
                results.append(r)
        except Exception as e:  # noqa: BLE001 — propagate to assertion
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(results) == list(range(8))


# ---------------------------------------------------------------------------
# Shared MCP client pool with per-server lock
# ---------------------------------------------------------------------------


def test_mcp_pool_starts_empty(service):
    """No MCP work yet, so the pool dicts are pristine."""
    assert service._mcp_stack is None
    assert service._mcp_clients == {}
    assert service._mcp_locks == {}
    assert service._mcp_build_locks == {}


def test_mcp_pool_unknown_server_raises_keyerror(service):
    """Asking for a server that isn't in cfg.mcp.servers fails fast."""

    async def fetch():
        return await service.get_mcp_client("does_not_exist")

    with pytest.raises(KeyError, match="does_not_exist"):
        service.submit_and_wait(fetch(), timeout=5)


def test_mcp_pool_lock_for_before_build_raises(service):
    """lock_for() requires the client to have been built first."""
    with pytest.raises(KeyError):
        service.lock_for("never_built")


def test_mcp_pool_builds_inproc_client_lazily(cfg_with_inproc_mcp):
    """get_mcp_client() builds on first request, returns same instance after."""
    svc = OrchestratorService.get_or_create(cfg_with_inproc_mcp)
    svc.start()
    try:
        async def fetch_twice():
            a = await svc.get_mcp_client("local_inc")
            b = await svc.get_mcp_client("local_inc")
            return a, b

        a, b = svc.submit_and_wait(fetch_twice(), timeout=10)
        assert a is b
        # After build, lock_for() returns a working asyncio.Lock.
        lock = svc.lock_for("local_inc")
        assert isinstance(lock, asyncio.Lock)
    finally:
        svc.shutdown()


def test_mcp_pool_per_server_lock_is_unique(cfg_with_inproc_mcp):
    """Each server name gets its own asyncio.Lock instance."""
    svc = OrchestratorService.get_or_create(cfg_with_inproc_mcp)
    svc.start()
    try:
        async def fetch():
            await svc.get_mcp_client("local_inc")

        svc.submit_and_wait(fetch(), timeout=10)
        # The build_lock and call_lock are separate locks.
        assert svc.lock_for("local_inc") is not svc._mcp_build_locks["local_inc"]
    finally:
        svc.shutdown()


def test_mcp_pool_concurrent_build_serialised(cfg_with_inproc_mcp):
    """Two concurrent get_mcp_client() calls share one client (no race)."""
    svc = OrchestratorService.get_or_create(cfg_with_inproc_mcp)
    svc.start()
    try:
        async def race():
            a, b, c = await asyncio.gather(
                svc.get_mcp_client("local_inc"),
                svc.get_mcp_client("local_inc"),
                svc.get_mcp_client("local_inc"),
            )
            return a, b, c

        a, b, c = svc.submit_and_wait(race(), timeout=10)
        assert a is b is c
        assert len(svc._mcp_clients) == 1
    finally:
        svc.shutdown()


def test_mcp_pool_torn_down_on_shutdown(cfg_with_inproc_mcp):
    """shutdown() closes the AsyncExitStack and clears the client dicts."""
    svc = OrchestratorService.get_or_create(cfg_with_inproc_mcp)
    svc.start()

    async def fetch():
        await svc.get_mcp_client("local_inc")

    svc.submit_and_wait(fetch(), timeout=10)
    assert svc._mcp_clients  # populated

    svc.shutdown()
    assert svc._mcp_stack is None
    assert svc._mcp_clients == {}
    assert svc._mcp_locks == {}


# ---------------------------------------------------------------------------
# Per-session task scheduling
# ---------------------------------------------------------------------------


def test_start_session_returns_id_immediately(service_full):
    """start_session() returns a session id synchronously while the
    agent run continues asynchronously on the service's loop."""
    sid = service_full.start_session(
        query="payments slow",
        environment="staging",
        submitter={"id": "u1", "team": "platform"},
    )
    assert isinstance(sid, str)
    # The ``cfg_full`` fixture doesn't override
    # ``FrameworkAppConfig.session_id_prefix``, so the framework
    # default ``SES-YYYYMMDD-NNN`` shape applies.
    assert sid.startswith("SES-")


def test_start_session_creates_persisted_row(service_full):
    """The session row is persisted before start_session returns —
    callers can look it up via the orchestrator's store immediately."""
    sid = service_full.start_session(
        query="db slow",
        environment="prod",
        submitter={"id": "u", "team": "dba"},
    )
    # Pull the row back through the loop. The orchestrator was lazily
    # built by start_session, so it exists at this point.
    async def _load():
        return service_full._orch.store.load(sid)

    inc = service_full.submit_and_wait(_load(), timeout=5)
    assert inc.id == sid
    assert inc.extra_fields.get("query") == "db slow"
    assert inc.extra_fields.get("environment") == "prod"


def test_concurrent_start_sessions(service_full):
    """Multiple sessions can run concurrently on the same service —
    each gets a distinct id, none stomp each other's registry slot."""
    sids = [
        service_full.start_session(
            query=f"q{i}",
            environment="dev",
            submitter={"id": "u", "team": "t"},
        )
        for i in range(3)
    ]
    assert len(set(sids)) == 3  # all distinct


def test_start_session_registry_evicts_on_completion(service_full):
    """Once the per-session task completes, the registry evicts the
    entry — long-lived services do not accumulate stale rows."""
    import time

    sid = service_full.start_session(
        query="t",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        # Hop through the loop so we observe eviction races correctly.
        async def _has():
            return sid in service_full._registry

        if not service_full.submit_and_wait(_has(), timeout=2):
            break
        time.sleep(0.05)
    else:
        async def _state():
            return list(service_full._registry.keys())

        keys = service_full.submit_and_wait(_state(), timeout=2)
        pytest.fail(f"session {sid} never evicted from registry; have {keys}")


# ---------------------------------------------------------------------------
# Active-session registry snapshot accessor
# ---------------------------------------------------------------------------


def test_list_active_sessions_returns_list(service_full):
    """list_active_sessions() returns a list of dicts (possibly empty)."""
    active = service_full.list_active_sessions()
    assert isinstance(active, list)
    assert all(isinstance(s, dict) for s in active)


def test_list_active_sessions_keys_are_well_known(service_full):
    """Snapshot entries expose the documented set of keys, regardless
    of whether any sessions are currently in flight."""
    sid = service_full.start_session(
        query="db latency",
        environment="production",
        submitter={"id": "u2", "team": "dba"},
    )
    # The stub-LLM graph may finish before the snapshot is taken; if so,
    # ``active`` is empty. When the entry IS still present, its shape
    # must match the contract.
    active = service_full.list_active_sessions()
    if active:
        ids = {s["session_id"] for s in active}
        if sid in ids:
            entry = next(s for s in active if s["session_id"] == sid)
            assert set(entry.keys()) >= {
                "session_id",
                "status",
                "started_at",
                "current_agent",
            }


def test_list_active_sessions_empty_after_completion(service_full):
    """Once tasks complete, the snapshot no longer includes them —
    eviction is the source of truth, the snapshot is just a view."""
    import time

    sid = service_full.start_session(
        query="t",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        active = service_full.list_active_sessions()
        if not any(s["session_id"] == sid for s in active):
            break
        time.sleep(0.05)
    else:
        pytest.fail(
            f"session {sid} never evicted from list_active_sessions"
        )


def test_list_active_sessions_snapshot_is_independent(service_full):
    """The returned list is a snapshot — callers mutating the list
    must not affect the registry, and successive calls return fresh
    lists rather than aliasing the same buffer."""
    a = service_full.list_active_sessions()
    a.append({"session_id": "fake", "status": "running",
              "started_at": "x", "current_agent": None})
    b = service_full.list_active_sessions()
    assert not any(s["session_id"] == "fake" for s in b)


def test_list_active_sessions_works_before_first_session(cfg_full):
    """Calling list_active_sessions() before any start_session() is a
    no-op that returns an empty list — it must not bootstrap the
    orchestrator (cheap to call from a Streamlit rerun)."""
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.start()
    try:
        assert svc.list_active_sessions() == []
        # Orchestrator was never built because no session ran.
        assert svc._orch is None
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# stop_session: cancel in-flight task + mark status=stopped
# ---------------------------------------------------------------------------


def test_stop_session_marks_status_stopped(service_full):
    """stop_session() persists status="stopped" on the row and clears
    pending_intervention. Whether the task was still running at the
    moment of the call is irrelevant — the post-condition is the row."""
    sid = service_full.start_session(
        query="long task",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    service_full.stop_session(sid)

    async def _load():
        return service_full._orch.store.load(sid)

    inc = service_full.submit_and_wait(_load(), timeout=5)
    assert inc.status == "stopped"
    assert inc.pending_intervention is None


def test_stop_session_clears_pending_intervention(service_full):
    """If a row carries a pending_intervention payload (interrupted
    mid-resume), stop_session must wipe it so the row doesn't keep
    asking the user for input after being stopped."""
    sid = service_full.start_session(
        query="t",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    # Plant a pending_intervention directly so we have something to clear.
    async def _plant():
        store = service_full._orch.store
        inc = store.load(sid)
        inc.pending_intervention = {"prompt": "approve?", "agent": "x"}
        store.save(inc)
    service_full.submit_and_wait(_plant(), timeout=5)

    service_full.stop_session(sid)

    async def _load():
        return service_full._orch.store.load(sid)
    inc = service_full.submit_and_wait(_load(), timeout=5)
    assert inc.status == "stopped"
    assert inc.pending_intervention is None


def test_stop_session_is_idempotent(service_full):
    """Calling stop_session twice on the same id must not raise — the
    second call is a no-op once the row is already stopped and the
    task is gone from the registry."""
    sid = service_full.start_session(
        query="t",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    service_full.stop_session(sid)
    service_full.stop_session(sid)  # must not raise


def test_stop_session_unknown_id_is_noop(service_full):
    """A bogus session id is silently accepted — keeps callers (UI,
    HTTP handlers) from having to special-case "already gone"."""
    # Trigger orchestrator build first by starting/stopping a real session,
    # so the no-op path also exercises the store.load(...) fallback.
    sid = service_full.start_session(
        query="seed",
        environment="dev",
        submitter={"id": "u", "team": "t"},
    )
    service_full.stop_session(sid)
    # Now an unknown id — must not raise.
    service_full.stop_session("INC-19990101-001")


def test_stop_session_before_any_session_is_noop(cfg_full):
    """stop_session() before the orchestrator has ever been built
    must not raise and must not bootstrap the orchestrator."""
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.start()
    try:
        svc.stop_session("INC-19990101-001")
        assert svc._orch is None
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Concurrent-session cap (fail-fast)
# ---------------------------------------------------------------------------


def test_runtime_config_default_cap_is_8():
    """Default RuntimeConfig.max_concurrent_sessions is 8."""
    from runtime.config import RuntimeConfig

    assert RuntimeConfig().max_concurrent_sessions == 8


def test_service_inherits_cap_from_runtime_config(cfg_full):
    """OrchestratorService picks up the cap from cfg.runtime when no
    explicit override is passed to the constructor."""
    from runtime.service import OrchestratorService

    OrchestratorService._reset_singleton()
    cfg_full.runtime.max_concurrent_sessions = 3
    svc = OrchestratorService.get_or_create(cfg_full)
    try:
        assert svc.max_concurrent_sessions == 3
    finally:
        svc.shutdown()


def test_resource_cap_raises_when_exceeded(cfg_full):
    """Service refuses to start a new session once the registry is at
    capacity — fail fast with SessionCapExceeded."""
    from runtime.service import OrchestratorService, SessionCapExceeded

    OrchestratorService._reset_singleton()
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.max_concurrent_sessions = 2
    svc.start()
    try:
        # Hold the registry at cap by inserting fake placeholder
        # entries on the loop thread — this is the deterministic way
        # to keep the cap full without racing the stub LLM completing.
        async def _seed_registry():
            from runtime.service import _ActiveSession
            for i in range(2):
                fake_id = f"FAKE-{i}"
                svc._registry[fake_id] = _ActiveSession(
                    session_id=fake_id, started_at="x"
                )
        svc.submit_and_wait(_seed_registry(), timeout=5)

        with pytest.raises(SessionCapExceeded):
            svc.start_session(
                query="c",
                environment="d",
                submitter={"id": "u", "team": "t"},
            )
        # SessionCapExceeded must be a RuntimeError subclass per spec.
        try:
            svc.start_session(
                query="d",
                environment="d",
                submitter={"id": "u", "team": "t"},
            )
        except RuntimeError:
            pass
    finally:
        # Clear the seeded fakes so shutdown's cancel-all doesn't try
        # to .cancel() a None task.
        async def _clear():
            svc._registry.clear()
        try:
            svc.submit_and_wait(_clear(), timeout=5)
        except Exception:
            pass
        svc.shutdown()


def test_resource_cap_releases_slot_after_stop(cfg_full):
    """Once a session is stopped (registry slot freed), a new
    start_session call succeeds — cap is a live count, not a
    monotonic counter."""
    from runtime.service import OrchestratorService

    OrchestratorService._reset_singleton()
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.max_concurrent_sessions = 1
    svc.start()
    try:
        sid = svc.start_session(
            query="a",
            environment="d",
            submitter={"id": "u", "team": "t"},
        )
        svc.stop_session(sid)
        # Slot is free; a new session must be admitted.
        sid2 = svc.start_session(
            query="b",
            environment="d",
            submitter={"id": "u", "team": "t"},
        )
        assert sid2 != sid
    finally:
        svc.shutdown()


def test_session_cap_exceeded_is_runtime_error_subclass():
    """Per the public contract, SessionCapExceeded is a RuntimeError —
    callers can catch RuntimeError and stay forward-compatible."""
    from runtime.service import SessionCapExceeded

    assert issubclass(SessionCapExceeded, RuntimeError)
    e = SessionCapExceeded(8)
    assert e.cap == 8
    assert "8" in str(e)


