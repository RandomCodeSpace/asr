"""Long-lived orchestrator service.

Owns a background asyncio event loop and a shared FastMCP client pool.
All session execution will run as asyncio tasks on this loop. Sync callers
(Streamlit, FastAPI request handlers, CLI) submit coroutines via
``submit(coro) -> concurrent.futures.Future``.

Lifecycle::

    svc = OrchestratorService.get_or_create(cfg)
    svc.start()    # spins up background thread + loop
    fut = svc.submit(some_coro)
    result = fut.result(timeout=30)
    svc.shutdown() # cancels in-flight tasks, closes MCP clients, joins thread

Capabilities:
  - Skeleton + singleton + start/shutdown lifecycle.
  - ``submit()`` / ``submit_and_wait()`` thread-safe bridge.
  - Shared ``MCPClientPool`` with per-server ``asyncio.Lock``.
  - ``start_session()`` schedules a per-session asyncio task on the
    service's loop and returns the session id immediately (the agent run
    continues in the background). Active tasks are tracked in an
    in-memory registry that evicts on completion / cancellation.
  - ``list_active_sessions()`` returns a thread-safe snapshot of
    the in-flight registry; the snapshot coroutine runs on the loop so
    readers from any thread see a point-in-time consistent view.
  - ``stop_session(sid)`` cancels the in-flight task, waits up
    to 5 s for graceful exit, and persists ``status="stopped"`` on the
    row (clearing ``pending_intervention``). Idempotent — a no-op for
    unknown ids or already-completed sessions.
  - Hard cap on concurrent sessions. ``start_session`` raises
    ``SessionCapExceeded`` once ``len(self._registry) >=
    self.max_concurrent_sessions``. Fail fast; queueing is not supported.

The singleton is process-scoped and reset on ``shutdown()`` so that test
suites can build, tear down, and rebuild the service without leaking
state across cases.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, TypeVar

from runtime.config import AppConfig
from runtime.mcp_loader import build_fastmcp_client

_log = logging.getLogger("runtime.service")

T = TypeVar("T")


@dataclass
class _ActiveSession:
    """In-memory metadata for an in-flight session.

    Lives in ``OrchestratorService._registry``; mutated only on the
    loop thread so the dict itself needs no thread lock. Snapshots are
    produced via :meth:`OrchestratorService.list_active_sessions`,
    which submits a coroutine to the loop and returns a list of plain
    dicts to the calling thread.
    """

    session_id: str
    started_at: str
    status: str = "running"
    current_agent: str | None = None
    task: asyncio.Task | None = None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionCapExceeded(RuntimeError):
    """Raised by ``start_session`` when the service is already running
    ``max_concurrent_sessions`` sessions.

    Fail fast, do not queue. Callers (Streamlit, FastAPI handlers)
    catch this and surface a clear error — Streamlit shows a toast;
    the HTTP layer translates it to a 429 with ``Retry-After``.
    """

    def __init__(self, cap: int) -> None:
        super().__init__(
            f"OrchestratorService at capacity ({cap} concurrent); "
            f"reject incoming start_session"
        )
        self.cap = cap


class OrchestratorService:
    """Process-singleton orchestrator service.

    Surface: construction, singleton accessor, ``start()`` /
    ``shutdown()``, coroutine submission bridge, and the shared MCP
    client pool.

    Thread-safety (HARD-06): ``get_or_create()`` and
    ``_reset_singleton()`` serialise singleton mutation through a
    class-level ``threading.Lock``. Concurrent first-callers
    (Streamlit warmup + FastAPI startup hook racing during process
    boot) all observe the same instance — the loser of the race blocks
    on the lock briefly, then short-circuits on the
    ``_instance is None`` check inside the critical section.
    """

    # Class-level singleton state. Guarded by ``_lock`` so concurrent
    # ``get_or_create()`` callers can't double-construct the service.
    # Reset on ``shutdown()`` via :meth:`_reset_singleton`.
    _lock: threading.Lock = threading.Lock()
    _instance: "OrchestratorService | None" = None

    def __init__(
        self,
        cfg: AppConfig,
        max_concurrent_sessions: int | None = None,
    ) -> None:
        self.cfg = cfg
        # Resource cap. Prefer the explicit constructor arg; fall back
        # to ``cfg.runtime.max_concurrent_sessions``. Tests mutate this
        # attribute directly to drive cap behaviour deterministically.
        self.max_concurrent_sessions: int = (
            max_concurrent_sessions
            if max_concurrent_sessions is not None
            else cfg.runtime.max_concurrent_sessions
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        # Shared MCP client pool — built lazily on first ``get_mcp_client``
        # so processes that never touch MCP pay zero startup cost. All
        # mutations of ``_mcp_clients`` / ``_mcp_locks`` happen on the
        # background loop, so the dicts themselves don't need a thread
        # lock.
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_clients: dict[str, Any] = {}
        self._mcp_locks: dict[str, asyncio.Lock] = {}
        # Per-server-name asyncio.Lock guarding lazy build. Created on the
        # loop the first time the server is requested.
        self._mcp_build_locks: dict[str, asyncio.Lock] = {}
        # Shared Orchestrator (lazy-built on first session start) and
        # the in-flight session registry. The registry dict itself is
        # only mutated from the loop thread (writers go through
        # ``submit_and_wait``); readers also hop through the loop so the
        # snapshot is point-in-time consistent with concurrent mutators.
        self._orch: Any | None = None
        self._registry: dict[str, _ActiveSession] = {}
        # Lazily-built lock for serialising orchestrator construction
        # under concurrent ``start_session`` calls. Created on the loop.
        self._orch_build_lock: asyncio.Lock | None = None
        # Pending-approval timeout watchdog. Started in ``start()`` iff
        # ``cfg.runtime.gateway`` is configured; otherwise None and the
        # lifecycle hooks are no-ops.
        self._approval_watchdog: Any | None = None

    @classmethod
    def get_or_create(cls, cfg: AppConfig) -> "OrchestratorService":
        """Return the process-singleton service, building it on first call.

        Subsequent calls ignore the supplied ``cfg`` and return the
        existing instance — there is exactly one orchestrator service per
        Python process. To rebuild with a new config, call
        ``shutdown()`` first.

        Thread-safe (HARD-06): the check-and-construct pair runs inside
        a class-level ``threading.Lock``. A concurrent second caller
        either blocks until the first caller's ``__init__`` returns and
        then short-circuits on the ``_instance is not None`` check, or
        wins the race and constructs alone — no double construction.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    def start(self) -> None:
        """Spin up the background thread + asyncio loop.

        Idempotent: a no-op if the loop is already running. Blocks until
        the background thread reports the loop is ready (5s timeout) so
        callers can ``submit()`` immediately after ``start()`` returns.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="OrchestratorService",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("OrchestratorService loop failed to start within 5s")
        # Arm the pending-approval watchdog iff a gateway is configured.
        # The watchdog is harmless when no high-risk tool calls ever
        # fire (it scans the empty registry), but skipping the start
        # when the gateway is off keeps process startup quiet for apps
        # that have not opted into HITL.
        gateway_cfg = getattr(self.cfg.runtime, "gateway", None)
        if gateway_cfg is not None:
            from runtime.tools.approval_watchdog import ApprovalWatchdog

            timeout_s = getattr(
                gateway_cfg, "approval_timeout_seconds", 3600,
            )
            self._approval_watchdog = ApprovalWatchdog(
                self,
                approval_timeout_seconds=timeout_s,
            )
            self._approval_watchdog.start(self._loop)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            # Drain any remaining tasks before closing so no coroutine is
            # left dangling without a chance to clean up.
            try:
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                self._loop.close()

    def submit(
        self, coro: Awaitable[T]
    ) -> concurrent.futures.Future[T]:
        """Submit a coroutine to the background loop from any thread.

        Returns a ``concurrent.futures.Future`` whose ``.result()`` blocks
        the calling thread until the coroutine resolves on the loop. Safe
        to call concurrently from multiple threads.
        """
        if self._loop is None:
            raise RuntimeError(
                "OrchestratorService not started; call start() first"
            )
        if not self._loop.is_running():
            raise RuntimeError("OrchestratorService loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def submit_and_wait(
        self, coro: Awaitable[T], timeout: float | None = None
    ) -> T:
        """Submit a coroutine and block the caller until it resolves.

        Convenience wrapper for sync callers (Streamlit, FastAPI request
        handlers, CLI). Raises ``concurrent.futures.TimeoutError`` if the
        coroutine doesn't complete within ``timeout`` seconds.

        WARNING: do not call from an async function whose event loop is
        the same loop ``OrchestratorService`` is hosting (e.g. tests using
        ``httpx.AsyncClient + ASGITransport`` against the FastAPI app
        share the same loop the service runs on). The caller would block
        the loop while waiting for work scheduled onto that same loop —
        a deadlock. Use :meth:`submit_async` from async code.
        """
        return self.submit(coro).result(timeout=timeout)

    async def submit_async(self, coro: Awaitable[T]) -> T:
        """Bridge a coroutine onto the service's background loop, awaitable
        from any caller's loop.

        Async equivalent of :meth:`submit_and_wait`. ``asyncio.wrap_future``
        exposes the cross-thread ``concurrent.futures.Future`` returned by
        ``run_coroutine_threadsafe`` as awaitable on the calling loop, so
        the caller yields control while the work runs on the service's
        loop. Safe to call from a request handler whose event loop is the
        same one the service is hosting (no deadlock).
        """
        if self._loop is None:
            raise RuntimeError(
                "OrchestratorService not started; call start() first"
            )
        if not self._loop.is_running():
            raise RuntimeError("OrchestratorService loop is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(fut)

    async def get_mcp_client(self, server_name: str) -> Any:
        """Return the shared FastMCP client for ``server_name``, building
        on first request.

        Lookup is serialised via a per-server ``asyncio.Lock`` so two
        concurrent sessions racing for the same server don't double-build
        the client. The clients themselves are reused across all sessions
        for the lifetime of the service; teardown happens in
        :meth:`shutdown`.

        Raises ``KeyError`` if ``server_name`` is not declared in
        ``cfg.mcp.servers``.
        """
        # Build-lock dict mutation must happen on the loop; we *are* on
        # the loop here (this is an async method).
        if server_name not in self._mcp_build_locks:
            self._mcp_build_locks[server_name] = asyncio.Lock()
        async with self._mcp_build_locks[server_name]:
            if server_name in self._mcp_clients:
                return self._mcp_clients[server_name]
            server_cfg = next(
                (s for s in self.cfg.mcp.servers if s.name == server_name),
                None,
            )
            if server_cfg is None:
                raise KeyError(
                    f"MCP server {server_name!r} not declared in cfg.mcp.servers"
                )
            if self._mcp_stack is None:
                self._mcp_stack = AsyncExitStack()
                await self._mcp_stack.__aenter__()
            client = build_fastmcp_client(server_cfg)
            await self._mcp_stack.enter_async_context(client)
            self._mcp_clients[server_name] = client
            self._mcp_locks[server_name] = asyncio.Lock()
            return client

    def lock_for(self, server_name: str) -> asyncio.Lock:
        """Return the per-server ``asyncio.Lock`` that serialises tool
        calls against a single FastMCP client.

        Must be called after ``get_mcp_client(server_name)`` has built
        the client, otherwise ``KeyError``.
        """
        return self._mcp_locks[server_name]

    # ------------------------------------------------------------------
    # Per-session task scheduling + in-flight registry
    # ------------------------------------------------------------------

    async def _ensure_orchestrator(self) -> Any:
        """Lazily build the shared ``Orchestrator`` on the loop thread.

        Concurrent ``start_session`` calls coordinate through
        ``_orch_build_lock`` so we never build the orchestrator twice.
        Returns the cached instance on subsequent calls.
        """
        # Build-lock construction must happen on the loop. We *are* on
        # the loop here (this is an async method invoked via the bridge).
        if self._orch_build_lock is None:
            self._orch_build_lock = asyncio.Lock()
        async with self._orch_build_lock:
            if self._orch is None:
                # Lazy import to avoid a circular dependency at module
                # load time (orchestrator transitively imports a lot).
                from runtime.orchestrator import Orchestrator
                self._orch = await Orchestrator.create(self.cfg)
            return self._orch

    def start_session(
        self,
        *,
        query: str = "",
        state_overrides: dict | None = None,
        environment: str | None = None,
        submitter: dict | None = None,
        reporter_id: str | None = None,
        reporter_team: str | None = None,
        trigger: Any | None = None,
    ) -> str:
        """Start a new agent session. Returns the session id immediately.

        The session row is created (and the id minted) synchronously on
        the loop so the caller has a stable handle before this method
        returns. The actual graph run is launched as an ``asyncio.Task``
        on the same loop and runs in the background — the caller does
        **not** block on it. Listen via :meth:`list_active_sessions` and
        per-session state lookups for progress.

        ``state_overrides`` is a free-form dict of domain fields the app
        stamps onto the new session row. The framework only projects
        ``environment`` onto the storage column today; other keys ride
        through to app-specific MCP tools.

        ``submitter`` is a free-form dict the calling app interprets.
        For incident-management it is ``{"id": "...", "team": "..."}``;
        other apps can carry app-specific keys (e.g. code-review's
        ``{"id": "<github-username>", "pr_url": "..."}``). The framework
        only projects ``id``/``team`` onto the row's reporter columns.

        Deprecated kwargs (coerced and warned):
          * ``environment`` -> ``state_overrides={"environment": ...}``
          * ``reporter_id`` / ``reporter_team`` -> ``submitter``

        The registry entry is evicted by a ``Task.add_done_callback`` on
        completion, cancellation, or failure — so a session that crashes
        does not leak a stale entry.
        """
        from runtime.graph import GraphState  # local import: graph deps
        from runtime.orchestrator import (
            _coerce_state_overrides,
            _coerce_submitter,
        )

        # Resolve the generic ``submitter`` and ``state_overrides`` once
        # on the caller's thread — the deprecation warnings fire here
        # (in the user's frame), not deep inside the loop's ``_scheduler``.
        resolved_overrides = _coerce_state_overrides(
            state_overrides, environment,
        )
        resolved_submitter = _coerce_submitter(
            submitter, reporter_id, reporter_team
        )
        sub_id = (resolved_submitter or {}).get("id", "user-mock")
        sub_team = (resolved_submitter or {}).get("team", "platform")
        env = (resolved_overrides or {}).get("environment", "")

        async def _scheduler() -> str:
            # Enforce the concurrency cap on the loop thread so the
            # registry size check is race-free. Fail-fast with
            # ``SessionCapExceeded``; the exception propagates through
            # ``submit_and_wait`` -> ``Future.result()`` to the caller.
            if len(self._registry) >= self.max_concurrent_sessions:
                raise SessionCapExceeded(self.max_concurrent_sessions)
            orch = await self._ensure_orchestrator()
            # Allocate the row (and its id) synchronously on the loop
            # so the caller gets a stable id back. The graph then runs
            # in a separate task — registration happens here, before
            # the task is created, so ``list_active_sessions`` sees the
            # entry immediately.
            inc = orch.store.create(
                query=query,
                environment=env,
                reporter_id=sub_id,
                reporter_team=sub_team,
            )
            session_id = inc.id
            # Stamp trigger provenance onto the row before the graph
            # runs so any crash mid-graph still leaves an audit trail.
            # ``inc.findings`` is a JSON dict on the row.
            if trigger is not None:
                try:
                    received_at = trigger.received_at.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                except Exception:  # noqa: BLE001
                    received_at = _utc_iso_now()
                inc.findings["trigger"] = {
                    "name": getattr(trigger, "name", None),
                    "transport": getattr(trigger, "transport", None),
                    "target_app": getattr(trigger, "target_app", None),
                    "received_at": received_at,
                }
                orch.store.save(inc)
            entry = _ActiveSession(
                session_id=session_id,
                started_at=_utc_iso_now(),
            )
            self._registry[session_id] = entry

            async def _run() -> None:
                # Fail-fast on contention (D-03): if another task already
                # holds the session lock, refuse the new turn immediately.
                if orch._locks.is_locked(session_id):
                    from runtime.locks import SessionBusy  # noqa: PLC0415
                    raise SessionBusy(session_id)
                # Hold the per-session lock for the full graph turn,
                # including any HITL interrupt() pause (D-01).
                async with orch._locks.acquire(session_id):
                    try:
                        await orch.graph.ainvoke(
                            GraphState(
                                session=inc,
                                next_route=None,
                                last_agent=None,
                                error=None,
                            ),
                            config=orch._thread_config(session_id),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # Phase 11 (FOC-04 / D-11-04): GraphInterrupt is a
                        # pending-approval pause, not a failure. Don't stamp
                        # status='error' on the registry entry -- let
                        # LangGraph's checkpointer hold the paused state
                        # and let the UI's Approve/Reject action drive
                        # resume.
                        try:
                            from langgraph.errors import GraphInterrupt
                            if isinstance(exc, GraphInterrupt):
                                # Propagate so the underlying Task
                                # observer (stop_session etc.) still
                                # sees the exception, but skip the
                                # status='error' write.
                                raise
                        except ImportError:  # pragma: no cover
                            pass
                        # Mark the registry entry so any concurrent snapshot
                        # observes the failure before the done-callback
                        # evicts it. The exception itself is preserved on
                        # the task object for ``stop_session`` and any
                        # other observer that holds a Task reference.
                        e = self._registry.get(session_id)
                        if e is not None:
                            e.status = "error"
                        raise

            task = asyncio.create_task(_run(), name=f"session:{session_id}")
            entry.task = task

            # Eviction is loop-local: ``add_done_callback`` fires on the
            # loop thread, so the dict mutation is single-threaded.
            def _evict(_t: asyncio.Task) -> None:
                self._registry.pop(session_id, None)

            task.add_done_callback(_evict)
            return session_id

        return self.submit_and_wait(_scheduler(), timeout=30.0)

    # ------------------------------------------------------------------
    # stop_session — cancel in-flight task + persist stopped status
    # ------------------------------------------------------------------

    def stop_session(self, session_id: str) -> None:
        """Cancel an in-flight session and mark its row ``status="stopped"``.

        Idempotent: calling on an unknown id, an already-stopped session,
        or a session that completed naturally is a no-op (does not raise).
        Also clears ``pending_intervention`` so a session interrupted
        mid-resume doesn't leave a stale prompt on the row.

        Partial work (recorded ``tool_calls``, ``agents_run``) is
        preserved — they are written as they happen, and stopping is
        not a rollback.
        """

        async def _stop() -> None:
            entry = self._registry.get(session_id)
            task = entry.task if entry is not None else None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception:  # noqa: BLE001
                    # The graph itself may have raised; we still want to
                    # mark the row stopped below. Swallow here, but log
                    # so post-mortem reveals the underlying failure.
                    _log.warning(
                        "stop_session: graph raised during cancel-await for %s",
                        session_id,
                        exc_info=True,
                    )
            # Persist the stopped status. The orchestrator may not have
            # been built yet (caller passed an unknown id before any
            # session ran) — in that case there's nothing to persist.
            orch = self._orch
            if orch is not None:
                try:
                    inc = orch.store.load(session_id)
                except Exception:  # noqa: BLE001
                    # Unknown id: nothing to persist; treat as no-op. A
                    # genuine store failure is still observable via the log.
                    _log.debug(
                        "stop_session: store.load(%s) failed; treating as unknown id",
                        session_id,
                        exc_info=True,
                    )
                    inc = None
                if inc is not None:
                    inc.status = "stopped"
                    inc.pending_intervention = None
                    orch.store.save(inc)
            # Drop the registry entry if the done-callback didn't already
            # evict it (it always does, but be defensive).
            self._registry.pop(session_id, None)

        # If the loop isn't running (caller stopped the service), be a
        # silent no-op rather than raising — keeps idempotency guarantees.
        if self._loop is None or not self._loop.is_running():
            return
        self.submit_and_wait(_stop(), timeout=10.0)

    # ------------------------------------------------------------------
    # Active-session registry snapshot accessor
    # ------------------------------------------------------------------

    def list_active_sessions(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of in-flight sessions.

        The snapshot coroutine runs on the loop thread, so the view is
        point-in-time consistent w.r.t. concurrent registry mutators
        (which also run on the loop). Each entry is a plain ``dict``
        with ``session_id``, ``status``, ``started_at``, and
        ``current_agent`` keys — callers in any thread can pass it
        around without holding any asyncio resources.

        Returns an empty list when the service has never run a session
        or when every previously-started run has completed.
        """

        async def _snapshot() -> list[dict[str, Any]]:
            return [
                {
                    "session_id": e.session_id,
                    "status": e.status,
                    "started_at": e.started_at,
                    "current_agent": e.current_agent,
                }
                for e in self._registry.values()
            ]

        return self.submit_and_wait(_snapshot(), timeout=5.0)

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the loop, tear down MCP clients, join the thread,
        reset the singleton.

        Idempotent: safe to call multiple times, including after the
        loop has already been torn down. Resets the module-level
        singleton so ``get_or_create()`` will rebuild on the next call.
        """
        if self._loop is None:
            self._reset_singleton()
            return
        loop = self._loop
        thread = self._thread
        # Stop the watchdog before draining sessions so its scan
        # doesn't race against the registry teardown below.
        if loop.is_running() and self._approval_watchdog is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._approval_watchdog.stop(), loop,
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: shutdown must continue even if the watchdog
                # refuses to stop cleanly. Surface the cause so it doesn't
                # silently rot.
                _log.warning(
                    "shutdown: approval watchdog stop failed",
                    exc_info=True,
                )
            self._approval_watchdog = None
        # Cancel in-flight session tasks first so they observe a
        # CancelledError before the orchestrator's underlying
        # resources (DB engine, FastMCP transports) are torn down.
        if loop.is_running() and self._registry:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._cancel_all_sessions(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: a stuck task that ignores cancellation must
                # not block the loop teardown below. Surface for diagnosis.
                _log.warning(
                    "shutdown: cancel_all_sessions failed",
                    exc_info=True,
                )
        # Close the shared orchestrator on the loop, releasing its
        # checkpointer connection / MCP exit-stack.
        if loop.is_running() and self._orch is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._close_orchestrator(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: a misbehaving aclose() must not block
                # the loop / thread join below. Surface for diagnosis.
                _log.warning(
                    "shutdown: orchestrator close failed",
                    exc_info=True,
                )
        # Close MCP clients on the loop *before* stopping it.
        if loop.is_running() and self._mcp_stack is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._close_mcp_pool(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: don't block shutdown on a misbehaving
                # client. Log so diagnostics survive the silent cleanup.
                _log.warning(
                    "shutdown: MCP pool close failed",
                    exc_info=True,
                )
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)
        self._loop = None
        self._thread = None
        self._started.clear()
        self._mcp_stack = None
        self._mcp_clients.clear()
        self._mcp_locks.clear()
        self._mcp_build_locks.clear()
        self._orch = None
        self._orch_build_lock = None
        self._registry.clear()
        self._approval_watchdog = None
        self._reset_singleton()

    async def _cancel_all_sessions(self) -> None:
        """Cancel every in-flight session task and wait for them to exit.

        Runs on the loop thread. Each task gets up to 5s to honour the
        ``CancelledError``; misbehaving tasks that ignore cancellation
        do not block shutdown beyond that — ``run_loop`` will sweep
        them in its final ``gather`` pass.
        """
        tasks = [e.task for e in self._registry.values() if e.task is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._registry.clear()

    async def _close_orchestrator(self) -> None:
        if self._orch is None:
            return
        orch = self._orch
        self._orch = None
        try:
            await orch.aclose()
        except Exception:  # noqa: BLE001
            # Best-effort cleanup: a checkpointer / MCP exit-stack that
            # blew up on close still leaves the process to exit cleanly.
            # Surface so the failure is observable post-mortem.
            _log.warning(
                "_close_orchestrator: orch.aclose() failed",
                exc_info=True,
            )

    async def _close_mcp_pool(self) -> None:
        if self._mcp_stack is None:
            return
        stack = self._mcp_stack
        self._mcp_stack = None
        await stack.__aexit__(None, None, None)
        self._mcp_clients.clear()
        self._mcp_locks.clear()
        self._mcp_build_locks.clear()

    @classmethod
    def _reset_singleton(cls) -> None:
        """Clear the class-level singleton under the same lock that
        ``get_or_create`` uses — so a reset racing with a fresh
        ``get_or_create`` call cannot leak the stale instance.
        """
        with cls._lock:
            cls._instance = None
