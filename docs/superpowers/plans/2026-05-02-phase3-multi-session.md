# Phase 3 — Multi-Session Concurrency Implementation Plan

**Branch suggestion:** `phase3/multi-session`
**Estimated effort:** ~1 week (12 tasks; 1 sequencing-heavy refactor + 11 incremental).
**Prerequisite phase:** Phase 2 (`docs/superpowers/plans/2026-05-02-phase2-extensible-state-and-checkpointer.md`).

> **Phase 2 dependency note** — At time of writing, the Phase 2 plan exists at the path above but Phase 2 has **not yet been implemented**. Phase 3 sequencing assumes Phase 2's deliverables have landed in `main`:
> - `Generic[StateT]` `Orchestrator`, `SessionStore`, `HistoryStore`.
> - LangGraph `SqliteSaver` / `PostgresSaver` checkpointer wired to `storage.metadata.url` (WAL on for SQLite).
> - `Session` carries `pending_intervention` with dual-write semantics; gate node uses `interrupt()` + `Command(resume=...)`.
> - `runtime/incident.py`, the `incident` bridge alias on `GraphState`, and the legacy `IncidentRepository` shim are all gone.
>
> If Phase 3 implementation starts before Phase 2 lands, **stop and confirm** with the user before proceeding — the task code below references Phase 2's surface (`Orchestrator[StateT]`, `start_session`, `resume_session`, `SessionStore.list_active_sessions`, `interrupt()` gate semantics) and does not include compatibility shims for the pre-Phase-2 world.

---

## Locked decisions (NON-NEGOTIABLE)

These are inputs to this plan, not outputs of it. Do not relitigate during implementation.

- **P3.1 Concurrency model.** A single, long-lived `asyncio` event loop, hosted on a background thread inside the same Python process as the host app (Streamlit or FastAPI/uvicorn). Each session is one `asyncio.Task` running the LangGraph until completion or `interrupt()`. **Rejected:** thread-per-session, process-per-session, Celery/RQ workers, ProcessPoolExecutor.
- **P3.2 MCP-client sharing.** **One** shared FastMCP client pool, owned by a single `contextlib.AsyncExitStack` opened at service startup and closed at shutdown. All sessions reuse the same clients; tool calls are serialised per-client by an `asyncio.Lock` (see R3). **Rejected:** per-session client pool, per-task transport.
- **P3.3 UI shape.** Smallest delta to today's UI: extend the existing sidebar incident list with status badges (`running`, `awaiting_input`, `done`, `stopped`, `error`); clicking an entry opens the existing detail pane. **Deferred to a later UI iteration:** tabs, multi-pane, dedicated picker / command palette.

---

## 1. Goal + Scope

**Goal.** Run multiple sessions concurrently in one process while preserving Phase 2's clean generic core, the air-gapped deployment story (single-file bundle), and Streamlit's rerun-driven UI model.

**In scope.**
- A long-lived `OrchestratorService` that owns the asyncio loop + shared MCP client pool.
- Sync-callable API for Streamlit / FastAPI: `start_session`, `list_active_sessions`, `get_session_state`, `stop_session`, `resume_session`.
- Per-session `asyncio.Task` registry with a configurable concurrency cap.
- Sidebar UI changes: status badges per session, live polling on the detail pane.
- One concurrency integration test (3 sessions in parallel) plus targeted unit tests.

**Out of scope** (deferred):
- Server-Sent Events / WebSocket streaming — we poll `get_session_state` every 1–2 s.
- Tabs, multi-pane, command palette UI.
- Cross-process orchestrator (e.g. running uvicorn separately from Streamlit). Plan keeps that path open but does not implement it.
- Distributed scheduling (Celery, K8s jobs, etc.).
- Backpressure / queueing of over-cap sessions — see decision in §4 Risks (R5): **fail fast** in P3.

**Non-goals.** No new state-machine semantics. The `Session.status` field gains one new value (`"stopped"`) but the resume / interrupt machinery is unchanged from Phase 2.

---

## 2. Target Architecture After Phase 3

```
                                  ┌──────────────────────────┐
   Streamlit script (main thread) │                          │
   ───────────────────────────────│   OrchestratorService    │
   on every rerun:                │  (background thread)     │
     • list_active_sessions()  →──┤   ┌────────────────────┐ │
     • get_session_state(tid)  →──┤   │  asyncio loop      │ │
     • start_session(...)      →──┤   │  ───────────────── │ │
     • stop_session(tid)       →──┤   │  Task registry     │ │
                                  │   │   tid → Task       │ │
                                  │   │  Shared MCP pool   │ │
                                  │   │   AsyncExitStack   │ │
                                  │   │  Per-client locks  │ │
                                  │   └────────────────────┘ │
                                  └──────────────────────────┘
                                              │
                                  ┌───────────┴────────────┐
                                  │ Phase 2 Orchestrator   │
                                  │ [IncidentState]        │
                                  │  • start_session       │
                                  │  • resume_session      │
                                  │  • get_session_state   │
                                  └───────────┬────────────┘
                                              │
                            ┌─────────────────┼─────────────────┐
                            │                 │                 │
                       SessionStore     LangGraph saver   FastMCP clients
                       (SQLite WAL)     (SQLite WAL)      (stdio + http)
```

**Key invariants.**
1. **Exactly one event loop** per process. It runs on a daemon thread named `orch-asyncio`. Streamlit and FastAPI never run their own loops against the same orchestrator instance.
2. **Sync → async bridge** is `asyncio.run_coroutine_threadsafe(coro, loop)`. The returned `concurrent.futures.Future` is what Streamlit blocks on (with timeout) for fast operations; for `start_session` it does not block — only the registration completes synchronously, the LangGraph runs in the background.
3. **Per-session task** is created inside the loop via `loop.create_task(...)` and registered in a `dict[str, asyncio.Task]` guarded by an `asyncio.Lock`. Streamlit reads a thread-safe **snapshot** (list of (session_id, status, started_at, current_agent)) via the bridge.
4. **Shared MCP pool.** `Orchestrator` no longer constructs its own clients in this Phase. The service hands the orchestrator a pre-built `MCPClientPool` whose lifetime equals the loop's lifetime.
5. **Status surface.** The DB row's `Session.status` is the source of truth. The in-memory registry is a cache/index, not the truth — if the process restarts mid-session, the registry is empty but rows show `running` → P3-F handles that as `error: process restarted`.

---

## 3. Naming Map / API Changes

| Concern | Before (Phase 2) | After (Phase 3) |
|---|---|---|
| Top-level service | `Orchestrator` (per-app, owns loop transitively) | `OrchestratorService` (per-process) wraps `Orchestrator[StateT]` |
| Loop ownership | Implicit — caller's loop (Streamlit's per-rerun, uvicorn's, or tests') | Explicit — `OrchestratorService.loop`, on a dedicated thread |
| MCP client lifecycle | `Orchestrator.create()` opens an `AsyncExitStack` | `OrchestratorService.start()` opens **one** `AsyncExitStack`; `Orchestrator` receives clients |
| Start a session | `await orch.start_session(...)` | `service.start_session(...)` (sync; schedules a task, returns `session_id` immediately) |
| List in-flight | n/a (one at a time) | `service.list_active_sessions() -> list[ActiveSession]` |
| Cancel | n/a | `service.stop_session(thread_id) -> bool` |
| Read live state | `await orch.get_session_state(tid)` | `service.get_session_state(tid) -> StateT \| None` (sync) |
| Resume after `interrupt()` | `await orch.resume_session(tid, decision)` | `service.resume_session(tid, decision)` (sync) |
| `Session.status` enum | `new`, `running`, `awaiting_input`, `done`, `error` | adds `"stopped"` |
| Concurrency cap | n/a | `runtime.max_concurrent_sessions: int = 8` (in `RuntimeConfig`) |
| FastAPI endpoints | `POST /investigate`, `POST /resume`, `GET /incidents/...` | `POST /sessions`, `GET /sessions`, `DELETE /sessions/{id}`, `POST /sessions/{id}/resume` |

**Backwards compatibility.** The example app's existing FastAPI routes (`/investigate`, `/resume`, `/incidents/...`) stay as thin shims for one phase — flagged with `Deprecated` headers — and are removed in Phase 4. New deployments use the `/sessions` shape.

### File / module map

```
src/runtime/
├── orchestrator.py              # unchanged surface from Phase 2 (Generic[StateT])
├── service.py                   # NEW — OrchestratorService class
├── service_loop.py              # NEW — background-thread event loop helpers
├── service_registry.py          # NEW — in-memory active-task registry
├── mcp_pool.py                  # NEW — shared MCPClientPool extracted from Orchestrator
├── api.py                       # MODIFIED — endpoints rebuilt over service
└── config.py                    # MODIFIED — RuntimeConfig.max_concurrent_sessions

examples/incident_management/
├── ui.py                        # MODIFIED — sidebar reads service.list_active_sessions
└── service_factory.py           # NEW — build the singleton OrchestratorService for the example app

tests/
├── test_orch_service_loop.py    # NEW
├── test_orch_service_registry.py # NEW
├── test_orch_service_cap.py     # NEW
├── test_orch_service_stop.py    # NEW
├── test_mcp_pool_concurrency.py # NEW
├── test_api_sessions.py         # NEW
├── test_concurrent_sessions_e2e.py # NEW (integration: 3 in parallel)
└── ui/                          # if any UI tests exist; otherwise skip
```

---

## 4. Task Breakdown

Each task: TDD where feasible (write failing test → implement → green). Tasks land as one commit each unless noted. Conventional commit prefixes: `feat(service):`, `feat(api):`, `feat(ui):`, `test(service):`, `refactor(runtime):`.

> **Convention:** code blocks below show the **target shape** post-task. They are illustrative — exact lint-clean code lands during implementation. The plan is for review of intent and structure, not copy-paste-and-run.

---

### P3-A — Define `OrchestratorService` shape + `ActiveSession` DTO

Define the public surface only (no implementation yet). Pure typing/dataclass scaffolding so subsequent tasks have something to import.

**Files:**
- Create: `src/runtime/service.py` (skeleton: dataclass + method stubs raising `NotImplementedError`)
- Create: `src/runtime/service_loop.py` (empty for now, just the module docstring)
- Create: `tests/test_orch_service_shape.py`

**Steps:**

- [ ] **1. Add tests** to `tests/test_orch_service_shape.py`:

```python
from runtime.service import OrchestratorService, ActiveSession

def test_active_session_dto_fields():
    a = ActiveSession(
        session_id="s-001",
        status="running",
        started_at="2026-05-02T00:00:00Z",
        current_agent="entry",
    )
    assert a.session_id == "s-001"
    assert a.current_agent == "entry"

def test_service_public_surface():
    methods = {"start", "stop", "start_session", "stop_session",
               "resume_session", "get_session_state",
               "list_active_sessions"}
    assert methods <= set(dir(OrchestratorService))
```

- [ ] **2. Run** → `ImportError`.

- [ ] **3. Implementation** — `src/runtime/service.py`:

```python
"""Process-singleton orchestrator service.

Owns the asyncio loop, the shared MCP client pool, and the in-memory
registry of in-flight sessions. Sync-callable from Streamlit / FastAPI.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
from runtime.state import Session

StateT = TypeVar("StateT", bound=Session)

@dataclass(frozen=True)
class ActiveSession:
    session_id: str
    status: str          # "running" | "awaiting_input" | etc. (mirrors Session.status)
    started_at: str      # ISO8601 UTC
    current_agent: str | None

class OrchestratorService(Generic[StateT]):
    """Stub — see P3-B and onward for implementation."""
    def start(self) -> None: raise NotImplementedError
    def stop(self) -> None: raise NotImplementedError
    def start_session(self, **kwargs: object) -> str: raise NotImplementedError
    def stop_session(self, session_id: str) -> bool: raise NotImplementedError
    def resume_session(self, session_id: str, decision: dict) -> None: raise NotImplementedError
    def get_session_state(self, session_id: str) -> StateT | None: raise NotImplementedError
    def list_active_sessions(self) -> list[ActiveSession]: raise NotImplementedError
```

- [ ] **4. Run** → green.

- [ ] **5. Commit** — `feat(service): scaffold OrchestratorService surface (P3-A)`

---

### P3-B — Background-thread asyncio loop with sync→async bridge

Stand up the loop on a daemon thread; expose `submit(coro) -> Future` and `submit_blocking(coro, timeout=...)`. No business logic yet.

**Files:**
- Modify: `src/runtime/service_loop.py`
- Create: `tests/test_orch_service_loop.py`

**Steps:**

- [ ] **1. Add tests:**

```python
import asyncio, time
from runtime.service_loop import BackgroundLoop

def test_loop_starts_and_runs_coroutines():
    bl = BackgroundLoop(name="test-loop")
    bl.start()
    try:
        async def add(a, b): return a + b
        assert bl.submit_blocking(add(2, 3), timeout=1.0) == 5
    finally:
        bl.stop(timeout=2.0)

def test_loop_stop_is_idempotent():
    bl = BackgroundLoop(name="test-loop")
    bl.start()
    bl.stop(timeout=2.0)
    bl.stop(timeout=0.1)  # second call no-op

def test_loop_thread_is_daemon():
    bl = BackgroundLoop(name="test-loop")
    bl.start()
    try:
        assert bl.thread.daemon is True
        assert bl.thread.is_alive()
    finally:
        bl.stop(timeout=2.0)

def test_submit_returns_future_for_async_use():
    bl = BackgroundLoop(name="test-loop")
    bl.start()
    try:
        async def slow():
            await asyncio.sleep(0.05); return "ok"
        fut = bl.submit(slow())
        assert fut.result(timeout=1.0) == "ok"
    finally:
        bl.stop(timeout=2.0)

def test_submit_after_stop_raises():
    import pytest
    bl = BackgroundLoop(name="test-loop")
    bl.start(); bl.stop(timeout=2.0)
    async def noop(): return 1
    with pytest.raises(RuntimeError, match="not running"):
        bl.submit(noop())
```

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** — `src/runtime/service_loop.py`:

```python
"""Background-thread asyncio loop with thread-safe submit helpers.

Streamlit/FastAPI run on the main thread; the orchestrator's tasks all
live on this dedicated loop. Sync code calls ``submit_blocking`` (returns
the result) or ``submit`` (returns a ``concurrent.futures.Future``).
"""
from __future__ import annotations
import asyncio
import threading
from concurrent.futures import Future
from typing import Coroutine, Any

class BackgroundLoop:
    def __init__(self, *, name: str = "orch-asyncio") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()

    @property
    def thread(self) -> threading.Thread:
        assert self._thread is not None, "loop not started"
        return self._thread

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        assert self._loop is not None, "loop not started"
        return self._loop

    def start(self) -> None:
        if self._thread is not None:
            return  # idempotent
        def _run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._ready.set()
            try:
                loop.run_forever()
            finally:
                # Cancel any survivors, then close.
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
                self._stopped.set()
        self._thread = threading.Thread(target=_run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("BackgroundLoop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def submit_blocking(self, coro: Coroutine[Any, Any, Any], *, timeout: float | None = None):
        return self.submit(coro).result(timeout=timeout)

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._loop is None:
            return  # never started
        if self._loop.is_closed():
            return  # already stopped
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._stopped.wait(timeout=timeout)
```

- [ ] **4. Run** → green; thread leak check (`threading.enumerate()` should not contain `orch-asyncio` after stop).

- [ ] **5. Commit** — `feat(service): background-thread asyncio loop with sync bridge (P3-B)`

---

### P3-C — Extract shared `MCPClientPool` from `Orchestrator`

In Phase 2 the `Orchestrator` opens its own `AsyncExitStack` and registers FastMCP clients. Phase 3 lifts that to a process-singleton.

**Files:**
- Create: `src/runtime/mcp_pool.py`
- Modify: `src/runtime/orchestrator.py` — accept an injected `MCPClientPool` instead of building one
- Modify: `src/runtime/mcp_loader.py` — extract the client-build helpers into reusable functions
- Create: `tests/test_mcp_pool.py`

**Steps:**

- [ ] **1. Add tests** for `MCPClientPool`:
  - `test_pool_open_close` — open builds N clients, close releases all transports.
  - `test_pool_double_open_idempotent` — `await pool.open()` twice is a no-op.
  - `test_pool_close_idempotent` — close twice is a no-op.
  - `test_pool_get_returns_same_client` — `pool.get("observability")` returns the same instance across calls.
  - `test_pool_unknown_server_raises_keyerror`.

- [ ] **2. Run** → fail.

- [ ] **3. Implementation** — `src/runtime/mcp_pool.py`:

```python
"""Process-singleton FastMCP client pool.

Opened once at service startup, closed at service shutdown. All
orchestrators / sessions share the same client instances. Tool-call
serialisation per-client is handled by ``async_tool_call`` (P3-G note).
"""
from __future__ import annotations
import asyncio
from contextlib import AsyncExitStack
from typing import Any
from runtime.config import AppConfig
from runtime.mcp_loader import build_client_for_server  # extracted helper

class MCPClientPool:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._stack: AsyncExitStack | None = None
        self._clients: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def open(self) -> None:
        if self._stack is not None:
            return
        stack = AsyncExitStack()
        try:
            await stack.__aenter__()
            for srv in self._cfg.mcp.servers:
                client = await stack.enter_async_context(build_client_for_server(srv))
                self._clients[srv.name] = client
                self._locks[srv.name] = asyncio.Lock()
            self._stack = stack
        except BaseException:
            await stack.__aexit__(None, None, None)
            raise

    async def close(self) -> None:
        if self._stack is None:
            return
        s = self._stack
        self._stack = None
        await s.__aexit__(None, None, None)
        self._clients.clear()
        self._locks.clear()

    def get(self, server_name: str):
        return self._clients[server_name]

    def lock_for(self, server_name: str) -> asyncio.Lock:
        return self._locks[server_name]
```

- [ ] **4. Update `Orchestrator.__init__`** to accept `mcp_pool: MCPClientPool | None`. If `None`, fall back to the Phase 2 self-built path (preserves single-test ergonomics). All session methods route tool calls through `pool.lock_for(name)` when a pool is supplied.

- [ ] **5. Run** Phase 2 tests + new pool tests → green.

- [ ] **6. Commit** — `refactor(runtime): extract MCPClientPool for shared use across sessions (P3-C)`

---

### P3-D — In-memory active-task registry

Thread-safe `dict[session_id, asyncio.Task]` plus per-session metadata (started_at, current_agent). Lives inside the loop; mutators run as coroutines. Streamlit reads via a snapshot coroutine.

**Files:**
- Create: `src/runtime/service_registry.py`
- Create: `tests/test_orch_service_registry.py`

**Steps:**

- [ ] **1. Add tests:**
  - `test_register_unregister` — happy path.
  - `test_snapshot_returns_immutable_list` — caller mutating the list does not affect registry.
  - `test_concurrent_register_unregister` — 100 fake tasks via `asyncio.gather`; final size = 0.
  - `test_set_current_agent_visible_in_snapshot`.
  - `test_registry_size_reflects_active_only`.

- [ ] **2. Implementation:**

```python
"""In-memory in-flight session registry. All access is from the loop thread."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field, replace
from runtime.service import ActiveSession

@dataclass
class _Entry:
    session_id: str
    task: asyncio.Task
    started_at: str
    status: str = "running"
    current_agent: str | None = None

class TaskRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def register(self, sid: str, task: asyncio.Task, started_at: str) -> None:
        async with self._lock:
            self._entries[sid] = _Entry(sid, task, started_at)

    async def unregister(self, sid: str) -> None:
        async with self._lock:
            self._entries.pop(sid, None)

    async def set_status(self, sid: str, status: str) -> None:
        async with self._lock:
            e = self._entries.get(sid)
            if e: e.status = status

    async def set_current_agent(self, sid: str, agent: str | None) -> None:
        async with self._lock:
            e = self._entries.get(sid)
            if e: e.current_agent = agent

    async def snapshot(self) -> list[ActiveSession]:
        async with self._lock:
            return [
                ActiveSession(
                    session_id=e.session_id, status=e.status,
                    started_at=e.started_at, current_agent=e.current_agent,
                )
                for e in self._entries.values()
            ]

    async def size(self) -> int:
        async with self._lock:
            return len(self._entries)

    async def get_task(self, sid: str) -> asyncio.Task | None:
        async with self._lock:
            e = self._entries.get(sid)
            return e.task if e else None
```

- [ ] **3. Run** → green.

- [ ] **4. Commit** — `feat(service): in-memory active-session task registry (P3-D)`

---

### P3-E — Wire `OrchestratorService` end-to-end (start/stop/start_session)

Compose B + C + D inside the service. `start()` spins up the loop and opens the pool; `stop()` cancels in-flight tasks and closes the pool. `start_session()` schedules a task and returns the session id.

**Files:**
- Modify: `src/runtime/service.py`
- Create: `tests/test_orch_service_lifecycle.py`

**Steps:**

- [ ] **1. Add tests:**
  - `test_service_start_stop_clean` — start, stop, no thread leak, no open SQLite handle (`lsof`-ish via `gc`).
  - `test_start_session_returns_id_and_registers` — start_session schedules a task; `list_active_sessions()` reflects it within 50 ms.
  - `test_start_session_eventually_completes` — fake graph that returns immediately; `list_active_sessions()` empties; `Session.status == "done"` in store.
  - `test_get_session_state_returns_typed_state` — returns the concrete `IncidentState` Phase 2 supplies.
  - `test_resume_session_routes_through_loop` — interrupt, resume, status flips back to `running`.

- [ ] **2. Implementation** — service body:

```python
class OrchestratorService(Generic[StateT]):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._bg = BackgroundLoop(name="orch-asyncio")
        self._pool: MCPClientPool | None = None
        self._orch: Orchestrator[StateT] | None = None
        self._registry = TaskRegistry()
        self._cap = cfg.runtime.max_concurrent_sessions

    def start(self) -> None:
        self._bg.start()
        async def _bootstrap():
            self._pool = MCPClientPool(self._cfg)
            await self._pool.open()
            self._orch = await Orchestrator.create(self._cfg, mcp_pool=self._pool)
        self._bg.submit_blocking(_bootstrap(), timeout=30.0)

    def stop(self) -> None:
        async def _teardown():
            # Cancel in-flight tasks.
            for sid in list((await self._registry.snapshot())):
                await self._cancel_locked(sid.session_id)
            if self._orch is not None:
                await self._orch.aclose()
            if self._pool is not None:
                await self._pool.close()
        try:
            self._bg.submit_blocking(_teardown(), timeout=30.0)
        finally:
            self._bg.stop(timeout=10.0)

    def start_session(self, **kwargs) -> str:
        async def _go():
            assert self._orch is not None
            if await self._registry.size() >= self._cap:
                raise SessionCapExceeded(self._cap)  # P3-G
            sid = await self._orch.create_session(**kwargs)  # writes status="new"
            started_at = utc_iso_now()
            task = asyncio.current_task().get_loop().create_task(
                self._run_session(sid, kwargs), name=f"sess:{sid}"
            )
            await self._registry.register(sid, task, started_at)
            return sid
        return self._bg.submit_blocking(_go(), timeout=10.0)

    async def _run_session(self, sid: str, kwargs: dict) -> None:
        try:
            await self._orch.run_session(sid, **kwargs)  # drives LangGraph until done/interrupt
        except asyncio.CancelledError:
            await self._orch.mark_stopped(sid)
            raise
        except Exception as exc:
            await self._orch.mark_error(sid, exc)
        finally:
            await self._registry.unregister(sid)

    def list_active_sessions(self) -> list[ActiveSession]:
        return self._bg.submit_blocking(self._registry.snapshot(), timeout=2.0)

    def get_session_state(self, sid: str) -> StateT | None:
        async def _go():
            assert self._orch is not None
            return await self._orch.get_session_state(sid)
        return self._bg.submit_blocking(_go(), timeout=2.0)

    def stop_session(self, sid: str) -> bool:
        return self._bg.submit_blocking(self._cancel_locked(sid), timeout=5.0)

    def resume_session(self, sid: str, decision: dict) -> None:
        async def _go():
            assert self._orch is not None
            await self._orch.resume_session(sid, decision)
        self._bg.submit_blocking(_go(), timeout=10.0)
```

- [ ] **3. Run** → green (excluding cap and cancel tests, which land in P3-F and P3-G).

- [ ] **4. Commit** — `feat(service): OrchestratorService lifecycle and session scheduling (P3-E)`

---

### P3-F — `stop_session` with cancellation + status accounting

`stop_session(sid)`: cancels the asyncio task, waits up to 5 s, persists `status="stopped"`. Partial work (already-recorded `tool_calls`, agent runs) **is preserved** — no rollback.

**Files:**
- Modify: `src/runtime/service.py` (`_cancel_locked`)
- Modify: `src/runtime/orchestrator.py` (`mark_stopped(sid)`)
- Modify: `src/runtime/state.py` (literal: add `"stopped"` to `Session.status` if it's a `Literal`)
- Create: `tests/test_orch_service_stop.py`

**Steps:**

- [ ] **1. Add tests:**
  - `test_stop_session_cancels_task` — long-running fake graph; stop returns `True`; task is `cancelled()`.
  - `test_stop_session_persists_stopped_status` — DB row `status="stopped"`.
  - `test_stop_session_preserves_recorded_tool_calls` — fake graph that records 2 tool_calls then sleeps; after stop, `Session.tool_calls` length is 2 (not 0).
  - `test_stop_unknown_session_returns_false`.
  - `test_stop_after_natural_completion_returns_false`.
  - `test_stop_during_interrupt_state` — interrupted session, then stop; status becomes `"stopped"` not `"awaiting_input"`.

- [ ] **2. Implementation:**

```python
async def _cancel_locked(self, sid: str) -> bool:
    task = await self._registry.get_task(sid)
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return True
```

`Orchestrator.mark_stopped(sid)` writes `status="stopped"` and `updated_at=now()` via the `SessionStore`. Does NOT touch `tool_calls` or `agents_run` (Phase 2 wrote them as they happened).

- [ ] **3. Open question deferred to docs:** if cancellation lands inside an MCP tool call, FastMCP's stdio transport may emit a `BrokenPipeError` on subsequent calls. Mitigation: per-client lock (P3-C) ensures the offending tool call has already returned before `_run_session` is `await`ed; the cancellation point is between graph nodes, not mid-tool. Documented as R4.

- [ ] **4. Run** → green.

- [ ] **5. Commit** — `feat(service): cancel-and-mark-stopped on stop_session (P3-F)`

---

### P3-G — Resource cap (default 8) — fail fast

When `list_active_sessions().size >= cap`, `start_session` raises `SessionCapExceeded`. **Decision: fail, not queue.** Queueing is roadmap-aligned but adds an admission/notification surface; defer to Phase 4+. Streamlit catches and renders a clear error.

**Files:**
- Modify: `src/runtime/config.py` — add `RuntimeConfig.max_concurrent_sessions: int = 8`
- Modify: `src/runtime/service.py` — `SessionCapExceeded` exception, raise in `start_session`
- Modify: `src/runtime/api.py` — translate to HTTP 429 with `Retry-After: 30`
- Modify: `examples/incident_management/ui.py` — show toast on cap error
- Create: `tests/test_orch_service_cap.py`

**Steps:**

- [ ] **1. Add tests:**
  - `test_cap_default_is_8` — `RuntimeConfig().max_concurrent_sessions == 8`.
  - `test_cap_blocks_ninth_concurrent_start` — start 8 long-running, the 9th `start_session` raises `SessionCapExceeded`.
  - `test_cap_releases_slot_on_completion` — after 1 completes, a new start succeeds.
  - `test_api_returns_429_with_retry_after` — FastAPI test client.
  - `test_cap_configurable_per_app` — `cfg.runtime.max_concurrent_sessions = 2` enforced.

- [ ] **2. Implementation:**

```python
# runtime/service.py
class SessionCapExceeded(RuntimeError):
    def __init__(self, cap: int) -> None:
        super().__init__(f"max concurrent sessions reached ({cap})")
        self.cap = cap
```

```yaml
# config/config.yaml — framework default block
runtime:
  state_class: examples.incident_management.state:IncidentState
  max_concurrent_sessions: 8
```

- [ ] **3. Run** → green.

- [ ] **4. Commit** — `feat(service): max_concurrent_sessions cap with fail-fast (P3-G)`

---

### P3-H — FastAPI endpoints rebuilt over `OrchestratorService`

Replace per-session manual orchestration in `api.py` with thin handlers over `OrchestratorService`. Old routes stay for one phase as deprecation shims.

**Files:**
- Modify: `src/runtime/api.py`
- Create: `tests/test_api_sessions.py`

**Endpoints:**

| Method | Path | Body / params | Response |
|---|---|---|---|
| `POST` | `/sessions` | `{state init kwargs}` | `201 {session_id}` or `429` if cap hit |
| `GET`  | `/sessions` | `?status=running\|awaiting_input\|...` | `[ActiveSession]` (in-flight only) |
| `GET`  | `/sessions/{id}` | — | `Session` (full state, `404` if missing) |
| `DELETE` | `/sessions/{id}` | — | `200 {stopped: bool}` |
| `POST` | `/sessions/{id}/resume` | `{decision}` | `204` |

**Steps:**

- [ ] **1. Add tests** using `httpx.AsyncClient` against `build_app(cfg)`.
  - `test_post_sessions_returns_201_with_id`
  - `test_get_sessions_lists_active_only`
  - `test_get_session_by_id_returns_state`
  - `test_delete_session_cancels_and_marks_stopped`
  - `test_post_resume_unblocks_interrupt`
  - `test_post_sessions_429_at_cap`
  - `test_legacy_investigate_route_still_works_with_deprecation_header`

- [ ] **2. Implementation:** lifespan opens/closes the singleton `OrchestratorService` on `app.state.service`. Handlers call sync methods directly (FastAPI tolerates sync handlers; the underlying work happens on the orch loop, not uvicorn's).

- [ ] **3. Run** → green.

- [ ] **4. Commit** — `feat(api): /sessions endpoints over OrchestratorService (P3-H)`

---

### P3-I — Streamlit sidebar: status badges + active-session merge

Sidebar shows a unified list. Each entry has a status badge:
- `running` — pulse dot, color = accent.
- `awaiting_input` — amber.
- `done` — neutral.
- `stopped` — muted with strikethrough.
- `error` — red dot.

Source: `service.list_active_sessions()` ∪ `store.list_recent_sessions(status_filter)`. In-flight wins on duplicate ids.

**Files:**
- Modify: `examples/incident_management/ui.py` (`render_sidebar`)
- Create: `examples/incident_management/service_factory.py` — process-singleton getter using `st.cache_resource(ttl=None)` so the service survives reruns

**Steps:**

- [ ] **1. Add a thin unit test** for the merge logic in isolation (move it out of `render_sidebar` into a pure function `_merge_session_lists(active, recent)`):

```python
def test_merge_prefers_active_status_over_recent():
    active = [ActiveSession("s1", "running", "...", "entry")]
    recent = [{"id": "s1", "status": "new", ...}, {"id": "s2", "status": "done", ...}]
    merged = _merge_session_lists(active, recent)
    assert merged[0]["id"] == "s1" and merged[0]["status"] == "running"
    assert {m["id"] for m in merged} == {"s1", "s2"}
```

- [ ] **2. UI changes:**
  - Status badge rendered via small CSS pill (`.status-badge[data-status="running"]`) — no extra deps.
  - Honour `prefers-reduced-motion` for the running pulse (CSS `@media`).
  - Existing "click to open detail" behaviour unchanged.

- [ ] **3. `service_factory.py`:**

```python
import streamlit as st
from runtime.service import OrchestratorService
from runtime.config import load_config
from examples.incident_management.config import load_incident_app_config

@st.cache_resource(ttl=None)
def get_service() -> OrchestratorService:
    cfg = load_config()
    svc = OrchestratorService[cfg.runtime.state_class](cfg)
    svc.start()
    # NOTE: relying on Streamlit's process lifetime to clean up.
    # No atexit hook — the daemon thread + AsyncExitStack tolerate hard shutdowns.
    return svc
```

- [ ] **4. Run** smoke (`python -m examples.incident_management`); start two sessions in sequence, observe both in sidebar with `running` then `done` badges.

- [ ] **5. Commit** — `feat(ui): sidebar status badges + active-session merge (P3-I)`

---

### P3-J — Streamlit detail pane: 1–2 s polling while in-flight

The detail pane already reads the session by id. Replace the "load once, render" path with: if `status in {running, awaiting_input}`, schedule a `st.experimental_rerun()` (or the modern API `st.rerun()`) after `time.sleep(1.5)`. If status is terminal (`done | stopped | error`), do not poll.

**Files:**
- Modify: `examples/incident_management/ui.py` (`render_incident_detail` → `render_session_detail` per Phase 2 rename)

**Steps:**

- [ ] **1. Behaviour:** add a `_should_poll(state.status)` helper — pure function, easy to test.

- [ ] **2. Tests:** unit test for the helper:

```python
def test_should_poll_only_for_in_flight():
    assert _should_poll("running") is True
    assert _should_poll("awaiting_input") is True
    assert _should_poll("new") is True
    assert _should_poll("done") is False
    assert _should_poll("stopped") is False
    assert _should_poll("error") is False
```

- [ ] **3. UI:** wrap the polling block so it can be disabled in tests:

```python
def render_session_detail(svc: OrchestratorService, sid: str) -> None:
    state = svc.get_session_state(sid)
    if state is None:
        st.warning("Session not found."); return
    _render_state(state)
    if _should_poll(state.status) and not st.session_state.get("_disable_poll"):
        time.sleep(1.5)
        st.rerun()
```

- [ ] **4. Smoke:** start a session, watch agent transitions update without manual refresh; reach `awaiting_input`, click resume button, watch run continue.

- [ ] **5. Commit** — `feat(ui): poll session detail every 1.5s while in-flight (P3-J)`

---

### P3-K — Concurrency integration test (3 sessions in parallel)

End-to-end test: start 3 sessions back-to-back, each running a deterministic fake graph that takes ~1 s, verify all reach `done`, sidebar list reflects them as they progress, no DB lock errors.

**Files:**
- Create: `tests/test_concurrent_sessions_e2e.py`
- Create: `tests/fixtures/fake_state.py` — minimal `Session` subclass + `Orchestrator` test config that uses an in-memory MCP server

**Test sketch:**

```python
@pytest.mark.timeout(30)
def test_three_sessions_complete_concurrently(tmp_path):
    cfg = _build_test_cfg(tmp_path, max_concurrent=4)
    svc = OrchestratorService[FakeState](cfg)
    svc.start()
    try:
        ids = [svc.start_session(label=f"s-{i}") for i in range(3)]

        # Within 200ms all three should appear active.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            active = {a.session_id for a in svc.list_active_sessions()}
            if set(ids) <= active: break
            time.sleep(0.02)
        else:
            pytest.fail(f"three sessions never all active: saw {active}")

        # Wait for all to terminate.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not svc.list_active_sessions(): break
            time.sleep(0.05)
        else:
            pytest.fail("sessions did not finish in 10s")

        for sid in ids:
            state = svc.get_session_state(sid)
            assert state.status == "done"
            assert len(state.tool_calls) > 0  # graph ran
    finally:
        svc.stop()

@pytest.mark.timeout(30)
def test_stop_one_of_three_does_not_disturb_others(tmp_path):
    cfg = _build_test_cfg(tmp_path, max_concurrent=4, slow=True)  # ~5s graph
    svc = OrchestratorService[FakeState](cfg); svc.start()
    try:
        ids = [svc.start_session(label=f"s-{i}") for i in range(3)]
        time.sleep(0.3)
        assert svc.stop_session(ids[1]) is True

        # The other two complete.
        deadline = time.monotonic() + 15.0
        survivors = {ids[0], ids[2]}
        while time.monotonic() < deadline:
            done = {sid for sid in survivors if svc.get_session_state(sid).status == "done"}
            if done == survivors: break
            time.sleep(0.05)
        else:
            pytest.fail("survivors did not complete")

        assert svc.get_session_state(ids[1]).status == "stopped"
    finally:
        svc.stop()
```

- [ ] **1. Run** → green.

- [ ] **2. Run twice** to catch flakiness from R1 (stress concurrent SQLite WAL writes from N tasks).

- [ ] **3. Commit** — `test(service): concurrent sessions e2e — 3-in-parallel + cancel-one (P3-K)`

---

### P3-L — Final verification + README update

Sign-off task. Not strictly TDD.

**Steps:**

- [ ] **1. Full suite** — `pytest tests/ -v`. Expected: Phase 2 baseline + ~30 new tests.

- [ ] **2. Bundle build** — `python scripts/build_single_file.py` → bundle still produces `dist/app.py` and `dist/apps/incident-management.py`. New modules (`service.py`, `service_loop.py`, `service_registry.py`, `mcp_pool.py`) are included.

- [ ] **3. Grep / shape checks (must all return zero):**

```bash
# No lingering "one orchestrator per call" pattern in api.py
grep -n 'await Orchestrator.create' src/runtime/api.py     # → 0 lines
# Service is the only opener of MCPClientPool
grep -rn 'MCPClientPool(' src/ examples/ | grep -v 'service.py' | wc -l   # → 0
# No nested asyncio.run / asyncio.new_event_loop in app code
grep -rn 'asyncio.run\|new_event_loop' src/runtime/ examples/ | grep -v service_loop.py
```

- [ ] **4. Smoke** — start the example app:
  - Open two browser tabs to the same Streamlit URL.
  - Tab A: start session "DB latency".
  - Tab B (within 1 s): start session "Auth errors".
  - Both appear in both sidebars with `running` badge; both reach `awaiting_input` independently; resume each from its own tab.
  - Hit the cap (set `max_concurrent_sessions: 2`); 3rd `start_session` shows toast "session cap reached (2)".

- [ ] **5. Update `examples/incident_management/README.md`** — append a "Concurrency" section: how many sessions can run, where to tune the cap, what `Stop` does to partial work.

- [ ] **6. Commit** — `chore(p3): final verification + README updates (P3-L)`

---

## 5. Sequencing and Dependencies

```
                          P3-A  (service surface scaffold)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
      P3-B               P3-C                 P3-D
   (background-          (MCPClient-        (TaskRegistry)
    thread loop)          Pool)
        └───────────┬──────┴──────┬───────────┘
                    ▼             ▼
                 P3-E (compose: start/stop/start_session)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      P3-F        P3-G        P3-H
   (cancel +     (resource    (FastAPI
    stopped)      cap)         /sessions)
        └───────────┬───────────┘
                    ▼
                  P3-I (sidebar UI)
                    │
                    ▼
                  P3-J (detail-pane polling)
                    │
                    ▼
                  P3-K (e2e concurrency test)
                    │
                    ▼
                  P3-L (verify + readme)
```

**Parallelisable.** Within each level shown above, tasks are independent and can be split across subagents:
- P3-B / P3-C / P3-D — three subagents.
- P3-F / P3-G / P3-H — three subagents (each touches `service.py` but in non-overlapping methods; coordinate via small merge after).

**Hard sequence.** P3-E is the merge point; nothing after it can start until E lands. P3-K depends on every preceding implementation task.

---

## 6. Risks and Mitigations

**R1 — Streamlit reruns vs. background asyncio loop: race / staleness.**
The Streamlit script restarts top-to-bottom on every interaction. The orchestrator service must be a **process-level singleton**, not a per-rerun object — otherwise we'd be opening / closing a loop and a FastMCP pool on every click. Two-pronged mitigation:
1. **Singleton via `st.cache_resource(ttl=None)`** in `service_factory.get_service()`. Streamlit's resource cache survives reruns. The `BackgroundLoop` daemon thread keeps running.
2. **All cross-thread communication is via `asyncio.run_coroutine_threadsafe`.** No raw `queue.SimpleQueue`, no shared dict mutated from both threads. The registry's mutators are coroutines that run on the loop thread; the snapshot read is also a coroutine; the bridge gives us a `concurrent.futures.Future` for sync waiting. This is the only thread-safe primitive we need.
3. **Watch out for Streamlit's session_state interactions.** The script writes user choices into `st.session_state["selected_session"]`; that's per-Streamlit-session and never read by the orchestrator. No race surface.

Verification: run two browser tabs; rapidly click between active sessions; confirm no duplicate task creation, no "loop closed" errors in the journal.

**R2 — SQLite WAL under concurrent writes from N asyncio tasks.**
Phase 2 enables WAL on both the metadata engine and the LangGraph saver. WAL allows concurrent readers + one writer per process. N asyncio tasks all live in the same process, but they are **serialised on the loop thread**, so at any wall-clock instant only one task is executing Python. SQLite's writer-lock contention is therefore minimal: writes interleave at `await` boundaries.
- Mitigation: keep transaction scope short (`with engine.begin():` around the smallest possible block — Phase 2 SessionStore already does this).
- Verification: P3-K stress test with 3 parallel sessions, plus a focused `test_sqlite_wal_under_concurrent_session_writes` that hammers `SessionStore.upsert` from 8 tasks for 2 seconds and asserts no `database is locked` errors.
- If we ever see contention in real deployments, escalate to Postgres (Phase 2's `PostgresSaver` is already wired). SQLite is the dev/single-user backend; concurrent multi-user goes Postgres.

**R3 — FastMCP client thread-safety under shared pool.**
One FastMCP client transports many concurrent tool calls. stdio transport multiplexes by request id and is thread-safe-ish, but the underlying `subprocess` pipe is **not** safe to interleave writes on. Without a lock, two `await client.call_tool(...)` from different tasks can scramble bytes on the wire.
- Mitigation: `MCPClientPool` exposes one `asyncio.Lock` per server (`pool.lock_for(name)`). The orchestrator's `_invoke_tool` wraps the call in `async with pool.lock_for(srv): ...`. This serialises tool calls **per server**, not globally — different MCP servers proceed in parallel.
- Cost: a slow tool on server X blocks other sessions' calls to server X. Documented; if it becomes a hot path, the answer is per-session client (more complex; deferred).
- Verification: `test_mcp_pool_concurrency.py` — 4 tasks call the same fake server's `slow_tool` (200 ms), assert wall-clock ≥ 800 ms (serialised), then call different servers and assert <300 ms (parallel).

**R4 — Cancellation: partial state when session is stopped mid-tool-call.**
When `task.cancel()` fires while the task is `await`ing a tool call:
1. `asyncio` injects `CancelledError` at the next checkpoint.
2. If the await is inside `async with pool.lock_for(srv):`, the lock releases cleanly.
3. The graph node never sees the tool result; no partial result written to `tool_calls`.
4. Already-recorded tool_calls (from prior nodes) are persisted in the DB. We do **not** roll them back. The session row goes `status="stopped"`, `updated_at=now()`.

What's recorded:
- `tool_calls`: every tool call that **completed and was persisted** before cancellation. Preserved.
- `agents_run`: any agents whose end-of-turn write completed. Preserved.
- `pending_intervention`: cleared (since we won't resume).
- `findings`: whatever was written before stop. Preserved.

Documented in P3-F task code; tested by `test_stop_session_preserves_recorded_tool_calls`.

**R5 — Resource cap behaviour: fail vs. queue.**
Decision: **fail fast** with `SessionCapExceeded` → HTTP 429 / Streamlit toast.
- Pros: simple, no admission queue, no notification channel needed, no unfair scheduling. Aligns with the framework's "make backpressure visible" stance.
- Cons: user has to retry. Acceptable for an interactive use case where 8 concurrent sessions already cover most teams.
- Roadmap-aligned alternative: bounded `asyncio.Queue` of pending starts + a "queued" status. Documented as Phase 4+ work; not in scope here. If the user asks for queueing during P3 review, the change is local to `OrchestratorService.start_session` plus a new `Session.status` value; no other code is affected.

**R6 — Process restart leaves DB rows in `running` state.**
If the host process dies while sessions are mid-flight, their rows are stuck on `running`. The in-memory registry is gone, so `list_active_sessions()` won't show them, but `list_recent_sessions(status="running")` will — confusingly.
- Mitigation: at `OrchestratorService.start()`, sweep `SessionStore.list_active_sessions()` (DB-side, status in `{running, awaiting_input}`) and reconcile:
  - For each row not in our (empty) in-memory registry, if `status="running"` and last `updated_at` is older than 60 s, set `status="error"` with note `"abandoned (process restart)"`.
  - For `status="awaiting_input"`, leave alone — those are legitimately resumable via the Phase 2 checkpointer.
- Implementation falls into P3-E (`start()`); test `test_start_reconciles_orphaned_running_rows` covers it.

**R7 — `time.sleep(1.5)` in Streamlit detail pane blocks the script thread.**
This is fine — Streamlit's script thread is per-rerun, not shared. While we sleep, the user's tab shows the last rendered state. Not an issue for CPU; might be confusing for the user if they click away. If users complain, swap to `st.empty().container()` with `st.experimental_fragment` (Streamlit ≥1.32). Documented; not implemented in P3.

---

## 7. Done Criteria

A task is not done until **all** apply (scaled per `~/.claude/CLAUDE.md` §6):

- [ ] All Phase 2 tests still pass (no regression).
- [ ] All new tests in §4 pass — including the 3-session concurrency e2e (P3-K).
- [ ] `pytest tests/ -v` clean. Linters / type checks clean (treat warnings as errors).
- [ ] `python scripts/build_single_file.py` produces a working bundle including the new modules.
- [ ] `npm`-equivalent — Python `pip-audit` clean (per `rules/security.md`); High/Critical = block.
- [ ] Concurrency cap enforced and surfaced as 429 / toast.
- [ ] `stop_session` cancels mid-flight and preserves recorded tool_calls.
- [ ] Two-tab smoke (P3-L step 4) works end-to-end.
- [ ] `examples/incident_management/README.md` has a Concurrency section.
- [ ] No `asyncio.run` or `new_event_loop` outside `service_loop.py`.
- [ ] No FastMCP client built outside `MCPClientPool`.
- [ ] R6 reconciliation sweep runs at startup and is covered by a test.
- [ ] No flaky tests (retry-loop in CI is forbidden — fix or quarantine).
- [ ] Branch ready for review with each P3-x landing as a single conventional commit.

---

## 8. Open Questions (deferred, not blocking)

1. Should `OrchestratorService` expose a context-manager API (`with service:`) for tests? Probably yes; cheap to add; deferred unless a test explicitly needs it.
2. When the cap is hit, should we surface "estimated wait time" (median session duration)? Useful but adds telemetry. Defer.
3. Cross-process orchestrator (Streamlit + uvicorn separately, IPC over a Unix socket) — better isolation, harder to deploy. Revisit if single-process memory becomes a problem.
4. Per-tenant concurrency caps (e.g. `max_concurrent_sessions_per_user`)? Out of scope; would require auth context which is also out of scope.
5. SSE / WebSocket streaming for the detail pane (instead of polling) — postponed to a UI iteration after P3.
