---
phase: 01-concurrency-foundation
plan: 01
subsystem: infra
tags: [asyncio, locks, concurrency, fastapi, streamlit, session-management]

# Dependency graph
requires: []
provides:
  - SessionBusy(RuntimeError) exception with session_id attribute
  - SessionLockRegistry.is_locked(session_id) non-blocking predicate
  - Per-session task-reentrant lock held across full graph turn including HITL pause
  - HTTP 429 + Retry-After:1 on all three session-start/approval API callsites
  - UI retry hint on SessionBusy at investigation form submission
  - locks.py inlined into dist/ bundles
affects:
  - 01-02-concurrency-foundation  # approval_watchdog retry path uses SessionBusy

# Tech tracking
tech-stack:
  added: []
  patterns:
    - class-name match for exception handling in api.py (no hard import at module load)
    - task-reentrant asyncio lock with is_locked() fail-fast check before acquire()
    - D-09: dist/ regeneration in same atomic commit as src/ changes

key-files:
  created: []
  modified:
    - src/runtime/locks.py
    - src/runtime/service.py
    - src/runtime/api.py
    - src/runtime/ui.py
    - tests/test_session_lock.py
    - scripts/build_single_file.py
    - dist/app.py
    - dist/ui.py
    - dist/apps/incident-management.py
    - dist/apps/code-review.py

key-decisions:
  - "D-01: Lock held across entire graph turn including LangGraph interrupt() HITL pause"
  - "D-02: Single acquire site inside _run() closure, not at start_session() entry"
  - "D-03: Fail-fast contention — SessionBusy raised, not queued"
  - "D-04: Reads stay lock-free throughout"
  - "D-09: dist/ regenerated in same atomic commit as src/ changes"
  - "D-10: Direct atomic commit on refactor/prompt-vs-code-remediation branch"
  - "D-15: Slot eviction deferred to v2 — TODO comment added to _slots dict"
  - "D-16 (location override): SessionBusy raised inside _run() at acquire site, NOT at start_session() entry — start_session() mints fresh session_id so no pre-existing lock slot exists"
  - "D-17: EventLog stays lock-free"
  - "locks.py added to RUNTIME_MODULE_ORDER in build_single_file.py (was missing)"

patterns-established:
  - "Exception class-name matching pattern: e.__class__.__name__ in ('SessionCapExceeded', 'SessionBusy') — avoids hard import at module load time"
  - "is_locked() + acquire() pattern: check is_locked() first for fail-fast, then async with acquire() for the body — non-contending in steady state"
  - "asyncio_mode=auto: new async tests in tests/ do NOT need @pytest.mark.asyncio decorator"

requirements-completed:
  - PVC-01

# Metrics
duration: ~35min
completed: 2026-05-06
---

# Phase 01: Concurrency Foundation — Plan 01 Summary

**Per-session task-reentrant asyncio lock with fail-fast SessionBusy, HTTP 429/Retry-After mapping at all three API callsites, UI retry hint, and locks.py bundled into dist/**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-06T08:00:00Z
- **Completed:** 2026-05-06T08:35:00Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments
- `SessionBusy(RuntimeError)` exception and `is_locked()` predicate added to `locks.py`; 5 new unit tests pass (838 total)
- `service.py._run()` wrapped with per-session lock acquire; fail-fast contention check via `is_locked()` before `acquire()`
- All three FastAPI callsites (`/investigate`, `POST /sessions`, approval submission) now map `SessionBusy` → HTTP 429 + `Retry-After: 1`; UI shows `st.warning` + early return
- `locks.py` added to `RUNTIME_MODULE_ORDER` in `build_single_file.py` (was omitted); all four dist bundles regenerated with `SessionBusy`, `is_locked`, `_locks.acquire` present

## Task Commits

All tasks committed atomically in a single commit per D-09/D-10:

1. **Tasks 1-3: All changes** - `ea43964` (feat)

## Files Created/Modified
- `src/runtime/locks.py` - Added `SessionBusy` class, `is_locked()` predicate, TODO(v2) eviction note
- `src/runtime/service.py` - Wrapped `_run()` body with `async with orch._locks.acquire(session_id):`; `is_locked()` fail-fast guard
- `src/runtime/api.py` - Extended class-name match at 2 existing handlers + 1 new handler at approval submission callsite
- `src/runtime/ui.py` - SessionBusy try/except at `asyncio.run()` investigation form path
- `tests/test_session_lock.py` - 5 new tests for `is_locked()` + `SessionBusy` (no `@pytest.mark.asyncio` per asyncio_mode=auto)
- `scripts/build_single_file.py` - Added `(RUNTIME_ROOT, "locks.py")` before `orchestrator.py` in `RUNTIME_MODULE_ORDER`
- `dist/app.py`, `dist/ui.py`, `dist/apps/incident-management.py`, `dist/apps/code-review.py` - Regenerated with locks.py inlined

## Decisions Made
- D-16 location override confirmed: `SessionBusy` raised inside `_run()` not at `start_session()` entry — `start_session()` mints a fresh `session_id` so there is no pre-existing lock slot to check
- `locks.py` was missing from `RUNTIME_MODULE_ORDER` in the build script — added before `orchestrator.py` which instantiates `SessionLockRegistry`
- Used `is_locked()` as a pre-check before `acquire()` to satisfy D-03 fail-fast without blocking; the acquire() itself is non-contending in the steady state

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] locks.py missing from build_single_file.py RUNTIME_MODULE_ORDER**
- **Found during:** Task 3 (dist/ regeneration verification)
- **Issue:** `def is_locked`, `class SessionBusy` absent from `dist/app.py` after initial build; `locks.py` was not listed in `RUNTIME_MODULE_ORDER`
- **Fix:** Added `(RUNTIME_ROOT, "locks.py")` to `RUNTIME_MODULE_ORDER` before `orchestrator.py`; rebuilt all four bundles
- **Files modified:** `scripts/build_single_file.py`, all four dist files
- **Verification:** `grep -c "def is_locked" dist/app.py` → 1; `grep -c "class SessionBusy" dist/app.py` → 1; `grep -c "_locks\.acquire" dist/app.py` → 2
- **Committed in:** `ea43964` (same atomic commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — missing bundle entry)
**Impact on plan:** Essential fix for D-09 compliance. No scope creep.

## Issues Encountered
None beyond the locks.py bundle omission documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Per-session lock foundation complete; `SessionBusy` exception available for 01-02
- 01-02 (`approval_watchdog.py` retry path) can import `SessionBusy` from `runtime.locks` without circular import risk
- All 838 tests pass; ruff clean on all modified files

---
*Phase: 01-concurrency-foundation*
*Completed: 2026-05-06*
