"""Monitor agent kind — out-of-band scheduled observer (P6-E,F).

A monitor skill runs **outside** any session graph. The orchestrator
owns one :class:`MonitorRunner` (a singleton, per the locked decision
in §2.1 of the plan) which schedules registered monitor skills on a
small bounded :class:`concurrent.futures.ThreadPoolExecutor` (R6).
Each tick:

1. Calls every tool name in ``observe`` via the supplied callable
   (``observe_fn``); aggregates results into one dict keyed by tool.
2. Evaluates ``emit_signal_when`` against the observation using the
   stdlib safe-eval evaluator (R7).
3. If true, looks up ``trigger_target`` in the supplied trigger
   registry / fire callback and fires it with the observation as the
   payload.

APScheduler is intentionally *not* a dependency: the air-gapped target
env doesn't ship it (see ``rules/build.md``). We get away with a tiny
single-threaded scheduler thread because monitor schedules are coarse
(minute-resolution cron) and tool calls are dispatched into the
executor; the scheduler thread itself never blocks on tool I/O.
"""
from __future__ import annotations

import ast
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from runtime.skill import Skill, _validate_safe_expr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe-eval evaluator
# ---------------------------------------------------------------------------


class SafeEvalError(Exception):
    """Raised when a supposedly-validated expression fails to evaluate."""


def safe_eval(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate ``expr`` against ``ctx`` after a fresh AST whitelist check.

    The skill loader validates ``emit_signal_when`` at parse time; we
    re-validate here on every call to keep the threat model defensive
    against any future code path that might construct a Skill bypassing
    the loader's validators.
    """
    _validate_safe_expr(expr, source="monitor.emit_signal_when")
    code = compile(expr, "<safe-eval>", "eval")
    try:
        return eval(code, {"__builtins__": {}}, ctx)  # noqa: S307 — AST-whitelisted
    except Exception as exc:  # noqa: BLE001
        raise SafeEvalError(f"emit_signal_when {expr!r} raised: {exc}") from exc


# ---------------------------------------------------------------------------
# Cron parsing (minute-resolution; matches Skill._validate_cron grammar)
# ---------------------------------------------------------------------------


def _expand_cron_field(field: str, lo: int, hi: int) -> set[int]:
    """Expand a single cron field into the set of int values it matches.

    Supports ``*``, ``*/n``, ``a``, ``a-b``, ``a-b/n``, and
    comma-separated combinations of those — the grammar accepted by
    :func:`runtime.skill._validate_cron`.
    """
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            base, _, step_s = part.partition("/")
            step = int(step_s)
        else:
            base = part
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            a, _, b = base.partition("-")
            start, end = int(a), int(b)
        else:
            v = int(base)
            start, end = v, v
        out.update(range(start, end + 1, step))
    return {v for v in out if lo <= v <= hi}


def _cron_matches(expr: str, when: datetime) -> bool:
    """Return True if the given datetime satisfies the 5-field cron expression.

    Fields: minute, hour, day-of-month, month, day-of-week (0=Mon..6=Sun
    — Python's ``datetime.weekday()`` convention; cron itself uses
    0=Sun, but for our minute-resolution scheduler the convention only
    needs to be internally consistent and documented).
    """
    minute, hour, dom, month, dow = expr.split()
    return (
        when.minute in _expand_cron_field(minute, 0, 59)
        and when.hour in _expand_cron_field(hour, 0, 23)
        and when.day in _expand_cron_field(dom, 1, 31)
        and when.month in _expand_cron_field(month, 1, 12)
        and when.weekday() in _expand_cron_field(dow, 0, 6)
    )


# ---------------------------------------------------------------------------
# Monitor callable factory
# ---------------------------------------------------------------------------


def make_monitor_callable(
    *,
    skill: Skill,
    observe_fn: Callable[[str], Any],
    fire_trigger: Callable[[str, dict[str, Any]], None],
) -> Callable[[], None]:
    """Build the callable a :class:`MonitorRunner` runs per tick (P6-F).

    ``observe_fn(tool_name)`` is the seam through which the runner
    invokes a tool. Production wires this to the orchestrator's MCP
    tool registry; tests wire it to deterministic stubs.

    ``fire_trigger(name, payload)`` is the seam through which the
    runner fires a Phase-5 trigger. Production wires this to the
    trigger registry; tests wire it to a recorder.

    The returned callable is intentionally synchronous and exception-
    safe: a failed ``observe_fn`` or ``fire_trigger`` is logged and
    swallowed so one bad monitor cannot stall the runner (R6).
    """
    if skill.kind != "monitor":
        raise ValueError(
            f"make_monitor_callable called with non-monitor skill "
            f"{skill.name!r} (kind={skill.kind!r})"
        )

    def tick() -> None:
        observation: dict[str, Any] = {}
        for tool_name in skill.observe:
            try:
                observation[tool_name] = observe_fn(tool_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: observe tool %r raised %s; skipping",
                    skill.name, tool_name, exc,
                )
                observation[tool_name] = None
        ctx = {
            "observation": observation,
            "obs": observation,
        }
        try:
            should_emit = bool(safe_eval(skill.emit_signal_when or "False", ctx))
        except SafeEvalError as exc:
            logger.warning("monitor %s: %s", skill.name, exc)
            return
        if not should_emit:
            return
        try:
            fire_trigger(skill.trigger_target or "", {
                "monitor": skill.name,
                "observation": observation,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "monitor %s: fire_trigger(%s) raised %s",
                skill.name, skill.trigger_target, exc,
            )

    return tick


# ---------------------------------------------------------------------------
# MonitorRunner — orchestrator-level singleton
# ---------------------------------------------------------------------------


class _RegisteredMonitor:
    __slots__ = ("skill", "callable_", "next_run_ts")

    def __init__(self, skill: Skill, callable_: Callable[[], None]) -> None:
        self.skill = skill
        self.callable_ = callable_
        # Track the last *scheduled* minute we fired so we never fire
        # twice for the same wall-clock minute even if the scheduler
        # thread oversleeps.
        self.next_run_ts: datetime | None = None


class MonitorRunner:
    """Owns a bounded thread pool and a scheduler thread that ticks
    registered monitor skills on their cron schedules (P6-E).

    Exactly one ``MonitorRunner`` exists per ``OrchestratorService``
    instance (per the §2.1 locked decision); the runner is built at
    service startup and shut down at service teardown.

    Concurrency: each tick is dispatched to the
    :class:`~concurrent.futures.ThreadPoolExecutor` so the scheduler
    thread itself never blocks on a slow ``observe`` tool. The pool
    size defaults to ``4`` (R6); each tick has a per-monitor timeout
    sourced from the skill's ``tick_timeout_seconds``.
    """

    def __init__(
        self,
        *,
        observe_fn: Callable[[str], Any],
        fire_trigger: Callable[[str, dict[str, Any]], None],
        max_workers: int = 4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._observe_fn = observe_fn
        self._fire_trigger = fire_trigger
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="monitor",
        )
        self._monitors: dict[str, _RegisteredMonitor] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Injection seam for tests; default uses real wall-clock UTC.
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ----- registration -----

    def register(self, skill: Skill) -> None:
        if skill.kind != "monitor":
            raise ValueError(
                f"MonitorRunner.register: skill {skill.name!r} kind="
                f"{skill.kind!r} (expected 'monitor')"
            )
        callable_ = make_monitor_callable(
            skill=skill,
            observe_fn=self._observe_fn,
            fire_trigger=self._fire_trigger,
        )
        with self._lock:
            if skill.name in self._monitors:
                raise ValueError(f"monitor {skill.name!r} already registered")
            self._monitors[skill.name] = _RegisteredMonitor(skill, callable_)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._monitors.pop(name, None)

    def registered(self) -> list[str]:
        with self._lock:
            return sorted(self._monitors.keys())

    # ----- lifecycle -----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MonitorRunner",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """Halt the scheduler thread and shut down the executor.

        ``wait=True`` (default) blocks up to ``timeout`` seconds for
        in-flight ticks to drain. Daemon threads are still joined so
        pytest fixture teardown is deterministic.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and wait:
            thread.join(timeout=timeout)
        self._executor.shutdown(wait=wait)
        self._thread = None

    # ----- test hook -----

    def tick_once(self, when: datetime | None = None) -> None:
        """Fire any monitors whose cron expression matches ``when``.

        Useful in tests where freezing wall-clock time is awkward; the
        production scheduler loop calls this internally too.
        """
        when = when or self._clock()
        # Truncate to the minute so identical seconds within a minute
        # don't fire the same monitor twice.
        minute = when.replace(second=0, microsecond=0)
        with self._lock:
            entries = list(self._monitors.values())
        for entry in entries:
            try:
                if not _cron_matches(entry.skill.schedule or "* * * * *", minute):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: cron parse failed (%s); skipping tick",
                    entry.skill.name, exc,
                )
                continue
            if entry.next_run_ts == minute:
                # Already fired this minute; idempotent on oversleep.
                continue
            entry.next_run_ts = minute
            self._dispatch(entry)

    def _dispatch(self, entry: _RegisteredMonitor) -> None:
        timeout = float(entry.skill.tick_timeout_seconds or 30.0)
        future = self._executor.submit(entry.callable_)

        def _wait_and_log() -> None:
            try:
                future.result(timeout=timeout)
            except FuturesTimeout:
                logger.warning(
                    "monitor %s: tick exceeded %.1fs timeout",
                    entry.skill.name, timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: tick raised %s", entry.skill.name, exc,
                )

        # Watcher runs on a side thread so the scheduler loop never
        # blocks waiting for a slow tick — the executor handles
        # parallelism, the watcher handles per-tick timeout reporting.
        threading.Thread(
            target=_wait_and_log,
            name=f"monitor-watch:{entry.skill.name}",
            daemon=True,
        ).start()

    # ----- scheduler loop -----

    def _run(self) -> None:
        """Single-threaded scheduler. Wakes once per second, fires
        any monitor whose cron expression matches the current minute,
        marks each fired monitor for the minute so we never fire
        twice if we oversleep.
        """
        while not self._stop.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001 — never crash the loop
                logger.warning("MonitorRunner loop error: %s", exc)
            # Sleep with frequent wakeups so stop() returns promptly.
            self._stop.wait(timeout=1.0)


__all__ = [
    "MonitorRunner",
    "SafeEvalError",
    "make_monitor_callable",
    "safe_eval",
]
