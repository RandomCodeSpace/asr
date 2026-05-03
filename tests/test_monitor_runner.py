"""Tests for the monitor agent kind + ``MonitorRunner``.

Covers:

* Cron parsing matches expected datetimes.
* A monitor whose emit expression is true fires the configured trigger
  on each scheduled tick; the trigger receives the observation payload.
* A failing observe tool is logged and swallowed; the next tick still
  runs.
* ``MonitorRunner.stop()`` halts further ticks and shuts down the
  executor.
* The runner refuses to register non-monitor skills.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from runtime.agents.monitor import (
    MonitorRunner,
    SafeEvalError,
    _cron_matches,
    make_monitor_callable,
    safe_eval,
)
from runtime.skill import Skill


def _monitor(**overrides) -> Skill:
    base = dict(
        name="watch", description="d", kind="monitor",
        schedule="*/5 * * * *",
        observe=["error_rate"],
        emit_signal_when="observation['error_rate'] > 0.05",
        trigger_target="incident_high_error_rate",
        tick_timeout_seconds=5.0,
    )
    base.update(overrides)
    return Skill(**base)


# ---------------------------------------------------------------------------
# Cron matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr,when,expected", [
    ("* * * * *", datetime(2026, 5, 3, 12, 30, tzinfo=timezone.utc), True),
    ("*/5 * * * *", datetime(2026, 5, 3, 12, 5, tzinfo=timezone.utc), True),
    ("*/5 * * * *", datetime(2026, 5, 3, 12, 6, tzinfo=timezone.utc), False),
    ("0 9-17 * * *", datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc), True),
    ("0 9-17 * * *", datetime(2026, 5, 3, 18, 0, tzinfo=timezone.utc), False),
])
def test_cron_matches(expr, when, expected):
    assert _cron_matches(expr, when) is expected


# ---------------------------------------------------------------------------
# safe_eval
# ---------------------------------------------------------------------------


def test_safe_eval_basic():
    assert safe_eval("a > 1 and b == 'x'", {"a": 2, "b": "x"}) is True
    assert safe_eval("a > 1", {"a": 0}) is False


def test_safe_eval_rejects_runtime_dangerous():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')", {})


def test_safe_eval_runtime_error_wrapped():
    with pytest.raises(SafeEvalError):
        safe_eval("a / b", {"a": 1, "b": 0})


# ---------------------------------------------------------------------------
# make_monitor_callable
# ---------------------------------------------------------------------------


def test_callable_fires_trigger_when_condition_true():
    fired = []

    def observe(name):
        return {"error_rate": 0.10}["error_rate"] if name == "error_rate" else None

    def fire(name, payload):
        fired.append((name, payload))

    skill = _monitor()
    tick = make_monitor_callable(
        skill=skill, observe_fn=observe, fire_trigger=fire,
    )
    tick()
    assert fired and fired[0][0] == "incident_high_error_rate"
    assert fired[0][1]["monitor"] == "watch"
    assert fired[0][1]["observation"] == {"error_rate": 0.10}


def test_callable_no_fire_when_condition_false():
    fired = []
    skill = _monitor()
    tick = make_monitor_callable(
        skill=skill,
        observe_fn=lambda name: 0.01,
        fire_trigger=lambda *a: fired.append(a),
    )
    tick()
    assert fired == []


def test_callable_swallows_observe_failure():
    skill = _monitor()
    fired = []

    def boom(name):
        raise RuntimeError("upstream down")

    tick = make_monitor_callable(
        skill=skill, observe_fn=boom,
        fire_trigger=lambda *a: fired.append(a),
    )
    # No exception should escape; condition becomes False because
    # observation[error_rate] is None > 0.05 is also a TypeError swallowed
    # by safe_eval -> no fire.
    tick()
    assert fired == []


def test_make_monitor_callable_rejects_non_monitor_skill():
    s = Skill(name="x", description="d", kind="responsive", system_prompt="hi")
    with pytest.raises(ValueError, match="non-monitor"):
        make_monitor_callable(
            skill=s, observe_fn=lambda n: None, fire_trigger=lambda *a: None,
        )


# ---------------------------------------------------------------------------
# MonitorRunner — registration, lifecycle, ticks
# ---------------------------------------------------------------------------


def test_runner_register_and_unregister():
    runner = MonitorRunner(
        observe_fn=lambda n: 0,
        fire_trigger=lambda *a: None,
    )
    runner.register(_monitor())
    assert runner.registered() == ["watch"]
    runner.unregister("watch")
    assert runner.registered() == []
    runner.stop()


def test_runner_rejects_duplicate_registration():
    runner = MonitorRunner(
        observe_fn=lambda n: 0,
        fire_trigger=lambda *a: None,
    )
    skill = _monitor()
    runner.register(skill)
    with pytest.raises(ValueError, match="already registered"):
        runner.register(skill)
    runner.stop()


def test_runner_rejects_non_monitor_skill():
    runner = MonitorRunner(
        observe_fn=lambda n: 0,
        fire_trigger=lambda *a: None,
    )
    s = Skill(name="x", description="d", kind="responsive", system_prompt="hi")
    with pytest.raises(ValueError, match="kind="):
        runner.register(s)
    runner.stop()


def test_runner_tick_once_fires_trigger_on_match():
    fired: list[tuple[str, dict]] = []
    done = threading.Event()

    def observe(name):
        return 0.20

    def fire(name, payload):
        fired.append((name, payload))
        done.set()

    runner = MonitorRunner(observe_fn=observe, fire_trigger=fire)
    runner.register(_monitor(schedule="* * * * *"))  # match every minute
    runner.tick_once(when=datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc))
    # Tick is dispatched onto the pool; wait briefly for completion.
    assert done.wait(timeout=5.0), "monitor never fired"
    runner.stop()
    assert fired and fired[0][0] == "incident_high_error_rate"


def test_runner_tick_once_idempotent_within_minute():
    """Two tick_once calls in the same minute should fire the monitor only once."""
    fired: list[tuple[str, dict]] = []
    cv = threading.Condition()

    def fire(name, payload):
        with cv:
            fired.append((name, payload))
            cv.notify_all()

    runner = MonitorRunner(observe_fn=lambda n: 0.20, fire_trigger=fire)
    runner.register(_monitor(schedule="* * * * *"))
    when = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    runner.tick_once(when=when)
    runner.tick_once(when=when)
    # Wait for at least one fire; allow brief settle for any second.
    with cv:
        cv.wait_for(lambda: len(fired) >= 1, timeout=5.0)
    # Give any second tick a chance.
    with cv:
        cv.wait_for(lambda: len(fired) >= 2, timeout=0.5)
    runner.stop()
    assert len(fired) == 1


def test_runner_tick_does_not_fire_on_non_matching_minute():
    fired: list = []
    runner = MonitorRunner(
        observe_fn=lambda n: 0.20,
        fire_trigger=lambda *a: fired.append(a),
    )
    # Schedule: only at hour 9. Tick at hour 5 — no match.
    runner.register(_monitor(schedule="0 9 * * *"))
    runner.tick_once(when=datetime(2026, 5, 3, 5, 0, tzinfo=timezone.utc))
    runner.stop()
    assert fired == []


def test_runner_observe_failure_does_not_stop_subsequent_ticks():
    """A monitor whose tool blows up on tick #1 must still run on tick #2."""
    calls = []
    fired = []

    def observe(name):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return 0.20

    def fire(name, payload):
        fired.append((name, payload))

    runner = MonitorRunner(observe_fn=observe, fire_trigger=fire)
    runner.register(_monitor(schedule="* * * * *"))
    # Tick #1 — observe raises, fire is suppressed (because
    # observation[error_rate] is None > 0.05 -> safe_eval error).
    runner.tick_once(when=datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc))
    # Tick #2 — observe returns 0.20, fire runs.
    import time
    time.sleep(0.2)  # let executor drain tick #1
    runner.tick_once(when=datetime(2026, 5, 3, 12, 1, tzinfo=timezone.utc))
    deadline = time.monotonic() + 5.0
    while not fired and time.monotonic() < deadline:
        time.sleep(0.05)
    runner.stop()
    assert len(calls) >= 2
    assert fired, "second tick should have fired the trigger"


def test_runner_lifecycle_start_stop():
    runner = MonitorRunner(
        observe_fn=lambda n: 0,
        fire_trigger=lambda *a: None,
    )
    runner.start()
    assert runner._thread is not None and runner._thread.is_alive()
    runner.stop(timeout=2.0)
    # After stop, the thread reference is cleared.
    assert runner._thread is None
