"""ScheduleTransport tests — APScheduler in-process cron."""
from __future__ import annotations

import asyncio

import pytest

from runtime.triggers import TriggerRegistry
from runtime.triggers.config import ScheduleTriggerConfig
from runtime.triggers.transports.schedule import ScheduleTransport


@pytest.mark.asyncio
async def test_schedule_transport_starts_apscheduler():
    cfg = ScheduleTriggerConfig(
        name="cron",
        target_app="incident_management",
        transform="tests.test_triggers.conftest.transform_schedule_heartbeat",
        schedule="*/5 * * * *",
        payload={"query": "heartbeat"},
    )

    async def fake(*, trigger=None, **kw):
        return "INC-X"

    reg = TriggerRegistry.create([cfg], start_session_fn=fake)
    await reg.start_all()
    schedule_transports = [
        t for t in reg.transports if isinstance(t, ScheduleTransport)
    ]
    assert len(schedule_transports) == 1
    sched = schedule_transports[0].scheduler
    assert sched is not None
    assert sched.running
    jobs = sched.get_jobs()
    assert any(j.id == "trigger:cron" for j in jobs)
    await reg.stop_all()
    assert schedule_transports[0].scheduler is None


@pytest.mark.asyncio
async def test_schedule_fire_dispatches_to_registry():
    """Manually invoke the job target and verify dispatch is called."""
    cfg = ScheduleTriggerConfig(
        name="cron",
        target_app="incident_management",
        transform="tests.test_triggers.conftest.transform_schedule_heartbeat",
        schedule="*/5 * * * *",
        payload={"query": "heartbeat-now"},
    )
    started = []

    async def fake(*, trigger=None, **kw):
        started.append(kw)
        return "INC-1"

    reg = TriggerRegistry.create([cfg], start_session_fn=fake)
    await reg.start_all()
    transport = next(
        t for t in reg.transports if isinstance(t, ScheduleTransport)
    )
    # Invoke the fire callback directly — far cheaper than waiting for
    # the cron tick.
    await transport._fire(name="cron", payload={"query": "heartbeat-now"})
    await reg.stop_all()
    assert len(started) == 1
    assert started[0]["query"] == "heartbeat-now"
