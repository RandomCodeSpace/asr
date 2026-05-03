"""APScheduler-backed ``schedule`` transport.

Single ``AsyncIOScheduler`` per process, started during FastAPI lifespan
and stopped on shutdown. Each ``ScheduleTriggerConfig`` becomes one cron
job that calls ``registry.dispatch(name, payload)`` on fire.

Cron flavour: standard 5-field via ``CronTrigger.from_crontab``. The
6-field APScheduler-native form is rejected by ``ScheduleTriggerConfig``
itself; this transport never sees it.

Drift / accuracy: APScheduler in-process is good for ±1 minute under
normal load. Tighter SLOs need an external scheduler (Celery beat,
k8s CronJob) — out of scope for Phase 5.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import ScheduleTriggerConfig

if TYPE_CHECKING:
    pass
    from runtime.triggers.registry import TriggerRegistry  # noqa: F401

_log = logging.getLogger(__name__)


class ScheduleTransport(TriggerTransport):
    """In-process APScheduler driving cron-firing triggers."""

    def __init__(self, configs: list[ScheduleTriggerConfig]) -> None:
        self._configs = list(configs)
        self._scheduler: AsyncIOScheduler | None = None
        self._registry: "TriggerRegistry | None" = None

    @property
    def scheduler(self) -> AsyncIOScheduler | None:
        return self._scheduler

    async def start(self, registry: "TriggerRegistry") -> None:
        if self._scheduler is not None:
            return
        self._registry = registry
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        for cfg in self._configs:
            cron = CronTrigger.from_crontab(cfg.schedule, timezone=cfg.timezone)
            self._scheduler.add_job(
                self._fire,
                trigger=cron,
                kwargs={"name": cfg.name, "payload": dict(cfg.payload)},
                id=f"trigger:{cfg.name}",
                replace_existing=True,
            )
        self._scheduler.start()

    async def stop(self) -> None:
        if self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as exc:  # noqa: BLE001
            _log.warning("apscheduler shutdown raised: %s", exc)
        self._scheduler = None
        self._registry = None

    async def _fire(self, *, name: str, payload: dict) -> None:
        """APScheduler job target. Logs and swallows exceptions so a bad
        cron job doesn't poison the scheduler thread."""
        registry = self._registry
        if registry is None:
            _log.warning(
                "schedule trigger %r fired with no registry attached", name
            )
            return
        try:
            await registry.dispatch(name, payload)
        except Exception as exc:  # noqa: BLE001
            _log.exception(
                "schedule trigger %r dispatch failed: %s", name, exc
            )
