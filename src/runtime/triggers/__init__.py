"""Trigger registry — declarative session-start surface (Phase 5).

Generalizes session-start beyond the hand-coded ``POST /investigate``
route. After Phase 5 the framework can fire ``Orchestrator.start_session``
from three first-party transports (``api``, ``webhook``, ``schedule``)
and any number of plugin-defined transports — all driven by the
``triggers:`` block in app config.

Public surface:
    from runtime.triggers import (
        TriggerConfig, TriggerRegistry, TriggerTransport,
        TriggerInfo, IdempotencyStore,
    )
"""
from __future__ import annotations

from runtime.triggers.base import TriggerInfo, TriggerTransport
from runtime.triggers.config import (
    APITriggerConfig,
    PluginTriggerConfig,
    ScheduleTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)
from runtime.triggers.idempotency import IdempotencyStore
from runtime.triggers.registry import TriggerRegistry

__all__ = [
    "APITriggerConfig",
    "IdempotencyStore",
    "PluginTriggerConfig",
    "ScheduleTriggerConfig",
    "TriggerConfig",
    "TriggerInfo",
    "TriggerRegistry",
    "TriggerTransport",
    "WebhookTriggerConfig",
]
