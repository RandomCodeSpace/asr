"""ABC and DTOs shared by every trigger transport.

A ``TriggerTransport`` owns the inbound side of one transport flavour
(api / webhook / schedule / plugin). The lifecycle is exactly two
async methods so the FastAPI lifespan in ``runtime/api.py`` can sequence
startup / shutdown deterministically.

A ``TriggerInfo`` is the provenance record attached to every session
started via a trigger. It rides along through ``Orchestrator.start_session``
purely for traceability; the orchestrator does not branch on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.triggers.registry import TriggerRegistry  # noqa: F401


@dataclass(frozen=True)
class TriggerInfo:
    """Provenance attached to every session started via a trigger.

    Stamped onto ``Orchestrator.start_session(trigger=...)`` by every
    transport. The framework does not branch on the contents; the field
    exists so dashboards / audit logs can answer "where did this session
    come from?" without re-deriving from disjoint sources.
    """

    name: str          # the trigger name from config (``triggers[].name``)
    transport: str     # ``api`` / ``webhook`` / ``schedule`` / plugin kind
    target_app: str    # ``triggers[].target_app``
    received_at: datetime


class TriggerTransport(ABC):
    """Lifecycle interface for a transport (api / webhook / schedule / plugin).

    The registry calls ``start(registry)`` on lifespan-enter and ``stop()``
    on lifespan-exit. Transports own their own state (router, scheduler,
    background tasks) and must be safe to construct *before* ``start`` —
    the FastAPI app collects routers from webhook transports during
    ``build_app`` and mounts them once during the lifespan handshake.
    """

    @abstractmethod
    async def start(self, registry: "TriggerRegistry") -> None:
        """Begin accepting inbound traffic. Must be idempotent."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop accepting traffic and release resources. Must be idempotent."""
