"""Built-in ``api`` transport.

The api transport is a no-op lifecycle wrapper — the actual HTTP route
(``POST /investigate`` and ``POST /sessions``) is mounted directly on
the FastAPI app for back-compat. Existing in the registry purely so
operators can list ``api`` triggers in YAML for symmetry, and so the
provenance ``TriggerInfo.transport == "api"`` is available for sessions
created via the legacy route.
"""
from __future__ import annotations

from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import APITriggerConfig


class APITransport(TriggerTransport):
    """No-op transport that holds the ``api`` configs.

    Future: when the legacy ``POST /investigate`` route is removed, this
    class can mount its own router. For Phase 5 it exists as a marker.
    """

    def __init__(self, configs: list[APITriggerConfig]) -> None:
        self._configs = list(configs)

    @property
    def configs(self) -> list[APITriggerConfig]:
        return list(self._configs)

    async def start(self, registry) -> None:  # noqa: D401, ARG002
        return None

    async def stop(self) -> None:
        return None
