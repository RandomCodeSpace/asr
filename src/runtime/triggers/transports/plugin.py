"""Documentation hook for plugin trigger transports.

Plugin transports are concrete subclasses of
:class:`runtime.triggers.base.TriggerTransport` registered via either
the ``runtime.triggers`` setuptools entry-point group or the
``plugin_transports={kind: cls}`` kwarg to ``TriggerRegistry.create``.

This module is intentionally lightweight — the contract lives entirely
in ``base.py``. It exists as a single discoverable entry-point for
operators reading the source tree.

Example minimal plugin transport::

    from runtime.triggers.base import TriggerTransport
    from runtime.triggers.config import PluginTriggerConfig

    class SQSTransport(TriggerTransport):
        def __init__(self, config: PluginTriggerConfig) -> None:
            self._cfg = config
            self._task = None

        async def start(self, registry):
            self._task = asyncio.create_task(self._poll(registry))

        async def stop(self):
            if self._task:
                self._task.cancel()

        async def _poll(self, registry):
            ...  # poll SQS, call registry.dispatch(self._cfg.name, payload)

Register via ``pyproject.toml``::

    [project.entry-points."runtime.triggers"]
    sqs = "myapp.triggers:SQSTransport"

Or explicitly::

    TriggerRegistry.create(
        configs, start_session_fn=...,
        plugin_transports={"sqs": SQSTransport},
    )
"""
from __future__ import annotations

from runtime.triggers.base import TriggerTransport

__all__ = ["TriggerTransport"]
