"""TriggerRegistry — owns transport instances and dispatch.

The registry is the single sink every transport calls when it wants to
fire a session. It:

- Resolves dotted paths (``payload_schema`` / ``transform``) at init time.
- Holds an :class:`runtime.triggers.idempotency.IdempotencyStore`.
- Exposes a single async ``dispatch(name, payload, *, idempotency_key=None)``
  entrypoint that performs:

      transform(payload) -> kwargs
      orchestrator.start_session(**kwargs, trigger=info)

The registry also owns each transport's lifecycle: ``start_all`` and
``stop_all`` mirror FastAPI's lifespan handshake.

Plugin transports are merged from two sources:

1. Setuptools entry-points in group ``runtime.triggers``: ``key = kind``,
   ``value = importable subclass of TriggerTransport``.
2. Explicit registration via ``plugin_transports={...}`` on
   :meth:`TriggerRegistry.create`. Explicit wins on key collision.
"""
from __future__ import annotations

import importlib.metadata
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Type

from pydantic import BaseModel

from runtime.triggers.base import TriggerInfo, TriggerTransport
from runtime.triggers.config import (
    APITriggerConfig,
    PluginTriggerConfig,
    ScheduleTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)
from runtime.triggers.idempotency import IdempotencyStore
from runtime.triggers.resolve import resolve_payload_schema, resolve_transform

_log = logging.getLogger(__name__)

# Type aliases for clarity — ``StartSessionFn`` is the closure the
# registry calls; the FastAPI lifespan binds it to the orchestrator
# service so we don't take a hard import dependency on the Orchestrator
# class here (avoids a circular import).
StartSessionFn = Callable[..., Awaitable[str]]


class TriggerSpec:
    """Resolved (live) form of one ``TriggerConfig``.

    Built at registry init: dotted paths bound to live callables /
    classes; the original config is retained for transports to read.
    """

    __slots__ = (
        "config",
        "payload_schema",
        "transform",
    )

    def __init__(
        self,
        config: TriggerConfig,
        payload_schema: Type[BaseModel] | None,
        transform: Callable[..., dict] | None,
    ) -> None:
        self.config = config
        self.payload_schema = payload_schema
        self.transform = transform

    @property
    def name(self) -> str:
        return self.config.name


class TriggerRegistry:
    """Owns trigger lifecycle + dispatch.

    Construct via :meth:`create` so dotted-path resolution and transport
    instantiation happen together. Direct ``__init__`` use is reserved
    for unit tests that pre-build their own spec list.
    """

    def __init__(
        self,
        specs: dict[str, TriggerSpec],
        transports: list[TriggerTransport],
        start_session_fn: StartSessionFn,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        self._specs: dict[str, TriggerSpec] = specs
        self._transports: list[TriggerTransport] = transports
        self._start_session_fn = start_session_fn
        self._idempotency = idempotency
        self._started = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        configs: list[TriggerConfig],
        *,
        start_session_fn: StartSessionFn,
        idempotency: IdempotencyStore | None = None,
        plugin_transports: dict[str, Type[TriggerTransport]] | None = None,
    ) -> "TriggerRegistry":
        """Resolve dotted paths + instantiate transports.

        Raises ``ImportError`` / ``TypeError`` at startup for any bad
        dotted path — fail-fast, never at request time.
        """
        # 1. Resolve specs (dotted paths -> live objects).
        specs: dict[str, TriggerSpec] = {}
        for cfg in configs:
            schema: Type[BaseModel] | None = None
            transform_fn: Callable[..., dict] | None = None
            if isinstance(cfg, WebhookTriggerConfig):
                schema = resolve_payload_schema(cfg.payload_schema)
            if cfg.transform is not None:
                transform_fn = resolve_transform(cfg.transform)
            specs[cfg.name] = TriggerSpec(
                config=cfg, payload_schema=schema, transform=transform_fn
            )

        # 2. Resolve plugin kinds (entry-points + explicit, explicit wins).
        plugin_kinds: dict[str, Type[TriggerTransport]] = (
            cls._load_entry_point_transports()
        )
        if plugin_transports:
            plugin_kinds.update(plugin_transports)

        # 3. Bucket configs by transport flavour.
        api_cfgs: list[APITriggerConfig] = []
        webhook_cfgs: list[WebhookTriggerConfig] = []
        schedule_cfgs: list[ScheduleTriggerConfig] = []
        plugin_cfgs: list[PluginTriggerConfig] = []
        for cfg in configs:
            if isinstance(cfg, APITriggerConfig):
                api_cfgs.append(cfg)
            elif isinstance(cfg, WebhookTriggerConfig):
                webhook_cfgs.append(cfg)
            elif isinstance(cfg, ScheduleTriggerConfig):
                schedule_cfgs.append(cfg)
            elif isinstance(cfg, PluginTriggerConfig):
                plugin_cfgs.append(cfg)

        # 4. Instantiate transports. Lazy import to break import cycles.
        from runtime.triggers.transports.api import APITransport
        from runtime.triggers.transports.schedule import ScheduleTransport
        from runtime.triggers.transports.webhook import WebhookTransport

        transports: list[TriggerTransport] = []
        if api_cfgs:
            transports.append(APITransport(api_cfgs))
        if webhook_cfgs:
            transports.append(WebhookTransport(webhook_cfgs, specs, idempotency))
        if schedule_cfgs:
            transports.append(ScheduleTransport(schedule_cfgs))
        for pcfg in plugin_cfgs:
            kind_cls = plugin_kinds.get(pcfg.kind)
            if kind_cls is None:
                raise ImportError(
                    f"plugin trigger {pcfg.name!r} requested kind={pcfg.kind!r} "
                    f"but no transport with that kind is registered "
                    f"(known: {sorted(plugin_kinds)})"
                )
            # Plugin transports inherit from the abstract
            # ``TriggerTransport`` (no positional args declared on the
            # ABC) but every concrete subclass loaded via the entry-
            # point registry must accept the plugin's config object.
            # The ABC mismatch is a stub limitation, not a runtime bug.
            transports.append(kind_cls(pcfg))  # pyright: ignore[reportCallIssue]

        return cls(specs, transports, start_session_fn, idempotency)

    @staticmethod
    def _load_entry_point_transports() -> dict[str, Type[TriggerTransport]]:
        """Discover plugin transports via the ``runtime.triggers`` group.

        Defensive: a missing or malformed entry-point is logged and
        skipped rather than failing registry init. Apps that need strict
        binding pass ``plugin_transports`` explicitly.
        """
        out: dict[str, Type[TriggerTransport]] = {}
        try:
            eps = importlib.metadata.entry_points(group="runtime.triggers")
        except Exception:  # noqa: BLE001
            return out
        for ep in eps:
            try:
                obj = ep.load()
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "trigger entry-point %r failed to load: %s", ep.name, exc
                )
                continue
            if not (isinstance(obj, type) and issubclass(obj, TriggerTransport)):
                _log.warning(
                    "trigger entry-point %r did not resolve to a "
                    "TriggerTransport subclass; got %r",
                    ep.name,
                    obj,
                )
                continue
            out[ep.name] = obj
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def transports(self) -> list[TriggerTransport]:
        return list(self._transports)

    @property
    def specs(self) -> dict[str, TriggerSpec]:
        return dict(self._specs)

    @property
    def idempotency(self) -> IdempotencyStore | None:
        return self._idempotency

    async def start_all(self) -> None:
        """Start every transport. Idempotent."""
        if self._started:
            return
        for t in self._transports:
            await t.start(self)
        self._started = True

    async def stop_all(self) -> None:
        """Stop every transport. Idempotent."""
        if not self._started:
            return
        for t in self._transports:
            try:
                await t.stop()
            except Exception as exc:  # noqa: BLE001
                # Best-effort: one misbehaving transport mustn't block
                # the rest from cleaning up.
                _log.warning("trigger transport %r stop() failed: %s", t, exc)
        self._started = False

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        payload: Any,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Run ``transform(payload)`` and call ``start_session_fn``.

        Returns the session id. If ``idempotency_key`` is provided, the
        cached session id is returned on hit; on miss the call proceeds
        and the (key, session_id) mapping is recorded for the trigger's
        configured TTL.

        Raises ``KeyError`` for an unknown trigger name. Surfaces any
        ``ValueError`` / ``ValidationError`` from ``transform`` to the
        caller — transports translate to HTTP status codes (typically
        ``422 Unprocessable Entity``).
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unknown trigger: {name!r}")

        # Idempotency hit: return cached session id without invoking
        # transform / orchestrator. Per R3 in the plan, transform errors
        # are NOT cached — only successful dispatches.
        if idempotency_key and self._idempotency is not None:
            cached = self._idempotency.get(name, idempotency_key)
            if cached is not None:
                return cached

        # Resolve trigger payload -> start_session kwargs.
        if spec.transform is not None:
            kwargs = spec.transform(payload)
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"transform for trigger {name!r} returned "
                    f"{type(kwargs).__name__}, expected dict"
                )
        else:
            # api transport: payload is already the kwargs dict.
            kwargs = dict(payload) if payload else {}

        info = TriggerInfo(
            name=name,
            transport=spec.config.transport,
            target_app=spec.config.target_app,
            received_at=datetime.now(timezone.utc),
        )

        session_id = await self._start_session_fn(trigger=info, **kwargs)

        # Record successful dispatch for idempotency.
        if idempotency_key and self._idempotency is not None:
            ttl = (
                spec.config.idempotency_ttl_hours
                if isinstance(spec.config, WebhookTriggerConfig)
                else 24
            )
            self._idempotency.put(name, idempotency_key, session_id, ttl_hours=ttl)

        return session_id
