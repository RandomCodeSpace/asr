"""Webhook transport — ``POST /triggers/{name}``.

Mounted by ``runtime/api.py`` during the FastAPI lifespan. Each
``WebhookTriggerConfig`` becomes one route under the same ``/triggers``
prefix; per-route bearer auth is wired via ``runtime.triggers.auth``.

Per-request flow:

1. Bearer dep validates ``Authorization: Bearer <token>`` (when configured).
2. Body parsed against the resolved ``payload_schema`` (Pydantic).
   Validation failure -> ``422``.
3. Optional ``Idempotency-Key`` header is forwarded to
   :meth:`TriggerRegistry.dispatch`. A cache hit returns the existing
   session id; misses run the full transform + ``start_session`` path.
4. Transform errors (any exception from the ``transform`` callable)
   surface as ``422 Unprocessable Entity`` with the exception message;
   per the plan we do not auto-retry and do not cache the failure.
5. Success: ``202 Accepted`` with ``{"session_id": "..."}``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import ValidationError

from runtime.triggers.auth import make_bearer_dep
from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import WebhookTriggerConfig

if TYPE_CHECKING:
    pass
    from runtime.triggers.idempotency import IdempotencyStore  # noqa: F401
    from runtime.triggers.registry import (  # noqa: F401
        TriggerRegistry, TriggerSpec,
    )

_log = logging.getLogger(__name__)


class WebhookTransport(TriggerTransport):
    """FastAPI router exposing one ``POST /triggers/{name}`` per webhook."""

    def __init__(
        self,
        configs: list[WebhookTriggerConfig],
        specs: "dict[str, TriggerSpec]",
        idempotency: "IdempotencyStore | None",
    ) -> None:
        self._configs = {c.name: c for c in configs}
        self._specs = specs
        self._idempotency = idempotency
        self.router = APIRouter()
        self._registry: "TriggerRegistry | None" = None
        self._mounted = False

    async def start(self, registry: "TriggerRegistry") -> None:
        if self._mounted:
            return
        self._registry = registry
        for name, cfg in self._configs.items():
            self.router.add_api_route(
                f"/triggers/{name}",
                self._make_handler(name),
                methods=["POST"],
                dependencies=self._auth_deps(cfg),
                status_code=status.HTTP_202_ACCEPTED,
                name=f"trigger:{name}",
            )
        self._mounted = True

    async def stop(self) -> None:
        # The router lives on the FastAPI app for the process lifetime;
        # there's nothing to tear down (FastAPI cleans up its own state
        # when the app object is GC'd). Mark unmounted so a subsequent
        # ``start`` is a no-op only if we were already started.
        self._registry = None
        self._mounted = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _auth_deps(cfg: WebhookTriggerConfig) -> list[Any]:
        if cfg.auth == "none":
            return []
        # cfg.auth == "bearer": the model validator already enforced
        # auth_token_env presence at config load.
        assert cfg.auth_token_env is not None
        return [Depends(make_bearer_dep(cfg.auth_token_env))]

    def _make_handler(self, name: str) -> Callable:
        spec = self._specs[name]
        schema = spec.payload_schema
        assert schema is not None  # webhook configs always carry a schema

        async def handler(
            request: Request,
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        ) -> dict:
            registry = self._registry
            if registry is None:
                raise HTTPException(
                    status_code=503,
                    detail="trigger registry not started",
                )
            # Parse body. We read raw JSON then run through the schema
            # so we get a uniform 422 surface (FastAPI body-binding 422
            # has a different shape).
            try:
                raw = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid json body: {exc}",
                ) from exc
            try:
                payload = schema.model_validate(raw)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            # Dispatch. Translate transform/start_session errors to 422
            # (per R3: log + 422, no retry, no idempotency cache).
            try:
                session_id = await registry.dispatch(
                    name, payload, idempotency_key=idempotency_key
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (ValueError, TypeError, ValidationError) as exc:
                _log.warning(
                    "trigger %r transform/dispatch failed: %s", name, exc
                )
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"session_id": session_id}

        return handler
