"""Bearer auth dependency for webhook trigger routes.

A small FastAPI dependency that compares the inbound ``Authorization``
header against the env-var named in ``WebhookTriggerConfig.auth_token_env``.
Constant-time comparison via ``hmac.compare_digest``; missing/bad/wrong
header all answer ``401``.

Tokens are read once at app startup (when the dependency is built) so
rotating a secret requires a process restart — same model as every other
config-derived secret in the runtime.
"""
from __future__ import annotations

import hmac
import os
from typing import Callable

from fastapi import Header, HTTPException, status


def make_bearer_dep(token_env: str) -> Callable:
    """Return a FastAPI dependency that asserts the inbound bearer token
    matches ``$token_env``.

    Snapshots the env var at construction time and raises ``RuntimeError``
    if it isn't set — callers can't accidentally start a webhook without
    a configured secret.
    """
    expected = os.environ.get(token_env)
    if not expected:
        raise RuntimeError(
            f"env var {token_env!r} for webhook auth is not set"
        )

    async def _bearer_dep(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer",
            )
        token = authorization[len("Bearer "):].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer",
            )

    return _bearer_dep
