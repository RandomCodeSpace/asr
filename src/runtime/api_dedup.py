"""Dedup retraction HTTP routes (P7-H).

Exposes ``register_dedup_routes(app, *, store_provider)`` — a side-car
router so we don't need to inline these routes in ``runtime.api``. The
caller wires it into the main FastAPI app at lifespan startup; tests
construct a tiny FastAPI app and register only this router so they
don't pull in the full lifespan stack.

Endpoint:
  ``POST /sessions/{session_id}/un-duplicate``
    Body: ``{"retracted_by": str | None, "note": str | None}``
    On success: 200 with the updated session.
    409 when ``status != "duplicate"``.
    404 when the session id is unknown.

The endpoint never re-runs the agent graph — operators trigger that
explicitly. The audit row is inserted in the same transaction as the
status flip via :meth:`SessionStore.un_duplicate`.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class UnDuplicateRequest(BaseModel):
    """Request body for the retraction endpoint.

    Both fields are optional. ``retracted_by`` is self-claimed (the
    framework does not authenticate the operator id at P7); ``note`` is
    free text persisted on the audit row.
    """

    retracted_by: str | None = None
    note: str | None = Field(default=None, max_length=2000)


class UnDuplicateResponse(BaseModel):
    """Successful retraction payload."""

    session_id: str
    status: str
    parent_session_id: str | None
    original_match_id: str
    retracted_by: str | None = None
    note: str | None = None


def register_dedup_routes(
    app: FastAPI,
    *,
    store_provider: Callable[[], Any],
) -> None:
    """Register the un-duplicate route on ``app``.

    ``store_provider`` is a no-arg callable that returns the live
    ``SessionStore``. We accept a callable (rather than the store
    directly) so apps can defer construction until first request — the
    route handler itself never caches the store.
    """

    @app.post(
        "/sessions/{session_id}/un-duplicate",
        response_model=UnDuplicateResponse,
        status_code=200,
        tags=["dedup"],
    )
    async def un_duplicate(
        session_id: str,
        body: UnDuplicateRequest | None = None,
    ) -> UnDuplicateResponse:
        store = store_provider()
        # Pre-flight: capture the parent id BEFORE the flip so we can
        # echo it on the response. The store does the same capture
        # internally for the audit row.
        try:
            current = store.load(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            # ``load`` validates the id format; map malformed -> 404.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if current.status != "duplicate":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not a duplicate",
                    "status": current.status,
                },
            )
        original_match_id = current.parent_session_id or ""
        payload = body or UnDuplicateRequest()
        try:
            updated = store.un_duplicate(
                session_id,
                retracted_by=payload.retracted_by,
                note=payload.note,
            )
        except FileNotFoundError as exc:
            # Race: deleted between load and un_duplicate.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            # Race: status flipped between load and un_duplicate.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return UnDuplicateResponse(
            session_id=updated.id,
            status=updated.status,
            parent_session_id=updated.parent_session_id,
            original_match_id=original_match_id,
            retracted_by=payload.retracted_by,
            note=payload.note,
        )
