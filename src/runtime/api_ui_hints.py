"""UI hints endpoint — drives the React shell's brand and templates.

GET /api/v1/config/ui-hints returns the runtime-configured ui block plus
the environments list. Read once at React-app boot and cached for the
session lifetime via `useUiHints()`.

Registered only via :func:`runtime.api.build_app` (requires
``app.state.cfg``); not suitable for lightweight test fixtures that
construct a bare ``FastAPI()`` app — use ``build_app(cfg)`` for tests.
"""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Request


def add_routes(api_v1: APIRouter) -> None:
    """Mount the /config/ui-hints handler on the api_v1 router."""

    @api_v1.get("/config/ui-hints")
    async def get_ui_hints(request: Request) -> dict[str, Any]:
        cfg = request.app.state.cfg
        ui = cfg.ui
        return {
            "brand_name": ui.brand_name,
            "brand_logo_url": ui.brand_logo_url,
            "approval_rationale_templates": list(ui.approval_rationale_templates),
            "hitl_question_templates": dict(ui.hitl_question_templates),
            "environments": list(cfg.environments or []),
        }
