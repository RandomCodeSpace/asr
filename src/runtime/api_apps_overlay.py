"""App-overlay UI views discovery — Approach C extensibility.

GET /api/v1/apps/{app}/ui-views returns the app-registered overlay views.
The framework UI's Selected-detail panel renders matching views as
"App-specific views →" links. v2.0 ships with one app per deployment;
multi-app per-app filtering is v2.1 scope (the path's app_name is
currently informational).

Registered only via :func:`runtime.api.build_app` (requires
``app.state.cfg``); not suitable for lightweight test fixtures that
construct a bare ``FastAPI()`` app — use ``build_app(cfg)`` for tests.
"""
from __future__ import annotations
from fastapi import APIRouter, Request


def add_apps_overlay_routes(api_v1: APIRouter) -> None:
    """Mount the /apps/{app}/ui-views handler on the api_v1 router.

    Module-qualified name so the bundler can flatten alongside sibling
    ``api_*`` side-cars without ``add_routes`` collisions. See
    ``runtime.api_session_full.add_session_full_routes``.
    """

    @api_v1.get("/apps/{app_name}/ui-views")
    async def list_app_views(app_name: str, request: Request) -> list[dict]:
        # app_name is informational for now; v2.0 has one app per deploy.
        cfg = request.app.state.cfg
        return [v.model_dump() for v in cfg.ui.app_views]
