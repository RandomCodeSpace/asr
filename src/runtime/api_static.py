"""StaticFiles mount + SPA fallback for the React UI bundle.

The React build output lives at $ASR_WEB_DIST (default: ../web/dist relative
to repo root). FastAPI serves /assets/* and /fonts/* directly; any unknown
path that isn't /api/v1/*, /health, or /docs falls back to index.html so
the React Router can pick up the URL.

Registered only via :func:`runtime.api.build_app` (mounts onto the root
FastAPI app, not the api_v1 router). Must be invoked AFTER all API routes
are registered so the catch-all SPA fallback doesn't shadow them.
"""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles


_BUILD_HINT = """<!DOCTYPE html>
<html><body style="font-family:sans-serif;padding:32px;background:#fbfaf6;color:#15110a">
<h1>React UI not built yet</h1>
<p>Run <code style="background:#eee;padding:2px 6px">cd web && npm ci && npm run build</code>
to populate <code>web/dist/</code>.</p>
<p>Then refresh.</p>
</body></html>
"""

_NOT_FOUND_JSON = (
    '{"error":{"code":"not_found","message":"unknown api path","details":{}}}'
)


def mount_static_assets(app: FastAPI) -> None:
    """Mount static assets + SPA fallback. API routes must be registered first.

    Module-qualified name (vs. the bare ``mount`` we had pre-bundle-fix) so
    the bundler can flatten this alongside its sibling ``api_*`` side-cars
    without stepping on FastAPI's ``app.mount`` or any future bundled module
    that happens to define a ``mount`` symbol. See
    ``runtime.api_session_full.add_session_full_routes``.
    """
    web_dist_path = os.environ.get("ASR_WEB_DIST")
    if web_dist_path:
        web_dist = Path(web_dist_path)
    else:
        # Default: web/dist relative to repo root.
        web_dist = (
            Path(__file__).resolve().parent.parent.parent / "web" / "dist"
        )

    if not (web_dist / "index.html").exists():
        # Stub fallback when the bundle isn't built — useful in dev.
        @app.get("/", include_in_schema=False)
        async def _missing_root() -> HTMLResponse:
            return HTMLResponse(content=_BUILD_HINT, status_code=503)
        return

    # Serve assets at /assets/* (Vite output)
    assets_dir = web_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    # Serve fonts at /fonts/* (vendored)
    fonts_dir = web_dist / "fonts"
    if fonts_dir.exists():
        app.mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts")

    # SPA fallback: anything not matched by API or other static mounts → index.html
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str, request: Request) -> Response:
        # Reserve API/health/docs paths
        if (full_path.startswith("api/") or full_path == "health"
                or full_path.startswith("docs") or full_path == "openapi.json"):
            return Response(
                content=_NOT_FOUND_JSON,
                status_code=404,
                media_type="application/json",
            )
        return FileResponse(web_dist / "index.html")
