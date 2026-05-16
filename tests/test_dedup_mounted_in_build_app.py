"""Regression: register_dedup_routes is wired in production build_app.

The un-duplicate route lives in api_dedup.py (register_dedup_routes).
v1.x left the registration unwired in build_app — only the in-test
fixture mounted it — so the route was effectively dead in dist/app.py.
v2.0.0-rc2 (Gap C in the endpoint audit) closes the gap by calling
register_dedup_routes(api_v1, store_provider=...) inside build_app.

This test pins that wiring so a future refactor can't silently undo it.
"""
from fastapi.testclient import TestClient
from runtime.api import build_app
from tests.test_api_v1_url_move import _cfg


def test_un_duplicate_route_is_mounted_on_build_app(tmp_path):
    app = build_app(_cfg(tmp_path))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/api/v1/sessions/{session_id}/un-duplicate" in paths, (
        "register_dedup_routes is not wired in build_app — Gap C regression"
    )


def test_un_duplicate_returns_404_for_unknown_session(tmp_path):
    """Smoke through the live app — the route is reachable end-to-end,
    not just declared. 404 is the correct envelope for an unknown id."""
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/sessions/INC-29991231-001/un-duplicate",
            json={"retracted_by": "operator", "note": "smoke"},
        )
        assert r.status_code == 404, r.text
