"""GET /api/v1/apps/{app}/ui-views returns app-specific UI overlay links.

Apps register their bespoke pages via cfg.ui.app_views. The framework UI
lists these in the Selected-detail panel ("App-specific views →").
"""
from fastapi.testclient import TestClient
from runtime.api import build_app
from runtime.config import UIConfig, AppView
from tests.test_api_v1_url_move import _cfg


def test_app_views_returns_empty_list_when_unconfigured(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/apps/incident_management/ui-views")
        assert r.status_code == 200
        assert r.json() == []


def test_app_views_returns_configured_views(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ui = UIConfig(
        app_views=[
            AppView(id="deploy-diff", title="Deploy diff",
                    applies_to="agent:investigate",
                    url="/apps/incident_management/ui/deploy-diff"),
            AppView(id="topology", title="Service topology",
                    applies_to="always",
                    url="/apps/incident_management/ui/topology"),
        ],
    )
    app = build_app(cfg)
    with TestClient(app) as client:
        r = client.get("/api/v1/apps/incident_management/ui-views")
        body = r.json()
        assert len(body) == 2
        assert body[0]["id"] == "deploy-diff"
        assert body[0]["applies_to"] == "agent:investigate"
        assert body[1]["id"] == "topology"


def test_app_views_arbitrary_app_name_returns_same_list(tmp_path):
    """v2.0: one app per deployment, app_name is informational only.
    Multi-app routing is v2.1 scope."""
    cfg = _cfg(tmp_path)
    cfg.ui = UIConfig(
        app_views=[
            AppView(id="x", title="X", applies_to="always", url="/apps/x/ui/x"),
        ],
    )
    app = build_app(cfg)
    with TestClient(app) as client:
        r1 = client.get("/api/v1/apps/incident_management/ui-views")
        r2 = client.get("/api/v1/apps/code_review/ui-views")
        # Both return the same list — no per-app filtering yet.
        assert r1.json() == r2.json()
