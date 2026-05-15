"""GET /api/v1/config/ui-hints returns app-configured UI hints."""
from fastapi.testclient import TestClient
from runtime.api import build_app
from runtime.config import UIConfig
from tests.test_api_v1_url_move import _cfg


def test_ui_hints_returns_defaults_when_unconfigured(tmp_path):
    cfg = _cfg(tmp_path)
    app = build_app(cfg)
    with TestClient(app) as client:
        r = client.get("/api/v1/config/ui-hints")
        assert r.status_code == 200
        body = r.json()
        # Defaults: empty brand, no logo, no rationale templates
        assert body["brand_name"] == ""
        assert body["brand_logo_url"] is None
        assert body["approval_rationale_templates"] == []


def test_ui_hints_returns_configured_values(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ui = UIConfig(
        brand_name="Acme Agents",
        brand_logo_url="/static/acme.svg",
        approval_rationale_templates=["Standard runbook", "Off-hours"],
    )
    app = build_app(cfg)
    with TestClient(app) as client:
        r = client.get("/api/v1/config/ui-hints")
        body = r.json()
        assert body["brand_name"] == "Acme Agents"
        assert body["brand_logo_url"] == "/static/acme.svg"
        assert body["approval_rationale_templates"] == ["Standard runbook", "Off-hours"]


def test_ui_hints_includes_environments_list(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.environments = ["dev", "staging", "production"]
    app = build_app(cfg)
    with TestClient(app) as client:
        r = client.get("/api/v1/config/ui-hints")
        body = r.json()
        assert body["environments"] == ["dev", "staging", "production"]
