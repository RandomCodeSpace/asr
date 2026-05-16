"""Legacy /incidents/* endpoints get HTTP 308 redirects to /api/v1/sessions/*."""
from fastapi.testclient import TestClient
from runtime.api import build_app
from tests.test_api_v1_url_move import _cfg  # reuse fixture builder


def test_legacy_incidents_list_redirects(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app, follow_redirects=False) as client:
        r = client.get("/incidents")
        assert r.status_code == 308
        assert r.headers["location"] == "/api/v1/sessions"


def test_legacy_incident_detail_redirects(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app, follow_redirects=False) as client:
        r = client.get("/incidents/SES-20260515-001")
        assert r.status_code == 308
        assert r.headers["location"] == "/api/v1/sessions/SES-20260515-001"


def test_legacy_resume_redirects(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app, follow_redirects=False) as client:
        r = client.post("/incidents/SES-20260515-001/resume", json={"decision": "resume_with_input"})
        assert r.status_code == 308
        assert r.headers["location"] == "/api/v1/sessions/SES-20260515-001/resume"


def test_legacy_investigate_redirects(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app, follow_redirects=False) as client:
        r = client.post("/investigate", json={"query": "x", "environment": "dev"})
        assert r.status_code == 308
        assert r.headers["location"] == "/api/v1/investigate"
