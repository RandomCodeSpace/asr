"""StaticFiles mount at / serves the React SPA with a fallback to index.html."""
from fastapi.testclient import TestClient
from runtime.api import build_app
from tests.test_api_v1_url_move import _cfg


def test_root_serves_index_html(tmp_path, monkeypatch):
    # Stage a fake web/dist with an index.html
    web_dist = tmp_path / "web_dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<!DOCTYPE html><html>SPA OK</html>")
    monkeypatch.setenv("ASR_WEB_DIST", str(web_dist))

    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "SPA OK" in r.text


def test_unknown_path_falls_back_to_index_html(tmp_path, monkeypatch):
    """SPA routes (/sessions/abc, /settings) fall back to index.html."""
    web_dist = tmp_path / "web_dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<!DOCTYPE html><html>SPA OK</html>")
    monkeypatch.setenv("ASR_WEB_DIST", str(web_dist))

    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/sessions/SES-20260515-001")  # SPA route, not API
        assert r.status_code == 200
        assert "SPA OK" in r.text


def test_api_v1_paths_are_NOT_caught_by_spa_fallback(tmp_path, monkeypatch):
    """Unknown /api/v1/* paths return 404 JSON, not the SPA HTML."""
    web_dist = tmp_path / "web_dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<!DOCTYPE html><html>SPA OK</html>")
    monkeypatch.setenv("ASR_WEB_DIST", str(web_dist))

    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/no-such-endpoint")
        assert r.status_code == 404
        assert "SPA OK" not in r.text


def test_when_web_dist_missing_serves_helpful_message(tmp_path, monkeypatch):
    """If web/dist not built yet, root returns a 503 with build instructions."""
    monkeypatch.setenv("ASR_WEB_DIST", "/nonexistent")
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 503
        assert "npm run build" in r.text
