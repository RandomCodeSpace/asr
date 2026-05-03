"""Bearer auth dependency tests."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient

from runtime.triggers.auth import make_bearer_dep


def _make_app(token_env: str) -> FastAPI:
    """Spin a tiny FastAPI app with one bearer-protected route."""
    app = FastAPI()
    dep = make_bearer_dep(token_env)

    @app.post("/protected", dependencies=[Depends(dep)])
    async def protected():
        return {"ok": True}

    return app


def test_make_bearer_dep_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_BEARER_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        make_bearer_dep("MISSING_BEARER_TOKEN")


def test_missing_authorization_header_returns_401(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "abc")
    app = _make_app("TEST_TOKEN")
    client = TestClient(app)
    r = client.post("/protected")
    assert r.status_code == 401


def test_wrong_scheme_returns_401(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "abc")
    app = _make_app("TEST_TOKEN")
    client = TestClient(app)
    r = client.post(
        "/protected", headers={"Authorization": "Basic abc"}
    )
    assert r.status_code == 401


def test_wrong_token_returns_401(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "abc")
    app = _make_app("TEST_TOKEN")
    client = TestClient(app)
    r = client.post(
        "/protected", headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401


def test_correct_token_passes(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "abc")
    app = _make_app("TEST_TOKEN")
    client = TestClient(app)
    r = client.post(
        "/protected", headers={"Authorization": "Bearer abc"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
