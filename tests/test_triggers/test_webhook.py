"""WebhookTransport tests — POST /triggers/{name} happy + error paths."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from runtime.triggers import TriggerRegistry
from runtime.triggers.config import WebhookTriggerConfig
from runtime.triggers.transports.webhook import WebhookTransport


def _build_app(*, auth: str = "bearer", token_env: str | None = "TEST_WEBHOOK_TOKEN"):
    """Build a tiny app with one webhook trigger mounted."""
    cfg = WebhookTriggerConfig(
        name="pd",
        target_app="incident_management",
        payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
        transform="tests.test_triggers.conftest.transform_pagerduty",
        auth=auth,
        auth_token_env=token_env,
    )

    started = []

    async def fake_start(*, trigger=None, **kw):
        started.append((trigger, kw))
        return f"INC-{len(started):03d}"

    registry = TriggerRegistry.create([cfg], start_session_fn=fake_start)

    app = FastAPI()
    app.state.registry = registry
    app.state.started = started

    @app.on_event("startup")
    async def _on_start():
        await registry.start_all()
        for t in registry.transports:
            if isinstance(t, WebhookTransport):
                app.include_router(t.router)

    @app.on_event("shutdown")
    async def _on_stop():
        await registry.stop_all()

    return app, started


def test_webhook_happy_path(monkeypatch, pagerduty_token):
    app, started = _build_app(auth="bearer", token_env="TEST_WEBHOOK_TOKEN")
    with TestClient(app) as client:
        r = client.post(
            "/triggers/pd",
            headers={"Authorization": f"Bearer {pagerduty_token}"},
            json={
                "incident_id": "P-1",
                "summary": "API down",
                "severity": "high",
                "environment": "production",
            },
        )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["session_id"].startswith("INC-")
    assert len(started) == 1
    trig, kw = started[0]
    assert trig.transport == "webhook"
    assert kw["query"].startswith("PD P-1")


def test_webhook_missing_bearer_returns_401(pagerduty_token):
    app, started = _build_app(auth="bearer", token_env="TEST_WEBHOOK_TOKEN")
    with TestClient(app) as client:
        r = client.post(
            "/triggers/pd",
            json={
                "incident_id": "P-1",
                "summary": "boom",
                "environment": "production",
            },
        )
    assert r.status_code == 401
    assert started == []


def test_webhook_wrong_token_returns_401(pagerduty_token):
    app, started = _build_app(auth="bearer", token_env="TEST_WEBHOOK_TOKEN")
    with TestClient(app) as client:
        r = client.post(
            "/triggers/pd",
            headers={"Authorization": "Bearer wrong"},
            json={
                "incident_id": "P-1",
                "summary": "boom",
                "environment": "production",
            },
        )
    assert r.status_code == 401
    assert started == []


def test_webhook_invalid_payload_returns_422(pagerduty_token):
    app, started = _build_app(auth="bearer", token_env="TEST_WEBHOOK_TOKEN")
    with TestClient(app) as client:
        r = client.post(
            "/triggers/pd",
            headers={"Authorization": f"Bearer {pagerduty_token}"},
            json={"summary": "missing incident_id"},
        )
    assert r.status_code == 422
    assert started == []


def test_webhook_auth_none_no_header_required(monkeypatch):
    app, started = _build_app(auth="none", token_env=None)
    with TestClient(app) as client:
        r = client.post(
            "/triggers/pd",
            json={
                "incident_id": "P-1",
                "summary": "boom",
                "environment": "production",
            },
        )
    assert r.status_code == 202
    assert len(started) == 1


def test_webhook_idempotency_key_dedupes_via_registry(tmp_path, pagerduty_token):
    """Sending the same Idempotency-Key twice yields the same session id."""
    from runtime.triggers.idempotency import IdempotencyStore

    cfg = WebhookTriggerConfig(
        name="pd",
        target_app="incident_management",
        payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
        transform="tests.test_triggers.conftest.transform_pagerduty",
        auth="bearer",
        auth_token_env="TEST_WEBHOOK_TOKEN",
    )
    started: list = []

    async def fake_start(*, trigger=None, **kw):
        started.append(kw)
        return f"INC-{len(started):03d}"

    db_url = f"sqlite:///{tmp_path / 'idem.db'}"
    store = IdempotencyStore.connect(db_url)
    registry = TriggerRegistry.create(
        [cfg], start_session_fn=fake_start, idempotency=store
    )

    app = FastAPI()

    @app.on_event("startup")
    async def _on_start():
        await registry.start_all()
        for t in registry.transports:
            if isinstance(t, WebhookTransport):
                app.include_router(t.router)

    @app.on_event("shutdown")
    async def _on_stop():
        await registry.stop_all()

    payload = {
        "incident_id": "P-1",
        "summary": "boom",
        "environment": "production",
    }
    headers = {
        "Authorization": f"Bearer {pagerduty_token}",
        "Idempotency-Key": "K-1",
    }
    with TestClient(app) as client:
        r1 = client.post("/triggers/pd", headers=headers, json=payload)
        r2 = client.post("/triggers/pd", headers=headers, json=payload)
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["session_id"] == r2.json()["session_id"]
    assert len(started) == 1
