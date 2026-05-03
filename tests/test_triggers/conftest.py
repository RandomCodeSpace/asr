"""Shared fixtures for trigger registry tests."""
from __future__ import annotations

import pytest
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic schemas + transforms used by webhook tests.  Defined here (in a
# real importable module) so the dotted-path resolver in
# ``runtime.triggers.resolve`` can find them via
# ``tests.test_triggers.conftest:PagerDutyPayload``.
# ---------------------------------------------------------------------------


class PagerDutyPayload(BaseModel):
    """Toy PagerDuty webhook payload."""

    incident_id: str
    summary: str
    severity: str = "high"
    environment: str = "production"


def transform_pagerduty(payload: PagerDutyPayload) -> dict:
    """Convert payload -> ``Orchestrator.start_session`` kwargs."""
    return {
        "query": f"PD {payload.incident_id}: {payload.summary}",
        "environment": payload.environment,
        "reporter_id": "pagerduty",
        "reporter_team": "oncall",
    }


def transform_schedule_heartbeat(payload: dict) -> dict:
    """Schedule trigger transform — payload is the static cfg.payload dict."""
    return {
        "query": payload.get("query", "scheduled run"),
        "environment": payload.get("environment", "production"),
        "reporter_id": "scheduler",
        "reporter_team": "platform",
    }


@pytest.fixture
def pagerduty_token(monkeypatch):
    """Set the bearer token env var used by webhook auth tests."""
    monkeypatch.setenv("TEST_WEBHOOK_TOKEN", "secret-token-123")
    yield "secret-token-123"
