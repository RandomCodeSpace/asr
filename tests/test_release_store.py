"""L5 Release Context store.

Pin tests for ``ReleaseStore``. Cover seed fallback, sort order,
service filtering, the symmetric correlation window, and the
``L5ReleaseContext`` assembly used by the triage agent.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from examples.incident_management.asr.memory_state import L5ReleaseContext
from examples.incident_management.asr.release_store import ReleaseStore


# ---- Seed fallback ------------------------------------------------------


def test_seed_fallback_loads_when_root_empty(tmp_path: Path) -> None:
    store = ReleaseStore(tmp_path)
    all_releases = store.list_all()
    assert len(all_releases) >= 4
    services = {r["service"] for r in all_releases}
    assert {"payments", "ledger"} <= services


def test_releases_sorted_descending_by_deployed_at(tmp_path: Path) -> None:
    store = ReleaseStore(tmp_path)
    timestamps = [r["deployed_at"] for r in store.list_all()]
    assert timestamps == sorted(timestamps, reverse=True)


# ---- Custom dataset fixture --------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ReleaseStore:
    """Anchor every release on a known timestamp so tests are deterministic.

    Reference incident time: 2026-05-03T10:00:00Z.
    """
    records = [
        # Two payments releases, one inside the 60-minute window, one outside.
        {
            "id": "pay-in-window",
            "service": "payments",
            "sha": "aaa111",
            "deployed_at": "2026-05-03T09:30:00Z",  # 30min before incident
            "author": "alice",
            "summary": "in-window payments deploy",
        },
        {
            "id": "pay-old",
            "service": "payments",
            "sha": "bbb222",
            "deployed_at": "2026-05-02T08:00:00Z",  # >24h before
            "author": "alice",
            "summary": "old payments deploy",
        },
        # Ledger release inside the window.
        {
            "id": "ledger-in-window",
            "service": "ledger",
            "sha": "ccc333",
            "deployed_at": "2026-05-03T10:30:00Z",  # 30min after incident
            "author": "bob",
            "summary": "in-window ledger deploy",
        },
        # Unrelated service.
        {
            "id": "billing-recent",
            "service": "billing",
            "sha": "ddd444",
            "deployed_at": "2026-05-03T09:55:00Z",
            "author": "carol",
            "summary": "noise",
        },
        # Malformed: missing deployed_at — must be dropped.
        {
            "id": "broken",
            "service": "payments",
            "sha": "eee555",
            "author": "dave",
            "summary": "broken record",
        },
    ]
    (tmp_path / "recent.json").write_text(json.dumps(records))
    return ReleaseStore(tmp_path)


def test_malformed_records_are_dropped(store: ReleaseStore) -> None:
    ids = {r["id"] for r in store.list_all()}
    assert "broken" not in ids


def test_recent_for_service_respects_hours(tmp_path: Path) -> None:
    """``recent_for_service`` measures against now() — use a recent record."""
    now = datetime.now(timezone.utc)
    records = [
        {
            "id": "fresh",
            "service": "payments",
            "sha": "x",
            "deployed_at": (now - timedelta(hours=1)).isoformat().replace(
                "+00:00", "Z"
            ),
            "author": "a",
            "summary": "",
        },
        {
            "id": "stale",
            "service": "payments",
            "sha": "y",
            "deployed_at": (now - timedelta(hours=48)).isoformat().replace(
                "+00:00", "Z"
            ),
            "author": "a",
            "summary": "",
        },
    ]
    (tmp_path / "recent.json").write_text(json.dumps(records))
    store = ReleaseStore(tmp_path)

    fresh = store.recent_for_service("payments", hours=24)
    assert {r["id"] for r in fresh} == {"fresh"}

    nothing = store.recent_for_service("payments", hours=0)
    assert nothing == []


def test_suspect_at_window_filters_by_service_and_time(
    store: ReleaseStore,
) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    suspects = store.suspect_at(
        services=["payments", "ledger"],
        at=incident_at,
        window_minutes=60,
    )
    # ``ledger-in-window`` (10:30) is more recent than ``pay-in-window`` (09:30)
    # — the API contract is "descending by deployed_at".
    assert suspects == ["ledger-in-window", "pay-in-window"]


def test_suspect_at_excludes_other_services(store: ReleaseStore) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    suspects = store.suspect_at(
        services=["payments"],
        at=incident_at,
        window_minutes=60,
    )
    assert suspects == ["pay-in-window"]


def test_suspect_at_zero_window_returns_empty(store: ReleaseStore) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    assert store.suspect_at(
        services=["payments"], at=incident_at, window_minutes=0
    ) == []


def test_suspect_at_naive_datetime_is_treated_as_utc(
    store: ReleaseStore,
) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0)  # naive
    suspects = store.suspect_at(
        services=["payments"], at=incident_at, window_minutes=60
    )
    assert suspects == ["pay-in-window"]


def test_context_returns_l5releasecontext(store: ReleaseStore) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    ctx = store.context(["payments", "ledger"], incident_at)
    assert isinstance(ctx, L5ReleaseContext)
    # 24h window backwards from incident_at: pay-in-window only (ledger
    # one is *after* the incident, so excluded from recent).
    recent_ids = {r["id"] for r in ctx.recent_releases}
    assert "pay-in-window" in recent_ids
    assert "pay-old" not in recent_ids
    # Suspects use the symmetric +/-60min window.
    assert sorted(ctx.suspect_releases) == sorted(
        ["ledger-in-window", "pay-in-window"]
    )


def test_context_unknown_service_yields_empty(store: ReleaseStore) -> None:
    incident_at = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    ctx = store.context(["nonexistent"], incident_at)
    assert ctx.recent_releases == []
    assert ctx.suspect_releases == []
