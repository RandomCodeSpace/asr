import pytest
from orchestrator.incident import IncidentStore
from orchestrator.mcp_servers.incident import (
    set_state, lookup_similar_incidents, create_incident, update_incident,
)


@pytest.fixture(autouse=True)
def setup_store(tmp_path, monkeypatch):
    store = IncidentStore(tmp_path)
    set_state(store=store, similarity_threshold=0.3)
    yield store


@pytest.mark.asyncio
async def test_create_then_lookup_returns_match(setup_store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = await create_incident(query="api latency spike production", environment="production",
                                reporter_id="u", reporter_team="t")
    # mark resolved so it appears in lookup
    inc_loaded = setup_store.load(inc["id"])
    inc_loaded.status = "resolved"
    inc_loaded.resolution = {"summary": "scaled up"}
    setup_store.save(inc_loaded)

    result = await lookup_similar_incidents(query="api latency production", environment="production")
    assert result["matches"]
    assert result["matches"][0]["id"] == inc["id"]


@pytest.mark.asyncio
async def test_update_incident_appends_finding(setup_store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = await create_incident(query="x", environment="dev", reporter_id="u", reporter_team="t")
    await update_incident(incident_id=inc["id"], patch={"severity": "sev3", "category": "latency"})
    loaded = setup_store.load(inc["id"])
    assert loaded.severity == "sev3"
    assert loaded.category == "latency"
