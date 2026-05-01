import pytest
from orchestrator.incident import IncidentStore
from orchestrator.mcp_servers.incident import (
    set_state, lookup_similar_incidents, create_incident, update_incident,
    IncidentMCPServer,
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
    # severity is normalized to the canonical {low, medium, high} vocabulary
    assert loaded.severity == "medium"
    assert loaded.category == "latency"


@pytest.mark.asyncio
async def test_update_incident_normalizes_severity(setup_store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = await create_incident(query="y", environment="dev", reporter_id="u", reporter_team="t")
    for raw, want in [
        ("sev1", "high"), ("SEV2", "high"), ("p1", "high"), ("critical", "high"),
        ("sev3", "medium"), ("moderate", "medium"), ("Medium", "medium"),
        ("sev4", "low"), ("info", "low"), ("LOW", "low"),
    ]:
        await update_incident(incident_id=inc["id"], patch={"severity": raw})
        assert setup_store.load(inc["id"]).severity == want, raw


@pytest.mark.asyncio
async def test_two_servers_have_independent_stores(tmp_path, monkeypatch):
    """Two IncidentMCPServer instances must NOT share their store reference.

    The previous module-level ``_state`` dict made the second instance clobber
    the first; this test pins the per-instance scoping.
    """
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")

    store_a = IncidentStore(tmp_path / "a")
    store_b = IncidentStore(tmp_path / "b")
    server_a = IncidentMCPServer()
    server_b = IncidentMCPServer()
    server_a.configure(store=store_a, similarity_threshold=0.5)
    server_b.configure(store=store_b, similarity_threshold=0.5)

    inc_a = await server_a._tool_create_incident(
        query="a-side", environment="dev", reporter_id="u", reporter_team="t",
    )
    inc_b = await server_b._tool_create_incident(
        query="b-side", environment="dev", reporter_id="u", reporter_team="t",
    )
    # Each store sees only its own INC.
    assert {p.stem for p in (tmp_path / "a").glob("INC-*.json")} == {inc_a["id"]}
    assert {p.stem for p in (tmp_path / "b").glob("INC-*.json")} == {inc_b["id"]}
    # And the FastMCP instances are distinct objects (no shared global server).
    assert server_a.mcp is not server_b.mcp
