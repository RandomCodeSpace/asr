import pytest

from examples.incident_management.state import IncidentState
from orchestrator.config import MetadataConfig
from orchestrator.storage.engine import build_engine
from orchestrator.storage.history_store import HistoryStore
from orchestrator.storage.models import Base
from orchestrator.storage.session_store import SessionStore
from orchestrator.mcp_servers.incident import (
    set_state, lookup_similar_incidents, create_incident, update_incident,
    IncidentMCPServer,
)


def _make_repo(tmp_path, *, similarity_threshold: float = 0.3):
    """Create a keyword-based store + history (no embedder) for fast unit tests."""
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    store = SessionStore(engine=eng, state_cls=IncidentState, embedder=None)
    history = HistoryStore(engine=eng, state_cls=IncidentState, embedder=None,
                           similarity_threshold=similarity_threshold)
    return store, history


@pytest.fixture(autouse=True)
def setup_store(tmp_path):
    store, history = _make_repo(tmp_path)
    set_state(store=store, history=history)
    yield store


@pytest.mark.asyncio
async def test_create_then_lookup_returns_match(setup_store):
    inc = await create_incident(query="api latency spike production", environment="production",
                                reporter_id="u", reporter_team="t")
    # mark resolved so it appears in lookup
    inc_loaded = setup_store.load(inc["id"])
    inc_loaded.status = "resolved"
    inc_loaded.resolution = "scaled up"
    setup_store.save(inc_loaded)

    result = await lookup_similar_incidents(query="api latency production", environment="production")
    assert result["matches"]
    assert result["matches"][0]["id"] == inc["id"]


@pytest.mark.asyncio
async def test_update_incident_appends_finding(setup_store):
    inc = await create_incident(query="x", environment="dev", reporter_id="u", reporter_team="t")
    await update_incident(incident_id=inc["id"], patch={"severity": "sev3", "category": "latency"})
    loaded = setup_store.load(inc["id"])
    # severity is normalized to the canonical {low, medium, high} vocabulary
    assert loaded.severity == "medium"
    assert loaded.category == "latency"


@pytest.mark.asyncio
async def test_update_incident_normalizes_severity(setup_store):
    inc = await create_incident(query="y", environment="dev", reporter_id="u", reporter_team="t")
    for raw, want in [
        ("sev1", "high"), ("SEV2", "high"), ("p1", "high"), ("critical", "high"),
        ("sev3", "medium"), ("moderate", "medium"), ("Medium", "medium"),
        ("sev4", "low"), ("info", "low"), ("LOW", "low"),
    ]:
        await update_incident(incident_id=inc["id"], patch={"severity": raw})
        assert setup_store.load(inc["id"]).severity == want, raw


@pytest.mark.asyncio
async def test_two_servers_have_independent_repos(tmp_path):
    """Two IncidentMCPServer instances must NOT share their store/history references."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    store_a, history_a = _make_repo(tmp_path / "a")
    store_b, history_b = _make_repo(tmp_path / "b")
    server_a = IncidentMCPServer()
    server_b = IncidentMCPServer()
    server_a.configure(store=store_a, history=history_a)
    server_b.configure(store=store_b, history=history_b)

    inc_a = await server_a._tool_create_incident(
        query="a-side", environment="dev", reporter_id="u", reporter_team="t",
    )
    inc_b = await server_b._tool_create_incident(
        query="b-side", environment="dev", reporter_id="u", reporter_team="t",
    )
    # Each store sees only its own INC.
    ids_a = {i.id for i in store_a.list_all()}
    ids_b = {i.id for i in store_b.list_all()}
    assert ids_a == {inc_a["id"]}
    assert ids_b == {inc_b["id"]}
    # And the FastMCP instances are distinct objects (no shared global server).
    assert server_a.mcp is not server_b.mcp
