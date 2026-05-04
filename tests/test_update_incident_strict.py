import pytest
from sqlalchemy import create_engine

from examples.incident_management.mcp_server import IncidentMCPServer
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def server(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    s = SessionStore(engine=engine)
    srv = IncidentMCPServer()
    srv.configure(store=s)
    return srv, s


@pytest.mark.asyncio
async def test_unknown_patch_key_raises(server):
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="confidance"):
        await srv._tool_update_incident(inc.id, {"confidance": 0.8})


@pytest.mark.asyncio
async def test_status_field_rejected(server):
    """Status transitions go through mark_resolved/mark_escalated, NOT update_incident."""
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="status"):
        await srv._tool_update_incident(inc.id, {"status": "resolved"})


@pytest.mark.asyncio
async def test_resolution_field_rejected(server):
    """resolution is set by mark_resolved, not update_incident."""
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="resolution"):
        await srv._tool_update_incident(inc.id, {"resolution": "rolled back"})


@pytest.mark.asyncio
async def test_escalated_to_field_rejected(server):
    """escalated_to is set by mark_escalated, not update_incident."""
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="escalated_to"):
        await srv._tool_update_incident(inc.id, {"escalated_to": "platform-oncall"})


@pytest.mark.asyncio
async def test_severity_update_works(server):
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    await srv._tool_update_incident(inc.id, {"severity": "high"})
    fresh = store.load(inc.id)
    assert fresh.extra_fields["severity"] == "high"


@pytest.mark.asyncio
async def test_category_summary_tags_update_works(server):
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    await srv._tool_update_incident(inc.id, {
        "category": "availability",
        "summary": "api down",
        "tags": ["urgent", "production"],
    })
    fresh = store.load(inc.id)
    assert fresh.extra_fields["category"] == "availability"
    assert fresh.extra_fields["summary"] == "api down"
    assert fresh.extra_fields["tags"] == ["urgent", "production"]


@pytest.mark.asyncio
async def test_findings_dict_update_works(server):
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    await srv._tool_update_incident(inc.id, {
        "findings": {"triage": "investigating", "deep_investigator": "found root cause"},
    })
    fresh = store.load(inc.id)
    assert fresh.findings["triage"] == "investigating"
    assert fresh.findings["deep_investigator"] == "found root cause"


@pytest.mark.asyncio
async def test_legacy_findings_underscore_keys_rejected(server):
    """The old ``findings_<agent>`` underscore-prefix pattern is no
    longer supported — use the typed ``findings`` dict instead."""
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    with pytest.raises(ValueError, match="findings_triage"):
        await srv._tool_update_incident(inc.id, {"findings_triage": "investigating"})


@pytest.mark.asyncio
async def test_empty_patch_succeeds_as_noop(server):
    """An empty patch dict is a valid no-op."""
    srv, store = server
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    await srv._tool_update_incident(inc.id, {})  # no error
