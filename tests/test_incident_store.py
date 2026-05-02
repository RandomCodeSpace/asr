import pytest
from orchestrator.incident import IncidentStore


@pytest.fixture
def store(tmp_path) -> IncidentStore:
    return IncidentStore(tmp_path)


def test_create_assigns_sequential_id_for_today(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc1 = store.create(query="A", environment="dev", reporter_id="u1", reporter_team="t")
    inc2 = store.create(query="B", environment="dev", reporter_id="u1", reporter_team="t")
    assert inc1.id == "INC-20260430-001"
    assert inc2.id == "INC-20260430-002"


def test_save_roundtrip(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="Q", environment="prod", reporter_id="u", reporter_team="t")
    inc.summary = "updated"
    store.save(inc)
    loaded = store.load(inc.id)
    assert loaded.summary == "updated"


def test_list_recent_returns_newest_first(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    items = store.list_recent(limit=10)
    assert [i.id for i in items] == [b.id, a.id]


def test_load_invalid_id_raises_value_error(store):
    """IDs not matching INC-YYYYMMDD-NNN must be rejected before any path ops."""
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.load("../../etc/passwd")


def test_save_invalid_id_raises_value_error(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="Q", environment="dev", reporter_id="u", reporter_team="t")
    inc.id = "../../malicious"
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.save(inc)


def test_load_missing_raises(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    with pytest.raises(FileNotFoundError):
        store.load("INC-20260430-999")


@pytest.mark.parametrize("bad_id", [
    "../../etc/passwd",
    "../incidents/INC-20260430-001",
    "INC-20260430-001.json",
    "INC-ABC-001",
    "",
    "INC-20260430-1",  # too short sequence
])
def test_load_rejects_traversal_and_malformed_ids(store, bad_id):
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.load(bad_id)


def test_valid_id_loads_correctly(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso", lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="valid", environment="dev", reporter_id="u", reporter_team="t")
    loaded = store.load(inc.id)
    assert loaded.id == inc.id


def test_delete_marks_status_and_timestamp(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso",
                        lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    deleted = store.delete(inc.id)
    assert deleted.status == "deleted"
    assert deleted.deleted_at == "2026-04-30T10:00:00Z"
    reloaded = store.load(inc.id)
    assert reloaded.status == "deleted"
    assert reloaded.deleted_at == "2026-04-30T10:00:00Z"


def test_delete_is_idempotent(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso",
                        lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    first = store.delete(inc.id)
    second = store.delete(inc.id)
    assert first.deleted_at == second.deleted_at
    assert second.status == "deleted"


def test_delete_clears_pending_intervention(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso",
                        lambda: "2026-04-30T10:00:00Z")
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "awaiting_input"
    inc.pending_intervention = {"reason": "low_confidence"}
    store.save(inc)
    deleted = store.delete(inc.id)
    assert deleted.pending_intervention is None


def test_list_recent_excludes_deleted_by_default(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso",
                        lambda: "2026-04-30T10:00:00Z")
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    store.delete(a.id)
    items = store.list_recent(limit=10)
    assert [i.id for i in items] == [b.id]


def test_list_recent_include_deleted_returns_all(store, monkeypatch):
    monkeypatch.setattr("orchestrator.incident._utc_today", lambda: "20260430")
    monkeypatch.setattr("orchestrator.incident._utc_now_iso",
                        lambda: "2026-04-30T10:00:00Z")
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    store.delete(a.id)
    items = store.list_recent(limit=10, include_deleted=True)
    assert {i.id for i in items} == {a.id, b.id}


def test_delete_invalid_id_raises(store):
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.delete("../../etc/passwd")
