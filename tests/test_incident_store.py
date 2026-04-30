from pathlib import Path
import pytest
from orchestrator.incident import Incident, IncidentStore, Reporter


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


def test_load_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load("INC-DOESNOTEXIST")
