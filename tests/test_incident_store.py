"""Behavioural tests for SessionStore.

Exercises ``SessionStore`` directly with the example ``IncidentState``.
"""
import pytest

from examples.incident_management.state import IncidentState
from runtime.config import EmbeddingConfig, MetadataConfig, ProviderConfig
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


def _make_repo(tmp_path) -> SessionStore:
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, state_cls=IncidentState, embedder=embedder)


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return _make_repo(tmp_path)


def test_create_assigns_sequential_ids(store):
    inc1 = store.create(query="A", environment="dev", reporter_id="u1", reporter_team="t")
    inc2 = store.create(query="B", environment="dev", reporter_id="u1", reporter_team="t")
    # Both must match INC-YYYYMMDD-NNN pattern and be sequential
    assert inc1.id.startswith("INC-")
    assert inc2.id.startswith("INC-")
    seq1 = int(inc1.id.rsplit("-", 1)[1])
    seq2 = int(inc2.id.rsplit("-", 1)[1])
    assert seq2 == seq1 + 1


def test_save_roundtrip(store):
    inc = store.create(query="Q", environment="prod", reporter_id="u", reporter_team="t")
    inc.summary = "updated"
    store.save(inc)
    loaded = store.load(inc.id)
    assert loaded.summary == "updated"


def test_list_recent_returns_newest_first(store):
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    items = store.list_recent(limit=10)
    # b was created after a so it should appear first
    assert items[0].id == b.id
    assert items[1].id == a.id


def test_load_invalid_id_raises_value_error(store):
    """IDs not matching INC-YYYYMMDD-NNN must be rejected before any DB ops."""
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.load("../../etc/passwd")


def test_save_invalid_id_raises_value_error(store):
    inc = store.create(query="Q", environment="dev", reporter_id="u", reporter_team="t")
    inc.id = "../../malicious"
    with pytest.raises(ValueError, match="Invalid incident id"):
        store.save(inc)


def test_load_missing_raises(store):
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


def test_valid_id_loads_correctly(store):
    inc = store.create(query="valid", environment="dev", reporter_id="u", reporter_team="t")
    loaded = store.load(inc.id)
    assert loaded.id == inc.id


def test_delete_marks_status_and_timestamp(store):
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    deleted = store.delete(inc.id)
    assert deleted.status == "deleted"
    assert deleted.deleted_at is not None
    reloaded = store.load(inc.id)
    assert reloaded.status == "deleted"
    assert reloaded.deleted_at is not None


def test_delete_is_idempotent(store):
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    first = store.delete(inc.id)
    second = store.delete(inc.id)
    assert first.deleted_at == second.deleted_at
    assert second.status == "deleted"


def test_delete_clears_pending_intervention(store):
    inc = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "awaiting_input"
    inc.pending_intervention = {"reason": "low_confidence"}
    store.save(inc)
    deleted = store.delete(inc.id)
    assert deleted.pending_intervention is None


def test_list_recent_excludes_deleted_by_default(store):
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    store.delete(a.id)
    items = store.list_recent(limit=10)
    assert [i.id for i in items] == [b.id]


def test_list_recent_include_deleted_returns_all(store):
    a = store.create(query="A", environment="dev", reporter_id="u", reporter_team="t")
    b = store.create(query="B", environment="dev", reporter_id="u", reporter_team="t")
    store.delete(a.id)
    items = store.list_recent(limit=10, include_deleted=True)
    assert {i.id for i in items} == {a.id, b.id}


def test_delete_missing_id_raises(store):
    with pytest.raises(FileNotFoundError):
        store.delete("INC-20260430-999")
