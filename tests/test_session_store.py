"""Tests for SessionStore (active CRUD) and HistoryStore (similarity search)."""
import pytest
from sqlalchemy import create_engine

from runtime.state import Session
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


class _QueryableSession(Session):
    """Session subclass that surfaces ``query`` + ``environment`` so
    similarity search (keyword path + vector path) has indexable text
    and the ``filter_kwargs={"environment": ...}`` predicate matches on
    the hydrated state. Bare ``Session`` leaves both empty by virtue of
    not declaring the attributes.
    """

    query: str = ""
    environment: str = ""


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    e = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture()
def store(engine):
    return SessionStore(engine=engine, state_cls=_QueryableSession)


@pytest.fixture()
def history(engine):
    # Threshold deliberately low: the keyword fallback uses Jaccard-style
    # token overlap, and "payments timeout" vs "payments timeout upstream
    # restarted payments service" scores ~0.4. We're testing wiring, not
    # threshold tuning.
    return HistoryStore(engine=engine, state_cls=_QueryableSession,
                        embedder=None, vector_store=None,
                        similarity_threshold=0.3)


def test_session_store_create_returns_incident(store):
    inc = store.create(
        query="payments latency", environment="production",
        reporter_id="user-1", reporter_team="platform",
    )
    assert inc.id.startswith("INC-")
    assert inc.status == "new"


def test_session_store_load_roundtrip(store):
    inc = store.create(query="t", environment="staging",
                       reporter_id="u", reporter_team="t")
    loaded = store.load(inc.id)
    assert loaded.id == inc.id


def test_session_store_save_updates(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    assert store.load(inc.id).status == "resolved"


def test_session_store_delete_soft(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    deleted = store.delete(inc.id)
    assert deleted.status == "deleted"
    assert deleted.deleted_at is not None


def test_session_store_list_excludes_deleted_by_default(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    store.delete(inc.id)
    active = store.list_recent(50)
    assert not any(i.id == inc.id for i in active)


def test_history_store_keyword_fallback(history, store):
    inc = store.create(query="payments timeout upstream",
                       environment="production",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    results = history.find_similar(
        query="payments timeout",
        filter_kwargs={"environment": "production"},
        status_filter="resolved", limit=5,
    )
    assert any(r[0].id == inc.id for r in results)


def test_session_store_is_separate_from_history_store():
    from runtime.storage.session_store import SessionStore
    from runtime.storage.history_store import HistoryStore
    assert SessionStore is not HistoryStore


def test_history_store_generic_filter_kwargs(history, store):
    """find_similar accepts arbitrary filter_kwargs (env, severity, etc.)."""
    inc = store.create(query="login latency", environment="staging",
                       reporter_id="u", reporter_team="t")
    inc.status = "resolved"
    store.save(inc)
    results = history.find_similar(
        query="login latency", filter_kwargs={"environment": "staging"},
        status_filter="resolved", limit=5,
    )
    assert any(r[0].id == inc.id for r in results)


def test_history_store_filter_kwargs_rejects_unknown_column(history):
    """Filter keys must correspond to IncidentRow columns."""
    with pytest.raises(ValueError, match="unsupported filter_kwargs"):
        history.find_similar(
            query="x", filter_kwargs={"not_a_column": "y"},
            status_filter="resolved", limit=5,
        )


def test_history_store_no_incident_state_import():
    """HistoryStore module must not import IncidentState directly."""
    import runtime.storage.history_store as mod
    src = open(mod.__file__).read()
    assert "examples.incident_management.state" not in src, (
        "HistoryStore must not import IncidentState — pass state_cls/text_extractor instead"
    )
