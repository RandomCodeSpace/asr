from sqlalchemy import create_engine

from runtime.orchestrator import Orchestrator
from runtime.state import ToolCall
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore, StaleVersionError


def _make_orch_with_store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)
    class _O:
        def __init__(self, s): self.store = s
        _finalize_session_status = Orchestrator._finalize_session_status
    return _O(store), store


def test_finalize_with_mark_escalated_in_history_yields_escalated(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_escalated",
        args={"team": "platform-oncall"}, result={"status": "escalated"},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    new_status = orch._finalize_session_status(inc.id)
    assert new_status == "escalated"
    fresh = store.load(inc.id)
    assert fresh.status == "escalated"
    assert fresh.extra_fields.get("escalated_to") == "platform-oncall"


def test_finalize_with_mark_resolved_in_history_yields_resolved(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_resolved", args={},
        result={"status": "resolved"}, ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "resolved"


def test_finalize_with_no_terminal_tool_yields_needs_review(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "needs_review"


def test_finalize_does_not_clobber_terminal_status(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "escalated"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) is None
    assert store.load(inc.id).status == "escalated"


def test_finalize_returns_none_on_stale_version(tmp_path, monkeypatch):
    """If a concurrent finalize wins the race, save() raises
    StaleVersionError; this finalize must yield (return None) rather
    than propagate the exception up the async stream loop."""
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)

    def _raise_stale(_):
        raise StaleVersionError("concurrent writer settled first")

    monkeypatch.setattr(store, "save", _raise_stale)
    assert orch._finalize_session_status(inc.id) is None
