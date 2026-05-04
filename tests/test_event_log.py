import pytest
from sqlalchemy import create_engine

from runtime.storage.models import Base
from runtime.storage.event_log import EventLog, SessionEvent


@pytest.fixture
def log(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    return EventLog(engine=engine)


def test_append_and_iterate(log):
    log.append("INC-1", "status_changed", {"from": "new", "to": "in_progress"})
    log.append("INC-1", "tool_invoked", {"tool": "update_incident"})
    events = list(log.iter_for("INC-1"))
    assert [e.kind for e in events] == ["status_changed", "tool_invoked"]
    assert events[0].payload == {"from": "new", "to": "in_progress"}


def test_events_for_other_sessions_excluded(log):
    log.append("INC-1", "x", {})
    log.append("INC-2", "y", {})
    assert [e.kind for e in log.iter_for("INC-1")] == ["x"]


def test_events_have_monotonic_seq(log):
    log.append("INC-1", "a", {})
    log.append("INC-1", "b", {})
    log.append("INC-1", "c", {})
    seqs = [e.seq for e in log.iter_for("INC-1")]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


def test_iter_returns_session_event_dataclass(log):
    log.append("INC-1", "kind1", {"key": "value"})
    events = list(log.iter_for("INC-1"))
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, SessionEvent)
    assert e.session_id == "INC-1"
    assert e.kind == "kind1"
    assert e.payload == {"key": "value"}
    assert isinstance(e.seq, int)
    assert isinstance(e.ts, str) and e.ts  # non-empty ISO timestamp
