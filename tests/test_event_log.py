import pytest
from sqlalchemy import create_engine

from runtime.storage.models import Base
from runtime.storage.event_log import EventLog, SessionEvent, _VALID_EVENT_KINDS


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


# M2 — record(kind, **payload) helper + EventKind literal validation.

def test_record_helper_stamps_kind_and_payload(log):
    """``record(sid, "tool_invoked", tool="x", latency_ms=12)`` writes a
    row equivalent to ``append`` with the kwargs collected into a payload
    dict, and ``iter_for`` round-trips it."""
    log.record("INC-1", "tool_invoked", tool="x", latency_ms=12)
    events = list(log.iter_for("INC-1"))
    assert len(events) == 1
    e = events[0]
    assert e.kind == "tool_invoked"
    assert e.payload == {"tool": "x", "latency_ms": 12}
    # ts populated by append's _now() — sanity-check it's a non-empty str
    assert isinstance(e.ts, str) and e.ts


def test_event_kind_literal_rejects_unknown(log):
    """Passing a kind outside :data:`EventKind` raises ``ValueError`` at
    ``record`` time so typos don't silently pollute the log."""
    with pytest.raises(ValueError) as exc:
        log.record("INC-1", "totally_made_up_kind", foo="bar")  # type: ignore[arg-type]
    assert "totally_made_up_kind" in str(exc.value)
    # Sanity: no row was written.
    assert list(log.iter_for("INC-1")) == []


def test_event_kind_literal_lists_full_vocabulary():
    """Lock the vocabulary so adding a kind requires updating tests +
    callers in the same commit. If this fails after intentionally
    growing the vocabulary, bump the expected set here."""
    assert _VALID_EVENT_KINDS == frozenset({
        "agent_started",
        "agent_finished",
        "tool_invoked",
        "confidence_emitted",
        "route_decided",
        "gate_fired",
        "status_changed",
        "lesson_extracted",
        # v2.0 cross-session SSE kinds — drive the React UI's "Other
        # Sessions" monitor (api_recent_events.py).
        "session.created",
        "session.status_changed",
        "session.agent_running",
    })
