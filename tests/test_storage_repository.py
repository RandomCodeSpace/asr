"""SessionStore CRUD tests against file-based SQLite + stub embedder.

The active CRUD surface lives on ``SessionStore``. Tests use the
framework default ``Session`` state class — the row schema still
carries the (now-legacy) incident-shaped typed columns for back-compat,
but the Python state object only sees what ``Session`` declares.
"""
from __future__ import annotations
import pytest

from runtime.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def repo(tmp_path):
    db = tmp_path / "test.db"
    eng = build_engine(MetadataConfig(url=f"sqlite:///{db}"))
    Base.metadata.create_all(eng)
    return SessionStore(engine=eng)


def test_create_assigns_id_and_persists(repo):
    inc = repo.create(query="redis OOM", environment="production",
                      reporter_id="u1", reporter_team="platform")
    assert inc.id.startswith("INC-")
    loaded = repo.load(inc.id)
    assert loaded.id == inc.id
    assert loaded.status == "new"


def test_create_id_sequence(repo):
    a = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    b = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    seq_a = int(a.id.rsplit("-", 1)[1])
    seq_b = int(b.id.rsplit("-", 1)[1])
    assert seq_b == seq_a + 1


def test_save_round_trip_preserves_nested(repo):
    from runtime.state import AgentRun, ToolCall, TokenUsage
    inc = repo.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.findings = {"triage": {"k": "v"}}
    inc.agents_run.append(AgentRun(agent="intake", started_at=inc.created_at,
                                   ended_at=inc.created_at, summary="ok",
                                   token_usage=TokenUsage()))
    inc.tool_calls.append(ToolCall(agent="intake", tool="x", args={}, result={},
                                   ts=inc.created_at))
    repo.save(inc)
    loaded = repo.load(inc.id)
    assert loaded.findings == {"triage": {"k": "v"}}
    assert len(loaded.agents_run) == 1
    assert len(loaded.tool_calls) == 1


def test_load_missing_raises_filenotfound(repo):
    with pytest.raises(FileNotFoundError):
        repo.load("INC-20260102-999")


def test_load_invalid_id_raises_valueerror(repo):
    with pytest.raises(ValueError):
        repo.load("not-an-id")


def test_delete_is_soft(repo):
    inc = repo.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    out = repo.delete(inc.id)
    assert out.status == "deleted"
    assert out.deleted_at is not None
    again = repo.delete(inc.id)
    assert again.status == "deleted"


def test_list_recent_excludes_deleted_by_default(repo):
    a = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    b = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    repo.delete(a.id)
    listed = repo.list_recent(limit=10)
    ids = [i.id for i in listed]
    assert b.id in ids
    assert a.id not in ids
    listed_all = repo.list_recent(limit=10, include_deleted=True)
    assert {a.id, b.id}.issubset({i.id for i in listed_all})


def test_list_recent_orders_by_created_at_desc(repo):
    inc1 = repo.create(query="q1", environment="dev", reporter_id="u", reporter_team="t")
    inc2 = repo.create(query="q2", environment="dev", reporter_id="u", reporter_team="t")
    listed = repo.list_recent(limit=10)
    assert [i.id for i in listed[:2]] == [inc2.id, inc1.id]
