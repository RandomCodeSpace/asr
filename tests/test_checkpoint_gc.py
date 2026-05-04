import pytest
from sqlalchemy import create_engine, text

from runtime.storage.models import Base
from runtime.storage.checkpoint_gc import gc_orphaned_checkpoints
from runtime.storage.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    return SessionStore(engine=engine), engine


def test_gc_keeps_checkpoints_for_active_sessions(store):
    s, engine = store
    inc = s.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, checkpoint_id TEXT, parent_id TEXT, "
            " checkpoint BLOB, metadata BLOB, type TEXT, "
            " PRIMARY KEY (thread_id, checkpoint_id))"
        ))
        conn.execute(text(f"INSERT INTO checkpoints VALUES ('{inc.id}', 'c1', NULL, x'00', x'00', 'msgpack')"))
    removed = gc_orphaned_checkpoints(engine)
    assert removed == 0


def test_gc_removes_checkpoints_for_deleted_sessions(store):
    s, engine = store
    inc = s.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, checkpoint_id TEXT, parent_id TEXT, "
            " checkpoint BLOB, metadata BLOB, type TEXT, "
            " PRIMARY KEY (thread_id, checkpoint_id))"
        ))
        conn.execute(text("INSERT INTO checkpoints VALUES ('INC-DELETED', 'c1', NULL, x'00', x'00', 'msgpack')"))
    removed = gc_orphaned_checkpoints(engine)
    assert removed == 1
