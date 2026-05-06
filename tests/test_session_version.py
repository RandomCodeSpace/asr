import pytest
from sqlalchemy import create_engine

from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore, StaleVersionError


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    return SessionStore(engine=engine)


def test_save_increments_version(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    assert inc.version == 1
    store.save(inc)
    fresh = store.load(inc.id)
    assert fresh.version == 2


def test_save_with_stale_version_raises(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    a = store.load(inc.id)
    b = store.load(inc.id)
    store.save(a)  # bumps to 2
    with pytest.raises(StaleVersionError):
        store.save(b)


def test_create_starts_at_version_one(store):
    inc = store.create(query="q", environment="dev",
                       reporter_id="u", reporter_team="t")
    assert inc.version == 1
    fresh = store.load(inc.id)
    assert fresh.version == 1
