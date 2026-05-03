"""find_similar tests — embedding ordering and dialect dispatch.

Uses a FAISS-backed store + history with a stub embedder. Postgres
parity is gated behind ``POSTGRES_TEST_URL`` env var; if unset, the
postgres params skip.

P2-J: migrated off the legacy ``IncidentRepository`` facade. The active
CRUD lives on ``SessionStore``; similarity lookups on ``HistoryStore``.
"""
from __future__ import annotations
import pytest

from examples.incident_management.state import IncidentState
from runtime.config import (
    EmbeddingConfig, MetadataConfig, ProviderConfig, VectorConfig,
)
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store


@pytest.fixture
def stores(tmp_path):
    db = tmp_path / "test.db"
    eng = build_engine(MetadataConfig(url=f"sqlite:///{db}"))
    Base.metadata.create_all(eng)

    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=8),
        {"s": ProviderConfig(kind="stub")},
    )
    vec_path = str(tmp_path / "vs")
    vs = build_vector_store(
        VectorConfig(backend="faiss", path=vec_path,
                     collection_name="t", distance_strategy="cosine"),
        embedder,
    )
    store = SessionStore(
        engine=eng, state_cls=IncidentState,
        embedder=embedder, vector_store=vs,
        vector_path=vec_path, vector_index_name="t",
        distance_strategy="cosine",
    )
    history = HistoryStore(
        engine=eng, state_cls=IncidentState,
        embedder=embedder, vector_store=vs,
        distance_strategy="cosine", similarity_threshold=0.0,
    )
    return store, history


def test_find_similar_returns_self_first(stores):
    store, history = stores
    a = store.create(query="redis OOMKill in payments", environment="production",
                     reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    a.resolution = "raise memory"
    store.save(a)
    store.create(query="ssh down on bastion", environment="production",
                 reporter_id="u", reporter_team="t")  # noise
    hits = history.find_similar(query="redis OOMKill in payments",
                                filter_kwargs={"environment": "production"})
    assert hits, "no hits returned"
    top_inc, top_score = hits[0]
    assert top_inc.id == a.id
    assert top_score > 0.99


def test_find_similar_filters_by_environment(stores):
    store, history = stores
    a = store.create(query="match me", environment="production",
                     reporter_id="u", reporter_team="t")
    a.status = "resolved"
    store.save(a)
    b = store.create(query="match me", environment="staging",
                     reporter_id="u", reporter_team="t")
    b.status = "resolved"
    store.save(b)
    hits = history.find_similar(query="match me",
                                filter_kwargs={"environment": "production"})
    assert {h[0].id for h in hits} == {a.id}


def test_find_similar_excludes_unresolved(stores):
    store, history = stores
    a = store.create(query="hello", environment="dev",
                     reporter_id="u", reporter_team="t")
    store.save(a)
    hits = history.find_similar(query="hello",
                                filter_kwargs={"environment": "dev"})
    assert hits == []


def test_find_similar_keyword_fallback_when_no_embedder(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/k.db"))
    Base.metadata.create_all(eng)
    store = SessionStore(engine=eng, state_cls=IncidentState, embedder=None)
    history = HistoryStore(engine=eng, state_cls=IncidentState, embedder=None,
                           similarity_threshold=0.0)
    a = store.create(query="redis OOM", environment="production",
                     reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    store.save(a)
    hits = history.find_similar(query="redis OOM",
                                filter_kwargs={"environment": "production"})
    assert hits and hits[0][0].id == a.id


def test_find_similar_threshold_excludes_low_scores(stores):
    store, history = stores
    a = store.create(query="alpha", environment="dev",
                     reporter_id="u", reporter_team="t")
    a.status = "resolved"
    store.save(a)
    history.similarity_threshold = 0.999
    hits = history.find_similar(query="zeta differs entirely",
                                filter_kwargs={"environment": "dev"})
    assert hits == []
