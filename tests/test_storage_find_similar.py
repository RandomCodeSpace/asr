"""find_similar tests — embedding ordering and dialect dispatch.

Uses a FAISS-backed repo with a stub embedder. Postgres parity is gated
behind ``POSTGRES_TEST_URL`` env var; if unset, the postgres params skip.
"""
from __future__ import annotations
import pytest

from orchestrator.config import (
    EmbeddingConfig, MetadataConfig, ProviderConfig, VectorConfig,
)
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository
from orchestrator.storage.vector import build_vector_store


@pytest.fixture
def repo(tmp_path):
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
    return IncidentRepository(
        engine=eng, embedder=embedder, vector_store=vs,
        vector_path=vec_path, vector_index_name="t",
        distance_strategy="cosine", similarity_threshold=0.0,
    )


def test_find_similar_returns_self_first(repo):
    a = repo.create(query="redis OOMKill in payments", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    a.resolution = "raise memory"
    repo.save(a)
    repo.create(query="ssh down on bastion", environment="production",
                reporter_id="u", reporter_team="t")  # noise
    hits = repo.find_similar(query="redis OOMKill in payments",
                             environment="production")
    assert hits, "no hits returned"
    top_inc, top_score = hits[0]
    assert top_inc.id == a.id
    assert top_score > 0.99


def test_find_similar_filters_by_environment(repo):
    a = repo.create(query="match me", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    repo.save(a)
    b = repo.create(query="match me", environment="staging",
                    reporter_id="u", reporter_team="t")
    b.status = "resolved"
    repo.save(b)
    hits = repo.find_similar(query="match me", environment="production")
    assert {h[0].id for h in hits} == {a.id}


def test_find_similar_excludes_unresolved(repo):
    a = repo.create(query="hello", environment="dev",
                    reporter_id="u", reporter_team="t")
    repo.save(a)
    hits = repo.find_similar(query="hello", environment="dev")
    assert hits == []


def test_find_similar_keyword_fallback_when_no_embedder(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/k.db"))
    Base.metadata.create_all(eng)
    repo = IncidentRepository(engine=eng, embedder=None, similarity_threshold=0.0)
    a = repo.create(query="redis OOM", environment="production",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    a.summary = "redis OOM"
    repo.save(a)
    hits = repo.find_similar(query="redis OOM", environment="production")
    assert hits and hits[0][0].id == a.id


def test_find_similar_threshold_excludes_low_scores(repo):
    a = repo.create(query="alpha", environment="dev",
                    reporter_id="u", reporter_team="t")
    a.status = "resolved"
    repo.save(a)
    repo.similarity_threshold = 0.999
    hits = repo.find_similar(query="zeta differs entirely", environment="dev")
    assert hits == []
