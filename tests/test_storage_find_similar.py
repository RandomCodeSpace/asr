"""find_similar tests — embedding ordering and dialect dispatch.

Default backend is SQLite (with sqlite-vec). Postgres parity is gated
behind ``POSTGRES_TEST_URL`` env var; if unset, the postgres params skip.
"""
from __future__ import annotations
import os
import pytest

from orchestrator.config import (
    EmbeddingConfig, ProviderConfig, StorageConfig,
)
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository


def _engine_params() -> list[tuple[str, str]]:
    out = [("sqlite", "")]  # url filled in per-test from tmp_path
    pg = os.environ.get("POSTGRES_TEST_URL")
    if pg:
        out.append(("postgres", pg))
    return out


@pytest.fixture(params=_engine_params(), ids=lambda p: p[0])
def repo(request, tmp_path):
    kind, url = request.param
    if kind == "sqlite":
        url = f"sqlite:///{tmp_path}/test.db"
    eng = build_engine(StorageConfig(url=url))
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return IncidentRepository(engine=eng, embedder=embedder, similarity_threshold=0.0)


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
    eng = build_engine(StorageConfig(url=f"sqlite:///{tmp_path}/k.db"))
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
