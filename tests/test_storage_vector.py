"""build_vector_store factory + FAISS persistence + stub-embedder smoke."""
from __future__ import annotations
from pathlib import Path
import pytest

from orchestrator.config import EmbeddingConfig, ProviderConfig, VectorConfig
from orchestrator.storage.embeddings import build_embedder


def _stub_embedder(dim: int = 8):
    return build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=dim),
        {"s": ProviderConfig(kind="stub")},
    )


def test_build_vector_store_none():
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="none")
    assert build_vector_store(cfg, _stub_embedder()) is None


def test_build_vector_store_no_embedder_returns_none():
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="faiss", path="/tmp/x")
    assert build_vector_store(cfg, None) is None


def test_build_vector_store_faiss_roundtrip(tmp_path: Path):
    from langchain_core.documents import Document
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="faiss", path=str(tmp_path / "vs"),
                       collection_name="t", distance_strategy="cosine")
    embedder = _stub_embedder()
    vs = build_vector_store(cfg, embedder)
    assert vs is not None
    vs.add_documents(
        [Document(page_content="hello world", metadata={"id": "INC-1"})],
        ids=["INC-1"],
    )
    vs.save_local(folder_path=cfg.path, index_name=cfg.collection_name)
    # Reload from disk via the factory:
    vs2 = build_vector_store(cfg, embedder)
    hits = vs2.similarity_search_with_score("hello world", k=1)
    assert hits and hits[0][0].metadata["id"] == "INC-1"


def test_build_vector_store_faiss_delete(tmp_path: Path):
    from langchain_core.documents import Document
    from orchestrator.storage.vector import build_vector_store
    cfg = VectorConfig(backend="faiss", path=str(tmp_path / "vs"),
                       collection_name="t")
    embedder = _stub_embedder()
    vs = build_vector_store(cfg, embedder)
    vs.add_documents(
        [Document(page_content="x", metadata={"id": "A"}),
         Document(page_content="y", metadata={"id": "B"})],
        ids=["A", "B"],
    )
    vs.delete(ids=["A"])
    hits = vs.similarity_search_with_score("x", k=5)
    ids_returned = {h[0].metadata.get("id") for h in hits}
    assert "A" not in ids_returned


def test_distance_to_similarity_cosine():
    from orchestrator.storage.vector import distance_to_similarity
    assert distance_to_similarity(0.0, "cosine") == 1.0
    assert abs(distance_to_similarity(2.0, "cosine") - (-1.0)) < 1e-6


def test_distance_to_similarity_euclidean_monotonic():
    from orchestrator.storage.vector import distance_to_similarity
    a = distance_to_similarity(0.0, "euclidean")
    b = distance_to_similarity(1.0, "euclidean")
    c = distance_to_similarity(10.0, "euclidean")
    assert a > b > c
    assert 0.0 < c < b < a <= 1.0


def test_distance_to_similarity_unknown_raises():
    from orchestrator.storage.vector import distance_to_similarity
    with pytest.raises(ValueError, match="unknown distance strategy"):
        distance_to_similarity(0.5, "manhattan")
