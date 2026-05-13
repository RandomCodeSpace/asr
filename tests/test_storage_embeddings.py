"""Embedding facade tests — stub determinism and provider dispatch."""
from __future__ import annotations
import pytest


def test_stub_embeddings_determinism():
    from runtime.storage.embeddings import _StubEmbeddings
    e = _StubEmbeddings(dim=8)
    a = e.embed_query("hello world")
    b = e.embed_query("hello world")
    c = e.embed_query("different")
    assert a == b
    assert a != c
    assert len(a) == 8
    assert all(isinstance(x, float) for x in a)


def test_stub_embed_documents_returns_list_of_lists():
    from runtime.storage.embeddings import _StubEmbeddings
    e = _StubEmbeddings(dim=4)
    out = e.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 4 for v in out)


def test_build_embedder_stub():
    from runtime.config import EmbeddingConfig, ProviderConfig
    from runtime.storage.embeddings import build_embedder
    cfg = EmbeddingConfig(provider="s", model="x", dim=8)
    providers = {"s": ProviderConfig(kind="stub")}
    e = build_embedder(cfg, providers)
    assert e is not None
    v = e.embed_query("test")
    assert len(v) == 8


def test_build_embedder_none_returns_none():
    from runtime.storage.embeddings import build_embedder
    assert build_embedder(None, {}) is None


def test_build_embedder_unknown_kind_raises():
    from runtime.config import EmbeddingConfig, ProviderConfig
    from runtime.storage.embeddings import build_embedder
    cfg = EmbeddingConfig(provider="x", model="m")
    # Phase 13 (HARD-05): ollama now requires base_url at config-load,
    # so seed from a no-required-field kind (stub) and mutate to "nonsense"
    # to exercise the unknown-kind dispatch path.
    bad = ProviderConfig(kind="stub")
    bad.kind = "nonsense"  # bypass pydantic for the test
    with pytest.raises(ValueError, match="unknown provider kind"):
        build_embedder(cfg, {"x": bad})
