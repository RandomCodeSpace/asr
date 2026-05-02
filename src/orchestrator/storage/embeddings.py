"""LangChain ``Embeddings`` facade + deterministic stub for tests.

Construction is config-driven via :func:`build_embedder`. Provider kind
dispatches to ``OllamaEmbeddings`` / ``AzureOpenAIEmbeddings`` / a local
stub. Stubs are deterministic so tests can assert similarity ordering
without external services.
"""
from __future__ import annotations
import hashlib
import numpy as np
from langchain_core.embeddings import Embeddings

from orchestrator.config import EmbeddingConfig, ProviderConfig


class _StubEmbeddings(Embeddings):
    """Deterministic dummy embedder.

    Same text → same vector; different texts → different vectors. Useful
    for CI and unit tests without a network or model server.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        # Normalize to unit length so cosine similarity = dot product in [−1, 1].
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(
    cfg: EmbeddingConfig | None,
    providers: dict[str, ProviderConfig],
) -> Embeddings | None:
    """Build a LangChain ``Embeddings`` from config; ``None`` if not configured."""
    if cfg is None:
        return None
    p = providers[cfg.provider]
    if p.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.model,
            base_url=p.base_url or "http://localhost:11434",
        )
    if p.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=cfg.deployment,
            model=cfg.model,
            azure_endpoint=p.endpoint,
            api_version=p.api_version,
            api_key=p.api_key,
        )
    if p.kind == "stub":
        return _StubEmbeddings(dim=cfg.dim)
    raise ValueError(f"unknown provider kind: {p.kind!r}")
