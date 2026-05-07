"""LangChain ``VectorStore`` factory.

Backends
--------
- ``faiss``    -> ``langchain_community.vectorstores.FAISS`` (file-backed, dev).
- ``pgvector`` -> ``langchain_postgres.PGVector`` (DB-backed, prod).
- ``none``     -> ``None``; caller falls back to keyword similarity.

FAISS persistence: callers invoke :meth:`vector_store.save_local` after
each mutation. The factory loads from disk if a saved index exists at
the configured ``path``; otherwise it constructs an empty index by
seeding with a placeholder doc and immediately deleting it (LangChain's
FAISS constructor doesn't accept an empty docstore).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from runtime.config import VectorConfig


_PLACEHOLDER_ID = "__seed__"


def _faiss_distance_strategy(name: str):
    from langchain_community.vectorstores.utils import DistanceStrategy
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
    }[name]


def _pgvector_distance_strategy(name: str):
    from langchain_postgres.vectorstores import DistanceStrategy
    # ``langchain_postgres.DistanceStrategy.INNER_PRODUCT`` exists at
    # runtime (verified via the live module) but the langchain-postgres
    # stubs only expose ``COSINE`` / ``EUCLIDEAN``.
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN,
        "inner_product": DistanceStrategy.INNER_PRODUCT,  # pyright: ignore[reportAttributeAccessIssue]
    }[name]


def _build_faiss(cfg: VectorConfig, embedder: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    folder = Path(cfg.path)
    index_file = folder / f"{cfg.collection_name}.faiss"
    if index_file.exists():
        return FAISS.load_local(
            folder_path=str(folder),
            index_name=cfg.collection_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
            distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
        )
    folder.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(
        [Document(page_content=_PLACEHOLDER_ID, metadata={"id": _PLACEHOLDER_ID})],
        embedding=embedder,
        ids=[_PLACEHOLDER_ID],
        distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
    )
    vs.delete(ids=[_PLACEHOLDER_ID])
    return vs


def _build_pgvector(cfg: VectorConfig, embedder: Embeddings,
                    engine) -> VectorStore:
    from langchain_postgres import PGVector
    return PGVector(
        embeddings=embedder,
        collection_name=cfg.collection_name,
        connection=engine,
        distance_strategy=_pgvector_distance_strategy(cfg.distance_strategy),
        use_jsonb=True,
    )


def build_vector_store(
    cfg: VectorConfig,
    embedder: Optional[Embeddings],
    metadata_engine=None,
) -> Optional[VectorStore]:
    if cfg.backend == "none" or embedder is None:
        return None
    if cfg.backend == "faiss":
        return _build_faiss(cfg, embedder)
    if cfg.backend == "pgvector":
        if metadata_engine is None:
            raise ValueError(
                "pgvector backend requires metadata_engine (the SQLAlchemy "
                "engine, used as the connection)"
            )
        return _build_pgvector(cfg, embedder, metadata_engine)
    raise ValueError(f"unknown vector backend: {cfg.backend!r}")


def distance_to_similarity(distance: float, strategy: str) -> float:
    """Normalize a backend-native distance to a similarity in roughly [0, 1].

    - cosine: ``1 - distance`` (LangChain's cosine-distance in [0, 2]).
    - inner_product: returns ``distance`` unchanged (already a similarity).
    - euclidean: ``1 / (1 + distance)`` -- monotonic, compressed to (0, 1].
    """
    if strategy == "cosine":
        return 1.0 - distance
    if strategy == "inner_product":
        return distance
    if strategy == "euclidean":
        return 1.0 / (1.0 + distance)
    raise ValueError(f"unknown distance strategy: {strategy!r}")
