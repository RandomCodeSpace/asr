"""SQLAlchemy-backed storage layer for incidents and embeddings.

Public surface
--------------
- ``IncidentRepository`` — replaces the old JSON ``IncidentStore``.
- ``build_engine``       — engine factory (sqlite + sqlite-vec, postgres + pgvector).
- ``build_embedder``     — LangChain ``Embeddings`` factory.
- ``Base``, ``IncidentRow`` — declarative model (exposed for tests/migrations).
"""
from orchestrator.storage.engine import build_engine
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.models import Base, IncidentRow
from orchestrator.storage.repository import IncidentRepository
from orchestrator.storage.vector import build_vector_store

__all__ = [
    "Base",
    "IncidentRepository",
    "IncidentRow",
    "build_embedder",
    "build_engine",
    "build_vector_store",
]
