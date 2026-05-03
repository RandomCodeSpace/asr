"""SQLAlchemy-backed storage layer for incidents and embeddings.

Public surface
--------------
- ``SessionStore``       — active CRUD + vector write-through.
- ``HistoryStore``       — read-only similarity search over closed sessions.
- ``build_engine``       — engine factory (sqlite + sqlite-vec, postgres + pgvector).
- ``build_embedder``     — LangChain ``Embeddings`` factory.
- ``Base``, ``IncidentRow``, ``SessionRow`` — declarative model + generic alias.

P2-J removed the ``IncidentRepository`` facade; callers now consume
``SessionStore`` and ``HistoryStore`` directly.
"""
from runtime.storage.engine import build_engine
from runtime.storage.embeddings import build_embedder
from runtime.storage.history_store import HistoryStore
from runtime.storage.migrations import migrate_add_session_columns, migrate_tool_calls_audit
from runtime.storage.models import Base, IncidentRow, SessionRow
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store

__all__ = [
    "Base",
    "HistoryStore",
    "IncidentRow",
    "SessionRow",
    "SessionStore",
    "build_embedder",
    "build_engine",
    "build_vector_store",
    "migrate_add_session_columns",
    "migrate_tool_calls_audit",
]
