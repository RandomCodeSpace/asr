"""SQLAlchemy-backed storage layer for incidents and embeddings.

Public surface
--------------
- ``SessionStore``       — active CRUD + vector write-through.
- ``HistoryStore``       — read-only similarity search over closed sessions.
- ``EventLog``           — append-only per-session telemetry sink.
- ``build_engine``       — engine factory (sqlite + sqlite-vec, postgres + pgvector).
- ``build_embedder``     — LangChain ``Embeddings`` factory.
- ``Base``, ``IncidentRow``, ``SessionRow`` — declarative model + generic alias.

Callers consume ``SessionStore`` and ``HistoryStore`` directly.
"""
from runtime.storage.engine import build_engine
from runtime.storage.embeddings import build_embedder
from runtime.storage.event_log import EventLog
from runtime.storage.history_store import HistoryStore
from runtime.storage.lesson_store import LessonStore
from runtime.storage.migrations import (
    migrate_add_lesson_table,
    migrate_add_session_columns,
    migrate_tool_calls_audit,
)
from runtime.storage.models import Base, IncidentRow, SessionLessonRow, SessionRow
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store

__all__ = [
    "Base",
    "EventLog",
    "HistoryStore",
    "IncidentRow",
    "LessonStore",
    "SessionLessonRow",
    "SessionRow",
    "SessionStore",
    "build_embedder",
    "build_engine",
    "build_vector_store",
    "migrate_add_lesson_table",
    "migrate_add_session_columns",
    "migrate_tool_calls_audit",
]
