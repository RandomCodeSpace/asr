"""M5 — LessonStore + LessonExtractor.

Covers:
- ``test_add_persists_row_and_vector``: stub embedder, add one lesson,
  assert the row is in the DB and the vector store has it.
- ``test_find_similar_returns_recent``: add two lessons with distinct
  embedding texts, query, top hit matches.
- ``test_extractor_canonical_form_stable``: same session + same event
  log → identical ``embedding_text`` (snapshot lock).
- ``test_extractor_skips_non_terminal``: extractor called on a
  non-terminal session returns ``None`` and persists no row.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as SqlaSession

from runtime.learning import LessonExtractor
from runtime.state import AgentRun, Session
from runtime.storage import (
    EventLog,
    LessonStore,
    SessionLessonRow,
    migrate_add_lesson_table,
)
from runtime.storage.models import Base


class _DeterministicEmbedder(Embeddings):
    """Tiny embedder that maps each known string to a fixed vector.
    Unknown strings get the zero vector. Deterministic for snapshot
    tests — no real model load required."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Pick the first registered key that appears as a substring.
        for needle, vec in self._table.items():
            if needle in text:
                return vec
        return [0.0] * 4


class _InMemoryVectorStore:
    """Minimal in-memory VectorStore stand-in. Implements just enough
    of the langchain_core.vectorstores.VectorStore surface that
    :class:`LessonStore` uses: ``add_documents(docs, ids=...)`` and
    ``similarity_search_with_score(query, k=...)``."""

    def __init__(self, embedder: Embeddings) -> None:
        self._embedder = embedder
        self._docs: list[Document] = []
        self._vecs: list[list[float]] = []

    def add_documents(self, docs: list[Document], ids: list[str] | None = None) -> list[str]:
        for d, _id in zip(docs, ids or [None] * len(docs)):
            self._docs.append(d)
            self._vecs.append(self._embedder.embed_query(d.page_content))
        return ids or []

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        q = self._embedder.embed_query(query)
        # Cosine-distance equivalent: 1 - cos_sim. LessonStore converts
        # this back to similarity via distance_to_similarity("cosine").
        def _cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)
        scored = [
            (d, 1.0 - _cos(q, v)) for d, v in zip(self._docs, self._vecs)
        ]
        scored.sort(key=lambda t: t[1])
        return scored[:k]


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'lessons.db'}")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def event_log(engine):
    return EventLog(engine=engine)


def _seed_incident(engine, *, sid: str) -> None:
    """The lesson row's source_session_id is FK-constrained to
    incidents.id; for these unit tests we seed a minimal row so the
    insert doesn't violate FK."""
    from runtime.storage.models import IncidentRow
    from datetime import datetime, timezone
    with SqlaSession(engine) as s:
        with s.begin():
            s.add(IncidentRow(
                id=sid,
                status="resolved",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                query="seed",
                environment="dev",
                reporter_id="u", reporter_team="t",
            ))


def _make_terminal_session(*, sid: str, status: str = "resolved") -> Session:
    s = Session(
        id=sid,
        status=status,
        created_at="2026-05-12T00:00:00Z",
        updated_at="2026-05-12T00:00:00Z",
    )
    s.agents_run.append(AgentRun(
        agent="resolution",
        started_at="2026-05-12T00:00:00Z",
        ended_at="2026-05-12T00:00:05Z",
        summary="rolled back the bad deploy",
        confidence=0.91,
        confidence_rationale="rollback verified by service health probe",
        signal="success",
    ))
    return s


# ===================================================================
# Migration sanity
# ===================================================================

def test_migrate_add_lesson_table_idempotent(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'a.db'}")
    Base.metadata.create_all(eng, tables=[])  # don't create lessons here
    # Verify table is missing first.
    from sqlalchemy import inspect as sa_inspect
    assert "session_lessons" not in sa_inspect(eng).get_table_names()
    out = migrate_add_lesson_table(eng)
    assert out == {"tables_added": 1}
    assert "session_lessons" in sa_inspect(eng).get_table_names()
    # Second call is a no-op.
    out2 = migrate_add_lesson_table(eng)
    assert out2 == {"tables_added": 0}


# ===================================================================
# LessonStore.add persists row + vector
# ===================================================================

def test_add_persists_row_and_vector(engine, event_log):
    """One ``LessonStore.add`` writes the row to ``session_lessons``
    and the document to the vector store."""
    _seed_incident(engine, sid="INC-A")
    sess = _make_terminal_session(sid="INC-A")
    row = LessonExtractor.extract(session=sess, event_log=event_log)
    assert row is not None

    embedder = _DeterministicEmbedder({"INC-A": [1.0, 0.0, 0.0, 0.0]})
    vs = _InMemoryVectorStore(embedder)
    store = LessonStore(engine=engine, vector_store=vs)  # pyright: ignore[reportArgumentType]
    # Snapshot the id before add() — the SQL session detaches the row.
    lesson_id = row.id
    store.add(row)

    # Row landed in SQL.
    with SqlaSession(engine) as s:
        rows = s.execute(select(SessionLessonRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].source_session_id == "INC-A"
    assert rows[0].outcome_status == "resolved"
    assert rows[0].provenance["extractor_version"] == "1"

    # Vector landed in the store.
    assert len(vs._docs) == 1
    assert vs._docs[0].metadata["id"] == lesson_id


# ===================================================================
# LessonStore.find_similar k-NN
# ===================================================================

def test_find_similar_returns_recent(engine, event_log):
    """Two lessons with distinct embeddings → query routes to the
    closer one."""
    _seed_incident(engine, sid="INC-DB")
    _seed_incident(engine, sid="INC-CACHE")

    sess_db = _make_terminal_session(sid="INC-DB")
    sess_cache = _make_terminal_session(sid="INC-CACHE")

    embedder = _DeterministicEmbedder({
        "INC-DB":    [1.0, 0.0, 0.0, 0.0],
        "INC-CACHE": [0.0, 1.0, 0.0, 0.0],
    })
    vs = _InMemoryVectorStore(embedder)
    store = LessonStore(
        engine=engine, vector_store=vs,  # pyright: ignore[reportArgumentType]
        similarity_threshold=-1.0,  # accept any match for the test
    )

    row_db = LessonExtractor.extract(session=sess_db, event_log=event_log)
    row_cache = LessonExtractor.extract(session=sess_cache, event_log=event_log)
    assert row_db and row_cache
    store.add(row_db)
    store.add(row_cache)

    # The query string contains "INC-DB" so the embedder picks that vector.
    hits = store.find_similar(query="something about INC-DB and rollback", limit=2)
    assert hits, "expected ≥1 hit"
    top_row, _score = hits[0]
    assert top_row.source_session_id == "INC-DB"


# ===================================================================
# LessonExtractor canonical form is deterministic
# ===================================================================

def test_extractor_canonical_form_stable(event_log):
    """Same session + same event log → identical ``embedding_text``.

    Snapshot lock so M7's idempotency check (compare embedding_text to
    decide whether to re-extract) doesn't silently break when the
    composition formula drifts."""
    sess = _make_terminal_session(sid="INC-SNAP")
    # Seed a couple of tool_invoked events so the canonical form
    # captures a non-empty tool list.
    event_log.append("INC-SNAP", "tool_invoked", {"tool": "get_logs"})
    event_log.append("INC-SNAP", "tool_invoked", {"tool": "rollback_deploy"})

    a: Any = LessonExtractor.extract(session=sess, event_log=event_log)
    b: Any = LessonExtractor.extract(session=sess, event_log=event_log)
    assert a is not None and b is not None
    assert a.embedding_text == b.embedding_text

    expected = (
        f"{sess.to_agent_input()}\n\n"
        f"Outcome: resolved\n"
        f"Key tools: ['get_logs', 'rollback_deploy']\n"
        f"Confidence: 0.91"
    )
    assert a.embedding_text == expected


# ===================================================================
# LessonExtractor skips non-terminal sessions
# ===================================================================

def test_extractor_skips_non_terminal(event_log):
    """Non-terminal status with ``terminal_statuses`` configured -> None.
    No row written, no lesson_extracted event emitted."""
    sess = _make_terminal_session(sid="INC-INPROG", status="in_progress")
    out = LessonExtractor.extract(
        session=sess, event_log=event_log,
        terminal_statuses=frozenset({"resolved", "escalated"}),
    )
    assert out is None
    # No lesson_extracted event was emitted.
    kinds = [e.kind for e in event_log.iter_for("INC-INPROG")]
    assert "lesson_extracted" not in kinds


# ===================================================================
# LessonExtractor emits lesson_extracted on success
# ===================================================================

def test_extractor_emits_lesson_extracted_event(event_log):
    """Successful extraction appends a ``lesson_extracted`` event with
    the new row's id."""
    sess = _make_terminal_session(sid="INC-EMIT")
    row = LessonExtractor.extract(session=sess, event_log=event_log)
    assert row is not None
    events = list(event_log.iter_for("INC-EMIT"))
    emitted = [e for e in events if e.kind == "lesson_extracted"]
    assert len(emitted) == 1
    assert emitted[0].payload["lesson_id"] == row.id
    assert emitted[0].payload["outcome_status"] == "resolved"
