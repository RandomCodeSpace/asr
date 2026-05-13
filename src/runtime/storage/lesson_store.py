"""M5: vector-indexed corpus of past resolved sessions ("lessons").

``LessonStore`` mirrors :class:`HistoryStore`'s public surface — ``add``
persists a row + vector embedding, ``find_similar`` runs k-NN over the
corpus and returns the top hits above a threshold.

The relational rows live in ``session_lessons`` (see
:class:`SessionLessonRow`); the embeddings live in whatever LangChain
``VectorStore`` the caller wires (FAISS dir or pgvector collection,
typically ``<vector.path>/lessons`` or collection ``lessons``).

Both writes are best-effort serialised: the relational row is persisted
FIRST so a vector-store failure leaves a recoverable on-disk record
the M7 refresher can re-embed.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlaSession

from runtime.storage.models import SessionLessonRow

_log = logging.getLogger("runtime.storage.lesson_store")


class LessonStore:
    """Append-only lesson corpus with vector similarity lookup.

    Telemetry / refresher writes through ``add(row)``; the intake
    runner reads through ``find_similar(query=...)``.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        vector_store: Optional[VectorStore] = None,
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.7,
    ) -> None:
        self.engine = engine
        self.vector_store = vector_store
        self.distance_strategy = distance_strategy
        self.similarity_threshold = similarity_threshold

    def add(self, lesson: SessionLessonRow) -> None:
        """Persist ``lesson`` to the relational table AND vector store.

        Relational write goes first so a vector-store hiccup is
        recoverable from disk. Vector failures are logged at WARNING
        and swallowed — the row is still discoverable via SQL lookup
        and the M7 refresher can re-embed on next pass.
        """
        # Snapshot the fields the vector-store call needs BEFORE the
        # SQL transaction commits — once the session closes, the row
        # detaches and attribute access raises DetachedInstanceError.
        lesson_id = lesson.id
        embedding_text = lesson.embedding_text
        source_session_id = lesson.source_session_id
        outcome_status = lesson.outcome_status

        with SqlaSession(self.engine) as s:
            with s.begin():
                s.add(lesson)

        if self.vector_store is None:
            return
        try:
            self.vector_store.add_documents(
                [
                    Document(
                        page_content=embedding_text,
                        metadata={
                            "id": lesson_id,
                            "source_session_id": source_session_id,
                            "outcome_status": outcome_status,
                        },
                    )
                ],
                ids=[lesson_id],
            )
        except Exception:  # noqa: BLE001 — vector backends raise a variety
            _log.warning(
                "LessonStore.add: vector_store write failed for lesson %s; "
                "row is still queryable via SQL",
                lesson_id, exc_info=True,
            )

    def find_similar(
        self,
        *,
        query: str,
        limit: int = 3,
        threshold: Optional[float] = None,
    ) -> list[tuple[SessionLessonRow, float]]:
        """Return up to ``limit`` lessons whose vector similarity to the
        embedded ``query`` is at or above ``threshold``. Returns an
        empty list when no vector store is configured.

        Result tuples are ``(row, similarity)`` sorted by descending
        similarity. Soft-deleted source sessions are not filtered here
        — the caller decides whether to honour them (M9 e2e covers the
        soft-delete-suppression contract).
        """
        if self.vector_store is None:
            return []
        threshold = (
            self.similarity_threshold if threshold is None else threshold
        )
        from runtime.storage.vector import distance_to_similarity

        try:
            raw = self.vector_store.similarity_search_with_score(
                query, k=limit * 4,
            )
        except Exception:  # noqa: BLE001
            _log.warning(
                "LessonStore.find_similar: vector_store query failed",
                exc_info=True,
            )
            return []
        out: list[tuple[SessionLessonRow, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(
                float(distance), self.distance_strategy,
            )
            if score < threshold:
                continue
            lid = doc.metadata.get("id")
            if not lid:
                continue
            row = self._load(lid)
            if row is None:
                continue
            out.append((row, score))
            if len(out) >= limit:
                break
        return out

    def _load(self, lesson_id: str) -> Optional[SessionLessonRow]:
        with SqlaSession(self.engine) as s:
            return s.get(SessionLessonRow, lesson_id)
