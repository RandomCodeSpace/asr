"""Read-only similarity search over closed sessions.

``HistoryStore`` is the read-side companion of ``SessionStore``: it
operates on the same engine + vector store but never writes. The vector
path is preferred when both ``vector_store`` and ``embedder`` are
configured; otherwise it falls back to keyword similarity.

Like ``SessionStore``, ``HistoryStore`` is parametrised on ``StateT`` so
that find_similar / load surface the configured app state class rather
than the framework default.

``find_similar`` accepts an arbitrary ``filter_kwargs`` mapping — keys
must correspond to ``IncidentRow`` columns. This decouples the
framework from app-specific filter dimensions: apps with a
schema with a single status-tier field, a multi-tenant ``tenant_id`` schema, or
anything else, build their filter on the fly.
"""
from __future__ import annotations
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlaSession

from runtime.state import Session
from runtime.storage.models import IncidentRow

# Mirrors the bound on ``SessionStore.StateT`` — tightened from
# ``BaseModel`` to ``runtime.state.Session`` in Phase 19 (HARD-03) so
# pyright sees the typed fields (``id``, ``status``, ``deleted_at`` …)
# this store reads. The resolver in :mod:`runtime.state_resolver`
# already enforces a ``Session`` subclass at config time, and every
# in-tree caller passes either bare ``Session`` or a ``Session``
# subclass.
StateT = TypeVar("StateT", bound=Session)

# Allowed ``filter_kwargs`` keys = IncidentRow column names.
# Computed at module load so we can produce a precise error for typos.
_ALLOWED_FILTER_COLUMNS: frozenset[str] = frozenset(
    c.name for c in IncidentRow.__table__.columns
)


class HistoryStore(Generic[StateT]):
    """Read-only similarity search over the same row store, parametrised on ``StateT``.

    Never mutates. Reuses ``SessionStore``'s row->state converter via a
    private internal instance to avoid duplicating mapping logic; that
    converter inherits the same ``state_cls`` so hydration stays consistent.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        state_cls: Optional[Type[StateT]] = None,
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_threshold: float = 0.85,
        distance_strategy: str = "cosine",
    ) -> None:
        # Imported lazily so a bare ``HistoryStore`` import has no
        # side-effect of pulling SessionStore's heavier dep tree.
        from runtime.storage.session_store import SessionStore

        self.engine = engine
        self._state_cls = state_cls
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.distance_strategy = distance_strategy
        # Private converter helper. We never call its mutating methods.
        # ``state_cls=None`` lets SessionStore pick its own default
        # (``runtime.state.Session``).
        ss_kwargs: dict[str, Any] = {
            "engine": engine, "embedder": None, "vector_store": None,
            "distance_strategy": distance_strategy,
        }
        if state_cls is not None:
            ss_kwargs["state_cls"] = state_cls
        self._converter: SessionStore[StateT] = SessionStore(**ss_kwargs)

    def _row_to_incident(self, row: IncidentRow) -> StateT:
        return self._converter._row_to_incident(row)

    def _load(self, incident_id: str) -> StateT:
        with SqlaSession(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def _list_filtered(self, *, filter_kwargs: Mapping[str, Any]) -> list[StateT]:
        """List non-deleted rows matching the given column filters.

        Pure SQL prefilter — used by both vector and keyword paths.
        """
        with SqlaSession(self.engine) as session:
            stmt = select(IncidentRow).where(IncidentRow.deleted_at.is_(None))
            for col, val in filter_kwargs.items():
                stmt = stmt.where(getattr(IncidentRow, col) == val)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    @staticmethod
    def _validate_filter_kwargs(filter_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
        """Reject filter keys that aren't IncidentRow columns.

        Returns a plain ``dict`` (defensive copy). ``None`` is treated as
        an empty filter for ergonomic callers.
        """
        if filter_kwargs is None:
            return {}
        bad = [k for k in filter_kwargs if k not in _ALLOWED_FILTER_COLUMNS]
        if bad:
            raise ValueError(
                f"unsupported filter_kwargs key(s): {bad}. "
                f"Allowed columns: {sorted(_ALLOWED_FILTER_COLUMNS)}"
            )
        return dict(filter_kwargs)

    def find_similar(
        self, *, query: str,
        filter_kwargs: Mapping[str, Any] | None = None,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
        # Back-compat: accept ``environment=`` so existing callers keep
        # compiling. New code should pass ``filter_kwargs={"environment": ...}``.
        environment: Optional[str] = None,
    ) -> list[tuple[StateT, float]]:
        """Return up to ``limit`` similar sessions matching the given filters.

        ``filter_kwargs`` is a mapping of ``IncidentRow`` column -> value
        (e.g. ``{"environment": "production"}``); each entry becomes an
        equality predicate in the SQL prefilter. ``status_filter`` is
        also applied (defaulting to ``"resolved"``). Vector path uses
        the configured VectorStore when both ``vector_store`` and
        ``embedder`` are present; otherwise keyword similarity.
        """
        filter_dict = self._validate_filter_kwargs(filter_kwargs)
        if environment is not None and "environment" not in filter_dict:
            # Convenience for legacy callers — translate to the new dict.
            filter_dict["environment"] = environment

        if self.vector_store is None or self.embedder is None:
            return self._keyword_similar(
                query=query, filter_kwargs=filter_dict,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        threshold = self.similarity_threshold if threshold is None else threshold
        from runtime.storage.vector import distance_to_similarity
        vec = self.embedder.embed_query(query)
        # ``similarity_search_with_score_by_vector`` is provided by the
        # concrete FAISS / pgvector / langchain-postgres backends (and
        # validated by ``runtime.storage.vector.build_vector_store``)
        # but the abstract ``langchain_core.vectorstores.VectorStore``
        # base class does not declare it.
        raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit * 4)  # pyright: ignore[reportAttributeAccessIssue]
        out: list[tuple[StateT, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(float(distance), self.distance_strategy)
            if score < threshold:
                continue
            inc_id = doc.metadata.get("id")
            if inc_id is None:
                continue
            try:
                inc = self._load(inc_id)
            except (FileNotFoundError, ValueError):
                continue
            if getattr(inc, "status", None) != status_filter:
                continue
            if getattr(inc, "deleted_at", None) is not None:
                continue
            # Generic column-equality check via getattr — works for any
            # field declared on the configured state subclass.
            if not all(getattr(inc, k, None) == v for k, v in filter_dict.items()):
                continue
            out.append((inc, score))
            if len(out) >= limit:
                break
        return out

    def _keyword_similar(self, *, query, filter_kwargs, status_filter, threshold, limit):
        from runtime.similarity import KeywordSimilarity, find_similar
        # SQL prefilter narrows the row set; status is filtered in
        # Python because it lives on the row but apps occasionally
        # override it via custom states.
        all_filtered = self._list_filtered(filter_kwargs=filter_kwargs)
        candidates_inc = [
            i for i in all_filtered
            if getattr(i, "status", None) == status_filter
            and getattr(i, "deleted_at", None) is None
        ]
        def _ef(i, key, default: Any = ""):
            """Read a field from typed attribute first, then extra_fields."""
            val = getattr(i, key, None)
            if val:
                return val
            return (getattr(i, "extra_fields", None) or {}).get(key, default)

        candidates = [
            {
                "id": i.id,
                "text": " ".join(filter(None, [
                    _ef(i, "query", "") or "",
                    _ef(i, "summary", "") or "",
                    " ".join(_ef(i, "tags", []) or []),
                ])),
                "session": i,
            }
            for i in candidates_inc
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(c["session"], float(s)) for c, s in results]
