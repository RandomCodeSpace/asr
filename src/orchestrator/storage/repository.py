"""SQLAlchemy-backed Incident store.

Public methods mirror the previous JSON ``IncidentStore`` 1:1 so call
sites in the MCP server and orchestrator change minimally. The repository
also owns the embedder and vector_store; ``find_similar`` dispatches to
the vector path when both are present, else falls back to keyword similarity.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orchestrator.incident import (
    AgentRun, Incident, Reporter, TokenUsage, ToolCall,
)
from orchestrator.storage.models import IncidentRow

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")


def _embed_source(inc: "Incident") -> str:
    """Produce the text that represents an incident in the vector store.

    Uses query only so that ``find_similar(query=inc.query)`` retrieves the
    same vector (enabling self-match and cross-incident similarity by query
    semantics, regardless of how the summary evolves).
    """
    return (inc.query or "").strip()


def _embed_source_from_row(row: "IncidentRow") -> str:
    """Same as ``_embed_source`` but from a raw ORM row (used in save)."""
    return (row.query or "").strip()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now().strftime("%Y%m%d")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """DB datetime -> Incident model ISO string (UTC, 'Z' suffix)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Incident ISO string -> DB datetime (UTC-aware)."""
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _deserialize_resolution(raw: Optional[str]):
    """Attempt JSON parse of stored resolution; return raw string on failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class IncidentRepository:
    """SQLAlchemy-backed Incident store. Drop-in for the old ``IncidentStore``.

    Threading note: methods open short-lived sessions; safe for the
    orchestrator's coarse-grained concurrency model.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        vector_path: Optional[str] = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.85,
        severity_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_path = vector_path
        self.vector_index_name = vector_index_name
        self.distance_strategy = distance_strategy
        self.similarity_threshold = similarity_threshold
        self.severity_aliases = severity_aliases or {}

    # ---------- ID minting ----------
    def _next_id(self, session: Session) -> str:
        prefix = f"INC-{_today_str()}-"
        like = f"{prefix}%"
        rows = session.execute(
            select(IncidentRow.id).where(IncidentRow.id.like(like))
        ).scalars().all()
        max_seq = 0
        for r in rows:
            try:
                max_seq = max(max_seq, int(r.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"{prefix}{max_seq + 1:03d}"

    # ---------- public API ----------
    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> Incident:
        with Session(self.engine) as session:
            now = _now()
            inc_id = self._next_id(session)
            row = IncidentRow(
                id=inc_id,
                status="new",
                created_at=now,
                updated_at=now,
                query=query,
                environment=environment,
                reporter_id=reporter_id,
                reporter_team=reporter_team,
                summary="",
                tags=[],
                agents_run=[],
                tool_calls=[],
                findings={},
                user_inputs=[],
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            inc = self._row_to_incident(row)
        self._add_vector(inc)
        return inc

    def load(self, incident_id: str) -> Incident:
        if not _INC_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected INC-YYYYMMDD-NNN"
            )
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def save(self, incident: Incident) -> None:
        if not _INC_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected INC-YYYYMMDD-NNN"
            )
        incident.updated_at = _iso(_now())
        with Session(self.engine) as session:
            existing = session.get(IncidentRow, incident.id)
            prior_text = _embed_source_from_row(existing) if existing is not None else ""
            data = self._incident_to_row_dict(incident)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()
        self._refresh_vector(incident, prior_text=prior_text)

    def delete(self, incident_id: str) -> Incident:
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            if row.status != "deleted":
                row.status = "deleted"
                row.deleted_at = _now()
                row.pending_intervention = None
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    def list_all(self, *, include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_recent(self, limit: int = 20, *,
                    include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            stmt = stmt.order_by(
                desc(IncidentRow.created_at), desc(IncidentRow.id)
            ).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    # ---------- vector helpers ----------
    def _persist_vector(self) -> None:
        """If FAISS-backed (has save_local) and a path is configured, persist to disk."""
        if self.vector_store is None:
            return
        if not hasattr(self.vector_store, "save_local"):
            return
        if not self.vector_path:
            return
        from pathlib import Path
        folder = Path(self.vector_path)
        folder.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(
            folder_path=str(folder),
            index_name=self.vector_index_name,
        )

    def _add_vector(self, inc: "Incident") -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    def _refresh_vector(self, inc: "Incident", *, prior_text: str) -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        if prior_text == text:
            return
        self.vector_store.delete(ids=[inc.id])
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    # ---------- similarity search ----------
    def find_similar(
        self, *, query: str, environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Incident, float]]:
        """Return up to ``limit`` similar resolved incidents for the same env.

        Vector path: uses the configured VectorStore when both vector_store
        and embedder are present. Falls back to keyword similarity otherwise.
        """
        if self.vector_store is None or self.embedder is None:
            return self._keyword_similar(
                query=query, environment=environment,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        threshold = self.similarity_threshold if threshold is None else threshold
        from orchestrator.storage.vector import distance_to_similarity
        vec = self.embedder.embed_query(query)
        raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit * 4)
        out: list[tuple[Incident, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(float(distance), self.distance_strategy)
            if score < threshold:
                continue
            inc_id = doc.metadata.get("id")
            if inc_id is None:
                continue
            try:
                inc = self.load(inc_id)
            except (FileNotFoundError, ValueError):
                continue
            if (inc.environment != environment
                    or inc.status != status_filter
                    or inc.deleted_at is not None):
                continue
            out.append((inc, score))
            if len(out) >= limit:
                break
        return out

    def _keyword_similar(self, *, query, environment, status_filter, threshold, limit):
        from orchestrator.similarity import KeywordSimilarity, find_similar
        candidates_inc = [
            i for i in self.list_all()
            if i.environment == environment
            and i.status == status_filter
            and i.deleted_at is None
        ]
        candidates = [
            {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
             "incident": i}
            for i in candidates_inc
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(c["incident"], float(s)) for c, s in results]

    # ---------- mapping helpers ----------
    def _row_to_incident(self, row: IncidentRow) -> Incident:
        agents_run = [AgentRun.model_validate(a) for a in (row.agents_run or [])]
        tool_calls = [ToolCall.model_validate(t) for t in (row.tool_calls or [])]
        token_usage = TokenUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
        )
        return Incident(
            id=row.id,
            status=row.status,
            created_at=_iso(row.created_at),
            updated_at=_iso(row.updated_at),
            deleted_at=_iso(row.deleted_at) if row.deleted_at else None,
            query=row.query,
            environment=row.environment,
            reporter=Reporter(id=row.reporter_id, team=row.reporter_team),
            summary=row.summary or "",
            tags=list(row.tags or []),
            severity=row.severity,
            category=row.category,
            matched_prior_inc=row.matched_prior_inc,
            agents_run=agents_run,
            tool_calls=tool_calls,
            findings=dict(row.findings or {}),
            resolution=_deserialize_resolution(row.resolution),
            token_usage=token_usage,
            pending_intervention=row.pending_intervention,
            user_inputs=list(row.user_inputs or []),
        )

    def _incident_to_row_dict(self, inc: Incident) -> dict:
        return {
            "id": inc.id,
            "status": inc.status,
            "created_at": _parse_iso(inc.created_at),
            "updated_at": _parse_iso(inc.updated_at),
            "deleted_at": _parse_iso(inc.deleted_at) if inc.deleted_at else None,
            "query": inc.query,
            "environment": inc.environment,
            "reporter_id": inc.reporter.id,
            "reporter_team": inc.reporter.team,
            "summary": inc.summary or "",
            "severity": inc.severity,
            "category": inc.category,
            "matched_prior_inc": inc.matched_prior_inc,
            "resolution": (
                inc.resolution if inc.resolution is None or isinstance(inc.resolution, str)
                else json.dumps(inc.resolution)
            ),
            "tags": list(inc.tags),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
        }

