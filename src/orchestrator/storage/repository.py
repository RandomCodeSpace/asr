"""SQLAlchemy-backed Incident store.

Public methods mirror the previous JSON ``IncidentStore`` 1:1 so call
sites in the MCP server and orchestrator change minimally. The repository
also owns the embedder; ``find_similar`` (Task G) does the dialect dispatch.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.embeddings import Embeddings
from sqlalchemy import and_, desc, func, literal, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orchestrator.incident import (
    AgentRun, Incident, Reporter, TokenUsage, ToolCall,
)
from orchestrator.storage.models import IncidentRow

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")


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
        similarity_threshold: float = 0.85,
        severity_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
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
                embedding=self._maybe_embed(query),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

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
            new_embedding = self._compute_save_embedding(existing, incident)
            data = self._incident_to_row_dict(incident, embedding=new_embedding)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()

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

    # ---------- similarity search ----------
    def find_similar(
        self, *, query: str, environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Incident, float]]:
        """Return up to ``limit`` similar resolved incidents for the same env.

        Embedding path uses native vector ops (pgvector ``cosine_distance`` /
        sqlite-vec ``vec_distance_cosine``). Keyword path falls back to
        the existing ``KeywordSimilarity`` scorer to preserve behaviour
        when no embedder is configured.
        """
        if self.embedder is None:
            return self._keyword_similar(
                query=query, environment=environment,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        return self._vector_similar(
            query=query, environment=environment,
            status_filter=status_filter,
            threshold=threshold, limit=limit,
        )

    def _vector_similar(self, *, query, environment, status_filter, threshold, limit):
        import numpy as np
        vec = self.embedder.embed_query(query)
        threshold = self.similarity_threshold if threshold is None else threshold
        with Session(self.engine) as session:
            if self.engine.dialect.name == "postgresql":
                score = (literal(1.0) - IncidentRow.embedding.cosine_distance(vec)).label("score")
            else:
                blob = np.asarray(vec, dtype=np.float32).tobytes()
                score = (literal(1.0) - func.vec_distance_cosine(IncidentRow.embedding, blob)).label("score")
            stmt = (
                select(IncidentRow, score)
                .where(and_(
                    IncidentRow.deleted_at.is_(None),
                    IncidentRow.status == status_filter,
                    IncidentRow.environment == environment,
                    IncidentRow.embedding.is_not(None),
                ))
                .order_by(desc("score"))
                .limit(limit)
            )
            rows = session.execute(stmt).all()
        out: list[tuple[Incident, float]] = []
        for row, s in rows:
            s = float(s)
            if s < threshold:
                continue
            out.append((self._row_to_incident(row), s))
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
            embedding=row.embedding,
            agents_run=agents_run,
            tool_calls=tool_calls,
            findings=dict(row.findings or {}),
            resolution=row.resolution,
            token_usage=token_usage,
            pending_intervention=row.pending_intervention,
            user_inputs=list(row.user_inputs or []),
        )

    def _incident_to_row_dict(
        self, inc: Incident, *, embedding: Optional[list[float]],
    ) -> dict:
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
            "resolution": inc.resolution,
            "tags": list(inc.tags),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "embedding": embedding,
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
        }

    # ---------- embedding lifecycle ----------
    def _maybe_embed(self, text: str) -> Optional[list[float]]:
        if self.embedder is None or not text:
            return None
        return self.embedder.embed_query(text)

    def _compute_save_embedding(
        self, existing: Optional[IncidentRow], inc: Incident,
    ) -> Optional[list[float]]:
        """Re-embed only when the source text materially changed."""
        if self.embedder is None:
            return existing.embedding if existing is not None else None
        text = _embed_source(inc)
        if existing is not None:
            prior = _embed_source_from_row(existing)
            if prior == text and existing.embedding is not None:
                return existing.embedding
        return self.embedder.embed_query(text) if text else None


def _embed_source(inc: Incident) -> str:
    return (inc.query or "").strip()


def _embed_source_from_row(row: IncidentRow) -> str:
    return (row.query or "").strip()
