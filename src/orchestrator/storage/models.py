"""SQLAlchemy declarative model for the ``incidents`` table.

Hybrid schema: scalar/queryable fields as columns, nested Pydantic
structures as JSON columns (JSONB on Postgres, TEXT on SQLite).
Vector similarity lives in a separate LangChain VectorStore (landed in M3).
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IncidentRow(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(String, nullable=False)
    reporter_id: Mapped[str] = mapped_column(String, nullable=False)
    reporter_team: Mapped[str] = mapped_column(String, nullable=False)

    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    matched_prior_inc: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)

    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    agents_run: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    findings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pending_intervention: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    user_inputs: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_incidents_status_env_active", "status", "environment",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
        Index("ix_incidents_created_at_active", "created_at",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
    )
