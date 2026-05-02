"""SQLAlchemy engine factory.

Sync engine for SQLite (dev) or Postgres (prod). No vector-extension
loading — vectors live in a separate LangChain VectorStore (see
:mod:`orchestrator.storage.vector`, landed in M3).
"""
from __future__ import annotations
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool

from orchestrator.config import MetadataConfig


def build_engine(cfg: MetadataConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        return create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )
    return create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
