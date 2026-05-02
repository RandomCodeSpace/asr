"""Custom SQLAlchemy column types that bridge SQLite and Postgres.

- ``VectorColumn(dim)`` — ``pgvector.sqlalchemy.Vector(dim)`` on Postgres,
  ``LargeBinary`` (numpy float32 bytes) on SQLite. Python value is always
  ``list[float] | None`` regardless of dialect.
- ``JSONColumn`` — thin alias over SQLAlchemy ``JSON``; auto-routes to
  ``JSONB`` on Postgres, ``TEXT`` on SQLite. Centralized so we can adjust
  serialization (e.g. datetime support) in one place later.
"""
from __future__ import annotations
import numpy as np
from sqlalchemy import JSON, LargeBinary
from sqlalchemy.types import TypeDecorator


JSONColumn = JSON  # SQLAlchemy auto-dialects: JSONB on pg, TEXT on sqlite.


class VectorColumn(TypeDecorator):
    """Vector column backed by pgvector on Postgres, BLOB on SQLite.

    Python value: ``list[float] | None``.
    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.dim,):
            raise ValueError(
                f"vector dim {arr.shape[0]} != column dim {self.dim}"
            )
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return np.frombuffer(value, dtype=np.float32).tolist()
