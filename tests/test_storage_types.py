"""Round-trip serialization tests for the custom SQLAlchemy types.

We use a tiny in-memory SQLite DB rather than mocking dialects — closer to
production, less brittle than dialect mocks.
"""
from __future__ import annotations
import numpy as np
import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, select


@pytest.fixture
def sqlite_engine():
    return create_engine("sqlite:///:memory:")


def test_vector_column_round_trip_sqlite(sqlite_engine):
    from orchestrator.storage.types import VectorColumn
    md = MetaData()
    t = Table("v", md, Column("id", Integer, primary_key=True),
              Column("vec", VectorColumn(4), nullable=True))
    md.create_all(sqlite_engine)
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "vec": [0.1, 0.2, 0.3, 0.4]}])
        conn.commit()
        row = conn.execute(select(t.c.vec)).scalar_one()
    assert isinstance(row, list)
    assert len(row) == 4
    np.testing.assert_allclose(row, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)


def test_vector_column_handles_none_sqlite(sqlite_engine):
    from orchestrator.storage.types import VectorColumn
    md = MetaData()
    t = Table("v", md, Column("id", Integer, primary_key=True),
              Column("vec", VectorColumn(4), nullable=True))
    md.create_all(sqlite_engine)
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "vec": None}])
        conn.commit()
        assert conn.execute(select(t.c.vec)).scalar_one() is None


def test_json_column_round_trip_sqlite(sqlite_engine):
    from orchestrator.storage.types import JSONColumn
    md = MetaData()
    t = Table("j", md, Column("id", Integer, primary_key=True),
              Column("payload", JSONColumn, nullable=False))
    md.create_all(sqlite_engine)
    payload = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
    with sqlite_engine.connect() as conn:
        conn.execute(t.insert(), [{"id": 1, "payload": payload}])
        conn.commit()
        out = conn.execute(select(t.c.payload)).scalar_one()
    assert out == payload
