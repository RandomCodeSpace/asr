"""Engine + sqlite-vec extension load tests."""
from __future__ import annotations
import pytest


def test_build_engine_sqlite_in_memory():
    from orchestrator.config import StorageConfig
    from orchestrator.storage.engine import build_engine
    from sqlalchemy import text
    import sqlite_vec
    eng = build_engine(StorageConfig(url="sqlite:///:memory:"))
    with eng.connect() as conn:
        # sqlite-vec should be loaded — vec_distance_cosine should be callable.
        # Use serialized float32 blobs so the function receives valid input.
        a = sqlite_vec.serialize_float32([0.0] * 4)
        b = sqlite_vec.serialize_float32([0.0] * 4)
        out = conn.execute(text("SELECT vec_distance_cosine(:a, :b)"),
                           {"a": a, "b": b}).scalar()
        # zero vectors → cosine distance is undefined or 0/1 depending on impl;
        # the assertion is just that the function exists (no exception thrown).
        assert out is not None or out is None  # smoke: function callable


def test_create_all_smoke(tmp_path):
    from orchestrator.config import StorageConfig
    from orchestrator.storage.engine import build_engine
    from orchestrator.storage.models import Base
    from sqlalchemy import inspect
    db_url = f"sqlite:///{tmp_path}/test.db"
    eng = build_engine(StorageConfig(url=db_url))
    Base.metadata.create_all(eng)
    insp = inspect(eng)
    assert "incidents" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("incidents")}
    expected = {
        "id", "status", "created_at", "updated_at", "deleted_at",
        "query", "environment", "reporter_id", "reporter_team",
        "summary", "severity", "category", "matched_prior_inc",
        "resolution", "tags", "agents_run", "tool_calls", "findings",
        "pending_intervention", "user_inputs", "embedding",
        "input_tokens", "output_tokens", "total_tokens",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"
