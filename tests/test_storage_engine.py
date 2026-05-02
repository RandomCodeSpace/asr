"""Engine smoke tests."""
from __future__ import annotations
import pytest


def test_create_all_smoke(tmp_path):
    from orchestrator.config import MetadataConfig
    from orchestrator.storage.engine import build_engine
    from orchestrator.storage.models import Base
    from sqlalchemy import inspect
    db_url = f"sqlite:///{tmp_path}/test.db"
    eng = build_engine(MetadataConfig(url=db_url))
    Base.metadata.create_all(eng)
    insp = inspect(eng)
    assert "incidents" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("incidents")}
    expected = {
        "id", "status", "created_at", "updated_at", "deleted_at",
        "query", "environment", "reporter_id", "reporter_team",
        "summary", "severity", "category", "matched_prior_inc",
        "resolution", "tags", "agents_run", "tool_calls", "findings",
        "pending_intervention", "user_inputs",
        "input_tokens", "output_tokens", "total_tokens",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"
