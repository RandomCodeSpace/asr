"""Migration script test: walk fixture JSON dir -> upsert into SQL -> verify rows."""
from __future__ import annotations
import json
from pathlib import Path

from orchestrator.config import (
    AppConfig, IncidentConfig, LLMConfig, MCPConfig, MetadataConfig, Paths,
    StorageConfig,
)


def test_migration_script_idempotent(tmp_path: Path):
    from scripts.migrate_jsonl_to_sql import migrate
    src = tmp_path / "incidents"
    src.mkdir()
    for i, q in enumerate(["redis OOM", "ssh down"], start=1):
        inc_id = f"INC-20260101-{i:03d}"
        (src / f"{inc_id}.json").write_text(json.dumps({
            "id": inc_id, "status": "resolved",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "query": q, "environment": "production",
            "reporter": {"id": "u", "team": "t"},
            "summary": q, "tags": [], "agents_run": [], "tool_calls": [],
            "findings": {}, "user_inputs": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }))
    db_path = tmp_path / "out.db"
    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(),
        incidents=IncidentConfig(store_path=str(src)),
        storage=StorageConfig(metadata=MetadataConfig(url=f"sqlite:///{db_path}")),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(src)),
    )
    out1 = migrate(cfg, with_embeddings=False, dry_run=False)
    assert out1 == {"inserted": 2, "skipped": 0, "failed": 0}
    out2 = migrate(cfg, with_embeddings=False, dry_run=False)
    assert out2 == {"inserted": 0, "skipped": 2, "failed": 0}
