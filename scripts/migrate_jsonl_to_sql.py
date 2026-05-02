"""One-shot backfill of JSON-file incidents into the SQL store.

Usage:
    python scripts/migrate_jsonl_to_sql.py [--config PATH] [--with-embeddings] [--dry-run]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from orchestrator.config import AppConfig, load_config
from orchestrator.incident import Incident
from orchestrator.storage.embeddings import build_embedder
from orchestrator.storage.engine import build_engine
from orchestrator.storage.models import Base
from orchestrator.storage.repository import IncidentRepository


def migrate(cfg: AppConfig, *, with_embeddings: bool, dry_run: bool) -> dict[str, int]:
    """Walk ``cfg.paths.incidents_dir`` for INC-*.json files and upsert them
    into the SQL store. Idempotent on incident id.

    Returns counters: ``{"inserted": N, "skipped": M, "failed": K}``.
    """
    src = Path(cfg.paths.incidents_dir)
    engine = build_engine(cfg.storage.metadata)
    Base.metadata.create_all(engine)
    embedder = (
        build_embedder(cfg.llm.embedding, cfg.llm.providers)
        if with_embeddings else None
    )
    repo = IncidentRepository(engine=engine, embedder=embedder)
    counts = {"inserted": 0, "skipped": 0, "failed": 0}
    for path in sorted(src.glob("INC-*.json")):
        try:
            inc = Incident.model_validate(json.loads(path.read_text()))
        except Exception:
            counts["failed"] += 1
            continue
        try:
            repo.load(inc.id)
            counts["skipped"] += 1
            continue
        except FileNotFoundError:
            pass
        if not dry_run:
            repo.save(inc)
        counts["inserted"] += 1
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--with-embeddings", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = migrate(cfg, with_embeddings=args.with_embeddings, dry_run=args.dry_run)
    print(
        f"migrate: inserted={out['inserted']} "
        f"skipped={out['skipped']} failed={out['failed']}"
    )


if __name__ == "__main__":
    _cli()
