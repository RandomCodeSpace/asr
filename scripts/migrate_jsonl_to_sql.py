"""One-shot backfill of JSON-file incidents into the SQL store.

Usage:
    python scripts/migrate_jsonl_to_sql.py [--config PATH] [--with-embeddings] [--dry-run]

P2-J: migrated off the deleted ``IncidentRepository`` shim. Uses
``SessionStore`` directly with the example app's ``IncidentState``.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from examples.incident_management.state import IncidentState
from runtime.config import AppConfig, load_config
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store


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
    vector_store = (
        build_vector_store(cfg.storage.vector, embedder, engine)
        if with_embeddings else None
    )
    store = SessionStore(
        engine=engine,
        state_cls=IncidentState,
        embedder=embedder,
        vector_store=vector_store,
        vector_path=(cfg.storage.vector.path
                     if cfg.storage.vector.backend == "faiss" else None),
        vector_index_name=cfg.storage.vector.collection_name,
        distance_strategy=cfg.storage.vector.distance_strategy,
    )
    counts = {"inserted": 0, "skipped": 0, "failed": 0}
    for path in sorted(src.glob("INC-*.json")):
        try:
            inc = IncidentState.model_validate(json.loads(path.read_text()))
        except Exception:
            counts["failed"] += 1
            continue
        try:
            store.load(inc.id)
            counts["skipped"] += 1
            continue
        except FileNotFoundError:
            pass
        if not dry_run:
            store.save(inc)
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
