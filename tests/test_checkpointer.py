"""Tests for runtime.checkpointer factory."""
import os
import sqlite3

import pytest


def _make_cfg(url: str):
    """Build a minimal AppConfig with the given metadata URL."""
    from runtime.config import (
        AppConfig,
        LLMConfig,
        MCPConfig,
        MetadataConfig,
        Paths,
        StorageConfig,
    )

    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(metadata=MetadataConfig(url=url)),
        paths=Paths(skills_dir="config/skills"),
    )


@pytest.mark.asyncio
async def test_make_checkpointer_sqlite_returns_saver(tmp_path):
    """SQLite URL -> AsyncSqliteSaver instance.

    The orchestrator drives the graph via async ``ainvoke`` /
    ``astream_events``, so the saver must support ``aget_tuple`` /
    ``aput`` — the async variant of SqliteSaver does.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from runtime.checkpointer import make_checkpointer

    db_path = tmp_path / "ckpt.db"
    cfg = _make_cfg(f"sqlite:///{db_path}")

    saver, cleanup = await make_checkpointer(cfg)
    try:
        assert isinstance(saver, AsyncSqliteSaver)
    finally:
        await cleanup()


@pytest.mark.asyncio
async def test_make_checkpointer_sqlite_enables_wal(tmp_path):
    """The PRAGMA must be set so SQLAlchemy sessions don't collide on
    the same DB file."""
    from runtime.checkpointer import make_checkpointer

    db_path = tmp_path / "ckpt.db"
    cfg = _make_cfg(f"sqlite:///{db_path}")

    saver, cleanup = await make_checkpointer(cfg)
    try:
        # Re-open a *separate* connection and verify journal_mode=WAL is
        # persisted on the on-disk file (WAL is a property of the DB
        # file in SQLite — set per-connection but written into the
        # file header).
        verify = sqlite3.connect(str(db_path))
        mode = verify.execute("PRAGMA journal_mode").fetchone()[0]
        verify.close()
        assert mode.lower() == "wal", f"expected wal, got {mode!r}"
    finally:
        await cleanup()


@pytest.mark.asyncio
async def test_make_checkpointer_creates_parent_dir(tmp_path):
    """Parent directories should be created automatically."""
    from runtime.checkpointer import make_checkpointer

    db_path = tmp_path / "nested" / "deeper" / "ckpt.db"
    cfg = _make_cfg(f"sqlite:///{db_path}")

    saver, cleanup = await make_checkpointer(cfg)
    try:
        assert db_path.parent.exists()
    finally:
        await cleanup()


@pytest.mark.asyncio
async def test_make_checkpointer_unsupported_url_raises():
    """An unrecognised scheme must raise loudly, not silently fall
    through."""
    from runtime.checkpointer import make_checkpointer

    cfg = _make_cfg("mysql://localhost/foo")
    with pytest.raises(ValueError, match="unsupported"):
        await make_checkpointer(cfg)


@pytest.mark.asyncio
async def test_make_checkpointer_uses_separate_connection(tmp_path):
    """The checkpointer connection must be independent of the metadata
    SQLAlchemy engine — they're separate pools sharing the same file."""
    from runtime.checkpointer import make_checkpointer
    from runtime.config import MetadataConfig
    from runtime.storage.engine import build_engine

    db_path = tmp_path / "ckpt.db"
    cfg = _make_cfg(f"sqlite:///{db_path}")

    engine = build_engine(MetadataConfig(url=f"sqlite:///{db_path}"))
    saver, cleanup = await make_checkpointer(cfg)
    try:
        # Exercise both: write a row through the engine, then verify
        # the saver can still operate on its own connection.
        with engine.begin() as c:
            c.exec_driver_sql("CREATE TABLE IF NOT EXISTS probe(x INT)")
            c.exec_driver_sql("INSERT INTO probe VALUES (1)")
        # Idempotent setup — a no-op second call validates the saver
        # is still usable after the engine wrote.
        await saver.setup()
    finally:
        await cleanup()
        engine.dispose()


@pytest.mark.asyncio
async def test_make_checkpointer_accepts_metadata_config_directly(tmp_path):
    """The orchestrator passes a resolved ``MetadataConfig`` (not the
    full ``AppConfig``) so per-test ``tmp_path`` URLs land on the right
    file."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from runtime.checkpointer import make_checkpointer
    from runtime.config import MetadataConfig

    db_path = tmp_path / "via-meta.db"
    saver, cleanup = await make_checkpointer(
        MetadataConfig(url=f"sqlite:///{db_path}")
    )
    try:
        assert isinstance(saver, AsyncSqliteSaver)
        assert db_path.exists()
    finally:
        await cleanup()


# ----- Postgres path (gated; integration test) -----


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("LANGGRAPH_PG_TEST_URL"),
    reason=(
        "LANGGRAPH_PG_TEST_URL not set; postgres checkpointer not "
        "exercised in CI"
    ),
)
async def test_make_checkpointer_postgres():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from runtime.checkpointer import make_checkpointer

    cfg = _make_cfg(os.environ["LANGGRAPH_PG_TEST_URL"])
    saver, cleanup = await make_checkpointer(cfg)
    try:
        assert isinstance(saver, AsyncPostgresSaver)
    finally:
        await cleanup()
