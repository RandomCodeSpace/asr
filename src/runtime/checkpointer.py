"""LangGraph checkpointer factory.

Reuses ``cfg.storage.metadata.url`` so the LangGraph durable-state
checkpointer and the application metadata store live in the same
database (one URL to configure, easier ops). Per-backend a *separate*
connection pool is created so the two paths never deadlock:

- SQLite: dedicated ``aiosqlite.Connection`` with ``PRAGMA journal_mode=WAL``
  so the SQLAlchemy session pool and the checkpoint saver can both write
  to the same on-disk file without blocking each other.
- Postgres: a separate ``psycopg_pool.AsyncConnectionPool`` (filled in
  P2-G) rather than reusing SQLAlchemy's pool, so checkpointer writes
  don't contend with metadata writes on the same connection.

The factory is async because the orchestrator drives the graph through
async ``ainvoke`` / ``astream_events`` — and LangGraph's async Pregel
loop calls ``aget_tuple`` on the saver, which in turn requires an
asyncio-friendly DB driver (aiosqlite for SQLite,
``psycopg_pool.AsyncConnectionPool`` for Postgres).

Returns ``(saver, cleanup)`` where ``cleanup`` is an *async* callable
that closes the dedicated connection / pool. The caller owns lifecycle
and must await ``cleanup()`` on shutdown — typically via the
orchestrator's ``aclose()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Tuple
from urllib.parse import urlparse

from langgraph.checkpoint.base import BaseCheckpointSaver

from runtime.config import AppConfig, MetadataConfig


def _sqlite_path_from_url(url: str) -> str:
    """Extract the on-disk path from a sqlite SQLAlchemy URL.

    Accepts the SQLAlchemy ``sqlite:///<path>`` form (three slashes ->
    relative or absolute path begins after the third slash). The four-
    slash variant ``sqlite:////abs/path`` is also tolerated for callers
    who explicitly want an absolute path; sqlite3 itself accepts both.
    """
    parsed = urlparse(url)
    path = parsed.path
    if path.startswith("//"):
        # sqlite:////abs/path -> urlparse path "//abs/path" -> "/abs/path"
        return path[1:]
    # sqlite:///x -> urlparse path "/x". sqlite3 accepts both absolute
    # and relative; tests use absolute via tmp_path.
    return path


async def make_checkpointer(
    cfg: AppConfig | MetadataConfig,
) -> Tuple[BaseCheckpointSaver, Callable[[], Awaitable[None]]]:
    """Build a checkpointer for the configured metadata DB.

    Accepts either a full :class:`AppConfig` (``cfg.storage.metadata`` is
    read) or a :class:`MetadataConfig` directly. The orchestrator uses
    the direct form because it post-processes the raw URL (resolving
    the default ``incidents/incidents.db`` sentinel against
    ``cfg.paths.incidents_dir``) and needs to pass *that* resolved URL
    through so per-test ``tmp_path`` isolation lands on the same DB
    file as the SQLAlchemy engine.

    Branches on the URL scheme:

    - ``sqlite:`` -> :class:`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
    - ``postgresql:`` / ``postgres:`` -> Postgres path (P2-G)

    Returns ``(saver, cleanup)``. ``cleanup`` is an async callable that
    closes the dedicated connection / pool; the caller is expected to
    await it at orchestrator shutdown.
    """
    if isinstance(cfg, MetadataConfig):
        url = cfg.url
    else:
        url = cfg.storage.metadata.url

    if url.startswith("sqlite:"):
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = _sqlite_path_from_url(url)
        # Ensure the parent directory exists. The orchestrator may build
        # the checkpointer before any storage write that would otherwise
        # have created it (e.g. on first start in a fresh deploy dir).
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Dedicated aiosqlite connection — separate from the SQLAlchemy
        # engine's connection pool. WAL mode lets readers and writers
        # proceed concurrently on the same file.
        #
        # ``isolation_level=None`` puts the underlying ``sqlite3`` driver
        # in autocommit mode so Python's stdlib doesn't sneak a
        # ``BEGIN DEFERRED`` in front of our INSERTs. AsyncSqliteSaver's
        # ``aput`` then runs as an implicit single-statement write
        # followed by ``commit()``; SQLite handles the transaction
        # internally as ``BEGIN IMMEDIATE`` for any DML, so the saver
        # plays nicely with the SQLAlchemy engine's explicit
        # ``BEGIN IMMEDIATE`` (see ``runtime.storage.engine``).
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        # Set WAL + relaxed durability up front so the orchestrator's
        # other path (the SQLAlchemy engine) sees them; AsyncSqliteSaver
        # itself also enables WAL during setup() but doing it here makes
        # the pragma observable to verification tests immediately.
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        # SQLAlchemy and the saver share the same DB file via separate
        # connections. Without a generous busy_timeout, a contended
        # writer races straight into ``database is locked``. 30s matches
        # the engine-side setting (see ``runtime.storage.engine``) so
        # both writers wait the same amount before failing.
        await conn.execute("PRAGMA busy_timeout=30000")
        saver = AsyncSqliteSaver(conn)
        # Create checkpoint tables on first use. Idempotent.
        await saver.setup()
        return saver, conn.close

    if url.startswith("postgresql:") or url.startswith("postgres:"):
        # Filled in P2-G. Imported lazily so SQLite-only deploys don't
        # need psycopg_pool installed.
        from runtime.checkpointer_postgres import make_postgres_checkpointer

        return await make_postgres_checkpointer(url)

    raise ValueError(
        f"unsupported checkpointer URL scheme {url!r} — "
        "expected sqlite or postgresql"
    )
