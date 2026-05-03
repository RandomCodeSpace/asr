"""SQLAlchemy engine factory.

Sync engine for SQLite (dev) or Postgres (prod). No vector-extension
loading — vectors live in a separate LangChain VectorStore (see
:mod:`orchestrator.storage.vector`, landed in M3).

P2-FIX: when the metadata store and the LangGraph ``AsyncSqliteSaver``
checkpointer share a SQLite file, two writers contend on the same DB.
SQLite's default ``BEGIN DEFERRED`` transaction acquires SHARED on the
first read and only escalates to RESERVED on the first write — and the
escalation is **non-retryable** when the connection has already read
inside the same transaction (busy_timeout does *not* apply). The losing
writer raises ``database is locked`` immediately. The fix is to start
write transactions with ``BEGIN IMMEDIATE`` so the RESERVED lock is
acquired up front, before any reads, and busy_timeout's wait-and-retry
loop can correctly serialize the two writers. This is the
SQLAlchemy-recommended pattern for concurrent SQLite writers; see
https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl
"""
from __future__ import annotations
from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool

from runtime.config import MetadataConfig

# Generous timeout: tests can run under load with multiple async writers
# interleaving on the same DB file. 30s leaves headroom for the slowest
# checkpointer commit while still failing fast on a true deadlock.
_SQLITE_BUSY_TIMEOUT_MS = 30_000


def build_engine(cfg: MetadataConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False, "isolation_level": None},
        )
        _install_sqlite_concurrency_pragmas(engine)
        return engine
    return create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)


def _install_sqlite_concurrency_pragmas(engine: Engine) -> None:
    """Configure every new SQLite connection so it plays nicely with a
    concurrent LangGraph checkpointer on the same DB file.

    Three things happen here:

    1. ``connect_args["isolation_level"]=None`` (set in ``build_engine``)
       puts the underlying ``sqlite3.Connection`` in autocommit mode so
       Python's stdlib doesn't sneak a ``BEGIN`` in front of our SQL.
       SQLAlchemy then drives transactions explicitly via the ``begin``
       event hook below — this is the documented escape hatch for
       custom transaction modes.
    2. ``PRAGMA journal_mode=WAL`` + ``PRAGMA synchronous=NORMAL`` +
       ``PRAGMA busy_timeout`` are issued on every new connection so
       readers and writers don't block each other and writers wait
       briefly on contention rather than failing immediately.
    3. ``BEGIN IMMEDIATE`` on the ``begin`` hook acquires the RESERVED
       lock at transaction start (before any read), so the
       ``busy_timeout`` retry loop can wait out a concurrent writer
       cleanly. Without this, two writers each in a ``BEGIN DEFERRED``
       transaction race to escalate and the loser hits ``database is
       locked`` instantly.
    """
    @sa_event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _conn_record):  # noqa: ANN001 — sqlalchemy event sig
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
        finally:
            cur.close()

    @sa_event.listens_for(engine, "begin")
    def _on_begin(conn):  # noqa: ANN001 — sqlalchemy event sig
        # Replace SQLAlchemy's default ``BEGIN`` (deferred) with
        # ``BEGIN IMMEDIATE``. Required so concurrent writers serialize
        # cleanly via busy_timeout instead of failing on lock-escalation.
        conn.exec_driver_sql("BEGIN IMMEDIATE")
