"""Postgres checkpointer wrapper.

Loaded only when ``cfg.storage.metadata.url`` resolves to a Postgres
URL. Uses a *separate* :class:`psycopg_pool.AsyncConnectionPool` (not
SQLAlchemy's pool) so the LangGraph checkpoint saver doesn't contend
with the metadata-store writes on the same connection.

The pool's lifecycle is bound to the orchestrator via the returned
async cleanup callable; the orchestrator awaits it from ``aclose``.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver


async def make_postgres_checkpointer(
    url: str,
) -> Tuple[BaseCheckpointSaver, Callable[[], Awaitable[None]]]:
    """Build a Postgres checkpointer + async cleanup callable.

    The orchestrator runs in async mode, so we use the async variant
    (:class:`AsyncPostgresSaver`) backed by an
    :class:`AsyncConnectionPool` rather than the sync ``PostgresSaver``
    -- LangGraph's async Pregel loop calls ``aget_tuple`` which the
    sync saver does not support.

    The pool is configured with ``autocommit=True`` because LangGraph
    issues each checkpoint write as a single statement and the
    enclosing transaction would otherwise hold the row lock until
    explicit commit.
    """
    # ``langgraph-checkpoint-postgres`` is an optional extra (declared
    # under [project.optional-dependencies].postgres in pyproject) so
    # the wheel is not present in CI's SQLite-only install. The module
    # is only imported on the Postgres URL branch in production.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # pyright: ignore[reportMissingImports]
    from psycopg_pool import AsyncConnectionPool

    # Translate SQLAlchemy URL -> libpq connection string. SQLAlchemy
    # accepts ``postgresql+psycopg://...`` while psycopg's
    # AsyncConnectionPool wants the bare ``postgresql://...``. Strip
    # any dialect suffix on the scheme so both URL flavours work.
    if "+" in url.split("://", 1)[0]:
        _, rest = url.split("://", 1)
        url = f"postgresql://{rest}"

    pool = AsyncConnectionPool(
        conninfo=url,
        max_size=4,
        kwargs={"autocommit": True},
        # ``open=False`` defers the actual TCP connect to ``open()``;
        # we open immediately below so callers see real connection
        # errors at construction time, not on first request.
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type] — pool is the right shape per docs
    # Idempotent: creates the checkpoint tables if they don't yet exist.
    await saver.setup()
    return saver, pool.close
