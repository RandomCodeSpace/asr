"""SQLAlchemy engine factory + sqlite-vec extension loader.

Behaviour
---------
- ``sqlite://``     → engine with ``NullPool``, ``check_same_thread=False``,
                      and a ``connect`` event hook that loads sqlite-vec into
                      every new dbapi connection.
- ``postgresql://`` → engine with the configured pool size, plus a
                      one-time ``CREATE EXTENSION IF NOT EXISTS vector``.
"""
from __future__ import annotations
import ctypes
import ctypes.util
from sqlalchemy import event, text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool

from orchestrator.config import StorageConfig

# Python 3.14 on many distros is compiled without SQLITE_ENABLE_LOAD_EXTENSION,
# so conn.enable_load_extension / conn.load_extension don't exist as Python
# methods.  We call the underlying C functions directly via ctypes instead.
_libsqlite = ctypes.CDLL(ctypes.util.find_library("sqlite3"))
_libsqlite.sqlite3_enable_load_extension.restype = ctypes.c_int
_libsqlite.sqlite3_enable_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_int]
_libsqlite.sqlite3_load_extension.restype = ctypes.c_int
_libsqlite.sqlite3_load_extension.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
]
# The CPython sqlite3.Connection C struct begins with PyObject header
# (ob_refcnt + ob_type = 2 pointers), followed immediately by sqlite3 *db.
_DB_PTR_OFFSET = 2 * ctypes.sizeof(ctypes.c_void_p)


def _ctypes_load_vec(dbapi_conn) -> None:  # type: ignore[misc]
    """Load sqlite-vec into *dbapi_conn* using the C-level SQLite API.

    Required because CPython may be built without SQLITE_ENABLE_LOAD_EXTENSION,
    which removes the Python-level enable_load_extension / load_extension methods
    but leaves the underlying C functions available in libsqlite3.
    """
    import sqlite_vec
    db_ptr = ctypes.c_void_p.from_address(id(dbapi_conn) + _DB_PTR_OFFSET).value
    _libsqlite.sqlite3_enable_load_extension(db_ptr, 1)
    errmsg = ctypes.c_char_p()
    path = sqlite_vec.loadable_path().encode()
    rc = _libsqlite.sqlite3_load_extension(db_ptr, path, None, ctypes.byref(errmsg))
    _libsqlite.sqlite3_enable_load_extension(db_ptr, 0)
    if rc != 0:
        raise RuntimeError(f"sqlite3_load_extension failed: {errmsg.value!r}")


def _attach_sqlite_vec(engine: Engine) -> None:
    """Register the sqlite-vec loader on every new SQLite dbapi connection."""
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):  # type: ignore[misc]
        _ctypes_load_vec(dbapi_conn)


def _ensure_pgvector(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def build_engine(cfg: StorageConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )
        _attach_sqlite_vec(engine)
        return engine
    engine = create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
    _ensure_pgvector(engine)
    return engine
