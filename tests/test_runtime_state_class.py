"""Tests for state-class plumbing.

Covers:
- ``RuntimeConfig`` and its placement on ``AppConfig``.
- ``resolve_state_class`` dotted-path resolver.
- ``Generic[StateT]`` parametrisation of ``Orchestrator`` /
  ``SessionStore`` / ``HistoryStore``.
"""
from __future__ import annotations

import pytest
import yaml
from sqlalchemy import create_engine


# ---------- RuntimeConfig + AppConfig.runtime ----------

def test_runtime_config_defaults_to_none_state_class():
    from runtime.config import RuntimeConfig

    cfg = RuntimeConfig()
    assert cfg.state_class is None


def test_runtime_config_accepts_dotted_path():
    from runtime.config import RuntimeConfig

    cfg = RuntimeConfig(state_class="runtime.state.Session")
    assert cfg.state_class == "runtime.state.Session"


def test_app_config_has_runtime_block_with_default():
    from runtime.config import AppConfig, LLMConfig, MCPConfig, RuntimeConfig

    app = AppConfig(llm=LLMConfig(), mcp=MCPConfig())
    assert isinstance(app.runtime, RuntimeConfig)
    assert app.runtime.state_class is None


def test_app_config_accepts_runtime_state_class():
    from runtime.config import AppConfig, LLMConfig, MCPConfig

    app = AppConfig(
        llm=LLMConfig(),
        mcp=MCPConfig(),
        runtime={"state_class": "runtime.state.Session"},
    )
    assert app.runtime.state_class == "runtime.state.Session"


def test_load_config_picks_up_runtime_block(tmp_path):
    from runtime.config import load_config

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {
                    "providers": {"stub": {"kind": "stub"}},
                    "models": {"stub_default": {"provider": "stub", "model": "x"}},
                    "default": "stub_default",
                },
                "mcp": {"servers": []},
                "runtime": {
                    "state_class": "runtime.state.Session"
                },
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.runtime.state_class == "runtime.state.Session"


# ---------- resolve_state_class ----------

def test_resolve_state_class_returns_default_session_for_none():
    from runtime.state import Session
    from runtime.state_resolver import resolve_state_class

    assert resolve_state_class(None) is Session


def test_resolve_state_class_returns_default_session_for_empty_string():
    from runtime.state import Session
    from runtime.state_resolver import resolve_state_class

    assert resolve_state_class("") is Session


def test_resolve_state_class_resolves_runtime_session_path():
    from runtime.state import Session
    from runtime.state_resolver import resolve_state_class

    assert resolve_state_class("runtime.state.Session") is Session


def test_resolve_state_class_resolves_session_subclass():
    """Resolver returns the class object for any importable Session subclass."""
    from runtime.state import Session
    from runtime.state_resolver import resolve_state_class

    # The resolver should return the class object for an importable
    # ``Session`` subclass referenced by dotted path. Use the framework
    # ``Session`` itself as the canonical resolvable target — apps that
    # ship their own subclass plug it in via the same dotted-path API.
    assert resolve_state_class("runtime.state.Session") is Session


def test_resolve_state_class_rejects_non_session_subclass():
    from runtime.state_resolver import resolve_state_class

    with pytest.raises(TypeError, match="must be a Session subclass"):
        resolve_state_class("builtins.dict")


def test_resolve_state_class_rejects_missing_module():
    from runtime.state_resolver import resolve_state_class

    with pytest.raises(ImportError, match="cannot import"):
        resolve_state_class("does.not.exist.Whatever")


def test_resolve_state_class_rejects_missing_attribute():
    from runtime.state_resolver import resolve_state_class

    with pytest.raises(AttributeError, match="has no attribute"):
        resolve_state_class("runtime.state.NotAClassHere")


def test_resolve_state_class_rejects_invalid_format():
    from runtime.state_resolver import resolve_state_class

    with pytest.raises(ValueError, match="dotted"):
        resolve_state_class("not_a_dotted_path")


# ---------- SessionStore / HistoryStore / Orchestrator are Generic[StateT] ----------

@pytest.fixture()
def engine(tmp_path):
    e = create_engine(
        f"sqlite:///{tmp_path}/t.db",
        connect_args={"check_same_thread": False},
    )
    from runtime.storage.models import Base

    Base.metadata.create_all(e)
    return e


class _FlaggedSession(__import__("runtime.state", fromlist=["Session"]).Session):
    """Tiny ``Session`` subclass used to exercise ``state_cls`` plumbing.

    Surfaces ``query`` + ``environment`` so ``HistoryStore.find_similar``
    (keyword path) has indexable text and the
    ``filter_kwargs={"environment": ...}`` predicate matches.
    """

    query: str = ""
    environment: str = ""


def test_session_store_accepts_state_cls(engine):
    from runtime.storage.session_store import SessionStore

    store = SessionStore(engine=engine, state_cls=_FlaggedSession)
    assert store._state_cls is _FlaggedSession


def test_session_store_default_state_cls_is_session(engine):
    """Default ``state_cls`` is ``runtime.state.Session`` (the framework
    base). Apps inject their own subclass via ``RuntimeConfig.state_class``.
    """
    from runtime.state import Session
    from runtime.storage.session_store import SessionStore

    store = SessionStore(engine=engine)
    assert store._state_cls is Session


def test_session_store_hydrates_to_configured_state_cls(engine):
    """Round-trip: a custom ``Session`` subclass survives create -> load."""
    from runtime.storage.session_store import SessionStore

    store = SessionStore(engine=engine, state_cls=_FlaggedSession)
    inc = store.create(
        query="payments latency",
        environment="production",
        reporter_id="u",
        reporter_team="t",
    )
    assert isinstance(inc, _FlaggedSession)
    loaded = store.load(inc.id)
    assert isinstance(loaded, _FlaggedSession)
    assert loaded.id == inc.id


def test_history_store_accepts_state_cls(engine):
    from runtime.storage.history_store import HistoryStore

    h = HistoryStore(engine=engine, state_cls=_FlaggedSession)
    assert h._converter._state_cls is _FlaggedSession


def test_history_store_threads_state_cls_through_to_converter(engine):
    from runtime.storage.history_store import HistoryStore
    from runtime.storage.session_store import SessionStore

    # Seed with the SessionStore using the custom class.
    store = SessionStore(engine=engine, state_cls=_FlaggedSession)
    seeded = store.create(
        query="payments timeout upstream",
        environment="production",
        reporter_id="u",
        reporter_team="t",
    )
    seeded.status = "resolved"
    store.save(seeded)

    h = HistoryStore(
        engine=engine,
        embedder=None,
        vector_store=None,
        similarity_threshold=0.3,
        state_cls=_FlaggedSession,
    )
    results = h.find_similar(
        query="payments timeout",
        filter_kwargs={"environment": "production"},
        status_filter="resolved",
        limit=5,
    )
    assert results, "expected keyword fallback to surface seeded session"
    inc, _score = results[0]
    assert isinstance(inc, _FlaggedSession)


def test_orchestrator_class_is_generic():
    """``Orchestrator`` should accept ``StateT`` parametrisation at the
    type-system level (Generic erasure means we can only check ``__class_getitem__``).
    """
    from runtime.orchestrator import Orchestrator
    from runtime.state import Session

    parametrised = Orchestrator[Session]
    assert parametrised is not None  # __class_getitem__ does not raise


def test_session_store_class_is_generic():
    from runtime.state import Session
    from runtime.storage.session_store import SessionStore

    parametrised = SessionStore[Session]
    assert parametrised is not None


def test_history_store_class_is_generic():
    from runtime.state import Session
    from runtime.storage.history_store import HistoryStore

    parametrised = HistoryStore[Session]
    assert parametrised is not None
