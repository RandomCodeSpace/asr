"""Tests for Phase 2 state-class plumbing (P2-A, P2-B, P2-C).

Covers:
- ``RuntimeConfig`` and its placement on ``AppConfig`` (P2-A).
- ``resolve_state_class`` dotted-path resolver (P2-B).
- ``Generic[StateT]`` parametrisation of ``Orchestrator`` /
  ``SessionStore`` / ``HistoryStore`` (P2-C).
"""
from __future__ import annotations

import pytest
import yaml
from sqlalchemy import create_engine


# ---------- P2-A: RuntimeConfig + AppConfig.runtime ----------

def test_runtime_config_defaults_to_none_state_class():
    from runtime.config import RuntimeConfig

    cfg = RuntimeConfig()
    assert cfg.state_class is None


def test_runtime_config_accepts_dotted_path():
    from runtime.config import RuntimeConfig

    cfg = RuntimeConfig(state_class="examples.incident_management.state.IncidentState")
    assert cfg.state_class == "examples.incident_management.state.IncidentState"


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
        runtime={"state_class": "examples.incident_management.state.IncidentState"},
    )
    assert app.runtime.state_class == "examples.incident_management.state.IncidentState"


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
                    "state_class": "examples.incident_management.state.IncidentState"
                },
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.runtime.state_class == "examples.incident_management.state.IncidentState"


# ---------- P2-B: resolve_state_class ----------

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


def test_resolve_state_class_resolves_incident_state():
    from examples.incident_management.state import IncidentState
    from runtime.state_resolver import resolve_state_class

    assert (
        resolve_state_class("examples.incident_management.state.IncidentState")
        is IncidentState
    )


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


# ---------- P2-C: SessionStore / HistoryStore / Orchestrator are Generic[StateT] ----------

@pytest.fixture()
def engine(tmp_path):
    e = create_engine(
        f"sqlite:///{tmp_path}/t.db",
        connect_args={"check_same_thread": False},
    )
    from runtime.storage.models import Base

    Base.metadata.create_all(e)
    return e


def test_session_store_accepts_state_cls(engine):
    from examples.incident_management.state import IncidentState
    from runtime.storage.session_store import SessionStore

    store = SessionStore(engine=engine, state_cls=IncidentState)
    assert store._state_cls is IncidentState


def test_session_store_default_state_cls_is_session(engine):
    """P2-J: default ``state_cls`` is ``runtime.state.Session`` (the framework
    base). Apps inject their own subclass via ``RuntimeConfig.state_class``.
    """
    from runtime.state import Session
    from runtime.storage.session_store import SessionStore

    store = SessionStore(engine=engine)
    assert store._state_cls is Session


def test_session_store_hydrates_to_configured_state_cls(engine):
    """Round-trip: a custom ``IncidentState`` subclass survives create -> load."""
    from examples.incident_management.state import IncidentState
    from runtime.storage.session_store import SessionStore

    class FlaggedIncident(IncidentState):
        pass

    store = SessionStore(engine=engine, state_cls=FlaggedIncident)
    inc = store.create(
        query="payments latency",
        environment="production",
        reporter_id="u",
        reporter_team="t",
    )
    assert isinstance(inc, FlaggedIncident)
    loaded = store.load(inc.id)
    assert isinstance(loaded, FlaggedIncident)
    assert loaded.id == inc.id


def test_history_store_accepts_state_cls(engine):
    from examples.incident_management.state import IncidentState
    from runtime.storage.history_store import HistoryStore

    h = HistoryStore(engine=engine, state_cls=IncidentState)
    assert h._converter._state_cls is IncidentState


def test_history_store_threads_state_cls_through_to_converter(engine):
    from examples.incident_management.state import IncidentState
    from runtime.storage.history_store import HistoryStore
    from runtime.storage.session_store import SessionStore

    class FlaggedIncident(IncidentState):
        pass

    # Seed with the SessionStore using the custom class.
    store = SessionStore(engine=engine, state_cls=FlaggedIncident)
    seeded = store.create(
        query="payments timeout upstream",
        environment="production",
        reporter_id="u",
        reporter_team="t",
    )
    seeded.status = "resolved"
    seeded.summary = "restarted payments service"
    store.save(seeded)

    h = HistoryStore(
        engine=engine,
        embedder=None,
        vector_store=None,
        similarity_threshold=0.3,
        state_cls=FlaggedIncident,
    )
    results = h.find_similar(
        query="payments timeout",
        filter_kwargs={"environment": "production"},
        status_filter="resolved",
        limit=5,
    )
    assert results, "expected keyword fallback to surface seeded incident"
    inc, _score = results[0]
    assert isinstance(inc, FlaggedIncident)


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
