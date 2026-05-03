"""Smoke tests for the runtime package layout.

These tests assert that the framework's generic surface is importable from
``runtime``. The Session model in ``runtime.state`` lands in P1-B; until
then ``test_runtime_state_importable`` is expected to fail under TDD.
"""


def test_runtime_package_importable():
    import runtime  # noqa: F401


def test_runtime_config_importable():
    from runtime.config import AppConfig  # noqa: F401


def test_runtime_state_importable():
    from runtime.state import Session  # noqa: F401


def test_session_has_generic_fields():
    from runtime.state import Session
    s = Session(
        id="test-001",
        status="new",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
    )
    assert s.id == "test-001"
    assert s.agents_run == []
    assert s.tool_calls == []
    assert s.findings == {}
    assert s.pending_intervention is None
    assert s.token_usage.total_tokens == 0


def test_session_has_no_incident_fields():
    from runtime.state import Session
    fields = set(Session.model_fields.keys())
    incident_only = {
        "environment", "reporter", "query", "severity",
        "category", "matched_prior_inc", "embedding",
        "resolution", "tags",
    }
    leaked = incident_only & fields
    assert not leaked, f"Session leaks incident fields: {leaked}"


def test_graph_state_uses_session():
    from runtime.graph import GraphState
    from runtime.state import Session
    import typing
    hints = typing.get_type_hints(GraphState)
    assert hints.get("session") is Session, (
        f"GraphState.session should be Session, got {hints.get('session')}"
    )


def test_graph_state_no_incident_alias():
    from runtime.graph import GraphState
    import typing
    hints = typing.get_type_hints(GraphState)
    assert "session" in hints
    assert "incident" not in hints, (
        "GraphState.incident bridge alias should be removed in Phase 2"
    )


def test_session_row_alias_exists():
    from runtime.storage.models import IncidentRow, SessionRow
    assert SessionRow is IncidentRow


def test_runtime_config_no_incident_keys():
    """P1-E stripped incident-flavored keys off AppConfig — they live on
    examples.incident_management.config.IncidentAppConfig now."""
    from runtime.config import AppConfig

    fields = set(AppConfig.model_fields.keys())
    assert "incidents" not in fields
    assert "intervention" not in fields
    assert "environments" not in fields


def test_runtime_config_no_severity_aliases():
    """severity_aliases moved to IncidentAppConfig (P1-E)."""
    from runtime.config import OrchestratorConfig

    fields = set(OrchestratorConfig.model_fields.keys())
    assert "severity_aliases" not in fields


def test_runtime_canonical_imports():
    """The canonical framework import paths remain stable."""
    from runtime.config import AppConfig  # noqa: F401
    from runtime.storage.session_store import SessionStore  # noqa: F401
    from runtime.storage.history_store import HistoryStore  # noqa: F401
    from runtime.graph import GraphState  # noqa: F401
    from runtime.orchestrator import Orchestrator  # noqa: F401


# ---------- P2-J: legacy ``runtime.incident`` + ``IncidentRepository`` are gone ----------

def test_legacy_incident_module_removed():
    import importlib.util
    assert importlib.util.find_spec("runtime.incident") is None, (
        "runtime/incident.py should be deleted in P2-J"
    )


def test_incident_repository_shim_removed():
    import importlib.util
    spec = importlib.util.find_spec("runtime.storage.repository")
    # Must either not exist OR be a different module (not the IncidentRepository shim)
    if spec is not None:
        import runtime.storage.repository
        assert not hasattr(runtime.storage.repository, "IncidentRepository"), (
            "IncidentRepository shim should be removed in P2-J"
        )


def test_storage_init_does_not_export_incident_repository():
    import runtime.storage as s
    assert not hasattr(s, "IncidentRepository"), (
        "IncidentRepository should be removed from runtime.storage exports in P2-J"
    )


def test_orchestrator_storage_init_does_not_export_incident_repository():
    import runtime.storage as s
    assert not hasattr(s, "IncidentRepository"), (
        "IncidentRepository should be removed from orchestrator.storage exports in P2-J"
    )


def test_orchestrator_generic_methods_exist():
    from runtime.orchestrator import Orchestrator
    for m in ("start_session", "stream_session", "resume_session",
              "list_recent_sessions", "get_session", "delete_session"):
        assert hasattr(Orchestrator, m), f"missing generic method: {m}"


def test_orchestrator_incident_shims_exist():
    from runtime.orchestrator import Orchestrator
    for m in ("start_investigation", "stream_investigation",
              "resume_investigation", "list_recent_incidents",
              "get_incident", "delete_incident"):
        assert hasattr(Orchestrator, m), f"missing incident shim: {m}"


def test_paths_skills_dir_default_is_none():
    from runtime.config import Paths
    p = Paths()
    assert p.skills_dir is None, (
        "Paths.skills_dir default should be None; apps must set it"
    )


def test_orchestrator_raises_clear_error_without_skills_dir():
    import pytest
    from runtime.config import AppConfig, LLMConfig, MCPConfig, Paths
    from runtime.orchestrator import Orchestrator

    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        paths=Paths(skills_dir=None),
    )
    import asyncio
    with pytest.raises(RuntimeError, match="skills_dir"):
        asyncio.run(Orchestrator.create(cfg))
