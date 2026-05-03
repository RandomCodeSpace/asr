"""Tests for the generic ``submitter`` kwarg on the ``start_session`` API
surface (Wave 2 follow-up: drop incident vocabulary).

Covers:
  - submitter dict passes through to the persisted row's reporter columns;
  - the deprecated ``reporter_id``/``reporter_team`` kwargs are coerced
    into ``submitter`` and emit a single ``DeprecationWarning``;
  - passing both ``submitter`` and either legacy kwarg raises ``TypeError``;
  - calling without either kwarg leaves the row's reporter columns set
    to the historical defaults (``"user-mock"`` / ``"platform"``).
"""
from __future__ import annotations

import warnings

import pytest

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MetadataConfig,
    Paths,
    RuntimeConfig,
    StorageConfig,
)
from runtime.service import OrchestratorService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_full(tmp_path):
    """AppConfig wired to the in-process incident-management MCP server —
    sufficient for ``Orchestrator.create`` and ``start_session`` to run
    end-to-end against the stub LLM."""
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(
            servers=[
                MCPServerConfig(
                    name="local_inc",
                    transport="in_process",
                    module="examples.incident_management.mcp_server",
                    category="incident_management",
                ),
                MCPServerConfig(
                    name="local_obs",
                    transport="in_process",
                    module="runtime.mcp_servers.observability",
                    category="observability",
                ),
                MCPServerConfig(
                    name="local_rem",
                    transport="in_process",
                    module="runtime.mcp_servers.remediation",
                    category="remediation",
                ),
                MCPServerConfig(
                    name="local_user",
                    transport="in_process",
                    module="runtime.mcp_servers.user_context",
                    category="user_context",
                ),
            ]
        ),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db")
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )


@pytest.fixture
def service_full(cfg_full):
    """Started OrchestratorService; teardown calls shutdown()."""
    OrchestratorService._reset_singleton()
    svc = OrchestratorService.get_or_create(cfg_full)
    svc.start()
    try:
        yield svc
    finally:
        svc.shutdown()


@pytest.fixture(autouse=True)
def _reset_singleton():
    yield
    OrchestratorService._reset_singleton()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _load_row(svc: OrchestratorService, sid: str):
    async def _load():
        return svc._orch.store.load(sid)

    return svc.submit_and_wait(_load(), timeout=5)


def test_submitter_dict_passes_through(service_full):
    """submitter={'id': ..., 'team': ...} populates the row's reporter
    columns directly — no deprecation warning, no TypeError, no defaults."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sid = service_full.start_session(
            query="db slow",
            state_overrides={"environment": "prod"},
            submitter={"id": "alice", "team": "payments"},
        )
    inc = _load_row(service_full, sid)
    assert inc.reporter.id == "alice"
    assert inc.reporter.team == "payments"


def test_legacy_reporter_kwargs_coerced_with_deprecation_warning(service_full):
    """Passing the deprecated ``reporter_id``/``reporter_team`` pair must
    still create a row with those values *and* emit exactly one
    ``DeprecationWarning`` so callers get a clear migration signal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        sid = service_full.start_session(
            query="db slow",
            state_overrides={"environment": "prod"},
            reporter_id="bob",
            reporter_team="dba",
        )
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, (
        f"expected exactly one DeprecationWarning, got {len(deprecations)}: "
        f"{[str(w.message) for w in deprecations]}"
    )
    assert "reporter_id" in str(deprecations[0].message)
    inc = _load_row(service_full, sid)
    assert inc.reporter.id == "bob"
    assert inc.reporter.team == "dba"


def test_both_submitter_and_legacy_raises_type_error(service_full):
    """Supplying both ``submitter`` and either legacy kwarg is a caller
    bug — silent precedence would mask it. Fail fast with TypeError."""
    with pytest.raises(TypeError, match="submitter"):
        service_full.start_session(
            query="x",
            state_overrides={"environment": "dev"},
            submitter={"id": "a", "team": "b"},
            reporter_id="c",
        )
    with pytest.raises(TypeError, match="submitter"):
        service_full.start_session(
            query="x",
            state_overrides={"environment": "dev"},
            submitter={"id": "a", "team": "b"},
            reporter_team="d",
        )


def test_neither_kwarg_uses_defaults(service_full):
    """No submitter and no legacy kwargs: the framework falls back to
    its historical defaults (``user-mock`` / ``platform``) so existing
    callers that never set a reporter keep working unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sid = service_full.start_session(
            query="x", state_overrides={"environment": "dev"},
        )
    inc = _load_row(service_full, sid)
    assert inc.reporter.id == "user-mock"
    assert inc.reporter.team == "platform"
