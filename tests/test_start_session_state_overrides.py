"""Tests for the generic ``state_overrides`` kwarg on ``start_session``.

Replaces the legacy positional ``environment`` kwarg with a free-form
``state_overrides`` dict so non-incident apps (code-review, etc.)
don't have to pass mock environment values to satisfy the runtime
API surface. ``environment`` survives as a deprecated kwarg with a
``DeprecationWarning`` + auto-coercion path.
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
from runtime.orchestrator import (
    _coerce_state_overrides,
    Orchestrator,
)


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(
                name="local_inc", transport="in_process",
                module="examples.incident_management.mcp_server",
                category="incident_management",
            ),
            MCPServerConfig(
                name="local_obs", transport="in_process",
                module="examples.incident_management.mcp_servers.observability",
                category="observability",
            ),
            MCPServerConfig(
                name="local_rem", transport="in_process",
                module="examples.incident_management.mcp_servers.remediation",
                category="remediation",
            ),
            MCPServerConfig(
                name="local_user", transport="in_process",
                module="examples.incident_management.mcp_servers.user_context",
                category="user_context",
            ),
        ]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(
            state_class=None,
        ),
    )


# ---------------------------------------------------------------------------
# Pure-function coercion tests
# ---------------------------------------------------------------------------


def test_coerce_state_overrides_neither_supplied_returns_none():
    assert _coerce_state_overrides(None, None) is None


def test_coerce_state_overrides_only_overrides_returns_unchanged():
    overrides = {"environment": "staging", "extra": 42}
    out = _coerce_state_overrides(overrides, None)
    assert out is overrides


def test_legacy_environment_kwarg_deprecation_warning_and_coercion():
    """Passing only ``environment`` emits DeprecationWarning and the
    value is coerced into ``state_overrides``."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _coerce_state_overrides(None, "production")
    assert out == {"environment": "production"}
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "environment" in str(item.message)
        for item in w
    )


def test_environment_and_state_overrides_both_supplied_raises():
    """Passing both kwargs is a TypeError — silent precedence would
    mask caller bugs."""
    with pytest.raises(TypeError, match="environment"):
        _coerce_state_overrides({"environment": "x"}, "y")


# ---------------------------------------------------------------------------
# Integration — overrides reach the persisted session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_overrides_applied_to_session(cfg):
    """The generic kwarg path: ``state_overrides={"environment":
    "staging"}`` lands on the persisted session row."""
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="db pool exhausted",
            state_overrides={"environment": "staging"},
        )
        inc = orch.store.load(sid)
        assert inc.extra_fields.get("environment") == "staging"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_legacy_environment_kwarg_still_works_with_warning(cfg):
    """Back-compat: passing the legacy ``environment`` kwarg still
    creates the session and projects the value, but emits a single
    ``DeprecationWarning``."""
    orch = await Orchestrator.create(cfg)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sid = await orch.start_session(
                query="db pool", environment="dev",
            )
        inc = orch.store.load(sid)
        assert inc.extra_fields.get("environment") == "dev"
        assert any(
            issubclass(item.category, DeprecationWarning)
            and "environment" in str(item.message)
            for item in w
        )
    finally:
        await orch.aclose()
