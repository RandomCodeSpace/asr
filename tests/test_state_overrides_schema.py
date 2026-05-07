"""Boundary tests for ``OrchestratorConfig.state_overrides_schema``
(DECOUPLE-05 / D-08-01 / D-08-02).

Covers:

* Default (``state_overrides_schema=None``) -> no validation; arbitrary
  dicts pass through ``Orchestrator.start_session`` (D-08-02 backward
  compat).
* String-format validation at config-load time (no importlib hit).
* importlib resolution at ``Orchestrator.create()``: bad module path
  and bad class name both raise at boot with messages naming the
  failing path / attribute.
* ``start_session`` validates against the registered class:
  unknown-key rejection, wrong-type rejection, well-shaped
  acceptance.
* Cross-app rejection: incident schema rejects code_review-shaped
  dicts and vice versa.
* YAML wiring: each app's runtime YAML round-trips
  ``state_overrides_schema`` correctly.
* Terminal-tool ``match_args`` value-dispatch: ``set_recommendation``
  routes to ``approved`` / ``changes_requested`` / ``commented`` based
  on ``args.recommendation``.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MetadataConfig,
    OrchestratorConfig,
    Paths,
    RuntimeConfig,
    StorageConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.state import ToolCall


# ---------------------------------------------------------------------------
# AppConfig fixture builders.
# ---------------------------------------------------------------------------


_INCIDENT_SERVERS = [
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
]

_CODE_REVIEW_SERVERS = [
    MCPServerConfig(
        name="local_cr", transport="in_process",
        module="examples.code_review.mcp_server",
        category="code_review",
    ),
]


def _base_cfg(
    tmp_path,
    *,
    state_overrides_schema: str | None = None,
    skills_dir: str = "examples/incident_management/skills",
    mcp_servers: list[MCPServerConfig] | None = None,
) -> AppConfig:
    """Minimal AppConfig used to construct an in-memory Orchestrator."""
    if mcp_servers is None:
        mcp_servers = (_CODE_REVIEW_SERVERS
                       if "code_review" in skills_dir
                       else _INCIDENT_SERVERS)
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=mcp_servers),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir=skills_dir,
            incidents_dir=str(tmp_path),
        ),
        runtime=RuntimeConfig(state_class=None),
        orchestrator=OrchestratorConfig(
            state_overrides_schema=state_overrides_schema,
        ),
    )


# ---------------------------------------------------------------------------
# OrchestratorConfig field validation (config-load time, no importlib).
# ---------------------------------------------------------------------------


def test_orchestrator_config_state_overrides_schema_default_is_none():
    assert OrchestratorConfig().state_overrides_schema is None


def test_orchestrator_config_state_overrides_schema_format_invalid_raises():
    with pytest.raises(ValidationError):
        OrchestratorConfig(
            state_overrides_schema="not.a.valid.path-with hyphens spaces",
        )


def test_orchestrator_config_state_overrides_schema_format_empty_raises():
    with pytest.raises(ValidationError):
        OrchestratorConfig(state_overrides_schema="   ")


def test_orchestrator_config_state_overrides_schema_format_dotted_accepted():
    cfg = OrchestratorConfig(
        state_overrides_schema="examples.code_review.state.CodeReviewStateOverrides",
    )
    assert (cfg.state_overrides_schema
            == "examples.code_review.state.CodeReviewStateOverrides")


def test_orchestrator_config_state_overrides_schema_format_colon_accepted():
    cfg = OrchestratorConfig(
        state_overrides_schema="examples.code_review.state:CodeReviewStateOverrides",
    )
    assert (cfg.state_overrides_schema
            == "examples.code_review.state:CodeReviewStateOverrides")


# ---------------------------------------------------------------------------
# Orchestrator.create() boot-time importlib resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_create_resolves_dotted_path(tmp_path):
    from examples.incident_management.state import IncidentStateOverrides
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.incident_management.state.IncidentStateOverrides"
        ),
    )
    orch = await Orchestrator.create(cfg)
    try:
        assert orch._state_overrides_cls is IncidentStateOverrides
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_orchestrator_create_resolves_colon_path(tmp_path):
    from examples.code_review.state import CodeReviewStateOverrides
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.code_review.state:CodeReviewStateOverrides"
        ),
    )
    orch = await Orchestrator.create(cfg)
    try:
        assert orch._state_overrides_cls is CodeReviewStateOverrides
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_orchestrator_create_bad_dotted_path_raises_at_boot(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema="nonexistent.module.Foo",
    )
    with pytest.raises(RuntimeError, match="nonexistent.module"):
        await Orchestrator.create(cfg)


@pytest.mark.asyncio
async def test_orchestrator_create_bad_attribute_raises_at_boot(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.incident_management.state:DoesNotExist"
        ),
    )
    with pytest.raises(RuntimeError, match="DoesNotExist"):
        await Orchestrator.create(cfg)


@pytest.mark.asyncio
async def test_orchestrator_create_non_basemodel_class_raises(tmp_path):
    # ``Path`` is a class but not a pydantic BaseModel.
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema="pathlib.Path",
    )
    with pytest.raises(RuntimeError, match="pydantic|BaseModel"):
        await Orchestrator.create(cfg)


# ---------------------------------------------------------------------------
# start_session(state_overrides=...) validation hook.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_session_validates_when_schema_set(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.incident_management.state.IncidentStateOverrides"
        ),
    )
    orch = await Orchestrator.create(cfg)
    try:
        with pytest.raises(ValidationError, match="bogus_key"):
            await orch.start_session(
                query="x",
                state_overrides={"environment": "staging", "bogus_key": "x"},
            )
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_start_session_passthrough_when_schema_unset(tmp_path):
    cfg = _base_cfg(tmp_path, state_overrides_schema=None)
    orch = await Orchestrator.create(cfg)
    try:
        # Arbitrary keys MUST pass through (D-08-02 backward compat).
        sid = await orch.start_session(
            query="x",
            state_overrides={"anything": "goes", "extra_key": 42},
        )
        assert sid  # a session id was minted
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_start_session_accepts_well_shaped_overrides(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.incident_management.state.IncidentStateOverrides"
        ),
    )
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="db pool",
            state_overrides={"environment": "staging", "severity": "warning"},
        )
        inc = orch.store.load(sid)
        assert inc.extra_fields.get("environment") == "staging"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_cross_app_rejection_incident_rejects_code_review_shape(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.incident_management.state.IncidentStateOverrides"
        ),
    )
    orch = await Orchestrator.create(cfg)
    try:
        with pytest.raises(ValidationError):
            await orch.start_session(
                query="x",
                state_overrides={
                    "pr_url": "https://example/pr/1",
                    "repo": "foo/bar",
                },
            )
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_cross_app_rejection_code_review_rejects_incident_shape(tmp_path):
    cfg = _base_cfg(
        tmp_path,
        state_overrides_schema=(
            "examples.code_review.state.CodeReviewStateOverrides"
        ),
        skills_dir="examples/code_review/skills",
    )
    orch = await Orchestrator.create(cfg)
    try:
        with pytest.raises(ValidationError):
            await orch.start_session(
                query="x",
                state_overrides={
                    "environment": "prod",
                    "severity": "critical",
                },
            )
    finally:
        await orch.aclose()


# ---------------------------------------------------------------------------
# YAML round-trip.
# ---------------------------------------------------------------------------


def _load_app_config_from_yaml(path: str) -> AppConfig:
    """Full ``load_config`` round-trip (env interpolation +
    ``AppConfig(**raw)``). Used for self-contained YAMLs (
    ``config/code_review.runtime.yaml``, ``config/config.yaml``).
    """
    from runtime.config import load_config
    return load_config(path)


def _load_orchestrator_block(path: str) -> OrchestratorConfig:
    """Partial loader for app-overlay YAMLs that don't carry the
    full ``llm:``/``mcp:`` blocks (e.g. ``incident_management.yaml``
    is composed with ``config.yaml`` at deploy time).

    Validates ONLY the ``orchestrator:`` block — sufficient for a
    YAML round-trip of the schema field.
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return OrchestratorConfig.model_validate(data["orchestrator"])


def test_code_review_yaml_loads_with_schema():
    cfg = _load_app_config_from_yaml("config/code_review.runtime.yaml")
    assert (cfg.orchestrator.state_overrides_schema
            == "examples.code_review.state.CodeReviewStateOverrides")


def test_incident_yaml_loads_with_schema():
    orch = _load_orchestrator_block("config/incident_management.yaml")
    assert (orch.state_overrides_schema
            == "examples.incident_management.state.IncidentStateOverrides")


def test_default_yaml_loads_with_schema():
    cfg = _load_app_config_from_yaml("config/config.yaml")
    assert (cfg.orchestrator.state_overrides_schema
            == "examples.incident_management.state.IncidentStateOverrides")


# ---------------------------------------------------------------------------
# TerminalToolRule.match_args dispatch (DECOUPLE-07 / D-08-03).
# ---------------------------------------------------------------------------


@pytest.fixture
def code_review_orch_for_dispatch(tmp_path):
    """Build an Orchestrator from the code_review YAML so the
    ``terminal_tools`` rules with ``match_args`` are wired in."""
    cfg = _load_app_config_from_yaml("config/code_review.runtime.yaml")
    # Override storage paths to tmp_path so the test is hermetic.
    cfg.storage.metadata.url = f"sqlite:///{tmp_path}/cr.db"
    cfg.storage.vector.path = str(tmp_path / "faiss")
    return cfg


def _exec_toolcall(name: str, args: dict) -> ToolCall:
    return ToolCall(
        agent="recommender",
        tool=name,
        args=args,
        result={"ok": True},
        ts="2026-01-01T00:00:00Z",
        status="executed",
    )


@pytest.mark.asyncio
async def test_code_review_yaml_terminal_tools_dispatch_approve(
    code_review_orch_for_dispatch,
):
    cfg = code_review_orch_for_dispatch
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(query="review PR")
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_toolcall(
            "set_recommendation",
            {"recommendation": "approve", "summary": "LGTM"},
        ))
        # Replicate finalizer behaviour — call _infer directly so
        # we don't depend on the agent graph.
        decision = orch._infer_terminal_decision(inc.tool_calls)
        assert decision is not None
        new_status, _ = decision
        assert new_status == "approved"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_code_review_yaml_terminal_tools_dispatch_request_changes(
    code_review_orch_for_dispatch,
):
    cfg = code_review_orch_for_dispatch
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(query="review PR")
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_toolcall(
            "set_recommendation",
            {"recommendation": "request_changes", "summary": "fix"},
        ))
        decision = orch._infer_terminal_decision(inc.tool_calls)
        assert decision is not None
        new_status, _ = decision
        assert new_status == "changes_requested"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_code_review_yaml_terminal_tools_dispatch_comment(
    code_review_orch_for_dispatch,
):
    cfg = code_review_orch_for_dispatch
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(query="review PR")
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_toolcall(
            "set_recommendation",
            {"recommendation": "comment", "summary": "fyi"},
        ))
        decision = orch._infer_terminal_decision(inc.tool_calls)
        assert decision is not None
        new_status, _ = decision
        assert new_status == "commented"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_code_review_yaml_terminal_tools_dispatch_unknown_arg_no_match(
    code_review_orch_for_dispatch,
):
    """A ``set_recommendation`` call with an unknown ``recommendation``
    value should match no rule; finalizer falls through to
    ``default_terminal_status`` (``unreviewed``)."""
    cfg = code_review_orch_for_dispatch
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(query="review PR")
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_toolcall(
            "set_recommendation",
            {"recommendation": "abstain"},
        ))
        decision = orch._infer_terminal_decision(inc.tool_calls)
        assert decision is None
    finally:
        await orch.aclose()
