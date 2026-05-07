"""End-to-end smoke tests for the code_review example app
(DECOUPLE-07 / D-08-03).

Constructs an Orchestrator from ``config/code_review.runtime.yaml``,
exercises real wiring (terminal_tools registry consultation, state
overrides validation, mcp_server.set_recommendation), and asserts:

1. ``set_recommendation(recommendation=approve|request_changes|comment)``
   resolves to the matching code_review status (``approved`` /
   ``changes_requested`` / ``commented``) — not the v1.0
   "framework didn't recognize the terminal tool" failure mode that
   coerced everything to ``unreviewed`` / ``needs_review``.

2. ``state_overrides`` validation rejects unknown keys (DECOUPLE-05)
   and cross-app shapes (incident-vocabulary keys against the
   code_review schema).

3. The code_review session run does NOT pull in any
   ``incident_management`` modules (DECOUPLE-07's structural
   invariant).

Mocking philosophy: the actual LLM-driven multi-skill flow is
exercised by per-skill unit tests in ``test_agent_node.py`` /
``test_finalize_status_inference.py``. Here we drive the orchestrator
directly through ``mcp_server.set_recommendation`` (which writes the
ToolCall the rule matcher consumes) and ``_finalize_session_status_async``
to validate the YAML-driven dispatch.
"""
from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from runtime.config import LLMConfig, RuntimeConfig, load_config
from runtime.orchestrator import Orchestrator
from runtime.state import ToolCall


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_code_review_cfg(tmp_path):
    """Load ``config/code_review.runtime.yaml`` and override storage
    paths to ``tmp_path`` so the test is hermetic."""
    cfg = load_config("config/code_review.runtime.yaml")
    cfg.storage.metadata.url = f"sqlite:///{tmp_path}/cr.db"
    cfg.storage.vector.path = str(tmp_path / "faiss")
    cfg.paths.incidents_dir = str(tmp_path)
    # Force the bundled stub LLM so no provider env vars are required.
    cfg.llm = LLMConfig.stub()
    cfg.runtime = RuntimeConfig(state_class=None)
    return cfg


def _exec_set_recommendation(
    *, recommendation: str, summary: str = "stub",
) -> ToolCall:
    """Synthesize the executed ToolCall the recommender skill would
    emit. Same shape as a real LangGraph-emitted call so the
    rule-matcher path is exactly the production path."""
    return ToolCall(
        agent="recommender",
        tool="local_cr:set_recommendation",
        args={
            "session_id": "<auto>",
            "recommendation": recommendation,
            "summary": summary,
        },
        result={"ok": True, "recommendation": recommendation,
                "summary": summary},
        ts="2026-01-01T00:00:00Z",
        status="executed",
    )


# ---------------------------------------------------------------------------
# Recommendation-dispatch e2e.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_review_e2e_approves_clean_pr(tmp_path):
    pre_modules = set(sys.modules.keys())
    cfg = _build_code_review_cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="Review PR https://github.com/foo/bar/pull/1",
            state_overrides={
                "pr_url": "https://github.com/foo/bar/pull/1",
                "repo": "foo/bar",
                "pr_number": 1,
            },
        )
        inc = orch.store.load(sid)
        # Stamp a recommender terminal tool call onto the session row
        # — same shape the real LangGraph harvester writes when the
        # recommender skill runs against the live MCP server.
        inc.tool_calls.append(_exec_set_recommendation(
            recommendation="approve", summary="LGTM",
        ))
        inc.extra_fields["overall_recommendation"] = "approve"
        orch.store.save(inc)

        new_status = await orch._finalize_session_status_async(sid)
        assert new_status == "approved", (
            f"Expected status='approved' (terminal_tools rule on "
            f"set_recommendation+args.recommendation=approve); "
            f"got {new_status!r}. If status is 'unreviewed' the "
            f"rule didn't fire — verify match_args dispatch. If "
            f"status is 'needs_review' the framework fell through "
            f"to default — verify OrchestratorConfig.statuses "
            f"includes 'approved'."
        )
        loaded = orch.store.load(sid)
        assert loaded.status == "approved"
        assert loaded.status not in ("unreviewed", "needs_review")
        assert (loaded.extra_fields.get("overall_recommendation")
                == "approve")
    finally:
        await orch.aclose()

    # DECOUPLE-07 structural invariant: a code_review session must
    # not load any incident_management modules.
    new_modules = set(sys.modules.keys()) - pre_modules
    incident_imports = [
        m for m in new_modules
        if "incident" in m.lower()
    ]
    assert not incident_imports, (
        f"code_review session pulled in incident_management modules: "
        f"{incident_imports}. DECOUPLE-07 violated."
    )


@pytest.mark.asyncio
async def test_code_review_e2e_request_changes_on_critical(tmp_path):
    cfg = _build_code_review_cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="Review PR with critical bug",
            state_overrides={
                "pr_url": "https://github.com/foo/bar/pull/2",
                "repo": "foo/bar",
                "pr_number": 2,
            },
        )
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_set_recommendation(
            recommendation="request_changes",
            summary="Critical bug found",
        ))
        orch.store.save(inc)

        new_status = await orch._finalize_session_status_async(sid)
        assert new_status == "changes_requested"
        loaded = orch.store.load(sid)
        assert loaded.status == "changes_requested"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_code_review_e2e_comment_on_warnings(tmp_path):
    cfg = _build_code_review_cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="Review PR with minor warnings",
            state_overrides={
                "pr_url": "https://github.com/foo/bar/pull/3",
                "repo": "foo/bar",
                "pr_number": 3,
            },
        )
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_set_recommendation(
            recommendation="comment",
            summary="Minor nits",
        ))
        orch.store.save(inc)

        new_status = await orch._finalize_session_status_async(sid)
        assert new_status == "commented"
        loaded = orch.store.load(sid)
        assert loaded.status == "commented"
    finally:
        await orch.aclose()


# ---------------------------------------------------------------------------
# state_overrides validation (DECOUPLE-05).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_review_e2e_state_overrides_validates(tmp_path):
    """Unknown keys are rejected before the graph runs."""
    cfg = _build_code_review_cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        with pytest.raises(ValidationError):
            await orch.start_session(
                query="x",
                state_overrides={
                    "pr_url": "https://github.com/foo/bar/pull/9",
                    "definitely_unknown_key": "x",
                },
            )
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_code_review_e2e_state_overrides_cross_app_rejection(tmp_path):
    """Incident-shaped overrides (``environment`` / ``severity``) are
    rejected by the code_review schema (cross-app guard)."""
    cfg = _build_code_review_cfg(tmp_path)
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
# Structural-coupling guard (DECOUPLE-07 invariant).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_review_e2e_no_incident_imports(tmp_path):
    """End-to-end: a code_review session must not pull
    ``incident_management`` into ``sys.modules``."""
    pre_modules = set(sys.modules.keys())
    cfg = _build_code_review_cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="Review PR",
            state_overrides={"pr_url": "https://example/pr/1"},
        )
        inc = orch.store.load(sid)
        inc.tool_calls.append(_exec_set_recommendation(
            recommendation="approve",
        ))
        orch.store.save(inc)
        await orch._finalize_session_status_async(sid)
    finally:
        await orch.aclose()

    new_modules = set(sys.modules.keys()) - pre_modules
    incident_imports = [
        m for m in new_modules if "incident" in m.lower()
    ]
    assert not incident_imports, (
        f"code_review session pulled in incident_management modules: "
        f"{incident_imports}. DECOUPLE-07 violated."
    )
