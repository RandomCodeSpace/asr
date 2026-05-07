"""Integration tests for the generic terminal-tool registry.

Two app configurations are exercised:

  * ``incident_management`` — the existing 3-tool / 4-status registration
    (D-06-04). Proves no behavior regression: the same tool calls the
    v1.0 hard-coded ``_TERMINAL_TOOL_RULES`` table classified continue
    to classify the same way under the registry-driven path.
  * ``code_review`` — a synthetic ``submit_review -> approved`` registration.
    Proves DECOUPLE-02: a non-incident app's session is classified by
    ITS OWN registered rules, NOT falsely marked ``needs_review``
    because the framework only knew ASR vocabulary. A session with no
    ``submit_review`` call falls through to ``unreviewed`` (the app's
    own ``default_terminal_status``) — NOT ``needs_review``.

Test harness mirrors ``tests/test_finalize_status_inference.py`` (the
v1.0 finalize-only fixture style): a real ``SessionStore`` against an
in-memory SQLite, a tiny synthesized class that splices
``Orchestrator._finalize_session_status`` into a stub holding only
``store`` and ``cfg``. We exercise the real method body — no mocking.
"""
from __future__ import annotations

from sqlalchemy import create_engine

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    OrchestratorConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.state import ToolCall
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.terminal_tools import StatusDef, TerminalToolRule


# ---------------------------------------------------------------------------
# Fixture configurations
# ---------------------------------------------------------------------------

INCIDENT_STATUSES = {
    "open":         StatusDef(name="open",         terminal=False, kind="pending"),
    "escalated":    StatusDef(name="escalated",    terminal=True,  kind="escalation"),
    "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
    "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
}

# NOTE: tool names live in test fixtures freely — the leak ratchet
# whitelists ``tests/``. Production code reads them out of YAML only.
INCIDENT_RULES = [
    TerminalToolRule(tool_name="mark_resolved", status="resolved"),
    TerminalToolRule(
        tool_name="mark_escalated", status="escalated",
        extract_fields={"team": ["args.team", "result.team"]},
    ),
    TerminalToolRule(
        tool_name="notify_oncall", status="escalated",
        extract_fields={"team": ["args.team"]},
    ),
]

CODE_REVIEW_STATUSES = {
    "in_review":  StatusDef(name="in_review",  terminal=False, kind="pending"),
    "approved":   StatusDef(name="approved",   terminal=True,  kind="success"),
    "unreviewed": StatusDef(name="unreviewed", terminal=True,  kind="needs_review"),
}
CODE_REVIEW_RULES = [
    TerminalToolRule(tool_name="submit_review", status="approved"),
]


def _incident_app_cfg() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(
            statuses=INCIDENT_STATUSES,
            terminal_tools=INCIDENT_RULES,
            default_terminal_status="needs_review",
        ),
    )


def _code_review_app_cfg() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(
            statuses=CODE_REVIEW_STATUSES,
            terminal_tools=CODE_REVIEW_RULES,
            default_terminal_status="unreviewed",
        ),
    )


def _make_orch(tmp_path, app_cfg: AppConfig):
    """Build a finalize-only test harness against a real SQLite store.

    Mirrors the v1.0 ``_make_orch_with_store`` pattern in
    ``tests/test_finalize_status_inference.py`` — splices the real
    ``Orchestrator`` methods onto a stub holding only ``store`` and
    ``cfg`` so we exercise the actual method body without paying the
    ``Orchestrator.create()`` MCP-bringup cost.
    """
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)

    class _O:
        def __init__(self, s, c):
            self.store = s
            self.cfg = c
        _finalize_session_status = Orchestrator._finalize_session_status
        _save_or_yield = Orchestrator._save_or_yield
        _infer_terminal_decision = Orchestrator._infer_terminal_decision
        _extract_field = Orchestrator._extract_field

    return _O(store, app_cfg), store


# ---------------------------------------------------------------------------
# Test 2.1.a — mark_resolved -> resolved (registry-driven equivalent of v1.0
# behavior; proves the lookup path through self.cfg.orchestrator.terminal_tools)
# ---------------------------------------------------------------------------

def test_finalize_with_mark_resolved_yields_resolved(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_resolved", args={},
        result={"status": "resolved"}, ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "resolved"
    assert store.load(inc.id).status == "resolved"


# ---------------------------------------------------------------------------
# Test 2.1.b — mark_escalated -> escalated + extra_fields["team"] AND
# extra_fields["escalated_to"] (v1.0 compat mirror via kind=escalation)
# ---------------------------------------------------------------------------

def test_finalize_with_mark_escalated_yields_escalated_with_team(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_escalated",
        args={"team": "platform-oncall"}, result={"status": "escalated"},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "escalated"
    fresh = store.load(inc.id)
    assert fresh.status == "escalated"
    # Generic registry key (D-06-02) — ``team`` lives in extra_fields.
    assert fresh.extra_fields.get("team") == "platform-oncall"
    # v1.0 compat mirror via kind=escalation (D-06-05).
    assert fresh.extra_fields.get("escalated_to") == "platform-oncall"


# ---------------------------------------------------------------------------
# Test 2.1.c — notify_oncall -> escalated (legacy direct path preserved)
# ---------------------------------------------------------------------------

def test_finalize_with_notify_oncall_yields_escalated(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="triage", tool="notify_oncall",
        args={"team": "data-oncall"}, result={"paged": True},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "escalated"
    fresh = store.load(inc.id)
    assert fresh.extra_fields.get("team") == "data-oncall"
    assert fresh.extra_fields.get("escalated_to") == "data-oncall"


# ---------------------------------------------------------------------------
# Test 2.1.d — prefixed form (``local_inc:mark_resolved``) matches the same
# as bare. MCP tool names are server-prefixed; the registry's suffix check
# preserves v1.0 ``endswith(f":{bare}")`` semantics.
# ---------------------------------------------------------------------------

def test_finalize_with_prefixed_tool_name_matches(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="local_inc:mark_resolved", args={},
        result={"status": "resolved"}, ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "resolved"


# ---------------------------------------------------------------------------
# Test 2.1.e — no rule fires -> default_terminal_status (incident_management
# = needs_review) + extra_fields["needs_review_reason"]
# ---------------------------------------------------------------------------

def test_finalize_with_no_terminal_tool_falls_to_default(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "needs_review"
    fresh = store.load(inc.id)
    assert fresh.status == "needs_review"
    assert fresh.extra_fields.get("needs_review_reason") == (
        "graph completed without terminal tool call"
    )


# ---------------------------------------------------------------------------
# Test 2.1.f — DECOUPLE-02 acceptance proof: a code_review-style app with
# ``submit_review -> approved`` is finalized as ``approved``, NOT
# ``needs_review`` because the framework reads the registered rules.
# ---------------------------------------------------------------------------

def test_finalize_code_review_submit_review_yields_approved(tmp_path):
    orch, store = _make_orch(tmp_path, _code_review_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="reviewer", tool="submit_review", args={"verdict": "lgtm"},
        result={"status": "approved"}, ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "approved"
    assert store.load(inc.id).status == "approved"


# ---------------------------------------------------------------------------
# Test 2.1.g — code_review with no terminal call falls through to
# ``unreviewed`` (the app's own default), NOT ``needs_review``. Proves the
# framework reads ``self.cfg.orchestrator.default_terminal_status`` rather
# than hard-coding the incident vocabulary.
# ---------------------------------------------------------------------------

def test_finalize_code_review_no_terminal_call_yields_unreviewed(tmp_path):
    orch, store = _make_orch(tmp_path, _code_review_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "unreviewed"
    fresh = store.load(inc.id)
    assert fresh.status == "unreviewed"
    # The reason key is generic — ``needs_review_reason`` is the
    # historical name (carries through from v1.0 for stability) but
    # the value remains the same regardless of the app's status name.
    assert fresh.extra_fields.get("needs_review_reason") == (
        "graph completed without terminal tool call"
    )


# ---------------------------------------------------------------------------
# Test 2.1.h — extract_fields honors both ``args.X`` and ``result.X``;
# ``result.team`` wins when ``args.team`` is missing (preserves v1.0
# ``_extract_team`` semantics — D-06-02).
# ---------------------------------------------------------------------------

def test_extract_fields_falls_back_to_result_when_args_missing(tmp_path):
    orch, store = _make_orch(tmp_path, _incident_app_cfg())
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_escalated",
        args={},  # no team here
        result={"team": "security-oncall"},  # team in result instead
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "escalated"
    fresh = store.load(inc.id)
    assert fresh.extra_fields.get("team") == "security-oncall"
    assert fresh.extra_fields.get("escalated_to") == "security-oncall"


# ---------------------------------------------------------------------------
# Test 2.1.i — extract_fields with multiple destination keys populates each
# into ``extra_fields`` independently. Proves the dict shape (D-06-02) is
# honored beyond the single-``team`` v1.0 case.
# ---------------------------------------------------------------------------

def test_extract_fields_with_multiple_destination_keys(tmp_path):
    # Synthesize a custom app config where one rule extracts both
    # ``team`` and ``severity``. (Not used in production YAMLs but
    # the framework should support arbitrary key counts.)
    statuses = {
        "open":      StatusDef(name="open",      terminal=False, kind="pending"),
        "escalated": StatusDef(name="escalated", terminal=True,  kind="escalation"),
        "fallback":  StatusDef(name="fallback",  terminal=True,  kind="needs_review"),
    }
    rules = [
        TerminalToolRule(
            tool_name="page_with_priority", status="escalated",
            extract_fields={
                "team":     ["args.team"],
                "severity": ["args.severity", "result.severity"],
            },
        ),
    ]
    cfg = AppConfig(
        llm=LLMConfig(), mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(
            statuses=statuses, terminal_tools=rules,
            default_terminal_status="fallback",
        ),
    )
    orch, store = _make_orch(tmp_path, cfg)
    inc = store.create(query="q", environment="dev", reporter_id="u",
                       reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="page_with_priority",
        args={"team": "sre", "severity": "sev1"},
        result={"paged": True},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "escalated"
    fresh = store.load(inc.id)
    assert fresh.extra_fields.get("team") == "sre"
    assert fresh.extra_fields.get("severity") == "sev1"
    # ``escalated_to`` mirror still fires because the matched rule's
    # status is kind=escalation — generic dispatch (D-06-05).
    assert fresh.extra_fields.get("escalated_to") == "sre"
