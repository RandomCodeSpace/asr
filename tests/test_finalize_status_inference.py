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
from runtime.storage.session_store import SessionStore, StaleVersionError
from runtime.terminal_tools import StatusDef, TerminalToolRule


# Registry-driven equivalent of the v1.0 hardcoded ``_TERMINAL_TOOL_RULES``
# table that lived in orchestrator.py. The framework now reads this from
# ``self.cfg.orchestrator.terminal_tools`` (Phase 6 / DECOUPLE-02).
_INCIDENT_STATUSES = {
    "open":         StatusDef(name="open",         terminal=False, kind="pending"),
    "escalated":    StatusDef(name="escalated",    terminal=True,  kind="escalation"),
    "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
    "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
}
_INCIDENT_RULES = [
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


def _incident_app_cfg() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(
            statuses=_INCIDENT_STATUSES,
            terminal_tools=_INCIDENT_RULES,
            default_terminal_status="needs_review",
        ),
    )


def _make_orch_with_store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)
    cfg = _incident_app_cfg()
    class _O:
        def __init__(self, s, c):
            self.store = s
            self.cfg = c
        _finalize_session_status = Orchestrator._finalize_session_status
        _save_or_yield = Orchestrator._save_or_yield
        _infer_terminal_decision = Orchestrator._infer_terminal_decision
        _extract_field = Orchestrator._extract_field
    return _O(store, cfg), store


def test_finalize_with_mark_escalated_in_history_yields_escalated(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_escalated",
        args={"team": "platform-oncall"}, result={"status": "escalated"},
        ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    new_status = orch._finalize_session_status(inc.id)
    assert new_status == "escalated"
    fresh = store.load(inc.id)
    assert fresh.status == "escalated"
    assert fresh.extra_fields.get("escalated_to") == "platform-oncall"


def test_finalize_with_mark_resolved_in_history_yields_resolved(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.tool_calls.append(ToolCall(
        agent="resolution", tool="mark_resolved", args={},
        result={"status": "resolved"}, ts="t", status="executed",
    ))
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "resolved"


def test_finalize_with_no_terminal_tool_yields_needs_review(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) == "needs_review"


def test_finalize_does_not_clobber_terminal_status(tmp_path):
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "escalated"
    store.save(inc)
    assert orch._finalize_session_status(inc.id) is None
    assert store.load(inc.id).status == "escalated"


def test_finalize_returns_none_on_stale_version(tmp_path, monkeypatch):
    """If a concurrent finalize wins the race, save() raises
    StaleVersionError; this finalize must yield (return None) rather
    than propagate the exception up the async stream loop."""
    orch, store = _make_orch_with_store(tmp_path)
    inc = store.create(query="q", environment="dev", reporter_id="u", reporter_team="t")
    inc.status = "in_progress"
    store.save(inc)

    def _raise_stale(_):
        raise StaleVersionError("concurrent writer settled first")

    monkeypatch.setattr(store, "save", _raise_stale)
    assert orch._finalize_session_status(inc.id) is None
