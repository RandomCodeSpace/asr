"""Phase 20 (HARD-09): UI tests for the agent step / event display path.

Targets:
  * ``_format_event``                — streaming event → display line
  * ``_summary_attribution``         — attribution line composition
  * ``_field`` / ``_resolve_field``  — top-level vs extra_fields routing
  * ``_badge_field_slots``           — UIConfig → badge slot pair
  * ``_retry_button_state_for``      — RetryDecision.reason → button label/disabled

Pure functions; no Streamlit runtime needed.
"""
from __future__ import annotations

from runtime.config import (
    FrameworkAppConfig,
    UIBadge,
    UIConfig,
    UIDetailField,
)


# ---------------------------------------------------------------------------
# _format_event — streaming events to one-liners
# ---------------------------------------------------------------------------


def test_format_event_investigation_started() -> None:
    from runtime.ui import _format_event
    line = _format_event({
        "event": "investigation_started",
        "ts": "2026-05-07T10:00:00Z",
        "incident_id": "INC-1",
    })
    assert line is not None
    assert "INC-1" in line
    assert "start" in line


def test_format_event_investigation_completed() -> None:
    from runtime.ui import _format_event
    line = _format_event({
        "event": "investigation_completed",
        "ts": "2026-05-07T10:01:00Z",
        "incident_id": "INC-9",
    })
    assert line is not None
    assert "done" in line
    assert "INC-9" in line


def test_format_event_chain_start_filtered_by_agent_names() -> None:
    """``on_chain_start`` events for nodes NOT in the configured agent
    set are suppressed (returns None) to keep the timeline focused.
    """
    from runtime.ui import _format_event

    agents = frozenset({"triage", "investigator"})
    ev_visible = {"event": "on_chain_start", "node": "triage", "ts": "T"}
    ev_hidden = {"event": "on_chain_start", "node": "internal_helper", "ts": "T"}

    assert _format_event(ev_visible, agents) is not None
    assert "triage" in _format_event(ev_visible, agents)
    assert _format_event(ev_hidden, agents) is None


def test_format_event_empty_agent_set_shows_all() -> None:
    """Safe fallback — when agent_names is empty (caller didn't have
    the list handy), every chain event is shown."""
    from runtime.ui import _format_event
    line = _format_event(
        {"event": "on_chain_end", "node": "anything", "ts": "T"},
        frozenset(),
    )
    assert line is not None
    assert "anything" in line


def test_format_event_tool_end_truncates_long_output() -> None:
    """Tool-end snippets are clipped to 120 chars to keep the live
    timeline readable when an MCP tool returns a giant payload."""
    from runtime.ui import _format_event

    huge = "x" * 500
    line = _format_event({
        "event": "on_tool_end",
        "node": "search_logs",
        "ts": "T",
        "data": {"output": huge},
    })
    assert line is not None
    # The clipped snippet must be at most 120 chars; raw 500-char output
    # would inflate the line beyond that snippet length.
    snippet_part = line.split("search_logs", 1)[1]
    assert len(snippet_part.strip()) <= 121  # 120 chars + leading space


def test_format_event_unknown_event_returns_none() -> None:
    from runtime.ui import _format_event
    assert _format_event({"event": "totally_made_up", "ts": "T"}) is None


# ---------------------------------------------------------------------------
# _summary_attribution — UIConfig-driven detail fields
# ---------------------------------------------------------------------------


def test_summary_attribution_returns_empty_when_no_summary_fields() -> None:
    from runtime.ui import _summary_attribution
    app_cfg = FrameworkAppConfig(ui=UIConfig(detail_fields=[]))
    assert _summary_attribution({"id": "INC-1"}, app_cfg) == ""


def test_summary_attribution_builds_by_clause() -> None:
    """First non-empty summary-section field becomes ``by <value>``;
    subsequent ones render as ``(extra1, extra2)``.
    """
    from runtime.ui import _summary_attribution

    app_cfg = FrameworkAppConfig(ui=UIConfig(
        detail_fields=[
            UIDetailField(key="reporter.id", label="Reporter", section="summary"),
            UIDetailField(key="reporter.team", label="Team", section="summary"),
            UIDetailField(key="component", label="Component", section="meta"),
        ],
    ))
    sess = {
        "extra_fields": {
            "reporter": {"id": "alice", "team": "platform"},
            "component": "billing",
        },
    }
    result = _summary_attribution(sess, app_cfg)
    assert result.startswith("by alice")
    assert "platform" in result
    # 'meta'-section field must NOT appear
    assert "billing" not in result


def test_summary_attribution_skips_empty_fields() -> None:
    """Missing fields (resolved to "") drop out — no stray commas."""
    from runtime.ui import _summary_attribution

    app_cfg = FrameworkAppConfig(ui=UIConfig(
        detail_fields=[
            UIDetailField(key="reporter.id", label="Reporter", section="summary"),
            UIDetailField(key="missing.key", label="Missing", section="summary"),
        ],
    ))
    sess = {"extra_fields": {"reporter": {"id": "bob"}}}
    assert _summary_attribution(sess, app_cfg) == "by bob"


# ---------------------------------------------------------------------------
# _field / _resolve_field — top-level + extra_fields routing
# ---------------------------------------------------------------------------


def test_field_reads_top_level_first() -> None:
    from runtime.ui import _field
    assert _field({"summary": "top-level"}, "summary") == "top-level"


def test_field_falls_back_to_extra_fields() -> None:
    from runtime.ui import _field
    assert (
        _field({"extra_fields": {"summary": "from-extra"}}, "summary")
        == "from-extra"
    )


def test_field_returns_default_when_missing() -> None:
    from runtime.ui import _field
    assert _field({}, "missing", default="—") == "—"


def test_field_coerces_non_string_to_str() -> None:
    """Numeric / bool fields end up rendered into markdown — the helper
    coerces so callers don't have to."""
    from runtime.ui import _field
    assert _field({"count": 42}, "count") == "42"


def test_resolve_field_walks_dotted_path_into_extra_fields() -> None:
    from runtime.ui import _resolve_field
    sess = {"extra_fields": {"reporter": {"id": "alice"}}}
    assert _resolve_field(sess, "reporter.id") == "alice"


def test_resolve_field_returns_empty_string_for_missing_path() -> None:
    from runtime.ui import _resolve_field
    sess = {"extra_fields": {"reporter": {"id": "alice"}}}
    assert _resolve_field(sess, "reporter.team") == ""
    assert _resolve_field(sess, "totally.absent.key") == ""


# ---------------------------------------------------------------------------
# _badge_field_slots
# ---------------------------------------------------------------------------


def test_badge_field_slots_picks_first_two_non_status_keys() -> None:
    from runtime.ui import _badge_field_slots
    app_cfg = FrameworkAppConfig(ui=UIConfig(badges={
        "status": {"open": UIBadge(label="OPEN", color="red")},
        "severity": {"sev1": UIBadge(label="SEV1", color="red")},
        "category": {"network": UIBadge(label="NETWORK", color="blue")},
        "third": {"x": UIBadge(label="X", color="gray")},
    }))
    primary, secondary = _badge_field_slots(app_cfg)
    assert primary == "severity"
    assert secondary == "category"


def test_badge_field_slots_returns_blanks_when_only_status_configured() -> None:
    from runtime.ui import _badge_field_slots
    app_cfg = FrameworkAppConfig(ui=UIConfig(badges={
        "status": {"open": UIBadge(label="OPEN", color="red")},
    }))
    primary, secondary = _badge_field_slots(app_cfg)
    assert primary == ""
    assert secondary == ""


# ---------------------------------------------------------------------------
# _retry_button_state_for — RetryDecision.reason → (label, disabled)
# ---------------------------------------------------------------------------


def test_retry_button_state_auto_retry_is_enabled() -> None:
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="auto_retry", retry_count=1, cap=3,
        last_confidence=0.9, threshold=0.5,
    )
    assert label == "Retry"
    assert disabled is False


def test_retry_button_state_max_retries_disabled_with_count() -> None:
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="max_retries_exceeded", retry_count=3, cap=3,
        last_confidence=0.9, threshold=0.5,
    )
    assert disabled is True
    assert "3/3" in label


def test_retry_button_state_low_confidence_renders_percentages() -> None:
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="low_confidence_no_retry", retry_count=0, cap=3,
        last_confidence=0.32, threshold=0.75,
    )
    assert disabled is True
    assert "32%" in label
    assert "75%" in label


def test_retry_button_state_unknown_reason_disabled_with_label() -> None:
    """Future-proofing: a reason the UI doesn't recognise still renders
    a disabled button rather than crashing."""
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="some_future_reason", retry_count=0, cap=3,
        last_confidence=None, threshold=0.5,
    )
    assert disabled is True
    assert "some_future_reason" in label
