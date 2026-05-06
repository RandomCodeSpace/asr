"""Boundary-validation tests for the generic status vocabulary
+ terminal-tool registry on ``OrchestratorConfig``.

Covers DECOUPLE-03 acceptance: framework rejects unknown statuses
and unknown default at config-load (not at gateway-eval time).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import OrchestratorConfig
from runtime.terminal_tools import StatusDef, TerminalToolRule


# ---- Test 1.1: TerminalToolRule + StatusDef shape ----

def test_terminal_tool_rule_minimal_construction():
    r = TerminalToolRule(tool_name="mark_resolved", status="resolved")
    assert r.tool_name == "mark_resolved"
    assert r.status == "resolved"
    assert r.extract_fields == {}


def test_terminal_tool_rule_with_extract_fields_round_trip():
    r = TerminalToolRule(
        tool_name="mark_escalated",
        status="escalated",
        extract_fields={"team": ["args.team", "result.team"]},
    )
    dumped = r.model_dump()
    assert dumped == {
        "tool_name": "mark_escalated",
        "status": "escalated",
        "extract_fields": {"team": ["args.team", "result.team"]},
    }


def test_terminal_tool_rule_extra_forbid():
    with pytest.raises(ValidationError):
        TerminalToolRule(tool_name="x", status="y", garbage="bad")  # type: ignore[call-arg]


def test_status_def_minimal_construction():
    s = StatusDef(name="resolved", terminal=True, kind="success")
    assert s.name == "resolved"
    assert s.terminal is True
    assert s.kind == "success"


def test_status_def_kind_must_be_literal():
    with pytest.raises(ValidationError):
        StatusDef(name="x", terminal=True, kind="bogus")  # type: ignore[arg-type]


def test_status_def_extra_forbid_color_rejected():
    # Per D-06-05 rejected alternative: ``color`` is a UI-presentation
    # leak; ``UIConfig.badges`` owns colours.
    with pytest.raises(ValidationError):
        StatusDef(name="x", terminal=True, kind="success", color="red")  # type: ignore[call-arg]


# ---- Test 1.2: OrchestratorConfig cross-field validation ----

@pytest.fixture
def basic_statuses() -> dict[str, StatusDef]:
    return {
        "resolved": StatusDef(name="resolved", terminal=True, kind="success"),
        "escalated": StatusDef(name="escalated", terminal=True, kind="escalation"),
        "needs_review": StatusDef(name="needs_review", terminal=True, kind="needs_review"),
        "in_progress": StatusDef(name="in_progress", terminal=False, kind="pending"),
    }


def test_orchestrator_config_happy_path(basic_statuses):
    cfg = OrchestratorConfig(
        statuses=basic_statuses,
        terminal_tools=[
            TerminalToolRule(tool_name="mark_resolved", status="resolved"),
            TerminalToolRule(
                tool_name="mark_escalated",
                status="escalated",
                extract_fields={"team": ["args.team", "result.team"]},
            ),
        ],
        default_terminal_status="needs_review",
    )
    assert len(cfg.terminal_tools) == 2
    assert cfg.default_terminal_status == "needs_review"


def test_orchestrator_config_bare_default_constructs():
    # Framework default ``OrchestratorConfig()`` — empty registry,
    # bare app loads still parse so unconfigured deploys are valid
    # (see config/config.yaml in Wave 2).
    cfg = OrchestratorConfig()
    assert cfg.terminal_tools == []
    assert cfg.statuses == {}
    assert cfg.default_terminal_status is None


def test_orchestrator_config_default_without_statuses_rejected():
    with pytest.raises(ValidationError, match="default_terminal_status is set"):
        OrchestratorConfig(default_terminal_status="resolved")


def test_orchestrator_config_terminal_tools_without_statuses_rejected():
    with pytest.raises(ValidationError, match="terminal_tools is non-empty"):
        OrchestratorConfig(
            terminal_tools=[TerminalToolRule(tool_name="x", status="y")],
        )


def test_orchestrator_config_default_required_when_statuses_set(basic_statuses):
    with pytest.raises(ValidationError, match="default_terminal_status is required"):
        OrchestratorConfig(statuses=basic_statuses)


def test_orchestrator_config_default_must_reference_known_status(basic_statuses):
    with pytest.raises(ValidationError, match="default_terminal_status='ghost'"):
        OrchestratorConfig(
            statuses=basic_statuses,
            default_terminal_status="ghost",
        )


def test_orchestrator_config_default_must_be_terminal(basic_statuses):
    # ``in_progress`` is terminal=False — cannot be the default.
    with pytest.raises(ValidationError, match="references a non-terminal status"):
        OrchestratorConfig(
            statuses=basic_statuses,
            default_terminal_status="in_progress",
        )


def test_orchestrator_config_rule_status_must_reference_known_status(basic_statuses):
    with pytest.raises(ValidationError, match=r"terminal_tools\[0\].status='ghost'"):
        OrchestratorConfig(
            statuses=basic_statuses,
            default_terminal_status="needs_review",
            terminal_tools=[TerminalToolRule(tool_name="x", status="ghost")],
        )


def test_orchestrator_config_rule_index_in_error_for_second_rule(basic_statuses):
    with pytest.raises(ValidationError, match=r"terminal_tools\[1\].status='ghost'"):
        OrchestratorConfig(
            statuses=basic_statuses,
            default_terminal_status="needs_review",
            terminal_tools=[
                TerminalToolRule(tool_name="ok", status="resolved"),
                TerminalToolRule(tool_name="bad", status="ghost"),
            ],
        )


def test_orchestrator_config_extra_forbid_survives(basic_statuses):
    with pytest.raises(ValidationError):
        OrchestratorConfig(
            statuses=basic_statuses,
            default_terminal_status="needs_review",
            garbage="not allowed",  # type: ignore[call-arg]
        )
