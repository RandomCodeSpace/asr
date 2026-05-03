"""Tests for the ``Skill.kind`` discriminator and per-kind validators.

* Each ``kind`` parses with a minimal valid config.
* Each forbidden field combination is rejected with a message naming
  the field.
* Missing required fields are rejected per kind.
* Cron and emit_signal_when expressions are validated at parse time.
* Old YAML without ``kind`` parses as responsive (back-compat).
"""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from runtime.skill import (
    DispatchRule,
    Skill,
    _validate_cron,
    _validate_safe_expr,
    load_skill,
)


# ---------------------------------------------------------------------------
# Back-compat: kind defaults to responsive
# ---------------------------------------------------------------------------


def test_kind_defaults_to_responsive():
    s = Skill(name="x", description="d", system_prompt="hi")
    assert s.kind == "responsive"


def test_yaml_without_kind_loads_as_responsive(tmp_path):
    d = tmp_path / "intake"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d",
        "tools": {"local": ["t"]},
        "routes": [{"when": "default", "next": "next_agent"}],
    }))
    (d / "system.md").write_text("you are an agent")
    skill = load_skill(d)
    assert skill.kind == "responsive"
    assert skill.system_prompt.startswith("you are an agent")


# ---------------------------------------------------------------------------
# Responsive: rejects fields belonging to other kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("subordinates", ["other"]),
    ("dispatch_prompt", "pick one"),
    ("dispatch_rules", [{"when": "True", "target": "x"}]),
    ("schedule", "* * * * *"),
    ("observe", ["tool_a"]),
    ("emit_signal_when", "True"),
    ("trigger_target", "trigger_a"),
])
def test_responsive_rejects_other_kind_fields(field, value):
    payload = {
        "name": "x", "description": "d", "kind": "responsive",
        "system_prompt": "hi", field: value,
    }
    with pytest.raises(ValidationError) as exc:
        Skill(**payload)
    assert field in str(exc.value)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def test_supervisor_minimal_llm_strategy_parses():
    s = Skill(
        name="router", description="d", kind="supervisor",
        subordinates=["a", "b"],
        dispatch_strategy="llm",
        dispatch_prompt="choose one",
    )
    assert s.kind == "supervisor"
    assert s.subordinates == ["a", "b"]


def test_supervisor_minimal_rule_strategy_parses():
    s = Skill(
        name="router", description="d", kind="supervisor",
        subordinates=["a", "b"],
        dispatch_strategy="rule",
        dispatch_rules=[
            DispatchRule(when="status == 'open'", target="a"),
            DispatchRule(when="True", target="b"),
        ],
    )
    assert s.dispatch_strategy == "rule"
    assert len(s.dispatch_rules) == 2


def test_supervisor_requires_subordinates():
    with pytest.raises(ValidationError, match="subordinates"):
        Skill(name="x", description="d", kind="supervisor",
              dispatch_strategy="llm", dispatch_prompt="p")


def test_supervisor_llm_requires_dispatch_prompt():
    with pytest.raises(ValidationError, match="dispatch_prompt"):
        Skill(name="x", description="d", kind="supervisor",
              subordinates=["a"], dispatch_strategy="llm")


def test_supervisor_rule_requires_dispatch_rules():
    with pytest.raises(ValidationError, match="dispatch_rules"):
        Skill(name="x", description="d", kind="supervisor",
              subordinates=["a"], dispatch_strategy="rule")


def test_supervisor_dispatch_rule_target_must_be_subordinate():
    with pytest.raises(ValidationError, match="not found in subordinates"):
        Skill(name="x", description="d", kind="supervisor",
              subordinates=["a"], dispatch_strategy="rule",
              dispatch_rules=[DispatchRule(when="True", target="ghost")])


def test_supervisor_dispatch_rule_unsafe_expression_rejected():
    with pytest.raises(ValidationError, match="disallowed"):
        Skill(name="x", description="d", kind="supervisor",
              subordinates=["a"], dispatch_strategy="rule",
              dispatch_rules=[DispatchRule(when="__import__('os')", target="a")])


@pytest.mark.parametrize("field,value", [
    ("system_prompt", "hi"),
    ("tools", {"local": ["t"]}),
    ("routes", [{"when": "default", "next": "x"}]),
    ("stub_response", "hi"),
    ("schedule", "* * * * *"),
    ("observe", ["tool_a"]),
    ("emit_signal_when", "True"),
    ("trigger_target", "trigger_a"),
])
def test_supervisor_rejects_other_kind_fields(field, value):
    payload = {
        "name": "x", "description": "d", "kind": "supervisor",
        "subordinates": ["a"],
        "dispatch_strategy": "llm",
        "dispatch_prompt": "p",
        field: value,
    }
    with pytest.raises(ValidationError) as exc:
        Skill(**payload)
    assert field in str(exc.value)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def test_monitor_minimal_parses():
    s = Skill(
        name="watch", description="d", kind="monitor",
        schedule="*/5 * * * *",
        observe=["error_rate"],
        emit_signal_when="observation['error_rate'] > 0.05",
        trigger_target="incident_high_error_rate",
    )
    assert s.kind == "monitor"
    assert s.schedule == "*/5 * * * *"


def test_monitor_requires_schedule():
    with pytest.raises(ValidationError, match="schedule"):
        Skill(name="x", description="d", kind="monitor",
              observe=["a"], emit_signal_when="True", trigger_target="t")


def test_monitor_rejects_malformed_cron():
    with pytest.raises(ValidationError, match="cron"):
        Skill(name="x", description="d", kind="monitor",
              schedule="every 5 minutes",
              observe=["a"], emit_signal_when="True", trigger_target="t")


def test_monitor_requires_observe():
    with pytest.raises(ValidationError, match="observe"):
        Skill(name="x", description="d", kind="monitor",
              schedule="* * * * *", observe=[],
              emit_signal_when="True", trigger_target="t")


def test_monitor_requires_emit_signal_when():
    with pytest.raises(ValidationError, match="emit_signal_when"):
        Skill(name="x", description="d", kind="monitor",
              schedule="* * * * *", observe=["a"],
              trigger_target="t")


def test_monitor_rejects_unsafe_emit_expression():
    with pytest.raises(ValidationError, match="disallowed"):
        Skill(name="x", description="d", kind="monitor",
              schedule="* * * * *", observe=["a"],
              emit_signal_when="__import__('os').system('rm -rf /')",
              trigger_target="t")


def test_monitor_requires_trigger_target():
    with pytest.raises(ValidationError, match="trigger_target"):
        Skill(name="x", description="d", kind="monitor",
              schedule="* * * * *", observe=["a"],
              emit_signal_when="True")


@pytest.mark.parametrize("field,value", [
    ("system_prompt", "hi"),
    ("routes", [{"when": "default", "next": "x"}]),
    ("stub_response", "hi"),
    ("subordinates", ["a"]),
    ("dispatch_prompt", "p"),
    ("dispatch_rules", [{"when": "True", "target": "a"}]),
])
def test_monitor_rejects_other_kind_fields(field, value):
    payload = {
        "name": "x", "description": "d", "kind": "monitor",
        "schedule": "* * * * *",
        "observe": ["t"],
        "emit_signal_when": "True",
        "trigger_target": "trig",
        field: value,
    }
    with pytest.raises(ValidationError) as exc:
        Skill(**payload)
    assert field in str(exc.value)


# ---------------------------------------------------------------------------
# Bounds and helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 11, -3])
def test_max_dispatch_depth_bounds(depth):
    with pytest.raises(ValidationError, match="max_dispatch_depth"):
        Skill(name="x", description="d", kind="supervisor",
              subordinates=["a"], dispatch_strategy="llm", dispatch_prompt="p",
              max_dispatch_depth=depth)


@pytest.mark.parametrize("expr,ok", [
    ("* * * * *", True),
    ("*/5 * * * *", True),
    ("0 9-17 * * 1-5", True),
    ("0 0 1 1 *", True),
    ("every 5 minutes", False),
    ("* * * *", False),
    ("a b c d e", False),
])
def test_validate_cron_accepts_valid_rejects_invalid(expr, ok):
    if ok:
        _validate_cron(expr)
    else:
        with pytest.raises(ValueError):
            _validate_cron(expr)


def test_validate_safe_expr_allows_simple_comparisons():
    _validate_safe_expr("a > 1 and b == 'open'", source="t")
    _validate_safe_expr("session['status'] in ['new', 'open']", source="t")


@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "open('/etc/passwd').read()",
    "lambda x: x",
    "[i for i in range(10)]",
    "x.__class__",
])
def test_validate_safe_expr_rejects_dangerous(expr):
    with pytest.raises(ValueError):
        _validate_safe_expr(expr, source="t")


# ---------------------------------------------------------------------------
# Loader: kind=monitor skills do not need .md prompts
# ---------------------------------------------------------------------------


def test_monitor_skill_loads_without_md_files(tmp_path):
    d = tmp_path / "watcher"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d",
        "kind": "monitor",
        "schedule": "*/5 * * * *",
        "observe": ["error_rate"],
        "emit_signal_when": "observation['error_rate'] > 0.05",
        "trigger_target": "high_error_trigger",
    }))
    skill = load_skill(d)
    assert skill.kind == "monitor"
    assert skill.system_prompt == ""


def test_supervisor_skill_loads_without_md_files(tmp_path):
    d = tmp_path / "router"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d",
        "kind": "supervisor",
        "subordinates": ["a", "b"],
        "dispatch_strategy": "llm",
        "dispatch_prompt": "choose one",
    }))
    skill = load_skill(d)
    assert skill.kind == "supervisor"


def test_unknown_kind_rejected(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d", "kind": "router",  # unknown
    }))
    (d / "system.md").write_text("hi")
    with pytest.raises(ValidationError):
        load_skill(d)
