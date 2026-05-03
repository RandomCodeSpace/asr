"""Skill.runner default fallback for kind=supervisor."""
from __future__ import annotations

import pytest

from runtime.skill import Skill


def test_supervisor_skill_without_runner_falls_back_to_framework_default() -> None:
    s = Skill.model_validate({
        "name": "intake",
        "description": "test",
        "kind": "supervisor",
        "subordinates": ["triage"],
        "dispatch_strategy": "rule",
        "dispatch_rules": [{"when": "True", "target": "triage"}],
    })
    assert s.runner == "runtime.intake:default_intake_runner"


def test_supervisor_skill_with_explicit_runner_keeps_it() -> None:
    s = Skill.model_validate({
        "name": "intake",
        "description": "test",
        "kind": "supervisor",
        "subordinates": ["triage"],
        "dispatch_strategy": "rule",
        "dispatch_rules": [{"when": "True", "target": "triage"}],
        "runner": "runtime.intake:default_intake_runner",
    })
    assert s.runner == "runtime.intake:default_intake_runner"


def test_responsive_skill_runner_field_still_forbidden() -> None:
    """The runner field is supervisor-only (P9-9h). Responsive skills
    must not default — they must reject the field outright."""
    with pytest.raises(Exception):
        Skill.model_validate({
            "name": "broken",
            "description": "test",
            "kind": "responsive",
            "tools": {"local": ["t1"]},
            "runner": "runtime.intake:default_intake_runner",
        })
