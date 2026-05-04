"""Golden-prompt assertions: the skill markdown must reference the
typed terminal tools, not the legacy update_incident({"status":...})
path. This catches future prompt drift that re-introduces the bug
class we just remediated."""
from pathlib import Path


SKILLS = Path("examples/incident_management/skills")


def test_resolution_prompt_calls_mark_resolved_or_escalated():
    text = (SKILLS / "resolution" / "system.md").read_text()
    assert "mark_resolved" in text
    assert "mark_escalated" in text
    assert 'status": "resolved"' not in text  # no legacy guidance
    assert 'status": "escalated"' not in text


def test_deep_investigator_prompt_calls_submit_hypothesis():
    text = (SKILLS / "deep_investigator" / "system.md").read_text()
    assert "submit_hypothesis" in text


def test_resolution_yaml_lists_typed_terminal_tools():
    yaml_text = (SKILLS / "resolution" / "config.yaml").read_text()
    assert "mark_resolved" in yaml_text
    assert "mark_escalated" in yaml_text


def test_deep_investigator_yaml_lists_submit_hypothesis():
    yaml_text = (SKILLS / "deep_investigator" / "config.yaml").read_text()
    assert "submit_hypothesis" in yaml_text


def test_common_confidence_md_removed():
    assert not (SKILLS / "_common" / "confidence.md").exists()
