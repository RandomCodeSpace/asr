import pytest
import yaml
from orchestrator.skill import Skill, RouteRule, load_skill, load_all_skills


_BASE_CONFIG = {
    "name": "intake",
    "description": "First-line agent",
    "model": "llama3.1:70b",
    "temperature": 0.2,
    "tools": ["lookup_similar_incidents", "create_incident", "get_user_context"],
    "routes": [
        {"when": "matched_known_issue", "next": "resolution"},
        {"when": "default", "next": "triage"},
    ],
}


def _write_agent(base, name, *, config=None, prompts=None):
    """Helper: drop a {config.yaml + N .md} pair into base/<name>/."""
    d = base / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(config or {**_BASE_CONFIG, "name": name}))
    for filename, body in (prompts or {"system.md": "You are the agent.\nBe terse."}).items():
        (d / filename).write_text(body)
    return d


def test_load_skill_reads_dir(tmp_path):
    d = _write_agent(tmp_path, "intake")
    skill = load_skill(d)
    assert isinstance(skill, Skill)
    assert skill.name == "intake"
    assert skill.model == "llama3.1:70b"
    assert skill.temperature == 0.2
    assert skill.tools == [
        "lookup_similar_incidents", "create_incident", "get_user_context",
    ]
    assert skill.routes == [
        RouteRule(when="matched_known_issue", next="resolution"),
        RouteRule(when="default", next="triage"),
    ]
    assert "agent" in skill.system_prompt


def test_load_skill_concatenates_multiple_md_in_alphabetical_order(tmp_path):
    d = _write_agent(tmp_path, "intake", prompts={
        "01_system.md": "System: be helpful.",
        "02_confidence.md": "Always emit confidence.",
        "03_output.md": "Reply tersely.",
    })
    skill = load_skill(d)
    assert skill.system_prompt == "System: be helpful.\n\nAlways emit confidence.\n\nReply tersely."


def test_load_all_skills_indexes_by_name(tmp_path):
    _write_agent(tmp_path, "intake")
    _write_agent(tmp_path, "triage", config={**_BASE_CONFIG, "name": "triage", "description": "t"})
    skills = load_all_skills(tmp_path)
    assert set(skills) == {"intake", "triage"}
    assert isinstance(skills["intake"], Skill)


def test_load_all_skills_skips_dirs_without_config_yaml(tmp_path):
    _write_agent(tmp_path, "intake")
    (tmp_path / "_notes").mkdir()
    (tmp_path / "_notes" / "draft.md").write_text("ignore me")
    skills = load_all_skills(tmp_path)
    assert set(skills) == {"intake"}


def test_load_skill_missing_config_raises(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "system.md").write_text("body")
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_skill(d)


def test_load_skill_missing_md_raises(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump(_BASE_CONFIG))
    with pytest.raises(FileNotFoundError, match=".md"):
        load_skill(d)


def test_load_all_skills_rejects_duplicate_names(tmp_path):
    _write_agent(tmp_path, "intake_a", config={**_BASE_CONFIG, "name": "intake"})
    _write_agent(tmp_path, "intake_b", config={**_BASE_CONFIG, "name": "intake"})
    with pytest.raises(ValueError, match="Duplicate"):
        load_all_skills(tmp_path)


def test_load_all_skills_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_skills(tmp_path / "nope")
