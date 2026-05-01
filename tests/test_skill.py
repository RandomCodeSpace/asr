import pytest
import yaml
from pydantic import ValidationError
from orchestrator.skill import Skill, RouteRule, load_skill, load_all_skills


_SAMPLE_DICT = {
    "name": "intake",
    "description": "First-line agent",
    "model": "llama3.1:70b",
    "temperature": 0.2,
    "tools": ["lookup_similar_incidents", "create_incident", "get_user_context"],
    "routes": [
        {"when": "matched_known_issue", "next": "resolution"},
        {"when": "default", "next": "triage"},
    ],
    "system_prompt": "You are the **Intake** agent.\n\nDo the things.",
}


def test_load_skill_validates_dict():
    skill = load_skill(_SAMPLE_DICT)
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
    assert "Intake" in skill.system_prompt


def test_load_all_skills_indexes_by_name(tmp_path):
    f = tmp_path / "skills.yaml"
    f.write_text(yaml.safe_dump({"skills": [
        _SAMPLE_DICT,
        {**_SAMPLE_DICT, "name": "triage", "description": "Triage agent"},
    ]}))
    skills = load_all_skills(f)
    assert set(skills) == {"intake", "triage"}
    assert isinstance(skills["intake"], Skill)


def test_load_all_skills_rejects_duplicate_names(tmp_path):
    f = tmp_path / "skills.yaml"
    f.write_text(yaml.safe_dump({"skills": [_SAMPLE_DICT, _SAMPLE_DICT]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_all_skills(f)


def test_load_all_skills_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_skills(tmp_path / "does_not_exist.yaml")


def test_missing_required_field_raises():
    bad = {k: v for k, v in _SAMPLE_DICT.items() if k != "name"}
    with pytest.raises(ValidationError):
        load_skill(bad)
