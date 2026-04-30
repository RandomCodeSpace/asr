from pathlib import Path
import pytest
from orchestrator.skill import Skill, RouteRule, load_skill, load_all_skills

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "skills"


def test_parse_single_skill():
    skill = load_skill(FIXTURE_DIR / "sample.md")
    assert skill.name == "intake"
    assert skill.model == "llama3.1:70b"
    assert skill.temperature == 0.2
    assert skill.tools == ["lookup_similar_incidents", "create_incident", "get_user_context"]
    assert skill.routes == [
        RouteRule(when="matched_known_issue", next="resolution"),
        RouteRule(when="default", next="triage"),
    ]
    assert "Intake agent" in skill.system_prompt


def test_load_all_skills_indexes_by_name():
    skills = load_all_skills(FIXTURE_DIR)
    assert "intake" in skills
    assert isinstance(skills["intake"], Skill)


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("---\ndescription: no name\n---\nbody")
    with pytest.raises(ValueError, match="name"):
        load_skill(bad)
