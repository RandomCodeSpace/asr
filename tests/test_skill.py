import pytest
import yaml
from orchestrator.skill import Skill, RouteRule, load_skill, load_all_skills


_BASE_CONFIG = {
    # NOTE: no "name" — agent identity is the directory name (single source of truth).
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
    """Drop a {config.yaml + N .md} pair into base/<name>/."""
    d = base / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(config or _BASE_CONFIG))
    for filename, body in (prompts or {"system.md": "You are the agent.\nBe terse."}).items():
        (d / filename).write_text(body)
    return d


def test_load_skill_uses_directory_name(tmp_path):
    d = _write_agent(tmp_path, "intake")
    skill = load_skill(d)
    assert isinstance(skill, Skill)
    assert skill.name == "intake"
    assert skill.model == "llama3.1:70b"
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
    _write_agent(tmp_path, "triage", config={**_BASE_CONFIG, "description": "t"})
    skills = load_all_skills(tmp_path)
    assert set(skills) == {"intake", "triage"}
    assert isinstance(skills["intake"], Skill)


def test_load_all_skills_skips_dirs_without_config_yaml(tmp_path):
    _write_agent(tmp_path, "intake")
    (tmp_path / "_notes").mkdir()
    (tmp_path / "_notes" / "draft.md").write_text("ignore me")
    skills = load_all_skills(tmp_path)
    assert set(skills) == {"intake"}


def test_common_prompt_is_appended_to_every_agent(tmp_path):
    _write_agent(tmp_path, "intake")
    _write_agent(tmp_path, "triage", config={**_BASE_CONFIG, "description": "t"})
    common = tmp_path / "_common"
    common.mkdir()
    (common / "confidence.md").write_text("## Confidence\nbe calibrated")
    (common / "output.md").write_text("## Output\nbe terse")
    skills = load_all_skills(tmp_path)
    for name in ("intake", "triage"):
        sp = skills[name].system_prompt
        assert sp.startswith("You are the agent."), sp
        assert sp.endswith("## Confidence\nbe calibrated\n\n## Output\nbe terse"), sp


def test_underscore_prefixed_dirs_are_never_agents(tmp_path):
    _write_agent(tmp_path, "intake")
    # _-prefixed dirs are reserved for shared content; never an agent even with
    # a config.yaml present.
    _write_agent(tmp_path, "_drafts")
    skills = load_all_skills(tmp_path)
    assert set(skills) == {"intake"}


def test_no_common_dir_leaves_prompt_untouched(tmp_path):
    _write_agent(tmp_path, "intake")
    skills = load_all_skills(tmp_path)
    assert skills["intake"].system_prompt == "You are the agent.\nBe terse."


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


def test_load_all_skills_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_skills(tmp_path / "nope")


def test_config_yaml_with_name_key_raises(tmp_path):
    """config.yaml must NOT declare 'name' — that's a redundancy that drifts
    when a directory is renamed. Loader rejects it explicitly."""
    d = tmp_path / "intake"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({**_BASE_CONFIG, "name": "intake"}))
    (d / "system.md").write_text("body")
    with pytest.raises(ValueError, match="must not declare 'name'"):
        load_skill(d)


@pytest.mark.parametrize("bad_name", [
    "Intake",            # uppercase
    "deep-investigator", # hyphen
    "1agent",            # leading digit
    "_hidden",           # leading underscore reserved for shared content
    "agent name",        # space
    "agent.name",        # dot
    "x" * 65,            # over 64 chars
])
def test_load_skill_rejects_invalid_directory_name(tmp_path, bad_name):
    d = tmp_path / bad_name
    try:
        d.mkdir()
    except OSError:
        pytest.skip(f"filesystem rejects {bad_name!r}")
    (d / "config.yaml").write_text(yaml.safe_dump(_BASE_CONFIG))
    (d / "system.md").write_text("body")
    with pytest.raises(ValueError, match="invalid agent name"):
        load_skill(d)
