"""Tests for examples/code_review scaffold.

The typed ``CodeReviewState`` / ``PullRequest`` pydantic shape is gone:
code-review session data now rides through ``Session.extra_fields`` like
every other app. What remains here is the framework-shaped config
loader contract and the skills-directory layout — both survive the
typed-state-class deletion.
"""


def test_code_review_app_config_defaults():
    """The bundled code-review YAML carries framework knobs under the
    ``framework:`` block which AppConfig binds directly.

    Domain-only knobs (``severity_categories``, ``auto_request_changes_on``,
    ``repos_in_scope``, ``review_max_diff_kb``, ``similarity_method``) are
    not modelled on AppConfig — they live as raw YAML the
    example-internal mcp_server / skills read off the file directly.
    """
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("config/code_review.yaml").read_text())
    framework = raw.get("framework") or {}
    # The framework knob round-trips through the YAML's ``framework:`` block.
    assert framework.get("similarity_threshold") == 0.3


def test_code_review_skills_dir_exists():
    from pathlib import Path
    skills = Path("examples/code_review/skills")
    assert skills.is_dir()
    assert (skills / "_common").is_dir()
    assert (skills / "intake").is_dir()
    assert (skills / "analyzer").is_dir()
    assert (skills / "recommender").is_dir()
