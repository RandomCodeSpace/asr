"""Tests for examples/code_review scaffold.

The typed ``CodeReviewState`` / ``PullRequest`` pydantic shape is gone:
code-review session data now rides through ``Session.extra_fields`` like
every other app. What remains here is the framework-shaped config
loader contract and the skills-directory layout — both survive the
typed-state-class deletion.
"""


def test_code_review_app_config_defaults():
    """The code-review YAML loader returns a framework-shaped config.

    Domain-only knobs (``severity_categories``, ``auto_request_changes_on``,
    ``repos_in_scope``, ``review_max_diff_kb``, ``similarity_method``) are
    no longer mirrored on a typed BaseModel — they live in the YAML and
    the example-internal mcp_server / skills read them off the file.
    """
    from examples.code_review.config import load_app_config
    cfg = load_app_config()
    assert cfg.similarity_threshold == 0.3


def test_code_review_skills_dir_exists():
    from pathlib import Path
    skills = Path("examples/code_review/skills")
    assert skills.is_dir()
    assert (skills / "_common").is_dir()
    assert (skills / "intake").is_dir()
    assert (skills / "analyzer").is_dir()
    assert (skills / "recommender").is_dir()
