"""Tests for examples/code_review scaffold."""


def test_code_review_state_importable():
    from examples.code_review.state import CodeReviewState  # noqa: F401


def test_code_review_state_inherits_session():
    from runtime.state import Session
    from examples.code_review.state import CodeReviewState
    assert issubclass(CodeReviewState, Session)


def test_code_review_state_has_domain_fields():
    from examples.code_review.state import CodeReviewState, PullRequest
    s = CodeReviewState(
        id="CR-2026-001", status="new",
        created_at="2026-05-03T00:00:00Z", updated_at="2026-05-03T00:00:00Z",
        pr=PullRequest(repo="org/repo", number=42, title="Fix",
                       author="alice", base_sha="abc", head_sha="def"),
    )
    assert s.pr.number == 42
    assert s.review_findings == []
    assert s.overall_recommendation is None


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


def test_code_review_uses_runtime_state_no_runtime_import():
    """Verify the example only imports runtime.state, not runtime.incident or other domain types."""
    state_text = open("examples/code_review/state.py").read()
    assert "from runtime.state import" in state_text
    assert "incident_management" not in state_text
    assert "from runtime.incident" not in state_text
