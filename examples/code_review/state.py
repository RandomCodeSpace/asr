"""Code-review state-overrides schema (DECOUPLE-05 / D-08-01).

Registered via ``OrchestratorConfig.state_overrides_schema`` in
``config/code_review.runtime.yaml``. Fields mirror the contract of
``examples/code_review/mcp_server.py:_tool_fetch_pr_diff(repo, number)``
and the legacy ``PullRequest`` shape stored in
``Session.extra_fields["pr"]``.
"""
from __future__ import annotations

from pydantic import BaseModel


class CodeReviewStateOverrides(BaseModel):
    """Per-session overrides for code_review.

    All fields are Optional so callers can stamp only what they have
    at session-start (e.g. a webhook trigger may know the PR URL but
    not the base branch). ``extra='forbid'`` catches typos.
    """

    model_config = {"extra": "forbid"}

    pr_url: str | None = None
    repo: str | None = None
    base_branch: str | None = None
    pr_number: int | None = None
