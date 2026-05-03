"""Code-review domain state."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from runtime.state import Session


CodeReviewStatus = Literal["new", "fetching", "analyzing", "awaiting_decision",
                            "approved", "rejected", "merged", "closed", "deleted"]


class PullRequest(BaseModel):
    repo: str   # e.g. "org/repo"
    number: int
    title: str
    author: str
    base_sha: str
    head_sha: str
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0


class ReviewFinding(BaseModel):
    severity: Literal["info", "warning", "error", "critical"]
    file: str
    line: int | None = None
    category: str
    message: str
    suggestion: str | None = None


class CodeReviewState(Session):
    """Code-review-specific session fields."""
    pr: PullRequest
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    overall_recommendation: Literal["approve", "request_changes", "comment"] | None = None
    review_summary: str = ""
    review_token_budget: int = 0

    # PR-shaped agent-input preamble so reviewer agents see the diff
    # context (repo, number, title, author, change-volume) without the
    # framework needing to know the code-review schema.
    def to_agent_input(self) -> str:
        pr = self.pr
        base = (
            f"Pull Request {self.id}\n"
            f"Repo: {pr.repo}\n"
            f"PR #{pr.number}: {pr.title}\n"
            f"Author: {pr.author}\n"
            f"Base SHA: {pr.base_sha}\n"
            f"Head SHA: {pr.head_sha}\n"
            f"Diff: +{pr.additions}/-{pr.deletions} across "
            f"{pr.files_changed} file(s)\n"
            f"Status: {self.status}\n"
        )
        for agent_key, finding in self.findings.items():
            base += f"Findings ({agent_key}): {finding}\n"
        if self.user_inputs:
            bullets = "\n".join(f"- {ui}" for ui in self.user_inputs)
            base += (
                "\nUser-provided context (appended via intervention):\n"
                f"{bullets}\n"
            )
        return base

    # P8-C: code-review has its own id namespace so two apps running
    # against the same metadata DB cannot collide on session ids.
    # ``CR-YYYYMMDD-NNN`` mirrors the incident shape but keeps the
    # prefix distinct.
    @classmethod
    def id_format(cls, *, seq: int) -> str:
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"CR-{today}-{seq:03d}"
