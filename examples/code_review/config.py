"""Code-review app config."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
import yaml

from runtime.config import FrameworkAppConfig


_CODE_REVIEW_DEDUP_SYSTEM_PROMPT = (
    "You are deduplicating pull-request reviews. Two reviews are "
    "duplicates only if they cover the same repository, the same "
    "branch/commit range, and surface overlapping findings. Return a "
    "single JSON object: {\"is_duplicate\": bool, \"confidence\": "
    "float in [0,1], \"rationale\": \"1-2 sentences\"}."
)


def _default_code_review_framework_cfg() -> FrameworkAppConfig:
    """Default ``FrameworkAppConfig`` for the code-review app.

    Reviewers don't have a paging roster the way the incident-management
    desk does; ``escalation_teams`` is left empty and apps that wire up
    a code-owners rotation can override via YAML. Severity vocabulary
    is review-specific (info/warning/error/critical), and the dedup
    prompt is PR-shaped.
    """
    return FrameworkAppConfig(
        confidence_threshold=0.7,
        similarity_threshold=0.3,
        escalation_teams=[],
        severity_aliases={
            "info": "info",
            "warning": "warning",
            "warn": "warning",
            "error": "error",
            "err": "error",
            "critical": "critical",
            "crit": "critical",
            "blocker": "critical",
        },
        dedup_system_prompt=_CODE_REVIEW_DEDUP_SYSTEM_PROMPT,
    )


class CodeReviewAppConfig(BaseModel):
    """All code-review app keys."""
    framework: FrameworkAppConfig = Field(
        default_factory=_default_code_review_framework_cfg,
    )

    similarity_method: Literal["keyword", "embedding"] = "keyword"
    severity_categories: list[str] = Field(
        default_factory=lambda: ["info", "warning", "error", "critical"]
    )
    auto_request_changes_on: list[str] = Field(
        default_factory=lambda: ["critical", "error"]
    )
    repos_in_scope: list[str] = Field(default_factory=list)
    review_max_diff_kb: int = 500

    @property
    def similarity_threshold(self) -> float:
        return self.framework.similarity_threshold


_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_code_review_app_config(path: str | Path | None = None) -> CodeReviewAppConfig:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return CodeReviewAppConfig()
    raw = yaml.safe_load(p.read_text()) or {}
    # Back-compat: lift flat top-level framework keys
    # (similarity_threshold, confidence_threshold, escalation_teams,
    # severity_aliases, dedup_system_prompt) into the composed
    # FrameworkAppConfig.
    fw_kwargs: dict[str, object] = {}
    if "framework" in raw:
        fw_block = raw.pop("framework") or {}
        fw_kwargs.update(fw_block)
    for legacy_key in (
        "confidence_threshold",
        "similarity_threshold",
        "escalation_teams",
        "severity_aliases",
        "dedup_system_prompt",
    ):
        if legacy_key in raw:
            fw_kwargs[legacy_key] = raw.pop(legacy_key)
    if fw_kwargs:
        defaults = _default_code_review_framework_cfg()
        merged = defaults.model_dump()
        merged.update(fw_kwargs)
        raw["framework"] = FrameworkAppConfig(**merged)
    return CodeReviewAppConfig(**raw)


def framework_app_config_provider() -> FrameworkAppConfig:
    """Provider hook referenced from ``RuntimeConfig.framework_app_config_path``.

    Returns the framework view embedded in ``CodeReviewAppConfig`` so
    the runtime reads code-review-tuned values without importing
    ``CodeReviewAppConfig``.
    """
    return load_code_review_app_config().framework
