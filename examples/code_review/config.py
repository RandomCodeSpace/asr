"""Code-review app config.

Mirrors the incident-management module: provides the YAML
<-> ``FrameworkAppConfig`` lift plus provider hooks the runtime
resolves via ``RuntimeConfig.framework_app_config_path``. The runtime
never imports this module directly.
"""
from __future__ import annotations
from pathlib import Path

import yaml

from runtime.config import FrameworkAppConfig


_CODE_REVIEW_DEDUP_SYSTEM_PROMPT = (
    "You are deduplicating pull-request reviews. Two reviews are "
    "duplicates only if they cover the same repository, the same "
    "branch/commit range, and surface overlapping findings. Return a "
    "single JSON object: {\"is_duplicate\": bool, \"confidence\": "
    "float in [0,1], \"rationale\": \"1-2 sentences\"}."
)


_CODE_REVIEW_SEVERITY_ALIASES: dict[str, str] = {
    "info": "info",
    "warning": "warning",
    "warn": "warning",
    "error": "error",
    "err": "error",
    "critical": "critical",
    "crit": "critical",
    "blocker": "critical",
}


_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "code_review.yaml"
)


def _read_yaml(path: str | Path | None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _default_framework_cfg() -> FrameworkAppConfig:
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
        severity_aliases=dict(_CODE_REVIEW_SEVERITY_ALIASES),
        dedup_system_prompt=_CODE_REVIEW_DEDUP_SYSTEM_PROMPT,
    )


def load_app_config(path: str | Path | None = None) -> FrameworkAppConfig:
    """Load the code-review YAML and return a ``FrameworkAppConfig``.

    Same lift semantics as the incident-management loader: top-level
    framework-flavored keys (and an optional ``framework:`` block, and
    a ``ui:`` block) merge into the returned ``FrameworkAppConfig``.
    Domain-only knobs (``severity_categories``, ``auto_request_changes_on``,
    ``repos_in_scope``, ``review_max_diff_kb``, ``similarity_method``)
    are intentionally not surfaced through this loader; they are
    example-internal and read directly off the YAML by the code-review
    MCP server / skills if needed.
    """
    raw = _read_yaml(path)
    if not raw:
        return _default_framework_cfg()

    fw_kwargs: dict[str, object] = {}
    if "framework" in raw:
        fw_kwargs.update(raw.get("framework") or {})
    for legacy_key in (
        "confidence_threshold",
        "similarity_threshold",
        "escalation_teams",
        "severity_aliases",
        "dedup_system_prompt",
        "ui",
    ):
        if legacy_key in raw:
            fw_kwargs[legacy_key] = raw[legacy_key]

    defaults = _default_framework_cfg()
    if not fw_kwargs:
        return defaults
    merged = defaults.model_dump()
    merged.update(fw_kwargs)
    return FrameworkAppConfig(**merged)


def framework_app_config_provider() -> FrameworkAppConfig:
    """Provider for ``RuntimeConfig.framework_app_config_path``."""
    return load_app_config()
