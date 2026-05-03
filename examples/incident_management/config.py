"""App-level config for the incident-management example.

The framework reads incident-flavored knobs (similarity threshold,
intervention confidence threshold, escalation roster, severity aliases,
dedup pipeline, environments) generically via ``FrameworkAppConfig`` and
``RuntimeConfig.{dedup,environments}_provider_path``. This module owns
the YAML <-> ``FrameworkAppConfig`` lift plus the provider hooks
referenced from ``config/code_review.runtime.yaml`` /
``config/config.yaml``; the runtime never imports ``IncidentAppConfig``
because that class no longer exists.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from runtime.config import FrameworkAppConfig
from runtime.dedup import DedupConfig


# The Stage-2 dedup system prompt for the incident-management app.
# Stays here (rather than on ``DedupConfig``) so that the framework
# default in ``FrameworkAppConfig.dedup_system_prompt`` can be a
# generic agent-orchestration prompt while the example app keeps its
# SRE-specific phrasing.
_INCIDENT_DEDUP_SYSTEM_PROMPT = (
    "You are deduplicating incident reports for an SRE platform. "
    "Two reports are duplicates only if they describe the same root cause "
    "AND the same service/environment AND overlap in time-of-occurrence. "
    "Surface-level keyword overlap is NOT enough. "
    "Respond with a single JSON object matching this schema: "
    '{"is_duplicate": bool, "confidence": float in [0,1], "rationale": '
    '"1-2 sentences"}.'
)


_INCIDENT_SEVERITY_ALIASES: dict[str, str] = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium",
    "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low",
    "informational": "low", "low": "low",
}


_INCIDENT_ESCALATION_TEAMS: list[str] = [
    "platform-oncall",
    "data-oncall",
    "security-oncall",
]


_INCIDENT_ENVIRONMENTS: list[str] = ["production", "staging", "dev", "local"]


_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "incident_management.yaml"
)


def _read_yaml(path: str | Path | None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _default_framework_cfg() -> FrameworkAppConfig:
    """Default ``FrameworkAppConfig`` populated with incident-tuned values."""
    return FrameworkAppConfig(
        confidence_threshold=0.75,
        similarity_threshold=0.2,
        escalation_teams=list(_INCIDENT_ESCALATION_TEAMS),
        severity_aliases=dict(_INCIDENT_SEVERITY_ALIASES),
        dedup_system_prompt=_INCIDENT_DEDUP_SYSTEM_PROMPT,
    )


def load_app_config(path: str | Path | None = None) -> FrameworkAppConfig:
    """Load the incident app YAML and return a ``FrameworkAppConfig``.

    The YAML may carry framework-flavored keys at the top level
    (``confidence_threshold``, ``similarity_threshold``,
    ``escalation_teams``, ``severity_aliases``, ``dedup_system_prompt``,
    ``ui``) and/or under a ``framework:`` block; both are folded into
    the returned ``FrameworkAppConfig``. Domain-specific top-level keys
    (``dedup``, ``environments``, ``store_path``, ``similarity_method``)
    are ignored here — they are surfaced via the dedicated provider
    hooks below. Falls back to ``_default_framework_cfg()`` when the
    file is missing.
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


# ---------------------------------------------------------------------------
# Provider hooks referenced from ``config/config.yaml``'s ``runtime:`` block.
# The runtime resolves these dotted paths into no-arg callables so it
# never imports ``examples.incident_management.config`` directly.
# ---------------------------------------------------------------------------


def framework_app_config_provider() -> FrameworkAppConfig:
    """Provider for ``RuntimeConfig.framework_app_config_path``."""
    return load_app_config()


def dedup_config_provider() -> DedupConfig | None:
    """Provider for ``RuntimeConfig.dedup_config_path``.

    Returns the configured dedup pipeline shape (or ``None`` when the
    bundled YAML disables it) so the runtime can build the pipeline
    without importing an app-specific config module.
    """
    raw = _read_yaml(None)
    block = raw.get("dedup")
    if block is None:
        return None
    return DedupConfig(**block)


def environments_provider() -> list[str]:
    """Provider for ``RuntimeConfig.environments_provider_path``."""
    raw = _read_yaml(None)
    envs = raw.get("environments")
    if envs is None:
        return list(_INCIDENT_ENVIRONMENTS)
    return list(envs)
