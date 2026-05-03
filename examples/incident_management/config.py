"""App-level config for the incident-management example.

Holds the incident-flavored knobs (similarity threshold, intervention
confidence threshold, escalation roster, environments, severity aliases)
that used to live in the framework's ``runtime.config.AppConfig`` before
P1-E.

The framework reads these via ``load_incident_app_config()``; tests that
need to override values can either write a temporary YAML file and pass
its path, or monkey-patch the loader to return a stubbed config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

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


def _default_incident_framework_cfg() -> FrameworkAppConfig:
    """Default ``FrameworkAppConfig`` populated with incident-tuned values."""
    return FrameworkAppConfig(
        confidence_threshold=0.75,
        similarity_threshold=0.2,
        escalation_teams=list(_INCIDENT_ESCALATION_TEAMS),
        severity_aliases=dict(_INCIDENT_SEVERITY_ALIASES),
        dedup_system_prompt=_INCIDENT_DEDUP_SYSTEM_PROMPT,
    )


class IncidentAppConfig(BaseModel):
    """Domain-level configuration for the incident-management app.

    Defaults are tuned to match the values currently shipped in
    ``config/config.yaml`` (notably ``similarity_threshold=0.2``, which
    is the operative default — the old framework default of 0.85 was
    masked by the YAML override and was never actually used).

    The cross-cutting fields the framework reads
    (``confidence_threshold``, ``similarity_threshold``,
    ``escalation_teams``, ``severity_aliases``,
    ``dedup_system_prompt``) live on a composed
    :class:`runtime.config.FrameworkAppConfig` accessible via
    ``IncidentAppConfig().framework``. The flat top-level attributes
    are kept as read-through proxies for back-compat with existing
    callers.
    """

    framework: FrameworkAppConfig = Field(
        default_factory=_default_incident_framework_cfg,
    )

    store_path: str = "incidents"
    similarity_method: Literal["keyword", "embedding"] = "keyword"
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"],
    )
    # P7-C: optional two-stage dedup pipeline. Off-by-default at the
    # framework level via ``DedupConfig.enabled=False``; the bundled
    # YAML for the incident-management example opts in. ``None`` means
    # "framework default" (off).
    dedup: DedupConfig | None = None

    # ------------------------------------------------------------------
    # Read-through proxies for the framework-level fields. Existing
    # callers (UI, MCP server, tests) read these as flat attributes;
    # the runtime now reads them off ``self.framework`` directly.
    # ------------------------------------------------------------------
    @property
    def similarity_threshold(self) -> float:
        return self.framework.similarity_threshold

    @property
    def confidence_threshold(self) -> float:
        return self.framework.confidence_threshold

    @property
    def escalation_teams(self) -> list[str]:
        return self.framework.escalation_teams

    @property
    def severity_aliases(self) -> dict[str, str]:
        return self.framework.severity_aliases


_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_incident_app_config(
    path: str | Path | None = None,
) -> IncidentAppConfig:
    """Load the incident app config from ``path`` (or the example default).

    Falls back to a fully-defaulted ``IncidentAppConfig`` when the file is
    missing — this keeps unit tests deterministic without forcing every
    ``AppConfig(...)``-constructed test to also create an app config file.

    Top-level YAML keys that match the legacy flat shape
    (``confidence_threshold``, ``similarity_threshold``,
    ``escalation_teams``, ``severity_aliases``) are folded into the
    composed :class:`FrameworkAppConfig` automatically so existing
    config.yaml files keep loading.
    """
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return IncidentAppConfig()
    raw = yaml.safe_load(p.read_text()) or {}

    # Back-compat: lift flat top-level keys that now live on the
    # composed FrameworkAppConfig.
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
        # Start from the incident defaults, then overlay user values.
        defaults = _default_incident_framework_cfg()
        merged = defaults.model_dump()
        merged.update(fw_kwargs)
        raw["framework"] = FrameworkAppConfig(**merged)
    return IncidentAppConfig(**raw)


def framework_app_config_provider() -> FrameworkAppConfig:
    """Provider hook referenced from ``RuntimeConfig.framework_app_config_path``.

    Returns the same :class:`FrameworkAppConfig` view the loaded
    :class:`IncidentAppConfig` exposes via ``.framework`` — the runtime
    reads this directly without ever importing ``IncidentAppConfig``.
    """
    return load_incident_app_config().framework


def dedup_config_provider() -> DedupConfig | None:
    """Provider hook referenced from ``RuntimeConfig.dedup_config_path``.

    Returns the configured dedup pipeline shape (or ``None`` when the
    bundled config disables it) so the runtime can build the pipeline
    without importing ``IncidentAppConfig``.
    """
    return load_incident_app_config().dedup


def environments_provider() -> list[str]:
    """Provider hook for ``RuntimeConfig.environments_provider_path``.

    Returns the incident-management environments roster the
    ``GET /environments`` endpoint surfaces to UI clients.
    """
    return list(load_incident_app_config().environments)
