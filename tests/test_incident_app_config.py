"""Smoke tests for the incident-management app's config loader.

The incident-flavored knobs (similarity threshold, intervention
confidence threshold, escalation roster, severity aliases) live on a
``FrameworkAppConfig`` populated by
``examples.incident_management.config.load_app_config``. Domain-only
knobs (``environments``, ``dedup``) are exposed via dedicated provider
hooks (``environments_provider``, ``dedup_config_provider``).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def test_incident_config_loader_importable():
    from examples.incident_management.config import (  # noqa: F401
        load_app_config,
        framework_app_config_provider,
        dedup_config_provider,
        environments_provider,
    )


def test_load_app_config_default_values():
    from examples.incident_management.config import load_app_config

    cfg = load_app_config()
    assert cfg.similarity_threshold == 0.2
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
    assert "data-oncall" in cfg.escalation_teams
    assert "security-oncall" in cfg.escalation_teams
    # Severity aliases collapse domain-specific labels onto canonical levels.
    assert cfg.severity_aliases["sev1"] == "high"
    assert cfg.severity_aliases["p3"] == "medium"
    assert cfg.severity_aliases["info"] == "low"


def test_environments_provider_default():
    from examples.incident_management.config import environments_provider

    envs = environments_provider()
    assert "production" in envs
    assert "staging" in envs


def test_load_from_yaml_file(tmp_path: Path):
    from examples.incident_management.config import load_app_config

    payload = {
        "similarity_threshold": 0.5,
        "confidence_threshold": 0.9,
        "escalation_teams": ["alpha-team", "beta-team"],
        "severity_aliases": {"crit": "high"},
    }
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(payload))

    cfg = load_app_config(path)
    assert cfg.similarity_threshold == 0.5
    assert cfg.confidence_threshold == 0.9
    assert cfg.escalation_teams == ["alpha-team", "beta-team"]
    assert cfg.severity_aliases == {"crit": "high"}


def test_load_default_yaml_returns_populated_config():
    """The bundled config/incident_management.yaml must populate the
    loaded ``FrameworkAppConfig`` with the operative defaults — this is
    what keeps the framework readers deterministic when the example app
    is wired up."""
    from examples.incident_management.config import load_app_config

    cfg = load_app_config()
    assert cfg.similarity_threshold == 0.2
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
