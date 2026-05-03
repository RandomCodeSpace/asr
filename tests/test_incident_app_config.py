"""Smoke tests for the IncidentAppConfig moved out of runtime.config.

P1-E moves the incident-flavored knobs (similarity threshold, intervention
confidence threshold, escalation roster, environments, severity aliases)
out of ``runtime.config.AppConfig`` into ``examples.incident_management.config``.
These tests pin the new surface.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def test_incident_app_config_importable():
    from examples.incident_management.config import (  # noqa: F401
        IncidentAppConfig,
        load_incident_app_config,
    )


def test_incident_app_config_default_values():
    from examples.incident_management.config import IncidentAppConfig

    cfg = IncidentAppConfig()
    assert cfg.similarity_threshold == 0.2
    assert cfg.similarity_method == "keyword"
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
    assert "data-oncall" in cfg.escalation_teams
    assert "security-oncall" in cfg.escalation_teams
    assert "production" in cfg.environments
    assert "staging" in cfg.environments
    # Severity aliases collapse domain-specific labels onto canonical levels.
    assert cfg.severity_aliases["sev1"] == "high"
    assert cfg.severity_aliases["p3"] == "medium"
    assert cfg.severity_aliases["info"] == "low"


def test_load_from_yaml_file(tmp_path: Path):
    from examples.incident_management.config import load_incident_app_config

    payload = {
        "similarity_threshold": 0.5,
        "confidence_threshold": 0.9,
        "escalation_teams": ["alpha-team", "beta-team"],
        "environments": ["prod"],
        "severity_aliases": {"crit": "high"},
    }
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(payload))

    cfg = load_incident_app_config(path)
    assert cfg.similarity_threshold == 0.5
    assert cfg.confidence_threshold == 0.9
    assert cfg.escalation_teams == ["alpha-team", "beta-team"]
    assert cfg.environments == ["prod"]
    assert cfg.severity_aliases == {"crit": "high"}


def test_load_default_yaml_returns_populated_config():
    """The example app's bundled config.yaml must populate IncidentAppConfig
    with the operative defaults — this is what keeps the framework readers
    deterministic when the example app is wired up."""
    from examples.incident_management.config import load_incident_app_config

    cfg = load_incident_app_config()
    assert cfg.similarity_threshold == 0.2
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
    assert "production" in cfg.environments
