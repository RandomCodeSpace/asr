"""Tests for ``runtime.config.FrameworkAppConfig`` (post-merge refactor).

Pins the generic, framework-level cross-cutting config shape that
replaces the four ``examples.incident_management.config`` imports the
runtime used to make. Apps compose this inside their own AppConfig and
expose a no-arg provider via ``RuntimeConfig.framework_app_config_path``.
"""
from __future__ import annotations

import pytest

from runtime.config import (
    FrameworkAppConfig,
    RuntimeConfig,
    resolve_framework_app_config,
)


# ---------------------------------------------------------------------------
# Default values — bare ``FrameworkAppConfig()`` is the framework fallback.
# ---------------------------------------------------------------------------


def test_default_values_are_framework_neutral():
    cfg = FrameworkAppConfig()
    assert cfg.confidence_threshold == 0.75
    assert cfg.similarity_threshold == 0.2
    assert cfg.escalation_teams == []
    assert cfg.severity_aliases == {}
    # Default dedup prompt is generic — no incident-management leakage.
    assert "incident" not in cfg.dedup_system_prompt.lower()
    assert "agent-orchestration" in cfg.dedup_system_prompt.lower()
    assert "is_duplicate" in cfg.dedup_system_prompt


def test_runtime_config_carries_optional_provider_path():
    rc = RuntimeConfig()
    assert rc.framework_app_config_path is None
    rc2 = RuntimeConfig(framework_app_config_path="some.module:provider")
    assert rc2.framework_app_config_path == "some.module:provider"


# ---------------------------------------------------------------------------
# Provider resolution — dotted ``module:callable`` resolves to a
# FrameworkAppConfig instance.
# ---------------------------------------------------------------------------


def test_resolver_returns_default_when_path_is_none():
    cfg = resolve_framework_app_config(None)
    assert isinstance(cfg, FrameworkAppConfig)
    # Same as a bare FrameworkAppConfig() — sanity check.
    assert cfg.confidence_threshold == 0.75


def test_resolver_rejects_malformed_path():
    with pytest.raises(ValueError, match="must be in 'module.path:callable' form"):
        resolve_framework_app_config("not_a_dotted_path")


def test_resolver_resolves_incident_provider():
    """The incident-management example must expose a working provider."""
    cfg = resolve_framework_app_config(
        "examples.incident_management.config:framework_app_config_provider"
    )
    assert isinstance(cfg, FrameworkAppConfig)
    assert cfg.confidence_threshold == 0.75
    assert "platform-oncall" in cfg.escalation_teams
    assert cfg.severity_aliases["sev1"] == "high"


def test_resolver_resolves_code_review_provider():
    """Code-review example must expose a working provider — different from incident."""
    cfg = resolve_framework_app_config(
        "examples.code_review.config:framework_app_config_provider"
    )
    assert isinstance(cfg, FrameworkAppConfig)
    # code-review has its own roster — must not be incident's roster.
    assert "platform-oncall" not in cfg.escalation_teams


def test_resolver_rejects_provider_returning_wrong_type(tmp_path, monkeypatch):
    """A provider that returns the wrong type fails loud."""
    # Build an ad-hoc module under a temp path to act as a bad provider.
    import sys
    import types

    mod = types.ModuleType("_test_bad_provider_mod")

    def _bad_provider():
        return "not a FrameworkAppConfig"

    mod.bad = _bad_provider
    sys.modules["_test_bad_provider_mod"] = mod
    try:
        with pytest.raises(TypeError, match="expected FrameworkAppConfig"):
            resolve_framework_app_config("_test_bad_provider_mod:bad")
    finally:
        sys.modules.pop("_test_bad_provider_mod", None)


# ---------------------------------------------------------------------------
# Composition — apps wrap FrameworkAppConfig inside their own config.
# ---------------------------------------------------------------------------


def test_incident_app_config_composes_framework():
    from examples.incident_management.config import (
        IncidentAppConfig,
        framework_app_config_provider,
    )

    app = IncidentAppConfig()
    fw = app.framework
    assert isinstance(fw, FrameworkAppConfig)
    # Incident-tuned values land on the framework layer.
    assert fw.confidence_threshold == 0.75
    assert "platform-oncall" in fw.escalation_teams
    # Provider returns the same shape.
    via_provider = framework_app_config_provider()
    assert via_provider.escalation_teams == fw.escalation_teams
    assert via_provider.severity_aliases == fw.severity_aliases


def test_code_review_app_config_composes_framework():
    from examples.code_review.config import (
        CodeReviewAppConfig,
        framework_app_config_provider,
    )

    app = CodeReviewAppConfig()
    fw = app.framework
    assert isinstance(fw, FrameworkAppConfig)
    via_provider = framework_app_config_provider()
    assert isinstance(via_provider, FrameworkAppConfig)


def test_code_review_framework_cfg_differs_from_incident():
    """Each app's provider returns its own tuned values — the runtime
    must never mix them up. Catches the original bug: graph baking in
    incident defaults via load_incident_app_config()."""
    inc = resolve_framework_app_config(
        "examples.incident_management.config:framework_app_config_provider"
    )
    cr = resolve_framework_app_config(
        "examples.code_review.config:framework_app_config_provider"
    )
    # Different rosters — incident has on-call teams, code-review has
    # an empty default (apps that wire up code-owners override via YAML).
    assert "platform-oncall" in inc.escalation_teams
    assert cr.escalation_teams == []
    # Different similarity thresholds.
    assert inc.similarity_threshold != cr.similarity_threshold
    # Severity vocabularies don't overlap on key types.
    assert "sev1" in inc.severity_aliases
    assert "sev1" not in cr.severity_aliases
    assert "warn" in cr.severity_aliases
    # Dedup prompts are tuned to each domain.
    assert "incident" in inc.dedup_system_prompt.lower()
    assert "pull-request" in cr.dedup_system_prompt.lower()


def test_code_review_yaml_round_trips_through_loader():
    """The bundled code-review YAML loads cleanly via the back-compat
    lift path (legacy flat keys lifted into the composed FrameworkAppConfig)."""
    from examples.code_review.config import load_code_review_app_config

    cfg = load_code_review_app_config()
    # YAML sets similarity_threshold=0.3; lift folds it onto
    # framework.similarity_threshold and the property proxy keeps
    # the legacy attribute access working.
    assert cfg.similarity_threshold == 0.3
    assert cfg.framework.similarity_threshold == 0.3
