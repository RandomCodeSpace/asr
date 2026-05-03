"""Tests for ``runtime.config.FrameworkAppConfig``.

Pins the generic, framework-level cross-cutting config shape. Apps
configure these knobs under the ``framework:`` block of their YAML
which AppConfig binds directly; the legacy
``RuntimeConfig.framework_app_config_path`` provider-callable
indirection has been removed (apps no longer ship a per-app
``config.py``).
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


def test_incident_yaml_loads_into_framework_block():
    """The bundled incident-management YAML's ``framework:`` block
    binds straight onto ``AppConfig.framework``."""
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("config/incident_management.yaml").read_text())
    fw = FrameworkAppConfig(**(raw.get("framework") or {}))
    assert fw.confidence_threshold == 0.75
    assert "platform-oncall" in fw.escalation_teams
    assert fw.severity_aliases["sev1"] == "high"


def test_code_review_yaml_loads_into_framework_block():
    """Code-review YAML's ``framework:`` block produces a different
    ``FrameworkAppConfig`` than the incident YAML's — no leakage."""
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("config/code_review.yaml").read_text())
    fw = FrameworkAppConfig(**(raw.get("framework") or {}))
    assert fw.escalation_teams == []
    assert "platform-oncall" not in fw.escalation_teams


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


def test_app_yamls_do_not_leak_into_each_other():
    """Each app's bundled YAML produces its own tuned
    ``FrameworkAppConfig`` — no cross-pollination. Catches the
    original bug where the runtime baked in incident defaults
    irrespective of the loaded app."""
    from pathlib import Path

    import yaml

    inc_raw = yaml.safe_load(Path("config/incident_management.yaml").read_text())
    cr_raw = yaml.safe_load(Path("config/code_review.yaml").read_text())
    inc = FrameworkAppConfig(**(inc_raw.get("framework") or {}))
    cr = FrameworkAppConfig(**(cr_raw.get("framework") or {}))
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
