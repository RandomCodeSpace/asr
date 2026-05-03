"""Tests for the gateway config schema (P4-A).

The framework-side ``GatewayConfig`` declares per-tool risk levels,
plus an optional ``ProdOverrides`` block that can force HITL approval
on tools matched by glob in production-like environments.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import (
    AppConfig,
    GatewayConfig,
    ProdOverrides,
    RuntimeConfig,
)


def test_gateway_config_defaults_are_empty_and_safe():
    """Default ``GatewayConfig`` is permissive: empty policy, no overrides."""
    cfg = GatewayConfig()
    assert cfg.policy == {}
    assert cfg.notify_channel is None
    assert cfg.prod_overrides is None


def test_gateway_config_accepts_full_policy_map():
    cfg = GatewayConfig(
        policy={
            "create_incident": "low",
            "update_incident": "medium",
            "apply_fix": "high",
        },
        notify_channel="#ops-bot",
    )
    assert cfg.policy["create_incident"] == "low"
    assert cfg.policy["update_incident"] == "medium"
    assert cfg.policy["apply_fix"] == "high"
    assert cfg.notify_channel == "#ops-bot"


def test_gateway_config_rejects_unknown_risk_level():
    """Pydantic ``Literal`` enforces the {low, medium, high} vocabulary."""
    with pytest.raises(ValidationError):
        GatewayConfig(policy={"weird_tool": "EXTREME"})  # type: ignore[arg-type]


def test_gateway_config_rejects_uppercase_risk_level():
    with pytest.raises(ValidationError):
        GatewayConfig(policy={"x": "HIGH"})  # type: ignore[arg-type]


def test_prod_overrides_defaults():
    p = ProdOverrides()
    assert p.prod_environments == ["production"]
    assert p.resolution_trigger_tools == []


def test_prod_overrides_accepts_globs_and_env_list():
    p = ProdOverrides(
        prod_environments=["production", "prod-eu", "prod-us"],
        resolution_trigger_tools=["update_incident", "remediation:*"],
    )
    assert "production" in p.prod_environments
    assert "remediation:*" in p.resolution_trigger_tools


def test_runtime_config_carries_optional_gateway():
    """``RuntimeConfig.gateway`` is optional — Phase 4 must not break Phase 3 callers."""
    rc = RuntimeConfig()
    assert rc.gateway is None

    rc2 = RuntimeConfig(
        gateway=GatewayConfig(
            policy={"x": "high"},
            prod_overrides=ProdOverrides(
                prod_environments=["production"],
                resolution_trigger_tools=["x"],
            ),
        ),
    )
    assert rc2.gateway is not None
    assert rc2.gateway.policy["x"] == "high"
    assert rc2.gateway.prod_overrides is not None
    assert rc2.gateway.prod_overrides.prod_environments == ["production"]


def test_app_config_can_load_gateway_from_dict_shape():
    """End-to-end: building an ``AppConfig`` with a nested gateway block works."""
    raw = {
        "llm": {
            "default": "stub_default",
            "providers": {"stub": {"kind": "stub"}},
            "models": {"stub_default": {"provider": "stub", "model": "stub-1"}},
        },
        "mcp": {"servers": []},
        "runtime": {
            "gateway": {
                "policy": {
                    "create_incident": "low",
                    "apply_fix": "high",
                },
                "prod_overrides": {
                    "prod_environments": ["production"],
                    "resolution_trigger_tools": ["apply_fix", "remediation:*"],
                },
            },
        },
    }
    cfg = AppConfig(**raw)
    assert cfg.runtime.gateway is not None
    assert cfg.runtime.gateway.policy["apply_fix"] == "high"
    assert cfg.runtime.gateway.prod_overrides is not None
    assert "remediation:*" in cfg.runtime.gateway.prod_overrides.resolution_trigger_tools


def test_runtime_config_back_compat_legacy_callers():
    """Code constructing ``RuntimeConfig`` without ``gateway`` still works."""
    rc = RuntimeConfig(state_class="examples.incident_management.state.IncidentState")
    assert rc.gateway is None
    assert rc.max_concurrent_sessions == 8
