"""Tests for ``effective_action``.

The resolver is a PURE function: given a tool name, the live session's
environment, and a ``GatewayConfig``, return one of:

  * ``"auto"``    -> execute without any HITL plumbing
  * ``"notify"``  -> execute, but emit a soft notification (no graph pause)
  * ``"approve"`` -> require human approval via LangGraph ``interrupt()``

Resolution order (CRITICAL — the prod override runs FIRST so it can only
TIGHTEN the action, never relax it):

  1. If ``cfg.prod_overrides`` is set AND ``env`` matches one of the
     listed prod environments AND ``tool_name`` matches one of the
     ``resolution_trigger_tools`` globs, return ``"approve"`` immediately.
  2. Otherwise look up ``cfg.policy[tool_name]`` and map:
        low    -> auto
        medium -> notify
        high   -> approve
  3. Tools absent from the policy map default to ``auto``.
"""
from __future__ import annotations

from runtime.config import GatewayConfig, ProdOverrides
from runtime.tools.gateway import effective_action


def test_unknown_tool_with_empty_policy_is_auto():
    """No policy entry => safe default of ``auto``."""
    cfg = GatewayConfig()
    assert effective_action("anything", env="dev", gateway_cfg=cfg) == "auto"


def test_low_risk_maps_to_auto():
    cfg = GatewayConfig(policy={"create_incident": "low"})
    assert effective_action("create_incident", env="dev", gateway_cfg=cfg) == "auto"


def test_medium_risk_maps_to_notify():
    cfg = GatewayConfig(policy={"update_incident": "medium"})
    assert effective_action("update_incident", env="dev", gateway_cfg=cfg) == "notify"


def test_high_risk_maps_to_approve():
    cfg = GatewayConfig(policy={"apply_fix": "high"})
    assert effective_action("apply_fix", env="dev", gateway_cfg=cfg) == "approve"


def test_no_gateway_cfg_returns_auto():
    """When the framework opts out of the gateway, every tool runs as ``auto``."""
    assert effective_action("apply_fix", env="production", gateway_cfg=None) == "auto"


def test_prod_override_forces_approve_for_low_risk_tool():
    """Prod env + matched glob = ``approve`` even though the tool is low risk."""
    cfg = GatewayConfig(
        policy={"update_incident": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],
        ),
    )
    assert effective_action("update_incident", env="production", gateway_cfg=cfg) == "approve"


def test_prod_override_does_not_apply_in_non_prod_env():
    """Same config but env is staging => fall back to risk-tier (low -> auto)."""
    cfg = GatewayConfig(
        policy={"update_incident": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],
        ),
    )
    assert effective_action("update_incident", env="staging", gateway_cfg=cfg) == "auto"


def test_prod_override_glob_match():
    """``remediation:*`` matches ``remediation:rollback`` and ``remediation:scale``."""
    cfg = GatewayConfig(
        policy={},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["remediation:*"],
        ),
    )
    assert effective_action("remediation:rollback", env="production", gateway_cfg=cfg) == "approve"
    assert effective_action("remediation:scale", env="production", gateway_cfg=cfg) == "approve"
    # Negative: a non-matching tool falls back to risk-tier (empty policy -> auto)
    assert effective_action("observability:get_logs", env="production", gateway_cfg=cfg) == "auto"


def test_prod_override_with_no_overrides_block_is_noop():
    """``prod_overrides=None`` => never tighten."""
    cfg = GatewayConfig(policy={"update_incident": "low"}, prod_overrides=None)
    assert effective_action("update_incident", env="production", gateway_cfg=cfg) == "auto"


def test_env_can_be_none_or_missing_safely():
    """If ``session.environment`` is missing (non-incident apps), prod override is a no-op."""
    cfg = GatewayConfig(
        policy={"x": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["x"],
        ),
    )
    assert effective_action("x", env=None, gateway_cfg=cfg) == "auto"
    assert effective_action("x", env="", gateway_cfg=cfg) == "auto"


def test_function_is_pure_no_side_effects():
    """Calling ``effective_action`` repeatedly with the same args is idempotent
    and never mutates the config."""
    cfg = GatewayConfig(
        policy={"a": "high", "b": "medium", "c": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["a*"],
        ),
    )
    snapshot = cfg.model_dump()
    for _ in range(5):
        assert effective_action("a", env="production", gateway_cfg=cfg) == "approve"
        assert effective_action("b", env="dev", gateway_cfg=cfg) == "notify"
        assert effective_action("c", env="dev", gateway_cfg=cfg) == "auto"
    assert cfg.model_dump() == snapshot


def test_multiple_prod_environments():
    cfg = GatewayConfig(
        policy={},
        prod_overrides=ProdOverrides(
            prod_environments=["production", "prod-eu", "prod-us"],
            resolution_trigger_tools=["apply_fix"],
        ),
    )
    assert effective_action("apply_fix", env="production", gateway_cfg=cfg) == "approve"
    assert effective_action("apply_fix", env="prod-eu", gateway_cfg=cfg) == "approve"
    assert effective_action("apply_fix", env="prod-us", gateway_cfg=cfg) == "approve"
    assert effective_action("apply_fix", env="staging", gateway_cfg=cfg) == "auto"
