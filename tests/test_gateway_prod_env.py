"""End-to-end prod-env HITL override tests.

Locked decision EXTRA: when the live ``Session.environment`` is in
``prod_overrides.prod_environments`` AND the tool name matches one of
the ``resolution_trigger_tools`` globs, the gateway forces
``require-approval`` regardless of the risk-tier lookup. The override
runs BEFORE the tier check so it can ONLY tighten, never relax.

The four required cases (verbatim from the locked spec):

  1. non-prod env, high-risk tool                      -> still ``approve``
                                                          (risk wins on its own)
  2. prod env, low-risk tool, NOT in trigger list      -> ``auto``
                                                          (override doesn't apply)
  3. prod env, low-risk tool, IN trigger list          -> ``approve``
                                                          (override forces it)
  4. prod env, medium-risk tool, IN trigger list       -> ``approve``
                                                          (override tightens medium -> high)

Plus the ``ToolPolicy``-style validator behaviour: the framework rejects
unknown risk levels at config-load time so a typo in YAML never silently
relaxes HITL.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import GatewayConfig, ProdOverrides
from runtime.tools.gateway import effective_action


# ---------- Test cases for the locked four-cell prod-override matrix ----------


def _cfg_with_overrides() -> GatewayConfig:
    """Common config: every tool low-risk by default, prod overrides cover
    a small set of resolution triggers."""
    return GatewayConfig(
        policy={
            "create_incident": "low",
            "update_incident": "medium",
            "lookup_similar": "low",
            "apply_fix": "high",
        },
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident", "remediation:*"],
        ),
    )


def test_case_1_non_prod_high_risk_tool_still_requires_approval():
    """Risk wins on its own. The override path is irrelevant in non-prod."""
    cfg = _cfg_with_overrides()
    # apply_fix is high risk; staging is not in prod_environments.
    assert effective_action("apply_fix", env="staging", gateway_cfg=cfg) == "approve"
    assert effective_action("apply_fix", env="dev", gateway_cfg=cfg) == "approve"


def test_case_2_prod_low_risk_tool_not_in_trigger_list_is_auto():
    """Production env, but the tool is low-risk AND NOT a resolution trigger."""
    cfg = _cfg_with_overrides()
    # create_incident is low risk and not in resolution_trigger_tools.
    assert effective_action("create_incident", env="production", gateway_cfg=cfg) == "auto"
    # lookup_similar is also low risk and not in the trigger list.
    assert effective_action("lookup_similar", env="production", gateway_cfg=cfg) == "auto"


def test_case_3_prod_low_risk_tool_in_trigger_list_forces_approval():
    """Production env + matched glob => override forces ``approve`` even
    though the policy says low risk."""
    cfg = GatewayConfig(
        policy={"update_incident": "low"},  # explicitly low; override must still tighten
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],
        ),
    )
    assert effective_action("update_incident", env="production", gateway_cfg=cfg) == "approve"


def test_case_4_prod_medium_risk_tool_in_trigger_list_tightens_to_approve():
    """Production env + matched glob + medium risk => override tightens
    medium-notify-soft to require-approval."""
    cfg = GatewayConfig(
        policy={"update_incident": "medium"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],
        ),
    )
    # Without the override, medium would be ``notify``; the override must
    # tighten to ``approve``.
    assert effective_action("update_incident", env="production", gateway_cfg=cfg) == "approve"


# ---------- Composition / ordering invariants ------------------------


def test_override_runs_before_risk_lookup_so_it_can_only_tighten():
    """If the override path were applied AFTER the risk lookup, a
    high-risk tool already mapped to ``approve`` could not be 'tightened
    further' but a low-risk tool could only be tightened by the override
    running first. We verify that ordering pins the only-tighten property:

      * low + override-match = approve  (tighten)
      * high + no-override   = approve  (tier alone)
      * low + no-override    = auto     (no false positives)
    """
    cfg = GatewayConfig(
        policy={"x": "low", "y": "high"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["x"],
        ),
    )
    assert effective_action("x", env="production", gateway_cfg=cfg) == "approve"  # tightened
    assert effective_action("y", env="production", gateway_cfg=cfg) == "approve"  # tier
    assert effective_action("x", env="staging", gateway_cfg=cfg) == "auto"        # no false positive


def test_override_glob_matches_remediation_namespace():
    """``remediation:*`` covers every tool in the remediation MCP server."""
    cfg = _cfg_with_overrides()
    assert effective_action("remediation:rollback", env="production", gateway_cfg=cfg) == "approve"
    assert effective_action("remediation:scale_down", env="production", gateway_cfg=cfg) == "approve"
    # Negative: a tool outside the namespace is unaffected.
    assert effective_action("observability:get_logs", env="production", gateway_cfg=cfg) == "auto"


# ---------- Validator behaviour (ToolPolicy unknown-risk rejection) -----


def test_unknown_risk_level_rejected_at_config_construction():
    """A typo in YAML must fail loudly — never silently relax HITL."""
    with pytest.raises(ValidationError):
        GatewayConfig(policy={"apply_fix": "HIGH"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GatewayConfig(policy={"apply_fix": "extreme"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GatewayConfig(policy={"apply_fix": ""})  # type: ignore[arg-type]


def test_prod_override_with_empty_trigger_list_is_safe_noop():
    """An empty trigger list never forces approval, regardless of env."""
    cfg = GatewayConfig(
        policy={"x": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=[],
        ),
    )
    assert effective_action("x", env="production", gateway_cfg=cfg) == "auto"


def test_prod_override_with_empty_env_list_is_safe_noop():
    """If no env is declared 'prod', the override path can never fire."""
    cfg = GatewayConfig(
        policy={"x": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=[],
            resolution_trigger_tools=["x"],
        ),
    )
    assert effective_action("x", env="production", gateway_cfg=cfg) == "auto"
