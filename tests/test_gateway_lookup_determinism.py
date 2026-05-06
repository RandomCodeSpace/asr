"""Pin the deterministic resolution order of effective_action's prefix
fallback. Prefixed form wins over bare form when both are configured."""
from runtime.config import GatewayConfig, ProdOverrides
from runtime.tools.gateway import effective_action


def test_prefixed_form_wins_over_bare_when_both_configured():
    cfg = GatewayConfig(
        policy={
            "local_inc:update_incident": "low",
            "update_incident": "high",
        },
        prod_overrides=None,
    )
    # local_inc:update_incident -> low -> auto. The prefixed form is more
    # specific and wins; the bare-form fallback only fires when the
    # prefixed form has no entry.
    assert effective_action(
        "local_inc:update_incident", env="dev", gateway_cfg=cfg,
    ) == "auto"


def test_bare_used_when_only_bare_configured():
    cfg = GatewayConfig(policy={"update_incident": "high"}, prod_overrides=None)
    assert effective_action(
        "local_inc:update_incident", env="dev", gateway_cfg=cfg,
    ) == "approve"


def test_prod_override_prefers_prefixed_pattern_match():
    """A prod_override pattern matching the prefixed form fires before
    the bare-form fallback even when both forms could match."""
    cfg = GatewayConfig(
        policy={"local_inc:update_incident": "low"},  # would resolve to auto
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["local_inc:update_incident"],  # exact match
        ),
    )
    # Prod override fires first → approve, regardless of policy tier.
    assert effective_action(
        "local_inc:update_incident", env="production", gateway_cfg=cfg,
    ) == "approve"


def test_prod_override_falls_back_to_bare_pattern():
    """When the override pattern is bare but the tool is prefixed, the
    bare-form fallback inside the prod predicate matches."""
    cfg = GatewayConfig(
        policy={"local_inc:update_incident": "low"},
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident"],  # bare-form pattern
        ),
    )
    assert effective_action(
        "local_inc:update_incident", env="production", gateway_cfg=cfg,
    ) == "approve"


def test_no_match_falls_through_to_auto():
    cfg = GatewayConfig(policy={}, prod_overrides=None)
    assert effective_action(
        "local_x:unknown", env="dev", gateway_cfg=cfg,
    ) == "auto"
