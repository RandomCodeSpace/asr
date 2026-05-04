"""Resolution agent helpers + prod-HITL override.

Two concerns covered here:

1. ``playbook_to_tool_calls`` translates a playbook YAML dict into
   :class:`ToolCallSpec` entries the resolution agent can issue
   through the gateway.
2. The framework gateway's ``effective_action`` correctly enforces
   the prod-environment override locked in this batch's
   ``config/config.yaml``: ``update_incident`` (medium) and any
   ``remediation:*`` tool tighten to ``approve`` in ``production``.

Cross-checks: the on-disk YAML for the gateway block round-trips
through ``load_config`` and produces the same ``GatewayConfig`` the
example app expects, so a future drift in either the YAML or the
helper is caught here.
"""
from __future__ import annotations

from pathlib import Path


from runtime.memory.session_state import L7PlaybookSuggestion
from runtime.memory.playbook_store import PlaybookStore
from runtime.memory.resolution import (
    playbook_to_tool_calls,
    top_playbook,
)
from runtime.config import GatewayConfig, ProdOverrides, load_config
from runtime.tools.gateway import effective_action


# ---------------------------------------------------------------------------
# playbook_to_tool_calls
# ---------------------------------------------------------------------------


def test_playbook_translates_remediation_steps_to_tool_calls() -> None:
    pb = {
        "id": "pb-x",
        "remediation": [
            {"tool": "remediation:restart_service", "args": {"service": "payments"}},
            {"tool": "update_incident", "args": {"patch": {"status": "resolved"}}},
        ],
        "required_approval": True,
    }
    calls = playbook_to_tool_calls(pb)
    assert len(calls) == 2
    assert calls[0]["tool"] == "remediation:restart_service"
    assert calls[0]["args"] == {"service": "payments"}
    assert calls[0]["requires_approval"] is True
    assert calls[1]["tool"] == "update_incident"


def test_playbook_with_no_remediation_returns_empty() -> None:
    assert playbook_to_tool_calls({"id": "x"}) == []
    assert playbook_to_tool_calls({"id": "x", "remediation": []}) == []
    assert playbook_to_tool_calls({}) == []


def test_playbook_skips_malformed_entries() -> None:
    pb = {
        "remediation": [
            {"tool": "ok_tool"},
            "not-a-dict",
            {"args": {}},  # no tool
            {"tool": ""},  # empty tool name
            None,
            {"tool": "second_ok"},
        ],
        "required_approval": False,
    }
    calls = playbook_to_tool_calls(pb)
    tools = [c["tool"] for c in calls]
    assert tools == ["ok_tool", "second_ok"]
    for c in calls:
        assert c["requires_approval"] is False
        assert c["args"] == {}


def test_playbook_seed_payments_translates_cleanly(tmp_path: Path) -> None:
    """The bundled pb-payments-latency playbook produces at least one call."""
    store = PlaybookStore(tmp_path)  # falls back to seed bundle
    pb = store.get("pb-payments-latency")
    assert pb is not None
    calls = playbook_to_tool_calls(pb)
    assert len(calls) >= 1
    assert all(isinstance(c["tool"], str) and c["tool"] for c in calls)


def test_playbook_returns_typed_dict_with_required_keys() -> None:
    pb = {"remediation": [{"tool": "x"}], "required_approval": True}
    calls = playbook_to_tool_calls(pb)
    assert len(calls) == 1
    spec = calls[0]
    # ToolCallSpec is a TypedDict — runtime check via key membership.
    assert {"tool", "args", "requires_approval"} <= set(spec.keys())


# ---------------------------------------------------------------------------
# top_playbook
# ---------------------------------------------------------------------------


def test_top_playbook_picks_highest_score() -> None:
    suggestions = [
        L7PlaybookSuggestion(playbook_id="a", score=0.4, matched_signals=[]),
        L7PlaybookSuggestion(playbook_id="b", score=0.9, matched_signals=[]),
        L7PlaybookSuggestion(playbook_id="c", score=0.7, matched_signals=[]),
    ]
    assert top_playbook(suggestions) == "b"


def test_top_playbook_empty_returns_none() -> None:
    assert top_playbook([]) is None


# ---------------------------------------------------------------------------
# Prod-HITL override — locked behaviour
# ---------------------------------------------------------------------------


def _gateway_with_locked_overrides() -> GatewayConfig:
    """Mirror the gateway block locked in config/config.yaml."""
    return GatewayConfig(
        policy={
            "update_incident": "medium",
            "remediation:restart_service": "high",
            "remediation:rollback": "high",
        },
        prod_overrides=ProdOverrides(
            prod_environments=["production"],
            resolution_trigger_tools=["update_incident", "remediation:*"],
        ),
    )


def test_prod_override_forces_approval_on_medium_update_incident() -> None:
    """``update_incident`` is medium-risk by policy, but production
    override tightens it to ``approve``."""
    cfg = _gateway_with_locked_overrides()
    assert effective_action("update_incident", env="production", gateway_cfg=cfg) == "approve"


def test_prod_override_forces_approval_on_remediation_glob() -> None:
    """Any ``remediation:*`` tool in prod must always require approval."""
    cfg = _gateway_with_locked_overrides()
    assert effective_action(
        "remediation:restart_service", env="production", gateway_cfg=cfg
    ) == "approve"
    assert effective_action(
        "remediation:rollback", env="production", gateway_cfg=cfg
    ) == "approve"
    # Even an unknown remediation tool (no policy entry) is forced.
    assert effective_action(
        "remediation:scale_up", env="production", gateway_cfg=cfg
    ) == "approve"


def test_non_prod_keeps_per_tier_dispatch() -> None:
    """Override is prod-only — staging keeps medium=notify / high=approve."""
    cfg = _gateway_with_locked_overrides()
    assert effective_action("update_incident", env="staging", gateway_cfg=cfg) == "notify"
    assert effective_action(
        "remediation:restart_service", env="staging", gateway_cfg=cfg
    ) == "approve"  # high tier alone


def test_unknown_low_risk_tool_in_prod_is_auto_when_not_in_trigger_list() -> None:
    cfg = _gateway_with_locked_overrides()
    # A tool not in policy and not in the trigger list runs auto.
    assert effective_action(
        "observability:get_logs", env="production", gateway_cfg=cfg
    ) == "auto"


# ---------------------------------------------------------------------------
# Round-trip via on-disk config.yaml
# ---------------------------------------------------------------------------


def test_config_yaml_loads_with_locked_gateway_block(monkeypatch) -> None:
    """The shipped config/config.yaml round-trips and yields the locked
    prod-override behaviour."""
    # Stub env vars referenced by the YAML so load_config doesn't blow up.
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://x.test")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "x")
    monkeypatch.setenv("EXTERNAL_MCP_URL", "https://x.test")
    monkeypatch.setenv("EXT_TOKEN", "x")

    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg = load_config(cfg_path)
    gw = cfg.runtime.gateway
    assert gw is not None
    assert gw.policy.get("update_incident") == "medium"
    assert gw.policy.get("apply_fix") == "high"
    assert gw.prod_overrides is not None
    assert "production" in gw.prod_overrides.prod_environments
    assert "update_incident" in gw.prod_overrides.resolution_trigger_tools
    assert "apply_fix" in gw.prod_overrides.resolution_trigger_tools
    # The runtime contract still holds — bare AND prefixed tool names
    # both resolve to ``approve`` in production via the candidate-list
    # fallback in ``effective_action``.
    assert effective_action(
        "update_incident", env="production", gateway_cfg=gw,
    ) == "approve"
    assert effective_action(
        "local_inc:update_incident", env="production", gateway_cfg=gw,
    ) == "approve"
    assert effective_action(
        "local_remediation:apply_fix", env="production", gateway_cfg=gw,
    ) == "approve"


def test_config_yaml_entry_agent_is_intake(monkeypatch) -> None:
    """Framework default 'intake' is the entry agent — no override needed."""
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://x.test")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "x")
    monkeypatch.setenv("EXTERNAL_MCP_URL", "https://x.test")
    monkeypatch.setenv("EXT_TOKEN", "x")
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg = load_config(cfg_path)
    assert cfg.orchestrator.entry_agent == "intake"
