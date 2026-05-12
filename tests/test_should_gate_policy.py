"""Phase 11 (FOC-04) -- pure-function should_gate matrix.

The should_gate function is the SOLE place the framework decides whether
a tool call requires HITL approval. It composes three orthogonal inputs:

  * effective_action(tool, env, gateway_cfg)  -- preserves PVC-08
    prefixed-form lookup invariant
  * session.environment                       -- vs cfg.gate_policy.gated_environments
  * confidence                                -- vs cfg.gate_policy.confidence_threshold

This module pins:
  * All 5 GateDecision.reason literal values are exercised.
  * Purity (same inputs -> identical results, no I/O).
  * PVC-08 prefixed-form lookup wins over bare form.
  * Boundary conditions on confidence_threshold (strict <).
  * None confidence treated as "no signal yet" -> no low_confidence gate.
"""
from __future__ import annotations

from unittest.mock import patch

from runtime.policy import GateDecision, should_gate
from runtime.tools import gateway as gw

from tests._policy_helpers import (
    make_env_session,
    make_orch_cfg,
    make_tool_call,
)


def test_should_gate_returns_auto_when_low_risk_safe_env() -> None:
    """env=dev, conf=0.99, action=auto -> auto."""
    cfg = make_orch_cfg(policy={"foo": "low"})
    sess = make_env_session(env="dev")
    tc = make_tool_call("foo")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=False, reason="auto")


def test_should_gate_returns_auto_when_low_conf_but_safe_tool() -> None:
    """env=dev, conf=0.1, action=auto -> auto.

    A known-safe tool (low risk -> action=auto) must NOT gate even on
    very low confidence -- safe tools are safe.
    """
    cfg = make_orch_cfg(policy={"foo": "low"})
    sess = make_env_session(env="dev")
    tc = make_tool_call("foo")
    decision = should_gate(sess, tc, confidence=0.1, cfg=cfg)
    assert decision == GateDecision(gate=False, reason="auto")


def test_should_gate_high_risk_tool_gates_in_dev() -> None:
    """env=dev, conf=0.99, action=approve -> high_risk_tool."""
    cfg = make_orch_cfg(policy={"apply_fix": "high"})
    sess = make_env_session(env="dev")
    tc = make_tool_call("apply_fix")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="high_risk_tool")


def test_should_gate_high_risk_tool_gates_in_prod() -> None:
    """env=production, conf=0.99, action=approve -> high_risk_tool."""
    cfg = make_orch_cfg(policy={"apply_fix": "high"})
    sess = make_env_session(env="production")
    tc = make_tool_call("apply_fix")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="high_risk_tool")


def test_should_gate_gated_env_with_notify_tool() -> None:
    """env=production, conf=0.99, action=notify -> gated_env."""
    cfg = make_orch_cfg(policy={"update_incident": "medium"})
    sess = make_env_session(env="production")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="gated_env")


def test_should_gate_gated_env_with_auto_tool_does_not_gate() -> None:
    """env=production, conf=0.99, action=auto -> auto.

    A safe-rated tool stays safe even in a gated environment.
    """
    cfg = make_orch_cfg(policy={"read_logs": "low"})
    sess = make_env_session(env="production")
    tc = make_tool_call("read_logs")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=False, reason="auto")


def test_should_gate_low_confidence_with_notify_tool() -> None:
    """env=dev, conf=0.5, threshold=0.7, action=notify -> low_confidence."""
    cfg = make_orch_cfg(
        policy={"update_incident": "medium"},
        confidence_threshold=0.7,
    )
    sess = make_env_session(env="dev")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=0.5, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="low_confidence")


def test_should_gate_low_confidence_at_boundary() -> None:
    """env=dev, conf=0.7, threshold=0.7, action=notify -> auto.

    Strict-less-than predicate: at-threshold confidence does NOT gate.
    """
    cfg = make_orch_cfg(
        policy={"update_incident": "medium"},
        confidence_threshold=0.7,
    )
    sess = make_env_session(env="dev")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=0.7, cfg=cfg)
    assert decision == GateDecision(gate=False, reason="auto")


def test_should_gate_high_risk_beats_low_confidence() -> None:
    """env=dev, conf=0.1, action=approve -> high_risk_tool.

    high_risk_tool has higher precedence than low_confidence.
    """
    cfg = make_orch_cfg(policy={"apply_fix": "high"})
    sess = make_env_session(env="dev")
    tc = make_tool_call("apply_fix")
    decision = should_gate(sess, tc, confidence=0.1, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="high_risk_tool")


def test_should_gate_gated_env_beats_low_confidence() -> None:
    """env=production, conf=0.1, action=notify -> gated_env.

    gated_env has higher precedence than low_confidence.
    """
    cfg = make_orch_cfg(policy={"update_incident": "medium"})
    sess = make_env_session(env="production")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=0.1, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="gated_env")


def test_should_gate_custom_gated_environments() -> None:
    """env=staging, gated_environments={production,staging}, action=notify -> gated_env."""
    cfg = make_orch_cfg(
        policy={"update_incident": "medium"},
        gated_environments={"production", "staging"},
    )
    sess = make_env_session(env="staging")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="gated_env")


def test_should_gate_pvc08_prefixed_form_preserved() -> None:
    """tool=remediation:apply_fix, prefixed=high AND bare=low -> prefixed wins.

    Pins PVC-08: the prefixed-form lookup in effective_action wins over
    the bare suffix. should_gate MUST delegate to effective_action so
    this invariant survives unchanged.
    """
    cfg = make_orch_cfg(policy={
        "remediation:apply_fix": "high",  # prefixed wins
        "apply_fix": "low",               # bare loses
    })
    sess = make_env_session(env="dev")
    tc = make_tool_call("remediation:apply_fix")
    decision = should_gate(sess, tc, confidence=0.99, cfg=cfg)
    assert decision == GateDecision(gate=True, reason="high_risk_tool")


def test_should_gate_with_none_confidence_does_not_low_confidence_gate() -> None:
    """confidence=None, action=notify -> auto (no signal yet)."""
    cfg = make_orch_cfg(
        policy={"update_incident": "medium"},
        confidence_threshold=0.9,
    )
    sess = make_env_session(env="dev")
    tc = make_tool_call("update_incident")
    decision = should_gate(sess, tc, confidence=None, cfg=cfg)
    assert decision == GateDecision(gate=False, reason="auto")


def test_should_gate_blocked_literal_accepted_by_schema() -> None:
    """GateDecision(gate=True, reason='blocked') constructs OK.

    The 'blocked' literal is reserved on the schema for future hard-stop
    semantics; Phase 11 itself never produces it from a code path. The
    schema must accept it so future phases don't need a migration.
    """
    decision = GateDecision(gate=True, reason="blocked")
    assert decision.gate is True
    assert decision.reason == "blocked"


def test_should_gate_is_pure_no_io() -> None:
    """Same inputs 5x -> identical results. No mutation, no I/O."""
    cfg = make_orch_cfg(policy={"apply_fix": "high"})
    sess = make_env_session(env="production")
    tc = make_tool_call("apply_fix")
    results = [should_gate(sess, tc, confidence=0.5, cfg=cfg) for _ in range(5)]
    assert all(r == results[0] for r in results)
    # Inputs are unmutated: env still 'production', tool still 'apply_fix'.
    assert sess.environment == "production"
    assert tc.tool == "apply_fix"


def test_evaluate_gate_helper_uses_default_policy_when_none() -> None:
    """The wrap-level ``_evaluate_gate`` helper falls back to a default
    GatePolicy when callers haven't yet been threaded.

    Pins the legacy-callsite migration path: any pre-Phase-11 caller
    that still constructs ``wrap_tool`` without ``gate_policy=`` gets
    Phase-11 default behaviour (``gated_risk_actions={"approve"}``)
    rather than a hard ImportError or NoneType crash.
    """
    from runtime.tools.gateway import _evaluate_gate
    from runtime.config import GatewayConfig

    sess = make_env_session(env="dev")
    decision = _evaluate_gate(
        session=sess,
        tool_name="apply_fix",
        gate_policy=None,
        gateway_cfg=GatewayConfig(policy={"apply_fix": "high"}),
    )
    assert decision.gate is True
    assert decision.reason == "high_risk_tool"


def test_evaluate_gate_helper_threads_confidence_hint_from_session() -> None:
    """``_evaluate_gate`` reads ``session.turn_confidence_hint`` for
    the low_confidence branch."""
    from runtime.config import GatePolicy, GatewayConfig
    from runtime.tools.gateway import _evaluate_gate

    sess = make_env_session(env="dev")
    sess.turn_confidence_hint = 0.5  # low

    # notify-rated tool + low confidence -> low_confidence reason.
    decision = _evaluate_gate(
        session=sess,
        tool_name="update_incident",
        gate_policy=GatePolicy(confidence_threshold=0.7),
        gateway_cfg=GatewayConfig(policy={"update_incident": "medium"}),
    )
    assert decision.gate is True
    assert decision.reason == "low_confidence"


def test_evaluate_gate_returns_auto_when_no_policy_match() -> None:
    """_evaluate_gate's auto branch -- safe-rated tool with no match."""
    from runtime.config import GatePolicy, GatewayConfig
    from runtime.tools.gateway import _evaluate_gate

    sess = make_env_session(env="dev")
    decision = _evaluate_gate(
        session=sess,
        tool_name="some_unrated_tool",
        gate_policy=GatePolicy(),
        gateway_cfg=GatewayConfig(policy={}),
    )
    assert decision.gate is False
    assert decision.reason == "auto"


def test_evaluate_gate_returns_gated_env_for_notify_in_production() -> None:
    """_evaluate_gate's gated_env branch -- production-class env tightening."""
    from runtime.config import GatePolicy, GatewayConfig
    from runtime.tools.gateway import _evaluate_gate

    sess = make_env_session(env="production")
    decision = _evaluate_gate(
        session=sess,
        tool_name="update_incident",
        gate_policy=GatePolicy(),
        gateway_cfg=GatewayConfig(policy={"update_incident": "medium"}),
    )
    assert decision.gate is True
    assert decision.reason == "gated_env"


def test_find_pending_index_no_match_returns_none() -> None:
    """Phase 11 coverage hit: _find_pending_index walks past every row
    when no ``pending_approval`` matches the tool_name + ts pair.

    Pre-Phase-11 the no-match path was unreachable from existing wrap
    tests because every wrap-level test registers exactly one pending
    row. Asserting None directly closes the gateway.py 75% gap.
    """
    from runtime.state import ToolCall
    from runtime.tools.gateway import _find_pending_index

    rows = [
        ToolCall(
            agent="t", tool="other_tool", args={}, result=None,
            ts="2026-05-07T00:00:00Z", risk="low",
            status="executed",
        ),
    ]
    assert _find_pending_index(rows, "missing_tool", "2026-05-07T00:00:00Z") is None


def test_wrap_tool_sync_run_path_passes_should_gate_for_low_risk() -> None:
    """Phase 11: sync _run branch coverage -- safe tool runs through.

    Exercises the sync ``_run`` path explicitly so the wrap's auto
    branch (decision.gate=False) lands a coverage hit on the sync
    side. Existing wrap tests use the async path; the sync mirror was
    historically uncovered.
    """
    from typing import Any

    from langchain_core.tools import BaseTool
    from runtime.config import GatePolicy, GatewayConfig
    from runtime.state import Session
    from runtime.tools.gateway import wrap_tool

    class _Echo(BaseTool):
        name: str = "echo_tool"
        description: str = "echoes args"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return {"echoed": dict(kwargs)}

    sess = Session(
        id="S-cov-1",
        status="open",
        created_at="2026-05-07T00:00:00Z",
        updated_at="2026-05-07T00:00:00Z",
    )
    sess.__dict__["environment"] = "dev"  # type: ignore[index]
    cfg = GatewayConfig(policy={"echo_tool": "low"})
    wrapped = wrap_tool(
        _Echo(), session=sess, gateway_cfg=cfg, agent_name="t",
        gate_policy=GatePolicy(),
    )
    out = wrapped.invoke({"x": 1})
    assert out == {"echoed": {"x": 1}}
    # Auto branch -> no audit row.
    assert sess.tool_calls == []


def test_should_gate_only_reads_documented_inputs() -> None:
    """should_gate calls effective_action exactly once with documented args.

    Patches at the policy module's import namespace because policy.py
    binds effective_action by name (`from runtime.tools.gateway import
    effective_action`) -- patching the original symbol at the gateway
    module would not intercept the bound reference.
    """
    from runtime import policy as pol

    cfg = make_orch_cfg(policy={"apply_fix": "high"})
    sess = make_env_session(env="production")
    tc = make_tool_call("apply_fix")
    with patch.object(pol, "effective_action", wraps=gw.effective_action) as spy:
        should_gate(sess, tc, confidence=0.5, cfg=cfg)
        spy.assert_called_once_with(
            "apply_fix", env="production", gateway_cfg=cfg.gateway,
        )
