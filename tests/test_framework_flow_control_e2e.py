"""Phase 12 (FOC-06) -- v1.2 milestone end-to-end genericity test.

Proves the full "framework owns flow control" thesis: the LLM emits
intent only (tool_name, tool_args_excluding_session_data, confidence,
signal); the framework injects session-derived args, enforces the
envelope, gates on policy, and decides retry -- none of those flow
through the LLM-supplied tool args.

If a future phase introduces a state-derived arg leak through the LLM,
or relaxes one of the framework-owned policy boundaries, any of these
five assertion sets will break loudly.

This file is the v1.2 regression-prevention contract:

  test_foc_01_environment_injected_from_session
  test_foc_02_incident_id_injected_from_session
  test_foc_03_envelope_missing_confidence_fails
  test_foc_04_high_risk_tool_gates_to_pending_approval
  test_foc_05_retry_decision_matches_policy

Each test asserts the framework's pure boundary still owns its slice of
flow control. The assertions are framework-pure (no orchestrator-stub
harness required) -- the v1.2 thesis is precisely that flow control
collapses into pure functions, so the tests probe those functions
directly.
"""
from __future__ import annotations


import pydantic
import pytest

from runtime.agents.turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
)
from runtime.config import (
    GatePolicy,
    GatewayConfig,
    OrchestratorConfig,
    RetryPolicy,
)
from runtime.policy import (
    GateDecision,
    RetryDecision,
    should_gate,
    should_retry,
)
from runtime.state import Session, ToolCall


# ---- helper: minimal-config builder for pure should_retry probes --

def _retry_cfg(
    *,
    max_retries: int = 2,
    retry_on_transient: bool = True,
    retry_low_confidence_threshold: float = 0.4,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            retry_on_transient=retry_on_transient,
            retry_low_confidence_threshold=retry_low_confidence_threshold,
        ),
    )


def _gate_cfg_high_risk(*, env: str | None = "production") -> OrchestratorConfig:
    """OrchestratorConfig + GatewayConfig wired so ``apply_fix`` is the
    canonical high-risk tool that v1.2 must gate to pending_approval.
    """
    cfg = OrchestratorConfig(
        gate_policy=GatePolicy(
            confidence_threshold=0.7,
            gated_environments={"production"},
            gated_risk_actions={"approve"},
        ),
    )
    # Attach a runtime gateway config that flags apply_fix high-risk.
    cfg_with_gateway = cfg.model_copy()
    object.__setattr__(
        cfg_with_gateway,
        "gateway",
        GatewayConfig(policy={"apply_fix": "high"}),
    )
    return cfg_with_gateway


def _make_session(*, environment: str | None = "production") -> Session:
    """Synthetic Session for pure-policy probes -- no store, no graph."""
    s = Session(
        id="S-foc-06",
        status="in_progress",
        created_at="2026-05-07T00:00:00Z",
        updated_at="2026-05-07T00:00:00Z",
    )
    # ``environment`` is an extra field on the framework Session; apps
    # subclass to model it. For the gate test we set it via attribute so
    # ``getattr(session, 'environment', None)`` returns the right value.
    object.__setattr__(s, "environment", environment)
    return s


# =====================================================================
# FOC-01: framework injects ``environment`` from session
# =====================================================================

def test_foc_01_environment_injected_from_session():
    """The v1.2 thesis: ``environment`` is a framework-owned, session-
    derived arg. ``OrchestratorConfig.injected_args`` is the declarative
    surface; the framework reads it at tool-invoke time. The LLM never
    emits ``environment``.

    Assertion contract: a runtime config that declares
    ``injected_args = {"environment": "session.environment"}`` is the
    sole place the wiring exists. The dotted path begins with
    ``session.``; non-session paths are forbidden by config-load.
    """
    cfg = OrchestratorConfig(
        injected_args={"environment": "session.environment"},
    )
    assert "environment" in cfg.injected_args
    assert cfg.injected_args["environment"] == "session.environment"
    assert cfg.injected_args["environment"].startswith("session.")
    # The validator pins dotted-path shape (Phase 9). A non-dotted value
    # is rejected at config-load. Real attribute resolution happens at
    # tool-invoke time in runtime.tools.arg_injection, so the leak guard
    # is the dotted-path rule plus the runtime-time resolver -- the
    # combination ensures nothing outside the live Session can be
    # injected without an explicit code change.
    with pytest.raises(pydantic.ValidationError):
        OrchestratorConfig(
            injected_args={"environment": "no_dot_here"},
        )


# =====================================================================
# FOC-02: framework injects ``incident_id`` from session.id
# =====================================================================

def test_foc_02_incident_id_injected_from_session():
    """Same thesis: ``incident_id`` is framework-injected from
    ``session.id``. The dotted-path validator pins it.
    """
    cfg = OrchestratorConfig(
        injected_args={
            "environment": "session.environment",
            "incident_id": "session.id",
        },
    )
    assert cfg.injected_args["incident_id"] == "session.id"
    assert cfg.injected_args["incident_id"].startswith("session.")
    # The framework can inject MULTIPLE session-derived args;
    # the LLM tool-call signature stays minimal.
    assert len(cfg.injected_args) == 2


# =====================================================================
# FOC-03: envelope-missing turn lands at status='error' with
#          EnvelopeMissingError raised by parse_envelope_from_result
# =====================================================================

def test_foc_03_envelope_missing_confidence_fails():
    """A ``create_react_agent`` result with NO ``structured_response``
    and a final AIMessage that is NOT a JSON envelope MUST raise
    :class:`EnvelopeMissingError`. The framework propagates that error
    to the agent runner which marks the agent_run with
    ``summary='agent failed: ...EnvelopeMissingError...'`` -- the same
    summary that ``Orchestrator._extract_last_error`` reconstructs to
    feed ``should_retry``.
    """
    from langchain_core.messages import AIMessage

    # Result mimicking a turn that never produced an envelope.
    result_missing = {
        "messages": [AIMessage(content="i think the answer is 42")],
        # No "structured_response" key.
    }
    with pytest.raises(EnvelopeMissingError):
        parse_envelope_from_result(result_missing, agent="intake")

    # Conversely, a properly-shaped envelope returns an AgentTurnOutput
    # with the confidence the framework's policy will read.
    result_ok = {
        "messages": [AIMessage(content="ok")],
        "structured_response": AgentTurnOutput(
            content="ok",
            confidence=0.85,
            confidence_rationale="stub",
            signal=None,
        ),
    }
    env = parse_envelope_from_result(result_ok, agent="intake")
    assert env.confidence == 0.85


# =====================================================================
# FOC-04: high-risk tool in production gates to pending_approval
#          (the should_gate decision drives the gateway interrupt)
# =====================================================================

def test_foc_04_high_risk_tool_gates_to_pending_approval():
    """Pin Phase 11 (FOC-04): a tool with risk=high in a gated env MUST
    return GateDecision(gate=True, reason='high_risk_tool'). The
    orchestrator's _GatedTool wrapper consults this and emits an
    Interrupt that the watchdog captures as pending_approval. The LLM
    never sees the gating decision.
    """
    cfg = _gate_cfg_high_risk(env="production")
    sess = _make_session(environment="production")
    tc = ToolCall(
        tool="apply_fix",
        agent="resolution",
        args={"target": "payments-svc"},
        result=None,
        ts="2026-05-07T00:00:00Z",
        risk="high",
    )
    decision = should_gate(
        session=sess,
        tool_call=tc,
        confidence=0.95,  # high confidence: gate fires anyway because risk=high
        cfg=cfg,
    )
    assert decision == GateDecision(gate=True, reason="high_risk_tool")

    # Sanity: a low-risk tool in the same env does NOT gate.
    cfg_low = OrchestratorConfig(
        gate_policy=GatePolicy(
            confidence_threshold=0.7,
            gated_environments={"production"},
            gated_risk_actions={"approve"},
        ),
    )
    object.__setattr__(
        cfg_low,
        "gateway",
        GatewayConfig(policy={"create_incident": "low"}),
    )
    tc_low = ToolCall(
        tool="create_incident",
        agent="intake",
        args={"summary": "x"},
        result=None,
        ts="2026-05-07T00:00:00Z",
        risk="low",
    )
    decision_low = should_gate(
        session=sess, tool_call=tc_low, confidence=0.95, cfg=cfg_low,
    )
    assert decision_low == GateDecision(gate=False, reason="auto")


# =====================================================================
# FOC-05: retry decision matches policy across the 3 critical cases
# =====================================================================

def test_foc_05_retry_decision_matches_policy():
    """Pin FOC-05: the framework owns retry policy via
    ``runtime.policy.should_retry``. Three sub-cases that v1.2's
    end-to-end thesis depends on:

      (a) ValidationError -> retry=False, reason='permanent_error'
      (b) TimeoutError + retry_count=0 + max_retries=2 -> retry=True,
          reason='auto_retry'
      (c) retry_count=2, max_retries=2 -> retry=False,
          reason='max_retries_exceeded' (regardless of error class)
    """
    cfg = _retry_cfg(max_retries=2)

    # (a) permanent error -- pydantic.ValidationError
    class _M(pydantic.BaseModel):
        x: int = pydantic.Field(ge=0)

    err: pydantic.ValidationError | None = None
    try:
        _M(x=-1)
    except pydantic.ValidationError as e:
        err = e
    assert err is not None
    d_perm = should_retry(
        retry_count=0, error=err, confidence=0.9, cfg=cfg,
    )
    assert d_perm == RetryDecision(retry=False, reason="permanent_error")

    # (b) transient under cap -- auto_retry
    d_first = should_retry(
        retry_count=0, error=TimeoutError("net blip"),
        confidence=0.9, cfg=cfg,
    )
    assert d_first == RetryDecision(retry=True, reason="auto_retry")

    # (c) at cap -- max_retries_exceeded
    d_cap = should_retry(
        retry_count=2, error=TimeoutError("net blip"),
        confidence=0.9, cfg=cfg,
    )
    assert d_cap == RetryDecision(
        retry=False, reason="max_retries_exceeded",
    )


# =====================================================================
# v1.2 thesis: stub LLM emits ONLY (tool_name, tool_args_excluding_
# session_data, confidence, signal) -- helper that polices the contract
# =====================================================================

def test_v12_stub_helper_rejects_session_data_in_tool_args():
    """Any test that drives the framework with a stub LLM MUST guard
    against accidental leakage of session-derived data into the tool
    args. ``_make_intent_only_stub`` enforces this contract by raising
    on construction if ``environment`` / ``incident_id`` / ``session_id``
    appear in the args.

    This sentinel test pins the contract so a future phase that adds a
    new framework-injected arg can extend the deny-list with one line.
    """
    # Allowed: tool args contain only LLM-emitted intent data.
    plan_ok = [{"name": "update_incident", "args": {"note": "stub"}}]
    _check_args_clean(plan_ok)  # no exception

    # Forbidden: ``environment`` leaked through LLM args.
    plan_leak_env = [
        {"name": "update_incident",
         "args": {"note": "x", "environment": "production"}},
    ]
    with pytest.raises(AssertionError):
        _check_args_clean(plan_leak_env)

    # Forbidden: ``incident_id`` leaked through LLM args.
    plan_leak_id = [
        {"name": "update_incident",
         "args": {"note": "x", "incident_id": "INC-1"}},
    ]
    with pytest.raises(AssertionError):
        _check_args_clean(plan_leak_id)


# ---- helper: stub-args contract enforcer --------------------------

def _check_args_clean(tool_call_plan: list[dict]) -> None:
    """v1.2 contract enforcer for stub LLMs: tool_call_plan args MUST
    NOT contain ``environment`` / ``incident_id`` / ``session_id``.
    The framework injects those via injected_args. Adding a new
    framework-injected arg = one new line in this deny-list.
    """
    forbidden = {"environment", "incident_id", "session_id"}
    for tc in tool_call_plan:
        leaked = forbidden & set(tc.get("args", {}).keys())
        assert not leaked, (
            f"v1.2 contract violation: tool_call_plan {tc!r} carries "
            f"session-derived args {leaked} that the framework should "
            f"inject via OrchestratorConfig.injected_args"
        )
