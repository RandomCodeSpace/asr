"""Pure HITL gating policy (Phase 11 / FOC-04).

The :func:`should_gate` function is the SOLE place the framework decides
whether a tool call requires human-in-the-loop approval. It composes
three orthogonal inputs:

  1. ``effective_action(tool_call.tool, env=session.environment,
     gateway_cfg=cfg.gateway)`` -- preserves the v1.0 PVC-08
     prefixed-form lookup invariant.
  2. ``session.environment`` -- gated when in
     ``cfg.gate_policy.gated_environments``.
  3. ``confidence`` -- gated when below
     ``cfg.gate_policy.confidence_threshold``.

Pure: same inputs always yield identical :class:`GateDecision`; no I/O,
no skill-prompt input, no mutation.

Precedence (descending):

  1. ``effective_action`` returns a value in
     ``cfg.gate_policy.gated_risk_actions``
     -> ``GateDecision(gate=True, reason="high_risk_tool")``
  2. ``session.environment`` in ``cfg.gate_policy.gated_environments``
     AND ``effective_action != "auto"``
     -> ``GateDecision(gate=True, reason="gated_env")``
  3. ``confidence`` is not None AND
     ``confidence < cfg.gate_policy.confidence_threshold``
     AND ``effective_action != "auto"``
     -> ``GateDecision(gate=True, reason="low_confidence")``
  4. otherwise -> ``GateDecision(gate=False, reason="auto")``

The literal ``"blocked"`` is reserved on :class:`GateDecision.reason`
for future hard-stop semantics; Phase 11 itself never returns it from a
production code path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from runtime.tools.gateway import effective_action

# Phase 11 (FOC-04): forward-reference imports for the should_gate
# signature only; kept inside ``TYPE_CHECKING`` so the bundle's
# intra-import stripper does not remove a load-bearing import. The
# ``pass`` keeps the block syntactically valid after stripping.
if TYPE_CHECKING:  # pragma: no cover -- type checking only
    from runtime.config import OrchestratorConfig  # noqa: F401
    from runtime.state import ToolCall  # noqa: F401
    pass  # noqa: PIE790 -- bundle survives even if imports are stripped


GateReason = Literal[
    "auto",
    "high_risk_tool",
    "gated_env",
    "low_confidence",
    "blocked",
]


class GateDecision(BaseModel):
    """Outcome of a single gating evaluation."""

    model_config = ConfigDict(extra="forbid")
    gate: bool
    reason: GateReason


def should_gate(
    session: Any,
    tool_call: "ToolCall",
    confidence: float | None,
    cfg: "OrchestratorConfig",
) -> GateDecision:
    """Decide whether ``tool_call`` should pause for HITL approval.

    Pure -- delegates the per-tool risk lookup to
    :func:`runtime.tools.gateway.effective_action` (so the v1.0 PVC-08
    prefixed-form lookup invariant is preserved) and combines the
    result with ``session.environment`` and ``confidence`` per the
    precedence rules in the module docstring.

    ``session`` is typed as ``Any`` because the framework's base
    :class:`runtime.state.Session` does not own the ``environment``
    field (apps subclass and add it). The function reads
    ``session.environment`` and tolerates a missing attribute by
    treating it as ``None``.

    ``confidence=None`` means "no signal yet" -- treated internally as
    1.0 to avoid a false-positive low_confidence gate before any
    envelope/tool-arg has surfaced for the active turn.
    """
    # Read gateway config off the OrchestratorConfig. The runtime threads
    # it via cfg.gateway today (sibling of cfg.gate_policy in the
    # OrchestratorConfig namespace) -- gracefully tolerate the legacy
    # path where gateway is configured on RuntimeConfig instead.
    gateway_cfg = getattr(cfg, "gateway", None)
    env = getattr(session, "environment", None)

    risk_action = effective_action(
        tool_call.tool,
        env=env,
        gateway_cfg=gateway_cfg,
    )

    # 1. high-risk tool gates first.
    if risk_action in cfg.gate_policy.gated_risk_actions:
        return GateDecision(gate=True, reason="high_risk_tool")

    # 2. gated env: any non-"auto" risk in a gated environment.
    if (env in cfg.gate_policy.gated_environments
            and risk_action != "auto"):
        return GateDecision(gate=True, reason="gated_env")

    # 3. low confidence: only an actionable tool. None == "no signal yet".
    effective_conf = 1.0 if confidence is None else confidence
    if (effective_conf < cfg.gate_policy.confidence_threshold
            and risk_action != "auto"):
        return GateDecision(gate=True, reason="low_confidence")

    return GateDecision(gate=False, reason="auto")


# ---------------------------------------------------------------
# Phase 12 (FOC-05): pure should_retry policy.
# ---------------------------------------------------------------

import asyncio as _asyncio

import pydantic as _pydantic

from runtime.agents.turn_output import EnvelopeMissingError

RetryReason = Literal[
    "auto_retry",
    "max_retries_exceeded",
    "permanent_error",
    "low_confidence_no_retry",
    "transient_disabled",
]


class RetryDecision(BaseModel):
    """Outcome of a single retry-policy evaluation.

    Pure surface: produced by :func:`should_retry` from
    ``(retry_count, error, confidence, cfg)``. The orchestrator's
    ``_retry_session_locked`` consults this BEFORE running the retry;
    the UI consults the same value via
    ``Orchestrator.preview_retry_decision`` to render the button label /
    disabled state.
    """

    model_config = ConfigDict(extra="forbid")
    retry: bool
    reason: RetryReason


# Whitelist of exception types that are NEVER auto-retryable.
# Schema/validation errors -- the LLM produced bad data; retrying
# without addressing root cause burns budget. Adding a new entry is a
# one-line PR (D-12-02 explicit choice -- no new ToolError ABC).
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    _pydantic.ValidationError,
    EnvelopeMissingError,
)

# Whitelist of exception types that are ALWAYS auto-retryable
# (subject to max_retries). Network blips, asyncio timeouts,
# filesystem/socket transients. httpx is NOT imported because the
# runtime does not raise httpx errors today; built-in TimeoutError
# covers asyncio's 3.11+ alias.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    _asyncio.TimeoutError,
    TimeoutError,
    OSError,
    ConnectionError,
)


def _is_permanent_error(error: Exception | None) -> bool:
    if error is None:
        return False
    return isinstance(error, _PERMANENT_TYPES)


def _is_transient_error(error: Exception | None) -> bool:
    if error is None:
        return False
    return isinstance(error, _TRANSIENT_TYPES)


def should_retry(
    retry_count: int,
    error: Exception | None,
    confidence: float | None,
    cfg: "OrchestratorConfig",
) -> RetryDecision:
    """Decide whether the framework should auto-retry a failed turn.

    Pure -- same inputs always yield identical RetryDecision.

    Precedence (descending; first match wins):
      1. ``retry_count >= cfg.retry_policy.max_retries``
         -> ``RetryDecision(retry=False, reason="max_retries_exceeded")``
      2. ``error`` matches ``_PERMANENT_TYPES``
         -> ``RetryDecision(retry=False, reason="permanent_error")``
      3. ``confidence is not None`` AND
         ``confidence < cfg.retry_policy.retry_low_confidence_threshold``
         AND ``error`` is NOT in ``_TRANSIENT_TYPES``
         -> ``RetryDecision(retry=False, reason="low_confidence_no_retry")``
      4. ``error`` matches ``_TRANSIENT_TYPES`` AND
         ``cfg.retry_policy.retry_on_transient is False``
         -> ``RetryDecision(retry=False, reason="transient_disabled")``
      5. ``error`` matches ``_TRANSIENT_TYPES`` AND
         ``cfg.retry_policy.retry_on_transient is True``
         -> ``RetryDecision(retry=True, reason="auto_retry")``
      6. Default fall-through (no match) -> ``RetryDecision(
         retry=False, reason="permanent_error")`` -- fail-closed
         conservative default (D-12-02).

    ``retry_count`` is the count of PRIOR retries (0 on the first
    retry attempt). Caller is responsible for the bump.

    ``error`` may be ``None`` (caller has no exception object); that is
    treated as a permanent error for safety.

    ``confidence`` is the last AgentRun.confidence for the failed turn;
    ``None`` means "no signal recorded" and skips the low-confidence
    gate.
    """
    # 1. absolute cap -- regardless of error class
    if retry_count >= cfg.retry_policy.max_retries:
        return RetryDecision(retry=False, reason="max_retries_exceeded")

    # 2. permanent errors -- never auto-retry
    if _is_permanent_error(error):
        return RetryDecision(retry=False, reason="permanent_error")

    is_transient = _is_transient_error(error)

    # 3. low-confidence -- only when error is NOT transient (transient
    # errors are mechanical; the LLM's confidence in the business
    # decision is still trustworthy on retry).
    if (confidence is not None
            and confidence < cfg.retry_policy.retry_low_confidence_threshold
            and not is_transient):
        return RetryDecision(
            retry=False, reason="low_confidence_no_retry",
        )

    # 4 + 5. transient classification
    if is_transient:
        if not cfg.retry_policy.retry_on_transient:
            return RetryDecision(retry=False, reason="transient_disabled")
        return RetryDecision(retry=True, reason="auto_retry")

    # 6. fail-closed default
    return RetryDecision(retry=False, reason="permanent_error")


__all__ = [
    # Phase 11
    "GateDecision", "GateReason", "should_gate",
    # Phase 12
    "RetryDecision", "RetryReason", "should_retry",
]
