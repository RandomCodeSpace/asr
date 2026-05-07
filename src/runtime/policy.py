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


__all__ = ["GateDecision", "GateReason", "should_gate"]
