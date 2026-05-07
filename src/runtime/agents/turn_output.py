"""Phase 10 (FOC-03) — AgentTurnOutput envelope + reconciliation helpers.

The envelope is the structural contract every responsive agent invocation
must satisfy: content + confidence ∈ [0,1] + confidence_rationale + optional signal.
LangGraph's `create_react_agent(..., response_format=AgentTurnOutput)` enforces
the schema at the LLM boundary; the framework reads the resulting
``result["structured_response"]`` and persists it onto the ``AgentRun`` row.

D-10-02 — pydantic envelope wrapped via ``response_format``.
D-10-03 — when a typed-terminal-tool was called this turn, the framework
reconciles its ``confidence`` arg against the envelope's. Tolerance 0.05
inclusive; tool-arg wins on mismatch with an INFO log.

This is a leaf module: no imports from ``runtime.graph`` or
``runtime.orchestrator``. Both of those depend on it; the dependency
graph is acyclic.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ConfigDict, Field

_LOG = logging.getLogger("runtime.orchestrator")

# D-10-03 — heuristic tolerance for envelope-vs-tool-arg confidence mismatch.
# Inclusive boundary (|env - tool| <= 0.05 is silent). Documented for future
# tuning; widening is cheap, narrowing requires care because the LLM's
# self-reported turn confidence is naturally ~5pp noisier than its
# tool-call-time confidence.
_DEFAULT_TOLERANCE: float = 0.05


class AgentTurnOutput(BaseModel):
    """Structural envelope every agent invocation MUST emit.

    The framework wires this as ``response_format=AgentTurnOutput`` on both
    ``create_react_agent`` call sites (``runtime.graph`` and
    ``runtime.agents.responsive``). Pydantic's ``extra="forbid"`` keeps the
    contract narrow — adding fields is a deliberate schema migration, not a
    free-for-all.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        min_length=1,
        description="Final user-facing message text.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence in this turn's output: "
            "0.85+ strong, 0.5 hedged, <0.4 weak."
        ),
    )
    confidence_rationale: str = Field(
        min_length=1,
        description="One-sentence explanation of the confidence value.",
    )
    signal: str | None = Field(
        default=None,
        description=(
            "Optional next-state signal "
            "(e.g. success | failed | needs_input | default). "
            "Routing layer validates the vocabulary."
        ),
    )


class EnvelopeMissingError(Exception):
    """Raised by :func:`parse_envelope_from_result` when neither
    ``result["structured_response"]`` nor a JSON-shaped final AIMessage
    yields a valid :class:`AgentTurnOutput`.

    Carries structured cause attributes (``agent``, ``field``) so the
    runner can mark the agent_run as ``error`` with a precise reason.
    """

    def __init__(self, *, agent: str, field: str, message: str | None = None):
        self.agent = agent
        self.field = field
        super().__init__(message or f"envelope_missing: {field} (agent={agent})")


def parse_envelope_from_result(
    result: dict,
    *,
    agent: str,
) -> AgentTurnOutput:
    """Extract an :class:`AgentTurnOutput` from a ``create_react_agent`` result.

    Three-step defensive fallback (Risk #1 — Ollama may not honor
    ``response_format`` cleanly across all providers):

    1. ``result["structured_response"]`` — preferred path; LangGraph 1.1.x
       populates it when ``response_format`` is set and the LLM honors
       structured output.
    2. ``result["messages"][-1].content`` parsed as JSON, validated against
       :class:`AgentTurnOutput` — covers providers that stuff envelope JSON
       in the AIMessage body instead of a separate structured field.
    3. Both fail → :class:`EnvelopeMissingError` so the runner marks
       agent_run ``error`` with a structured cause.
    """
    # Path 1: structured_response (preferred)
    sr = result.get("structured_response")
    if isinstance(sr, AgentTurnOutput):
        return sr
    if isinstance(sr, dict):
        try:
            return AgentTurnOutput.model_validate(sr)
        except Exception:  # noqa: BLE001
            pass

    # Path 2: JSON-parse last AIMessage content
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            return AgentTurnOutput.model_validate(payload)
        except Exception:  # noqa: BLE001
            continue
        break

    # Path 3: fail loudly
    raise EnvelopeMissingError(
        agent=agent,
        field="structured_response",
        message=(
            f"envelope_missing: no structured_response or JSON-decodable "
            f"AIMessage envelope found (agent={agent})"
        ),
    )


def reconcile_confidence(
    envelope_value: float,
    tool_arg_value: float | None,
    *,
    agent: str,
    session_id: str,
    tool_name: str | None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    """Reconcile envelope confidence against typed-terminal-tool-arg confidence.

    D-10-03 contract:
    - When ``tool_arg_value`` is None: return envelope value silently.
    - When both present and ``|envelope - tool_arg| <= tolerance``: return
      tool-arg silently (tool-arg wins on the return regardless — it's the
      finer-grained, gated value).
    - When both present and ``|envelope - tool_arg| > tolerance``: log INFO
      with the verbatim format from CONTEXT.md / D-10-03 and return tool-arg.

    Log shape (preserved verbatim for grep-based observability assertions):
        ``runtime.orchestrator: turn.confidence_mismatch agent={a} turn_value={e:.2f} tool_value={t:.2f} tool={tn} session_id={sid}``
    """
    if tool_arg_value is None:
        return envelope_value
    diff = abs(envelope_value - tool_arg_value)
    if diff > tolerance:
        _LOG.info(
            "turn.confidence_mismatch "
            "agent=%s turn_value=%.2f tool_value=%.2f tool=%s session_id=%s",
            agent,
            envelope_value,
            tool_arg_value,
            tool_name,
            session_id,
        )
    return tool_arg_value


__all__ = [
    "AgentTurnOutput",
    "EnvelopeMissingError",
    "parse_envelope_from_result",
    "reconcile_confidence",
]
