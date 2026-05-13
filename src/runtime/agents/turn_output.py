"""Phase 10 (FOC-03) — AgentTurnOutput envelope + reconciliation helpers.

The envelope is the structural contract every responsive agent invocation
must satisfy: content + confidence in [0,1] + confidence_rationale + optional
signal. The framework wires it as ``response_format=AgentTurnOutput`` into
``langchain.agents.create_agent`` (see Phase 15 / LLM-COMPAT-01); the
agent loop terminates on the same turn the LLM emits the envelope-shaped
tool call, populating ``result["structured_response"]``, which the
framework reads and persists onto the ``AgentRun`` row.

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
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    ``create_agent`` call sites (``runtime.graph`` and
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


_HEADER_SPLIT = re.compile(r"^#{2,}\s+(\w+)\s*$", re.MULTILINE)
_CONF_LINE = re.compile(
    # Leftmost float (allows int form), optional rationale after em-dash /
    # ASCII dash / hyphen separator. ``re.DOTALL`` so a multi-line rationale
    # is captured wholesale.
    r"^\s*(-?[0-9]*\.?[0-9]+)\s*(?:[\u2014\-]+\s*(.*))?$",
    re.DOTALL,
)


def _clamp_unit(x: float) -> float:
    """Clamp a confidence float into [0, 1] without raising. The skill
    prompt asks for [0, 1]; an LLM occasionally emits ``1.05`` or
    ``-0.1`` — clamp rather than reject so the parse step is forgiving."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def parse_markdown_envelope(
    content: str, *, agent: str = "",
) -> "AgentTurnOutput":
    """D-22-03 — parse the trailing markdown contract block into an
    :class:`AgentTurnOutput`.

    Expects three sections ``## Response`` / ``## Confidence`` /
    ``## Signal`` (case-insensitive headers, ``##+`` accepted) at the
    end of the LLM's reply. The free-text body of ``## Response``
    becomes ``content``; the float on the ``## Confidence`` line
    becomes ``confidence`` (clamped to [0, 1]); the rest of that line
    after ``-`` / ``--`` / ``\u2014`` becomes ``confidence_rationale``;
    the ``## Signal`` body becomes ``signal`` (with ``none``/``null``/
    blank coerced to ``None``).

    Raises :class:`EnvelopeMissingError` only when the parse cannot
    produce a valid envelope at all — missing ``## Confidence`` line,
    unparseable confidence value, or empty ``## Response`` body.
    """
    if not isinstance(content, str) or not content.strip():
        raise EnvelopeMissingError(
            agent=agent, field="content",
            message=f"envelope_missing: empty content (agent={agent!r})",
        )

    # ``re.split`` with the header capture-group yields:
    #   [pre, h1, body1, h2, body2, ...]
    # Pre is the prose before the first heading; we ignore it for parse.
    parts = _HEADER_SPLIT.split(content)
    sections: dict[str, str] = {}
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            key = parts[i].strip().lower()
            body = parts[i + 1].strip()
            # Don't overwrite an earlier section with a later same-named
            # one (unusual but possible in pathological prompts).
            sections.setdefault(key, body)

    response_body = sections.get("response", "").strip()
    raw_conf = sections.get("confidence", "").strip()

    if not raw_conf:
        raise EnvelopeMissingError(
            agent=agent, field="confidence",
            message=(
                f"envelope_missing: confidence section absent or empty "
                f"(agent={agent!r})"
            ),
        )

    m = _CONF_LINE.match(raw_conf)
    if not m:
        raise EnvelopeMissingError(
            agent=agent, field="confidence",
            message=(
                f"envelope_missing: confidence section did not parse "
                f"(agent={agent!r}, raw={raw_conf!r})"
            ),
        )
    try:
        conf_value = _clamp_unit(float(m.group(1)))
    except (TypeError, ValueError) as exc:
        raise EnvelopeMissingError(
            agent=agent, field="confidence",
            message=(
                f"envelope_missing: confidence value not a float "
                f"(agent={agent!r}, raw={raw_conf!r})"
            ),
        ) from exc
    rationale = (m.group(2) or "").strip() or "(no rationale provided)"

    signal_raw = sections.get("signal", "").strip().lower() or None
    if signal_raw in {"none", "null", "", "n/a"}:
        signal_raw = None

    if not response_body:
        raise EnvelopeMissingError(
            agent=agent, field="content",
            message=(
                f"envelope_missing: response section empty (agent={agent!r})"
            ),
        )

    try:
        return AgentTurnOutput(
            content=response_body,
            confidence=conf_value,
            confidence_rationale=rationale,
            signal=signal_raw,
        )
    except ValidationError as exc:
        raise EnvelopeMissingError(
            agent=agent, field="validation",
            message=(
                f"envelope_missing: pydantic validation rejected the "
                f"parsed values (agent={agent!r}): {exc}"
            ),
        ) from exc


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
            # Path 1 produced a dict that doesn't match the envelope
            # schema. Fall through to Path 2 (parse last AIMessage), but
            # log so providers shipping malformed structured_response are
            # observable instead of silently degraded.
            _LOG.debug(
                "envelope path 1 (structured_response dict) failed validation; "
                "falling through to AIMessage JSON parse",
                exc_info=True,
            )

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

    # Path 4 (D-22-01): markdown-primary parse on the last AIMessage's
    # content. Producers that emit the new ``## Response / ## Confidence /
    # ## Signal`` section block land here when Paths 1+2 yield nothing.
    for msg in reversed(messages):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            return parse_markdown_envelope(content, agent=agent)
        except EnvelopeMissingError:
            # This AIMessage didnt carry a parseable markdown contract.
            # Continue scanning earlier messages on the off-chance a
            # nested loop emitted the contract one step back.
            continue
        break

    # Path 5 (terminal): fail loudly. None of the four paths produced a
    # valid envelope — this is a real prompt-drift signal worth
    # surfacing as a structured agent_run error.
    raise EnvelopeMissingError(
        agent=agent,
        field="structured_response",
        message=(
            f"envelope_missing: no structured_response, JSON-decodable, or "
            f"markdown-decodable AIMessage envelope found (agent={agent})"
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
    "parse_markdown_envelope",
    "reconcile_confidence",
]
