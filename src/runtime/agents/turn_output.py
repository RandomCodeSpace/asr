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

# Dash separators between the confidence number and its rationale \u2014
# accept the full Pd block so gpt-oss's preferred EN DASH (\u2013) and
# the spec's EM DASH (\u2014) both parse, plus the ASCII hyphen.
_DASH_CHARS = frozenset("\u2010\u2011\u2012\u2013\u2014\u2015-")


def _parse_confidence_line(raw: str) -> tuple[float, str] | None:
    """Procedural confidence-line parser. Returns ``(value, rationale)`` on
    success, ``None`` on shape mismatch.

    Replaces an earlier regex implementation that Sonar's S5852 flagged
    as polynomial-time backtracking-vulnerable. A linear scan over the
    leading whitespace + number + optional dash-prefixed rationale has
    no backtracking surface to attack.

    Accepted shapes (on the first non-empty line of the section body):
      * ``"0.85"``
      * ``"0.85 \u2014 rationale"`` (or ASCII ``--`` / ``-`` / any Pd dash)
      * ``"-0.5 - rationale"``
      * ``"1"`` / ``"1."`` / ``".5"``
    """
    body = raw.lstrip()
    if not body:
        return None

    # Pull the leading number token: optional minus, then any combination
    # of digits and at most one dot. Stop at the first character that
    # cannot be part of a number.
    pos = 0
    if body[pos] == "-":
        pos += 1
    num_start = pos
    saw_dot = False
    saw_digit = False
    while pos < len(body):
        ch = body[pos]
        if ch.isdigit():
            saw_digit = True
            pos += 1
            continue
        if ch == "." and not saw_dot:
            saw_dot = True
            pos += 1
            continue
        break
    if not saw_digit:
        return None
    try:
        value = float(body[: pos] if pos > num_start else "0")
    except ValueError:
        return None

    # Skip whitespace + dash-cluster + whitespace before the rationale.
    rest = body[pos:].lstrip()
    while rest and rest[0] in _DASH_CHARS:
        rest = rest[1:]
    rationale = rest.lstrip()
    return value, rationale


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

    parsed = _parse_confidence_line(raw_conf)
    if parsed is None:
        raise EnvelopeMissingError(
            agent=agent, field="confidence",
            message=(
                f"envelope_missing: confidence section did not parse "
                f"(agent={agent!r}, raw={raw_conf!r})"
            ),
        )
    conf_value = _clamp_unit(parsed[0])
    rationale = parsed[1].strip() or "(no rationale provided)"

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

    # Path 5 (D-22-fallback): some models (gpt-oss:20b in particular)
    # treat a terminal-tool call as the completion and emit an empty
    # closing AIMessage. Apps configure typed-terminal tools with
    # ``confidence`` + ``confidence_rationale`` args by contract —
    # synthesise an envelope from those args when the markdown path
    # yielded nothing. Skip if any earlier path succeeded.
    for msg in reversed(messages):
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            args = tc.get("args") or {}
            if not isinstance(args, dict):
                continue
            conf = args.get("confidence")
            rat = args.get("confidence_rationale") or args.get("rationale")
            if conf is None or not rat:
                continue
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                continue
            content_body = (
                args.get("resolution_summary")
                or args.get("reason")
                or args.get("summary")
                or f"terminal tool {tc.get('name', '<unknown>')} invoked"
            )
            if not isinstance(content_body, str) or not content_body.strip():
                content_body = f"terminal tool {tc.get('name', '<unknown>')} invoked"
            try:
                env = AgentTurnOutput(
                    content=content_body,
                    confidence=max(0.0, min(1.0, conf_f)),
                    confidence_rationale=str(rat),
                    signal=args.get("signal"),
                )
            except ValidationError:
                continue
            _LOG.info(
                "envelope_synthesized_from_tool_args agent=%s tool=%s "
                "confidence=%.2f",
                agent, tc.get("name"), env.confidence,
            )
            return env

    # Path 6 (D-22-fallback-permissive): the model called at least one
    # tool but never emitted the markdown contract or a typed-terminal
    # tool with confidence args. Rather than hard-failing the run,
    # synthesise a minimal envelope so the session reaches a terminal
    # status (typically default_terminal_status -> needs_review) and
    # the operator can review what happened. Confidence is intentionally
    # low (0.30) so any HITL gate fires.
    invoked_tool_names: list[str] = []
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            n = tc.get("name")
            if n:
                invoked_tool_names.append(n)
    if invoked_tool_names:
        body = (
            f"Agent {agent} invoked "
            f"{', '.join(invoked_tool_names)} but did not emit a final "
            f"summary. Review the tool-call audit for outcome."
        )
        try:
            env = AgentTurnOutput(
                content=body,
                confidence=0.30,
                confidence_rationale=(
                    "synthesized: agent invoked tools but emitted no "
                    "closing message"
                ),
                signal=None,
            )
            _LOG.warning(
                "envelope_synthesized_permissive agent=%s tools=%s",
                agent, invoked_tool_names,
            )
            return env
        except ValidationError:
            pass

    # Path 7 (terminal): fail loudly. None of the six paths produced a
    # valid envelope — this is a real prompt-drift signal worth
    # surfacing as a structured agent_run error. Log the last
    # AIMessage content (capped) so operators can diagnose whether
    # the LLM emitted the wrong shape vs nothing at all.
    last_ai_content = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            c = getattr(msg, "content", None)
            if isinstance(c, str):
                last_ai_content = c
                break
    # Also dump every tool_call across all AIMessages so we can see
    # whether the model called any tool with confidence args (Path 5
    # would have fired) or whether it called a non-terminal tool only.
    tool_call_summary = []
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            args = tc.get("args") or {}
            tool_call_summary.append({
                "name": tc.get("name"),
                "arg_keys": sorted(args.keys()) if isinstance(args, dict) else "?",
                "has_confidence": isinstance(args, dict) and "confidence" in args,
                "has_rationale": isinstance(args, dict) and (
                    "confidence_rationale" in args or "rationale" in args
                ),
            })
    # Detailed per-message dump so we can see exactly what shape the
    # conversation took when the parse failed.
    msg_summary = []
    for m in messages:
        kind = m.__class__.__name__
        c = getattr(m, "content", None)
        c_len = len(c) if isinstance(c, str) else (
            f"list[{len(c)}]" if isinstance(c, list) else type(c).__name__
        )
        tcs = [tc.get("name") for tc in (getattr(m, "tool_calls", None) or [])]
        msg_summary.append({"kind": kind, "content_len": c_len, "tool_calls": tcs})
    _LOG.warning(
        "envelope_missing: agent=%s last_ai_message_content=%r "
        "structured_response_type=%s message_count=%d tool_calls=%r "
        "msg_trace=%r",
        agent,
        last_ai_content[:1500] if last_ai_content else "<empty>",
        type(result.get("structured_response")).__name__,
        len(messages),
        tool_call_summary,
        msg_summary,
    )
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
