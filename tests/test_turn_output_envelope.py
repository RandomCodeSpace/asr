"""Phase 10 (FOC-03) — AgentTurnOutput envelope tests.

Coverage matrix:
- Schema validation (10 tests): missing/out-of-range/extra-field/empty rejections.
- Reconciliation (4 tests): match/mismatch/no-tool-arg/at-tolerance-boundary.
- Parser fallback (3 tests): structured_response → AIMessage JSON → EnvelopeMissingError.
- All-six-agent-kinds emit envelope (1 parametrized = 6 cases) covering
  intake, triage, deep_investigator, resolution, supervisor, monitor.

Reconciliation log shape (D-10-03 verbatim):
  INFO runtime.orchestrator: turn.confidence_mismatch agent={a} turn_value={e:.2f} tool_value={t:.2f} tool={tn} session_id={sid}
"""
from __future__ import annotations

import json
import logging

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from runtime.agents.turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
    reconcile_confidence,
)


# ---------------------------------------------------------------------------
# 1) Schema validation
# ---------------------------------------------------------------------------


class TestAgentTurnOutputSchema:
    def test_envelope_valid_minimum(self):
        env = AgentTurnOutput(
            content=".",
            confidence=0.0,
            confidence_rationale="x",
        )
        assert env.confidence == 0.0
        assert env.signal is None

    def test_envelope_valid_maximum(self):
        env = AgentTurnOutput(
            content="x",
            confidence=1.0,
            confidence_rationale="x",
        )
        assert env.confidence == 1.0

    def test_envelope_missing_confidence_raises(self):
        with pytest.raises(ValidationError) as exc:
            AgentTurnOutput(
                content="x",
                confidence_rationale="x",
            )  # type: ignore[call-arg]
        assert "confidence" in str(exc.value)

    def test_envelope_missing_rationale_raises(self):
        with pytest.raises(ValidationError) as exc:
            AgentTurnOutput(
                content="x",
                confidence=0.5,
            )  # type: ignore[call-arg]
        assert "confidence_rationale" in str(exc.value)

    def test_envelope_missing_content_raises(self):
        with pytest.raises(ValidationError) as exc:
            AgentTurnOutput(
                confidence=0.5,
                confidence_rationale="x",
            )  # type: ignore[call-arg]
        assert "content" in str(exc.value)

    def test_envelope_extra_field_forbidden(self):
        with pytest.raises(ValidationError) as exc:
            AgentTurnOutput(
                content="x",
                confidence=0.5,
                confidence_rationale="x",
                foo="bar",
            )  # type: ignore[call-arg]
        assert "foo" in str(exc.value).lower() or "extra" in str(exc.value).lower()

    def test_envelope_negative_confidence_raises(self):
        with pytest.raises(ValidationError):
            AgentTurnOutput(
                content="x",
                confidence=-0.1,
                confidence_rationale="x",
            )

    def test_envelope_above_one_confidence_raises(self):
        with pytest.raises(ValidationError):
            AgentTurnOutput(
                content="x",
                confidence=1.01,
                confidence_rationale="x",
            )

    def test_envelope_empty_rationale_raises(self):
        with pytest.raises(ValidationError):
            AgentTurnOutput(
                content="x",
                confidence=0.5,
                confidence_rationale="",
            )

    def test_envelope_signal_optional(self):
        # None accepted
        env = AgentTurnOutput(
            content="x", confidence=0.5, confidence_rationale="x", signal=None
        )
        assert env.signal is None
        # "success" accepted (string-typed; routing layer validates downstream)
        env2 = AgentTurnOutput(
            content="x",
            confidence=0.5,
            confidence_rationale="x",
            signal="success",
        )
        assert env2.signal == "success"
        # "bogus" accepted at the schema layer (routing validates separately)
        env3 = AgentTurnOutput(
            content="x",
            confidence=0.5,
            confidence_rationale="x",
            signal="bogus",
        )
        assert env3.signal == "bogus"


# ---------------------------------------------------------------------------
# 2) Reconciliation
# ---------------------------------------------------------------------------


class TestReconcileConfidence:
    def test_reconcile_match_silent(self, caplog):
        caplog.set_level(logging.INFO, logger="runtime.orchestrator")
        out = reconcile_confidence(
            envelope_value=0.83,
            tool_arg_value=0.85,
            agent="deep_investigator",
            session_id="INC-001",
            tool_name="submit_hypothesis",
        )
        assert out == 0.85  # tool-arg wins on the return value (D-10-03)
        # within tolerance → silent
        mismatch_logs = [
            r
            for r in caplog.records
            if "turn.confidence_mismatch" in r.getMessage()
        ]
        assert mismatch_logs == [], (
            f"expected silent on match within tolerance; got {[r.getMessage() for r in mismatch_logs]}"
        )

    def test_reconcile_mismatch_logs_and_tool_wins(self, caplog):
        caplog.set_level(logging.INFO, logger="runtime.orchestrator")
        out = reconcile_confidence(
            envelope_value=0.50,
            tool_arg_value=0.90,
            agent="deep_investigator",
            session_id="INC-002",
            tool_name="submit_hypothesis",
        )
        assert out == 0.90  # tool-arg wins
        # Find the mismatch log
        mismatch = [
            r.getMessage()
            for r in caplog.records
            if "turn.confidence_mismatch" in r.getMessage()
        ]
        assert len(mismatch) == 1
        msg = mismatch[0]
        assert "agent=deep_investigator" in msg
        assert "turn_value=0.50" in msg
        assert "tool_value=0.90" in msg
        assert "tool=submit_hypothesis" in msg
        assert "session_id=INC-002" in msg

    def test_reconcile_no_tool_arg_returns_envelope(self, caplog):
        caplog.set_level(logging.INFO, logger="runtime.orchestrator")
        out = reconcile_confidence(
            envelope_value=0.66,
            tool_arg_value=None,
            agent="triage",
            session_id="INC-003",
            tool_name=None,
        )
        assert out == 0.66
        mismatch = [
            r for r in caplog.records if "turn.confidence_mismatch" in r.getMessage()
        ]
        assert mismatch == []

    def test_reconcile_at_tolerance_boundary_silent(self, caplog):
        # |0.85 - 0.80| == 0.05 exactly → boundary inclusive → silent
        caplog.set_level(logging.INFO, logger="runtime.orchestrator")
        out = reconcile_confidence(
            envelope_value=0.80,
            tool_arg_value=0.85,
            agent="deep_investigator",
            session_id="INC-004",
            tool_name="submit_hypothesis",
        )
        assert out == 0.85
        mismatch = [
            r for r in caplog.records if "turn.confidence_mismatch" in r.getMessage()
        ]
        assert mismatch == [], "boundary 0.05 must be inclusive (no log)"


# ---------------------------------------------------------------------------
# 3) Parser fallback (3-step)
# ---------------------------------------------------------------------------


class TestParseEnvelopeFromResult:
    def test_parse_envelope_from_structured_response(self):
        env = AgentTurnOutput(
            content="hello",
            confidence=0.9,
            confidence_rationale="r",
            signal=None,
        )
        result = {"messages": [AIMessage(content="ignored")], "structured_response": env}
        parsed = parse_envelope_from_result(result, agent="triage")
        assert parsed is env

    def test_parse_envelope_from_last_aimessage_json(self):
        # No structured_response key — fall back to JSON-parse last AIMessage
        payload = {
            "content": "from-json",
            "confidence": 0.7,
            "confidence_rationale": "json fallback",
            "signal": "success",
        }
        result = {"messages": [AIMessage(content=json.dumps(payload))]}
        parsed = parse_envelope_from_result(result, agent="intake")
        assert parsed.content == "from-json"
        assert parsed.confidence == 0.7
        assert parsed.signal == "success"

    def test_parse_envelope_missing_raises_envelope_missing_error(self):
        # No structured_response, AIMessage content is not JSON
        result = {"messages": [AIMessage(content="just plain text, no JSON here")]}
        with pytest.raises(EnvelopeMissingError) as excinfo:
            parse_envelope_from_result(result, agent="supervisor")
        assert excinfo.value.agent == "supervisor"
        assert excinfo.value.field  # non-empty


# ---------------------------------------------------------------------------
# 4) All six agent kinds emit envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_kind",
    [
        "intake",
        "triage",
        "deep_investigator",
        "resolution",
        "supervisor",
        "monitor",
    ],
)
def test_all_six_agent_kinds_emit_envelope(agent_kind):
    """Each agent kind, when handed a structured_response, parses it back."""
    from tests._envelope_helpers import envelope_stub

    result = envelope_stub(
        content=f"{agent_kind} ran",
        confidence=0.82,
        rationale=f"{agent_kind} stub rationale",
        signal=None,
    )
    env = parse_envelope_from_result(result, agent=agent_kind)
    assert env.confidence == 0.82
    assert env.confidence_rationale == f"{agent_kind} stub rationale"
    assert env.content == f"{agent_kind} ran"
