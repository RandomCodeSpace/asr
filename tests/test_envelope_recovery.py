"""Coverage tests for ``runtime.graph._try_recover_envelope_from_raw`` (graph.py:583-610).

The recovery helper is invoked by the agent runner when LangGraph's
structured-output pass raises ``OutputParserException`` — it tries
several candidate substrings to dig an :class:`AgentTurnOutput` out of
free-form LLM text. The function is pure and behaves identically for
every input, so a table-driven test suite pins each branch.
"""
from __future__ import annotations

import json

import pytest

from runtime.agents.turn_output import AgentTurnOutput
from runtime.graph import _try_recover_envelope_from_raw


def _envelope_dict(*, content: str = "ok", confidence: float = 0.85,
                   rationale: str = "stub", signal: str | None = None) -> dict:
    return {
        "content": content,
        "confidence": confidence,
        "confidence_rationale": rationale,
        "signal": signal,
    }


def _envelope_json(**overrides) -> str:
    return json.dumps(_envelope_dict(**overrides))


class TestEmptyInput:
    @pytest.mark.parametrize("raw", ["", "   ", "\n\n  \t\n"])
    def test_empty_or_whitespace_returns_none(self, raw):
        assert _try_recover_envelope_from_raw(raw) is None


class TestPlainJsonInput:
    def test_valid_envelope_json_parses(self):
        out = _try_recover_envelope_from_raw(_envelope_json(content="hello"))
        assert isinstance(out, AgentTurnOutput)
        assert out.content == "hello"

    def test_valid_envelope_with_signal(self):
        out = _try_recover_envelope_from_raw(_envelope_json(signal="reconcile"))
        assert out is not None
        assert out.signal == "reconcile"


class TestMarkdownFencedJson:
    def test_fenced_with_json_tag(self):
        raw = f"```json\n{_envelope_json()}\n```"
        out = _try_recover_envelope_from_raw(raw)
        assert isinstance(out, AgentTurnOutput)

    def test_fenced_without_json_tag(self):
        raw = f"```\n{_envelope_json(confidence=0.42)}\n```"
        out = _try_recover_envelope_from_raw(raw)
        assert out is not None
        assert out.confidence == 0.42

    def test_fenced_with_surrounding_chatter(self):
        raw = (
            "Here is my structured response:\n\n"
            f"```json\n{_envelope_json(content='fenced')}\n```\n\n"
            "Hope that helps!"
        )
        out = _try_recover_envelope_from_raw(raw)
        assert out is not None
        assert out.content == "fenced"


class TestGreedyBraceMatch:
    def test_chatter_then_json_then_chatter(self):
        # No fences — should fall through to the greedy first-{...-last-} scan.
        raw = (
            f"Sure, here's the answer: {_envelope_json(content='greedy')} "
            "Let me know if you need more!"
        )
        out = _try_recover_envelope_from_raw(raw)
        assert out is not None
        assert out.content == "greedy"


class TestUnrecoverableInput:
    def test_invalid_json_returns_none(self):
        assert _try_recover_envelope_from_raw("{not valid json}") is None

    def test_no_braces_returns_none(self):
        assert _try_recover_envelope_from_raw("Just a plain sentence.") is None

    def test_json_array_not_dict_returns_none(self):
        # Greedy match would still find a substring, but `[1, 2, 3]`
        # has no braces. Use a real array text.
        assert _try_recover_envelope_from_raw('["a", "b"]') is None

    def test_valid_dict_missing_required_fields_returns_none(self):
        # `{"foo": "bar"}` parses but fails AgentTurnOutput validation.
        assert _try_recover_envelope_from_raw('{"foo": "bar"}') is None

    def test_dict_with_invalid_field_types_returns_none(self):
        # confidence must be 0..1; this should fail validation on every candidate.
        assert _try_recover_envelope_from_raw(
            '{"content": "x", "confidence": 5.0, "confidence_rationale": "y"}'
        ) is None
