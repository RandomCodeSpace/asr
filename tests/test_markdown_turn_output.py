"""Phase 22 — markdown turn output parser matrix.

Pure-function tests on parse_markdown_envelope. Covers the
section-by-section parse + the validation paths the framework relies
on (D-22-03 contract).

Pinning the contract here is the single point of compat truth — every
LLM that writes the documented ## Response / ## Confidence /
## Signal block round-trips through these tests verbatim.
"""
from __future__ import annotations

import pytest

from runtime.agents.turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
    parse_markdown_envelope,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_sections_present_em_dash_rationale():
    md = (
        "Some prose preamble that may or may not exist.\n\n"
        "## Response\n"
        "Looks like a deploy regression in payments-svc.\n\n"
        "## Confidence\n"
        "0.85 — strong correlation with v1.120 release timeline\n\n"
        "## Signal\n"
        "default\n"
    )
    env = parse_markdown_envelope(md, agent="triage")
    assert isinstance(env, AgentTurnOutput)
    assert env.content == "Looks like a deploy regression in payments-svc."
    assert env.confidence == 0.85
    assert env.confidence_rationale == (
        "strong correlation with v1.120 release timeline"
    )
    assert env.signal == "default"


def test_ascii_dash_rationale():
    md = (
        "## Response\nfoo\n"
        "## Confidence\n0.7 -- ok\n"
        "## Signal\nsuccess\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 0.7
    assert env.confidence_rationale == "ok"
    assert env.signal == "success"


def test_single_hyphen_rationale():
    md = (
        "## Response\nfoo\n"
        "## Confidence\n0.6 - some rationale\n"
        "## Signal\nfailed\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 0.6
    assert env.confidence_rationale == "some rationale"


# ---------------------------------------------------------------------------
# Confidence shape variants
# ---------------------------------------------------------------------------

def test_confidence_int_zero_clamps_in_range():
    md = (
        "## Response\nbody\n"
        "## Confidence\n0\n"
        "## Signal\nfailed\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 0.0
    # Missing rationale is replaced with placeholder.
    assert env.confidence_rationale == "(no rationale provided)"


def test_confidence_int_one():
    md = (
        "## Response\nbody\n"
        "## Confidence\n1\n"
        "## Signal\nsuccess\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 1.0


def test_confidence_above_range_clamps():
    """LLM occasionally emits 1.05 — clamp rather than reject."""
    md = (
        "## Response\nbody\n"
        "## Confidence\n1.05 — over\n"
        "## Signal\nsuccess\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 1.0


def test_confidence_below_range_clamps():
    md = (
        "## Response\nbody\n"
        "## Confidence\n-0.2 — under\n"
        "## Signal\nfailed\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 0.0


def test_confidence_unparseable_raises():
    md = (
        "## Response\nbody\n"
        "## Confidence\nhigh — strong\n"
        "## Signal\nsuccess\n"
    )
    with pytest.raises(EnvelopeMissingError) as exc:
        parse_markdown_envelope(md, agent="triage")
    assert exc.value.field == "confidence"
    assert "triage" in str(exc.value)


def test_confidence_section_missing_raises():
    md = (
        "## Response\nbody\n"
        "## Signal\nsuccess\n"
    )
    with pytest.raises(EnvelopeMissingError) as exc:
        parse_markdown_envelope(md)
    assert exc.value.field == "confidence"


# ---------------------------------------------------------------------------
# Signal vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "none", "None", "NULL", "  ", "n/a"])
def test_signal_blank_or_none_token_yields_none(blank: str):
    md = (
        "## Response\nbody\n"
        "## Confidence\n0.5 — mid\n"
        f"## Signal\n{blank}\n"
    )
    env = parse_markdown_envelope(md)
    assert env.signal is None


def test_signal_unknown_value_passed_through_lowercased():
    """Vocabulary validation lives in the routing layer, not the parser."""
    md = (
        "## Response\nbody\n"
        "## Confidence\n0.5 — mid\n"
        "## Signal\nWeirdSignal\n"
    )
    env = parse_markdown_envelope(md)
    assert env.signal == "weirdsignal"


def test_signal_section_missing_yields_none():
    md = (
        "## Response\nbody\n"
        "## Confidence\n0.4 — hedged\n"
    )
    env = parse_markdown_envelope(md)
    assert env.signal is None


# ---------------------------------------------------------------------------
# Response body shape
# ---------------------------------------------------------------------------

def test_multi_line_response_with_code_block():
    md = (
        "## Response\n"
        "Here is the fix:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Run pytest after applying.\n\n"
        "## Confidence\n"
        "0.92 — verified locally\n\n"
        "## Signal\n"
        "success\n"
    )
    env = parse_markdown_envelope(md)
    assert "```python" in env.content
    assert "Run pytest" in env.content
    assert env.confidence == 0.92


def test_response_section_missing_raises():
    md = (
        "## Confidence\n0.7 — ok\n"
        "## Signal\nsuccess\n"
    )
    with pytest.raises(EnvelopeMissingError) as exc:
        parse_markdown_envelope(md)
    assert exc.value.field == "content"


def test_empty_content_raises():
    with pytest.raises(EnvelopeMissingError) as exc:
        parse_markdown_envelope("")
    assert exc.value.field == "content"


# ---------------------------------------------------------------------------
# Header drift: H3 / lowercase
# ---------------------------------------------------------------------------

def test_h3_headers_accepted():
    """Watch-out: model may emit ### instead of ##. Parser accepts ##+."""
    md = (
        "### Response\nbody\n"
        "### Confidence\n0.5 — mid\n"
        "### Signal\ndefault\n"
    )
    env = parse_markdown_envelope(md)
    assert env.content == "body"
    assert env.confidence == 0.5
    assert env.signal == "default"


def test_lowercase_headers_accepted():
    md = (
        "## response\nbody\n"
        "## confidence\n0.5 — mid\n"
        "## signal\nsuccess\n"
    )
    env = parse_markdown_envelope(md)
    assert env.confidence == 0.5
    assert env.signal == "success"


# ---------------------------------------------------------------------------
# parse_envelope_from_result Path 4 integration
# ---------------------------------------------------------------------------

def _ai(content: str):
    """Construct an object that quacks like an AIMessage for path-4
    class-name walker."""
    obj = type("AIMessage", (), {})()
    obj.content = content  # type: ignore[attr-defined]
    return obj


def test_parse_envelope_from_result_falls_through_to_path_4_markdown():
    """No structured_response, no JSON in messages — Path 4 wins."""
    md = (
        "## Response\nbody from path 4\n"
        "## Confidence\n0.8 — ok\n"
        "## Signal\nsuccess\n"
    )
    result = {"messages": [_ai(md)], "structured_response": None}
    env = parse_envelope_from_result(result, agent="triage")
    assert env.content == "body from path 4"
    assert env.confidence == 0.8
    assert env.signal == "success"


def test_parse_envelope_from_result_path_1_still_wins_when_present():
    """Markdown parser doesn't shadow the structured_response path."""
    md = (
        "## Response\nshould not be picked\n"
        "## Confidence\n0.1 — wrong\n"
        "## Signal\nfailed\n"
    )
    sr = AgentTurnOutput(
        content="from structured_response",
        confidence=0.99,
        confidence_rationale="strong",
        signal="success",
    )
    result = {"messages": [_ai(md)], "structured_response": sr}
    env = parse_envelope_from_result(result, agent="triage")
    assert env.content == "from structured_response"
    assert env.confidence == 0.99


def test_parse_envelope_from_result_raises_when_no_path_yields():
    """Empty messages + no structured_response + no markdown → raises."""
    result = {"messages": [_ai("just prose, no sections")]}
    with pytest.raises(EnvelopeMissingError):
        parse_envelope_from_result(result, agent="triage")
