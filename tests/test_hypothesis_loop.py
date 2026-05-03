"""P9-9i — triage hypothesis-refinement helpers.

Pin tests for the deterministic primitives that gate the triage
agent's inner loop:

* :func:`score_hypothesis` — token-overlap heuristic.
* :func:`should_refine` — iteration / threshold predicate.

These tests must pass without an LLM; the boundary conditions are the
loop's safety net so we exercise them in isolation.
"""
from __future__ import annotations

import pytest

from examples.incident_management.asr.hypothesis_loop import (
    ACCEPT_THRESHOLD,
    MAX_ITERATIONS,
    score_hypothesis,
    should_refine,
)


# ---------------------------------------------------------------------------
# score_hypothesis
# ---------------------------------------------------------------------------


def test_score_full_overlap_is_one() -> None:
    out = score_hypothesis(
        "payments service p99 latency spike",
        ["payments service p99 latency spike post-deploy"],
    )
    assert out["score"] == 1.0
    assert "payments" in out["matched_terms"]


def test_score_no_overlap_is_zero() -> None:
    out = score_hypothesis(
        "payments service latency",
        ["billing connection refused"],
    )
    assert out["score"] == 0.0
    assert out["matched_terms"] == []


def test_score_partial_overlap_is_proportional() -> None:
    # Hypothesis tokens (after stop/short filter): {payments, service, p99, latency}
    # Evidence tokens: {payments, deploy, rollback}
    # Overlap: {payments} -> 1/4 = 0.25
    out = score_hypothesis(
        "payments service p99 latency",
        ["payments deploy rollback"],
    )
    assert out["score"] == pytest.approx(0.25, abs=1e-3)
    assert out["matched_terms"] == ["payments"]


def test_score_empty_hypothesis_is_zero() -> None:
    out = score_hypothesis("", ["anything"])
    assert out["score"] == 0.0
    assert "Empty hypothesis" in out["rationale"]


def test_score_empty_evidence_is_zero() -> None:
    out = score_hypothesis("payments slow", [])
    assert out["score"] == 0.0
    assert "No evidence" in out["rationale"]


def test_score_stopwords_filtered_out() -> None:
    # 'is' / 'the' / 'a' are stop words and do NOT count toward the
    # token total. Hypothesis non-stop tokens: {service, broken}.
    # Evidence: {broken}. Match: {broken} → 1/2 = 0.5.
    out = score_hypothesis(
        "the service is broken",
        ["broken broken broken"],
    )
    assert out["score"] == pytest.approx(0.5)
    # Neither 'is' nor 'the' is in matched_terms even if they happened
    # to appear in evidence: stopwords are removed pre-intersection.
    assert "is" not in out["matched_terms"]
    assert "the" not in out["matched_terms"]


def test_score_returns_sorted_matched_terms() -> None:
    out = score_hypothesis(
        "zeta alpha beta",
        ["alpha beta zeta"],
    )
    assert out["matched_terms"] == ["alpha", "beta", "zeta"]


# ---------------------------------------------------------------------------
# should_refine
# ---------------------------------------------------------------------------


def test_should_refine_below_threshold_below_cap() -> None:
    assert should_refine(score=0.4, iterations=0) is True
    assert should_refine(score=0.6, iterations=2) is True


def test_should_refine_at_threshold_stops() -> None:
    """Score == ACCEPT_THRESHOLD is acceptance, not refinement."""
    assert should_refine(score=ACCEPT_THRESHOLD, iterations=0) is False


def test_should_refine_above_threshold_stops() -> None:
    assert should_refine(score=0.95, iterations=0) is False


def test_should_refine_iteration_cap_stops() -> None:
    """Even a poor score must stop refining at the iteration cap."""
    assert should_refine(score=0.0, iterations=MAX_ITERATIONS) is False
    assert should_refine(score=0.5, iterations=MAX_ITERATIONS + 1) is False


def test_should_refine_clamps_negative_iterations() -> None:
    """Defensive: negative iteration counters are clamped to 0."""
    assert should_refine(score=0.0, iterations=-5) is True


def test_should_refine_clamps_score_above_one() -> None:
    """Defensive: score > 1.0 is treated as 1.0 (acceptance)."""
    assert should_refine(score=1.5, iterations=0) is False


def test_should_refine_clamps_score_below_zero() -> None:
    """Defensive: negative score is treated as 0.0 (still refine)."""
    assert should_refine(score=-0.2, iterations=0) is True
