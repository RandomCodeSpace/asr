"""ASR triage hypothesis-refinement loop helpers (P9-9i).

The triage agent runs an iterative pattern: generate a hypothesis →
gather evidence (L1 current findings, L3-equivalent past similar
incidents, L5 recent releases) → score → refine OR accept. The loop
is bounded so a stuck hypothesis doesn't spin forever.

This module ships the *deterministic* primitives that gate the loop:

* :func:`score_hypothesis` — token-overlap heuristic. Pure, no LLM.
  Returns a normalised score in ``[0.0, 1.0]`` plus a one-sentence
  rationale. Tests can assert exact behaviour.

* :func:`should_refine` — boolean decision based on the current score
  and the iteration counter. Refines while score < 0.7 AND
  iterations < 3.

The agent's LLM-driven generation step (the *hypothesis* itself) lives
in the system prompt at ``skills/triage/system.md``; only the scoring
and continue/stop predicates are deterministic Python so the loop's
boundary conditions are exercised in unit tests without spinning the
LLM.

Design note: tokenisation mirrors :mod:`runtime.similarity` — same
regex, same stopword list — so a hypothesis containing service names
and timing words ranks consistent with the dedup pipeline's notion of
"similar".
"""
from __future__ import annotations

import re
from typing import TypedDict


# Loop bounds (locked per P9-9i plan).
MAX_ITERATIONS: int = 3
ACCEPT_THRESHOLD: float = 0.7


# Mirror runtime.similarity's tokenisation so the score's notion of
# "overlap" matches the dedup / lookup_similar_incidents path.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "to", "and", "or",
    "is", "was", "with", "for", "be", "are", "as", "at",
    "by", "from", "has", "had", "have", "it", "that", "this",
    "we", "i", "you", "they", "but", "not", "no", "if",
})


class HypothesisScore(TypedDict):
    """Result returned by :func:`score_hypothesis`."""

    score: float
    rationale: str
    matched_terms: list[str]


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOP and len(t) > 1
    }


def score_hypothesis(
    hypothesis: str,
    evidence: list[str],
) -> HypothesisScore:
    """Score how well ``evidence`` supports ``hypothesis``.

    Token-overlap heuristic. The score is the fraction of hypothesis
    tokens that appear in *any* evidence string, capped at 1.0.

    - Empty hypothesis -> ``0.0`` (defensive; the LLM should never
      produce an empty hypothesis but the loop must not crash).
    - Empty evidence list -> ``0.0`` (no support).
    - Score is always in ``[0.0, 1.0]`` inclusive.

    Returns a :class:`HypothesisScore` with the score, a short
    machine-generated rationale, and the matched tokens (handy for
    rendering in the UI's hypothesis trail).
    """
    h_tokens = _tokens(hypothesis)
    if not h_tokens:
        return HypothesisScore(
            score=0.0,
            rationale="Empty hypothesis — no tokens to score against evidence.",
            matched_terms=[],
        )
    if not evidence:
        return HypothesisScore(
            score=0.0,
            rationale=f"No evidence supplied to support {len(h_tokens)} hypothesis terms.",
            matched_terms=[],
        )

    e_tokens: set[str] = set()
    for snippet in evidence:
        e_tokens |= _tokens(snippet)

    matched = h_tokens & e_tokens
    score = len(matched) / len(h_tokens)
    # Round to 3 dp to keep the audit trail readable; score is still a
    # float for callers that want a tighter comparison.
    score = round(score, 3)

    rationale = (
        f"Matched {len(matched)}/{len(h_tokens)} hypothesis terms "
        f"in {len(evidence)} evidence snippets."
    )
    return HypothesisScore(
        score=score,
        rationale=rationale,
        matched_terms=sorted(matched),
    )


def should_refine(score: float, iterations: int) -> bool:
    """Loop-control predicate: True when the agent should refine again.

    Refines while:

    * the current score is below :data:`ACCEPT_THRESHOLD`, AND
    * the iteration count is strictly less than :data:`MAX_ITERATIONS`.

    The iteration counter is the number of *completed* rounds — so
    ``should_refine(score=0.5, iterations=0)`` returns ``True`` (we've
    done 0 rounds, want at least 1), ``should_refine(score=0.5,
    iterations=3)`` returns ``False`` (cap hit).

    Defensive on bad inputs: negative iterations are clamped to 0;
    out-of-range scores are clamped to ``[0.0, 1.0]``.
    """
    if score is None:
        return iterations < MAX_ITERATIONS
    s = max(0.0, min(1.0, float(score)))
    n = max(0, int(iterations))
    return s < ACCEPT_THRESHOLD and n < MAX_ITERATIONS


__all__ = [
    "ACCEPT_THRESHOLD",
    "HypothesisScore",
    "MAX_ITERATIONS",
    "score_hypothesis",
    "should_refine",
]
