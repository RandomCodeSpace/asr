"""Similarity scoring for session matching."""
from __future__ import annotations
from typing import Protocol
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "the", "of", "in", "on", "to", "and", "or", "is", "was", "with", "for",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1}


class Scorer(Protocol):
    def score(self, a: str, b: str) -> float: ...


class KeywordSimilarity:
    """Jaccard overlap on tokenized text. Cheap, deterministic."""

    def score(self, a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


def find_similar(*, query: str, candidates: list[dict], text_field: str,
                 scorer: Scorer, threshold: float, limit: int) -> list[tuple[dict, float]]:
    scored = [(c, scorer.score(query, c[text_field])) for c in candidates]
    passing = [(c, s) for c, s in scored if s >= threshold]
    passing.sort(key=lambda x: x[1], reverse=True)
    return passing[:limit]
