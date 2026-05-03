import pytest
from runtime.similarity import KeywordSimilarity, find_similar


def test_keyword_overlap_score():
    sim = KeywordSimilarity()
    s = sim.score("api latency spike production", "api latency p99 production")
    assert 0.4 < s < 1.0
    assert sim.score("a b c", "x y z") == pytest.approx(0.0)
    assert sim.score("identical text here", "identical text here") == pytest.approx(1.0)


def test_find_similar_returns_threshold_passing_only():
    sim = KeywordSimilarity()
    candidates = [
        {"id": "INC-1", "text": "api latency spike production"},
        {"id": "INC-2", "text": "totally different topic"},
        {"id": "INC-3", "text": "api latency p99 prod"},
    ]
    results = find_similar(
        query="api latency production",
        candidates=candidates,
        text_field="text",
        scorer=sim,
        threshold=0.4,
        limit=5,
    )
    assert {r["id"] for r, _ in results}.issubset({"INC-1", "INC-3"})
    assert all(score >= 0.4 for _, score in results)
    # Sorted descending by score
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
