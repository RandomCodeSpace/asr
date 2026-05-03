"""L7 Playbook Store.

Pin tests for ``PlaybookStore``. Cover seed fallback, YAML schema
parsing, the case-insensitive signal matcher, score ranking, and the
:class:`L7PlaybookSuggestion` shape ready for
``IncidentState.memory.l7_playbooks``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from examples.incident_management.asr.memory_state import L7PlaybookSuggestion
from examples.incident_management.asr.playbook_store import PlaybookStore


# ---- Seed fallback ------------------------------------------------------


def test_seed_fallback_loads_when_root_empty(tmp_path: Path) -> None:
    store = PlaybookStore(tmp_path)
    ids = {p["id"] for p in store.list_all()}
    assert {"pb-payments-latency", "pb-ledger-error-rate"} <= ids


def test_seed_payments_playbook_has_required_fields(tmp_path: Path) -> None:
    store = PlaybookStore(tmp_path)
    pb = store.get("pb-payments-latency")
    assert pb is not None
    assert pb["title"]
    assert pb["match_signals"]["service"] == "payments"
    assert isinstance(pb["hypothesis_steps"], list)
    assert isinstance(pb["remediation"], list)
    assert pb["required_approval"] is True


# ---- Custom playbook fixture -------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> PlaybookStore:
    (tmp_path / "pb-a.yaml").write_text(
        """\
id: pb-a
title: "Service A latency"
match_signals:
  service: A
  metric: p99_latency
  threshold_breach: true
hypothesis_steps: ["check A"]
remediation:
  - tool: noop
    args: {}
required_approval: false
"""
    )
    (tmp_path / "pb-b.yaml").write_text(
        """\
id: pb-b
title: "Service B errors"
match_signals:
  service: B
  metric: error_rate
hypothesis_steps: ["check B"]
remediation: []
required_approval: true
"""
    )
    # Playbook with no match_signals — should be dropped from match().
    (tmp_path / "pb-empty.yaml").write_text(
        """\
id: pb-empty
title: "No signals"
match_signals: {}
hypothesis_steps: []
remediation: []
"""
    )
    # Malformed file — must not crash loader.
    (tmp_path / "garbage.yaml").write_text("::: not valid yaml :::\n - [")
    # Non-yaml file should be ignored.
    (tmp_path / "README.md").write_text("ignored")
    return PlaybookStore(tmp_path)


def test_custom_root_overrides_seed(store: PlaybookStore) -> None:
    ids = {p["id"] for p in store.list_all()}
    # Seed playbooks must NOT leak in once the user supplies their own.
    assert "pb-payments-latency" not in ids
    assert {"pb-a", "pb-b", "pb-empty"} <= ids


def test_get_returns_none_for_missing(store: PlaybookStore) -> None:
    assert store.get("does-not-exist") is None


def test_match_full_score(store: PlaybookStore) -> None:
    out = store.match({
        "service": "A",
        "metric": "p99_latency",
        "threshold_breach": True,
    })
    assert len(out) >= 1
    top = out[0]
    assert isinstance(top, L7PlaybookSuggestion)
    assert top.playbook_id == "pb-a"
    assert top.score == 1.0
    assert "service=A" in top.matched_signals


def test_match_partial_score(store: PlaybookStore) -> None:
    out = store.match({"service": "A"})
    # pb-a has 3 declared signals; only ``service`` matches => 1/3.
    pb_a = next(s for s in out if s.playbook_id == "pb-a")
    assert pb_a.score == pytest.approx(1 / 3)


def test_match_case_insensitive(store: PlaybookStore) -> None:
    out = store.match({"SERVICE": "a", "metric": "P99_LATENCY"})
    pb_a = next(s for s in out if s.playbook_id == "pb-a")
    # 2 of 3 declared signals matched.
    assert pb_a.score == pytest.approx(2 / 3)


def test_match_boolean_normalised(store: PlaybookStore) -> None:
    """``True``/``"true"`` should compare equal."""
    out = store.match({
        "service": "A",
        "metric": "p99_latency",
        "threshold_breach": "true",
    })
    pb_a = next(s for s in out if s.playbook_id == "pb-a")
    assert pb_a.score == 1.0


# ---- Boolean coercion regression (Gemini finding) ----------------------


def test_match_python_true_against_yaml_true(store: PlaybookStore) -> None:
    """Regression: Python ``True`` in signals must match YAML ``true`` in
    ``match_signals``. Before the fix, ``_normalise`` returned ``"True"``
    for the bool and ``"true"`` for the YAML scalar, so equality failed."""
    out = store.match({
        "service": "A",
        "metric": "p99_latency",
        "threshold_breach": True,  # Python bool, NOT a string
    })
    pb_a = next(s for s in out if s.playbook_id == "pb-a")
    assert pb_a.score == 1.0
    assert "threshold_breach=True" in pb_a.matched_signals


def test_match_python_false_against_yaml_false(tmp_path: Path) -> None:
    """``False`` must normalise to ``"false"`` for symmetric comparison."""
    (tmp_path / "pb-quiet.yaml").write_text(
        """\
id: pb-quiet
title: "Quiet hours"
match_signals:
  service: payments
  off_hours: false
hypothesis_steps: []
remediation: []
required_approval: false
"""
    )
    s = PlaybookStore(tmp_path)
    out = s.match({"service": "payments", "off_hours": False})
    pb = next(x for x in out if x.playbook_id == "pb-quiet")
    assert pb.score == 1.0


def test_match_mixed_string_and_bool_signals(tmp_path: Path) -> None:
    """Mixed dict — string scalar AND Python bool — must both match."""
    (tmp_path / "pb-mix.yaml").write_text(
        """\
id: pb-mix
title: "Mixed signals"
match_signals:
  environment: production
  high_traffic: true
hypothesis_steps: []
remediation: []
required_approval: false
"""
    )
    s = PlaybookStore(tmp_path)
    out = s.match({"environment": "production", "high_traffic": True})
    pb = next(x for x in out if x.playbook_id == "pb-mix")
    assert pb.score == 1.0
    assert "environment=production" in pb.matched_signals
    assert "high_traffic=True" in pb.matched_signals


def test_match_drops_zero_score(store: PlaybookStore) -> None:
    out = store.match({"service": "Z"})
    assert out == []


def test_match_drops_playbooks_with_no_signals(store: PlaybookStore) -> None:
    out = store.match({"service": "A", "metric": "p99_latency"})
    assert all(s.playbook_id != "pb-empty" for s in out)


def test_match_ranks_by_score_desc_then_id(store: PlaybookStore) -> None:
    out = store.match({
        "service": "A",
        "metric": "p99_latency",
        "threshold_breach": True,
    })
    # pb-a should come first (full match); pb-b should not appear at all
    # since none of its declared signals (service=B, metric=error_rate)
    # are present in the query.
    assert out[0].playbook_id == "pb-a"
    assert all(s.playbook_id != "pb-b" for s in out)


def test_match_empty_signals_returns_empty(store: PlaybookStore) -> None:
    assert store.match({}) == []
