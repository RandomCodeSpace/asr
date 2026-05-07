"""Phase 20 (HARD-09): UI tests for error / display formatting.

Targets:
  * ``_parse_iso``           — defensive ISO parser
  * ``_duration_seconds``    — duration math with bad inputs
  * ``_fmt_tokens`` / ``_fmt_tokens_short``
  * ``_fmt_duration``        — human-readable durations
  * ``_fmt_confidence_badge``— confidence-tier glyph + label

These are the value-formatting rails the entire detail pane runs
through. Pure functions; small but load-bearing.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


def test_parse_iso_returns_datetime_for_valid_z_suffix() -> None:
    from runtime.ui import _parse_iso
    out = _parse_iso("2026-05-07T10:30:45Z")
    assert out is not None
    assert (out.year, out.month, out.day, out.hour, out.minute) == (
        2026, 5, 7, 10, 30,
    )


@pytest.mark.parametrize("bad", [
    "", None, "not-a-date", "2026-13-99", "2026-05-07 10:30:45",
])
def test_parse_iso_returns_none_for_garbage(bad) -> None:
    from runtime.ui import _parse_iso
    assert _parse_iso(bad) is None


# ---------------------------------------------------------------------------
# _duration_seconds
# ---------------------------------------------------------------------------


def test_duration_seconds_simple_minute() -> None:
    from runtime.ui import _duration_seconds
    out = _duration_seconds("2026-05-07T10:00:00Z", "2026-05-07T10:01:00Z")
    assert out == 60


def test_duration_seconds_returns_zero_when_either_side_unparseable() -> None:
    from runtime.ui import _duration_seconds
    assert _duration_seconds("", "2026-05-07T10:00:00Z") == 0
    assert _duration_seconds("2026-05-07T10:00:00Z", "garbage") == 0
    assert _duration_seconds("garbage", "garbage") == 0


def test_duration_seconds_clamps_negative_to_zero() -> None:
    """End before start (clock skew) → 0, never a negative duration."""
    from runtime.ui import _duration_seconds
    out = _duration_seconds("2026-05-07T10:01:00Z", "2026-05-07T10:00:00Z")
    assert out == 0


# ---------------------------------------------------------------------------
# _fmt_tokens / _fmt_tokens_short
# ---------------------------------------------------------------------------


def test_fmt_tokens_uses_thousands_separators() -> None:
    from runtime.ui import _fmt_tokens
    assert _fmt_tokens(0) == "0"
    assert _fmt_tokens(999) == "999"
    assert _fmt_tokens(12_345) == "12,345"
    assert _fmt_tokens(1_234_567) == "1,234,567"


def test_fmt_tokens_short_compact_form() -> None:
    from runtime.ui import _fmt_tokens_short
    assert _fmt_tokens_short(0) == "0"
    assert _fmt_tokens_short(999) == "999"
    assert _fmt_tokens_short(1000) == "1.0k"
    assert _fmt_tokens_short(12_345) == "12.3k"


# ---------------------------------------------------------------------------
# _fmt_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"),
    (42, "42s"),
    (60, "1m 0s"),
    (185, "3m 5s"),
    (3600, "1h 0m"),
    (3720, "1h 2m"),
    (86_400, "1d 0h"),
    (90_000, "1d 1h"),
])
def test_fmt_duration_compacts_to_two_units(seconds: int, expected: str) -> None:
    from runtime.ui import _fmt_duration
    assert _fmt_duration(seconds) == expected


# ---------------------------------------------------------------------------
# _fmt_confidence_badge
# ---------------------------------------------------------------------------


def test_fmt_confidence_badge_none_renders_hard_error_marker() -> None:
    """Phase 10 (FOC-03): a missing envelope ⇒ structural failure ⇒
    distinct red badge — never the silent ⚪ fallback.
    """
    from runtime.ui import _fmt_confidence_badge
    out = _fmt_confidence_badge(None)
    assert "missing" in out.lower()
    # Sanity: not a green/amber glyph
    assert "🟢" not in out
    assert "🟡" not in out


def test_fmt_confidence_badge_high_is_green() -> None:
    from runtime.ui import _fmt_confidence_badge
    out = _fmt_confidence_badge(0.95)
    assert "🟢" in out
    assert "0.95" in out


def test_fmt_confidence_badge_amber_band() -> None:
    """0.5 ≤ conf < 0.75 → amber/yellow."""
    from runtime.ui import _fmt_confidence_badge
    assert "🟡" in _fmt_confidence_badge(0.5)
    assert "🟡" in _fmt_confidence_badge(0.74)


def test_fmt_confidence_badge_low_is_red() -> None:
    from runtime.ui import _fmt_confidence_badge
    out = _fmt_confidence_badge(0.10)
    assert "🔴" in out
    assert "0.10" in out


# ---------------------------------------------------------------------------
# _is_hypothesis_list — defensive type guard
# ---------------------------------------------------------------------------


def test_is_hypothesis_list_recognises_cause_keyed_dicts() -> None:
    from runtime.ui import _is_hypothesis_list
    assert _is_hypothesis_list([{"cause": "deploy", "evidence": []}]) is True


def test_is_hypothesis_list_rejects_non_lists_and_wrong_shapes() -> None:
    from runtime.ui import _is_hypothesis_list
    assert _is_hypothesis_list(None) is False
    assert _is_hypothesis_list([]) is False
    assert _is_hypothesis_list("not a list") is False
    assert _is_hypothesis_list([{"hypothesis": "no cause key"}]) is False
    assert _is_hypothesis_list([1, 2, 3]) is False
