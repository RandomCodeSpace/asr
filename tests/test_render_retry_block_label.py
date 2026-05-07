"""Phase 12 (FOC-05) -- targeted unit test for the 5-case label/disabled
selection in ``_render_retry_block``. Avoids spinning up a full
Streamlit harness by exercising the pure helper extracted from the
render-block: ``_retry_button_state_for(reason, retry_count, cap,
last_confidence, threshold) -> (label, disabled)``.

Pins the D-12-04 mapping:

  auto_retry              -> enabled, "Retry"
  max_retries_exceeded    -> disabled, "Max retries reached (rc/cap)"
  permanent_error         -> disabled, "Permanent error -- cannot auto-retry"
  low_confidence_no_retry -> disabled, "Confidence too low (N% < th%)"
  transient_disabled      -> disabled, "Auto-retry disabled in policy"
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "reason,expect_disabled,label_substr",
    [
        ("auto_retry", False, "Retry"),
        ("max_retries_exceeded", True, "Max retries"),
        ("permanent_error", True, "Permanent error"),
        ("low_confidence_no_retry", True, "Confidence too low"),
        ("transient_disabled", True, "disabled in policy"),
    ],
)
def test_retry_button_state_for_reason(
    reason, expect_disabled, label_substr,
):
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason=reason, retry_count=1, cap=2,
        last_confidence=0.2, threshold=0.4,
    )
    assert disabled is expect_disabled, (reason, label, disabled)
    assert label_substr in label, (reason, label)


def test_retry_button_state_for_unknown_reason_disables():
    """Future-proof: a never-before-seen reason (e.g. a v1.3 addition
    not yet wired into the UI) renders as disabled with a fallback
    label that includes the reason verbatim, so the user has at least
    a clue about the policy-side decision.
    """
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="some_future_reason", retry_count=0, cap=2,
        last_confidence=None, threshold=0.4,
    )
    assert disabled is True
    assert "some_future_reason" in label


def test_retry_button_state_for_max_retries_includes_count():
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="max_retries_exceeded", retry_count=2, cap=2,
        last_confidence=0.9, threshold=0.4,
    )
    assert disabled is True
    assert "2/2" in label


def test_retry_button_state_for_low_confidence_formats_percentages():
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="low_confidence_no_retry", retry_count=0, cap=2,
        last_confidence=0.2, threshold=0.4,
    )
    assert disabled is True
    assert "20%" in label
    assert "40%" in label


def test_retry_button_state_for_low_confidence_handles_none_conf():
    """If last_confidence is missing, the label falls back to a "?"
    placeholder so the message stays readable.
    """
    from runtime.ui import _retry_button_state_for
    label, disabled = _retry_button_state_for(
        reason="low_confidence_no_retry", retry_count=0, cap=2,
        last_confidence=None, threshold=0.4,
    )
    assert disabled is True
    assert "?" in label
    assert "40%" in label
