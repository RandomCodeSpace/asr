"""P9-9m-sliver — UI panel sanity checks.

Two non-Streamlit tests:

1. The ui.py module imports cleanly outside a Streamlit runtime — no
   side-effecting top-level code that needs a script run context.
2. The pure helpers (``_is_hypothesis_trail`` /
   ``_render_pending_approvals_block`` / ``_render_hypothesis_trail_block``)
   are addressable on the module — tests don't actually drive
   Streamlit, but importing without exception + presence of the panel
   helpers is the verification target.
"""
from __future__ import annotations


def test_ui_module_imports_cleanly() -> None:
    """``import examples.incident_management.ui`` must succeed headlessly."""
    import examples.incident_management.ui as ui  # noqa: F401
    # The renderer for the new panel exists.
    assert hasattr(ui, "_render_hypothesis_trail_block")
    # The Approval Inbox renderer (P4-H) is still present after the
    # P9-9m additions.
    assert hasattr(ui, "_render_pending_approvals_block")


def test_is_hypothesis_trail_recognises_triage_trail_shape() -> None:
    from examples.incident_management.ui import _is_hypothesis_trail

    trail = [
        {"iteration": 0, "hypothesis": "deploy caused it", "score": 0.4,
         "rationale": "1/3 terms matched"},
        {"iteration": 1, "hypothesis": "deploy caused it (refined)",
         "score": 0.8, "rationale": "all terms matched"},
    ]
    assert _is_hypothesis_trail(trail) is True


def test_is_hypothesis_trail_rejects_other_shapes() -> None:
    from examples.incident_management.ui import _is_hypothesis_trail

    # Empty / wrong type.
    assert _is_hypothesis_trail([]) is False
    assert _is_hypothesis_trail(None) is False
    assert _is_hypothesis_trail("not a list") is False
    # List of strings.
    assert _is_hypothesis_trail(["a", "b"]) is False
    # Wrong key set (legacy hypothesis list with ``cause`` only).
    assert _is_hypothesis_trail([{"cause": "x", "evidence": []}]) is False
    # Has ``iteration`` but no ``hypothesis``.
    assert _is_hypothesis_trail([{"iteration": 0, "score": 0.5}]) is False
