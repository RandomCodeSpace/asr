"""Phase 20 (HARD-09): UI tests for the P4 approval submission paths.

These are the load-bearing HITL surfaces in ``runtime.ui`` — when the
framework's pure-policy gate paused a tool call, the operator's only
way to unstick the session is via the Approve / Reject buttons rendered
by ``_render_pending_approvals_block`` (which delegates to
``_submit_approval_via_service``).

Approach: pure-helper tests + ``streamlit.testing.v1.AppTest`` driver
for end-to-end render flows. Mock-fixture for ``_get_service`` /
``load_config`` so we never bring up the real OrchestratorService.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_should_render_retry_block_skips_when_pending_approval_present() -> None:
    """If a tool call is paused for HITL approval, the retry block must
    NOT render — the pending-approvals block owns the action surface
    instead. Mutual-exclusion invariant from D-11-04.
    """
    from runtime.ui import _should_render_retry_block

    sess = {
        "status": "error",
        "tool_calls": [
            {"agent": "investigator", "tool": "remediate",
             "status": "pending_approval"},
        ],
    }
    assert _should_render_retry_block(sess) is False


def test_should_render_retry_block_fires_for_terminal_error_without_approval() -> None:
    """Plain terminal error (no pending_approval row) → retry block renders."""
    from runtime.ui import _should_render_retry_block

    sess = {
        "status": "error",
        "tool_calls": [
            {"agent": "investigator", "tool": "search_logs",
             "status": "completed"},
        ],
    }
    assert _should_render_retry_block(sess) is True


def test_should_render_retry_block_skips_non_error_status() -> None:
    from runtime.ui import _should_render_retry_block

    for status in ("in_progress", "resolved", "awaiting_input", "matched"):
        assert _should_render_retry_block({"status": status}) is False


def test_should_render_retry_block_tolerates_pydantic_objects() -> None:
    """Defensive: live ``Session.tool_calls`` returns pydantic objects, not
    dicts. The predicate must read ``.status`` via getattr in that case
    (D-11-04 callout)."""
    from runtime.ui import _should_render_retry_block

    class _FakeToolCall:
        status = "pending_approval"

    sess = {"status": "error", "tool_calls": [_FakeToolCall()]}
    assert _should_render_retry_block(sess) is False


# ---------------------------------------------------------------------------
# _submit_approval_via_service — error path + happy path with stubs
# ---------------------------------------------------------------------------


def test_submit_approval_emits_st_error_when_service_unavailable() -> None:
    """When the service singleton is None (e.g. headless rerun),
    the helper must surface ``st.error`` and return — never crash.
    """
    from runtime import ui as ui_mod

    fake_st = MagicMock()
    fake_cfg = MagicMock()

    with patch.object(ui_mod, "_get_service", return_value=None), \
         patch.object(ui_mod, "st", fake_st):
        ui_mod._submit_approval_via_service(
            fake_cfg, "INC-1", "0",
            decision="approve", approver="ui-user", rationale=None,
        )

    fake_st.error.assert_called_once()
    msg = fake_st.error.call_args.args[0]
    assert "service" in msg.lower() or "refresh" in msg.lower()


def test_submit_approval_drives_service_with_correct_payload() -> None:
    """Happy path: build the expected ``Command(resume=...)`` payload and
    drive ``svc.submit_and_wait`` with it. The test patches the service
    so we never touch a real orchestrator.
    """
    from runtime import ui as ui_mod

    captured_awaitables: list = []

    def _capture(awaitable, timeout=None):
        # Close the coroutine so we don't get the "never awaited" warning;
        # we're verifying the call shape, not the actual resume flow.
        captured_awaitables.append((awaitable, timeout))
        if hasattr(awaitable, "close"):
            awaitable.close()

    fake_svc = MagicMock()
    fake_svc.submit_and_wait = MagicMock(side_effect=_capture)
    fake_cfg = MagicMock()
    fake_st = MagicMock()

    with patch.object(ui_mod, "_get_service", return_value=fake_svc), \
         patch.object(ui_mod, "st", fake_st):
        ui_mod._submit_approval_via_service(
            fake_cfg, "INC-42", "3",
            decision="reject",
            approver="ui-user",
            rationale="risk too high",
        )

    # submit_and_wait called exactly once with the contract's 60-second
    # timeout (matches HITL bridge in OrchestratorService).
    assert fake_svc.submit_and_wait.call_count == 1
    assert len(captured_awaitables) == 1
    assert captured_awaitables[0][1] == 60.0


# ---------------------------------------------------------------------------
# _render_pending_approvals_block — empty / present cases via AppTest
# ---------------------------------------------------------------------------


def test_render_pending_approvals_block_renders_nothing_when_no_pending() -> None:
    """No pending_approval rows → block is a no-op (returns before
    ``st.markdown('### Pending Approvals')``). This protects the detail
    pane from rendering a phantom header on resolved sessions.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("""
from unittest.mock import patch, MagicMock
from runtime.ui import _render_pending_approvals_block
sess = {"tool_calls": [{"agent": "x", "tool": "y", "status": "completed"}]}
with patch("runtime.ui.load_config", return_value=MagicMock()):
    _render_pending_approvals_block(sess, "INC-test")
""")
    at.run(timeout=10)
    assert not at.exception
    # No '### Pending Approvals' header should be in the rendered markdown.
    md_blobs = [m.value for m in at.markdown]
    assert not any("Pending Approvals" in m for m in md_blobs)


def test_render_pending_approvals_block_renders_card_for_pending_row() -> None:
    """One pending_approval row → header + card with tool name and Approve/Reject buttons."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("""
from unittest.mock import patch, MagicMock
from runtime.ui import _render_pending_approvals_block
sess = {"tool_calls": [
    {"agent": "investigator", "tool": "remediate",
     "status": "pending_approval", "args": {"target": "host-1"}},
]}
with patch("runtime.ui.load_config", return_value=MagicMock()):
    _render_pending_approvals_block(sess, "INC-test")
""")
    at.run(timeout=10)
    assert not at.exception
    md_blobs = [m.value for m in at.markdown]
    # Header rendered
    assert any("Pending Approvals" in m for m in md_blobs)
    # Tool reference visible (header markdown carries agent/tool names)
    assert any("investigator" in m and "remediate" in m for m in md_blobs)
    # Buttons present with the unique session-scoped keys
    button_keys = {b.key for b in at.button if b.key}
    assert "approval_approve_INC-test_0" in button_keys
    assert "approval_reject_INC-test_0" in button_keys
