"""End-to-end regression test for the typed-terminal flow.

The pre-remediation bug: when an agent finished without calling a
terminal tool, ``_finalize_session_status`` blind-coerced the session
to ``status="resolved"`` with ``auto_resolved=True``. This silently
masked stuck/escalated sessions as resolved.

Post Task 4.1, the same end state must land at ``needs_review`` with
``needs_review_reason`` set. This test pins that contract through the
full Orchestrator startup → start_investigation → finalize stack.

Note: a companion "happy path" test (LLM calls mark_resolved → status
becomes resolved) is covered by the unit tests in
``tests/test_finalize_status_inference.py`` and
``tests/test_harvester_typed.py``. We don't duplicate it here.
"""
from __future__ import annotations

import pytest

from runtime.config import LLMConfig, RuntimeConfig, load_config
from runtime.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_finalize_on_real_orchestrator_lands_at_needs_review(
    tmp_path, monkeypatch,
):
    """Full Orchestrator.create() boots; a session with no terminal
    tool calls in its history then finalizes via the real async path.
    Must land at needs_review (not silently coerce to resolved).

    This exercises the WHOLE startup stack (MCP load, skill validator,
    checkpoint GC, lock registry) plus the lock-guarded async finalize
    against a real store — coverage the unit tests in
    ``test_finalize_status_inference.py`` deliberately bypass.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "noop")
    monkeypatch.setenv("AZURE_ENDPOINT", "noop")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "noop")
    monkeypatch.setenv("EXTERNAL_MCP_URL", "noop")
    monkeypatch.setenv("EXT_TOKEN", "noop")

    cfg = load_config("config/config.yaml.example")
    cfg.paths.incidents_dir = str(tmp_path)
    cfg.llm = LLMConfig.stub()
    cfg.runtime = RuntimeConfig(state_class=None)

    orch = await Orchestrator.create(cfg)
    try:
        # Bypass start_investigation (which would route through the
        # full graph and likely pause at HITL gates). We just need a
        # session in the store with status=in_progress and an empty
        # tool_calls history — the very shape that pre-remediation
        # would have been silently coerced to "resolved".
        inc = orch.store.create(
            query="some open investigation",
            environment="staging",
            reporter_id="u",
            reporter_team="t",
        )
        inc.status = "in_progress"
        orch.store.save(inc)

        new_status = await orch._finalize_session_status_async(inc.id)
        assert new_status == "needs_review", (
            f"expected needs_review, got {new_status!r}; "
            f"pre-remediation bug coerced this to 'resolved'"
        )

        fresh = orch.store.load(inc.id)
        assert fresh.status == "needs_review"
        assert fresh.extra_fields.get("needs_review_reason"), (
            "needs_review_reason must be set so operators see why"
        )
        assert "without terminal tool call" in fresh.extra_fields["needs_review_reason"]
        # Legacy auto_resolved must NOT be written.
        assert not fresh.extra_fields.get("auto_resolved"), (
            "auto_resolved was the pre-remediation sentinel; new "
            "sessions must not write it"
        )
    finally:
        await orch.aclose()
