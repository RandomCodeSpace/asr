"""Genericity ratchet — total leak count cannot grow.

If this test fails after a commit:
  - Either remove the new domain reference from src/runtime/ (preferred)
  - Or, if the reference is genuinely necessary AND replaces an old one
    (net-zero change), update BASELINE_TOTAL in this file in the same commit
    AFTER documenting why in the commit message.

Bumping BASELINE_TOTAL upward without an architecture rationale is a code-review red flag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from check_genericity import count_runtime_leaks, total  # noqa: E402


# Recorded after the FrameworkAppConfig refactor (Wave 2) landed.
# This is a downward-only ratchet. Lower numbers are better.
#
# History:
#   140 -> 144   submitter shim landed (transient ``reporter`` tokens
#                from the kwarg-coercion path; will drop again when
#                the legacy kwargs are removed entirely).
#   144 -> 139   Wave-3 follow-ups landed: ``state_overrides`` replaces
#                the ``environment`` kwarg on ``start_session``;
#                ``SessionStartBody`` drops ``reporter_id``/``reporter_team``
#                in favour of ``submitter``; the legacy ``/investigate``
#                endpoint and method aliases coerce internally so the
#                runtime deprecation paths never fire on hot routes.
#   139 -> 137   Phase-tag noise stripped from runtime/ comments —
#                cleanup removed two incidental ``incident``-token
#                mentions that lived inside historical phase narrative.
#   137 -> 139   Wave-1 strip-down (Tasks 1.A + 1.C) lifted the per-app
#                memory stores into ``runtime/memory/`` and the Streamlit
#                UI into ``runtime/ui.py``. The UI shell drops every
#                hardcoded severity/reporter colormap (now config-driven
#                via ``UIConfig.badges`` / ``detail_fields`` / ``tags``)
#                but the framework still references the field name
#                ``"severity"`` in two call sites and carries ``incident``
#                tokens in its memory-store docstrings (the L2/L5/L7
#                memory layers were always cross-app concepts; their
#                docstrings keep the historical "incident" example for
#                clarity). Net: +2 unavoidable tokens from generalising
#                code that previously lived under ``examples/``.
#   146 -> 147   ``Orchestrator.retry_session`` (post-failure manual retry)
#                added a single ``incident_id`` reference via the existing
#                ``_thread_config`` helper used to build the LangGraph
#                thread-id. Generic session-id terminology elsewhere; the
#                helper itself is older and keeps its parameter name for
#                callers in the same file.
#   147 -> 149   Phase 10 (FOC-03): mandatory per-turn confidence wrapped
#                each ``create_react_agent`` call site (graph.py, responsive.py)
#                in an envelope-parse + reconcile + EnvelopeMissingError-handler
#                block. The two new ``_handle_agent_failure(..., fallback=incident)``
#                calls reuse the pre-existing local ``incident`` variable name
#                (the runner's domain Session) on the new envelope-error
#                branch — no new domain concept, just two new uses of the
#                existing variable on a structurally required code path.
#   149 -> 153   Phase 11 (FOC-04): pure-policy HITL gating + GraphInterrupt-vs-error
#                fix. The runner's per-turn confidence-hint reset / update lines
#                in graph.py and responsive.py reuse the same ``incident`` local
#                variable name introduced in Phase 10 (the runner's domain
#                Session). Net +4 ``incident`` tokens, all reuses of the
#                existing local on structurally required code paths -- no new
#                domain concept introduced.
BASELINE_TOTAL = 153


def test_runtime_leaks_at_or_below_baseline():
    counts = count_runtime_leaks(Path("src/runtime"))
    t = total(counts)
    assert t <= BASELINE_TOTAL, (
        f"src/runtime/ has {t} domain-token references (incident/severity/reporter), "
        f"baseline is {BASELINE_TOTAL}. Either remove the new reference or "
        f"update BASELINE_TOTAL in this file with a rationale in the commit message."
    )


def test_individual_token_categories_visible():
    """Sanity: counts dict has all three tokens, each int."""
    counts = count_runtime_leaks(Path("src/runtime"))
    assert set(counts.keys()) == {"incident", "severity", "reporter"}
    for v in counts.values():
        assert isinstance(v, int) and v >= 0
