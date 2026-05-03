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
BASELINE_TOTAL = 139


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
