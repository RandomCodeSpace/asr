"""Concept-level leak ratchet (DECOUPLE-06).

Walks ``src/runtime/`` for incident-management-specific tool names
and asserts zero matches **except** for entries on a documented
``RATCHET_ALLOWLIST``. Each allowlist entry is annotated with the
phase that will remove it, so the ratchet tightens as Phase 7 +
Phase 8 ship (Resolution B / Option 3 — staged ratchet with
documented allowlist).

Forbidden tokens are the 6 ASR-app terminal/typed tools that the
Phase 6 refactor pushed out of the framework:

    mark_resolved, mark_escalated, notify_oncall,
    submit_hypothesis, update_incident, apply_fix

If this test fails, fix the leak (or add an allowlist entry with
a phase-handle). Do NOT relax the forbidden-token list.

Companion meta-test ``test_allowlist_entries_actually_match``
asserts every allowlist entry still matches a real line; when a
relocation phase ships and the offending file is gone, the meta-
test fails, forcing the developer to drop the stale entry.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "src" / "runtime"

FORBIDDEN_TOKENS: tuple[str, ...] = (
    "mark_resolved",
    "mark_escalated",
    "notify_oncall",
    "submit_hypothesis",
    "update_incident",
    "apply_fix",
)

# Word-boundary regex per token avoids false positives on substrings
# that happen to share characters (e.g. ``apply_fixture`` is fine).
PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(tok) for tok in FORBIDDEN_TOKENS) + r")\b"
)


# ---------------------------------------------------------------------------
# Allowlist (Resolution B — Option 3, staged ratchet).
#
# Each entry is keyed by ``(relative_path, matched_token)`` and maps to
# the phase-handle that will REMOVE the leak. When the listed phase
# ships, the entry MUST be deleted and the test re-run; the meta-test
# below catches stale entries by asserting each pattern still has at
# least one matching line in the file.
#
# Removing an allowlist entry without first removing the leak is
# always a regression and will fail this ratchet.
# ---------------------------------------------------------------------------
# Phase 8 (DECOUPLE-07) closed the milestone — every Phase 8
# docstring leak was scrubbed to use neutral placeholders
# (``<terminal_tool>``, ``set_recommendation``). The ratchet is
# now a binary assertion: ANY match under ``src/runtime/`` is a
# regression. To intentionally land code that matches a forbidden
# token, you must first remove the leak; the allowlist is no
# longer a release valve.
RATCHET_ALLOWLIST: dict[tuple[str, str], str] = {}


def _iter_runtime_py_files() -> list[Path]:
    # Exclude ``__pycache__`` automatically (rglob skips dotted dirs
    # but not __pycache__; filter explicitly).
    return [
        p for p in RUNTIME_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _collect_offenders() -> list[tuple[str, int, str, str]]:
    """Return list of ``(rel_path, lineno, token, line_excerpt)`` for
    every forbidden-token match under ``src/runtime/``.
    """
    offenders: list[tuple[str, int, str, str]] = []
    for path in _iter_runtime_py_files():
        rel = str(path.relative_to(REPO_ROOT))
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in PATTERN.finditer(line):
                offenders.append((rel, lineno, match.group(0),
                                  line.strip()[:120]))
    return offenders


def test_runtime_dir_exists():
    assert RUNTIME_DIR.is_dir(), f"src/runtime/ not found at {RUNTIME_DIR}"


def test_no_incident_specific_tool_leaks():
    """All forbidden-token matches under ``src/runtime/`` MUST appear
    in ``RATCHET_ALLOWLIST``. Anything else is a regression."""
    unallowlisted: list[str] = []
    for rel_path, lineno, token, excerpt in _collect_offenders():
        if (rel_path, token) in RATCHET_ALLOWLIST:
            continue
        unallowlisted.append(
            f"{rel_path}:{lineno}: {token} (line: {excerpt})"
        )
    assert not unallowlisted, (
        "src/runtime/ leaks incident-specific tool names "
        "(DECOUPLE-06 ratchet violation). Either move the leak to "
        "YAML / examples/incident_management/ OR add a documented "
        "RATCHET_ALLOWLIST entry with a phase-handle. Offending "
        "references:\n  " + "\n  ".join(unallowlisted)
    )


def test_allowlist_entries_actually_match():
    """Every allowlist entry must still match at least one real line.

    When a relocation phase ships (e.g. Phase 7 moves
    ``remediation.py`` out of the framework dir), the corresponding
    file disappears and the matched-token count for that entry
    drops to zero. This test then fails, forcing the developer to
    delete the stale entry — the allowlist is *append-only* with a
    forced-shrink mechanic, not a quiet drift surface.
    """
    matched_pairs: set[tuple[str, str]] = set()
    for rel_path, _lineno, token, _excerpt in _collect_offenders():
        matched_pairs.add((rel_path, token))

    stale: list[str] = []
    for (rel_path, token), phase in RATCHET_ALLOWLIST.items():
        if (rel_path, token) not in matched_pairs:
            stale.append(
                f"{rel_path} [{token}] (declared cleanup: {phase})"
            )
    assert not stale, (
        "RATCHET_ALLOWLIST contains entries that no longer match any "
        "line under src/runtime/. The leak was already removed — "
        "delete the stale allowlist entry:\n  " + "\n  ".join(stale)
    )


def test_forbidden_tokens_list_is_locked():
    # Sanity: catch accidental list shrinkage during conflict
    # resolution / refactors. If this needs to change, update the
    # constant + commit message + the test (one PR, one rationale).
    assert FORBIDDEN_TOKENS == (
        "mark_resolved",
        "mark_escalated",
        "notify_oncall",
        "submit_hypothesis",
        "update_incident",
        "apply_fix",
    )


def test_allowlist_phases_are_well_known():
    """All declared phase handles should be one of the planned
    relocation phases. Catches typos like 'Phase  8' or 'phase7'.
    """
    valid_phases = {"Phase 7", "Phase 8"}
    invalid = {
        v for v in RATCHET_ALLOWLIST.values() if v not in valid_phases
    }
    assert not invalid, (
        f"unknown phase handles in RATCHET_ALLOWLIST: {invalid}; "
        f"expected one of {valid_phases}"
    )
