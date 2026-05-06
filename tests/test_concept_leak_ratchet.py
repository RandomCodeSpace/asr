"""Concept-level leak ratchet (DECOUPLE-06).

Walks ``src/runtime/`` for incident-management-specific tool names
and asserts zero matches. Complements the existing token-level
ratchet (``tests/test_genericity_ratchet.py``) which counts
``incident``/``severity``/``reporter`` references with a downward-
only baseline. This ratchet is BINARY — adding any of these names
to ``src/runtime/`` is always a regression (D-06-07).

Forbidden tokens are the 6 ASR-app terminal/typed tools that the
Phase 6 refactor pushed out of the framework:

    mark_resolved, mark_escalated, notify_oncall,
    submit_hypothesis, update_incident, apply_fix

If this test fails, fix the leak — do NOT relax the list.
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


def _iter_runtime_py_files() -> list[Path]:
    # Exclude ``__pycache__`` automatically (rglob skips dotted dirs
    # but not __pycache__; filter explicitly).
    return [
        p for p in RUNTIME_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def test_runtime_dir_exists():
    assert RUNTIME_DIR.is_dir(), f"src/runtime/ not found at {RUNTIME_DIR}"


def test_no_incident_specific_tool_leaks():
    offenders: list[str] = []
    for path in _iter_runtime_py_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in PATTERN.finditer(line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: "
                    f"{match.group(0)} (line: {line.strip()[:120]})"
                )
    assert not offenders, (
        "src/runtime/ leaks incident-specific tool names "
        "(DECOUPLE-06 ratchet violation). Move them to YAML or "
        "examples/incident_management/. Offending references:\n  "
        + "\n  ".join(offenders)
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
