"""Phase 18 ratchet — no `except Exception: pass` (and equivalents) without
either (a) a logging call in the body or (b) a `noqa: BLE001 — <reason>`
rationale within 3 lines of the except.

This test walks every Python file under ``src/runtime/`` via AST. The
"production" assertion runs on the live tree; the four sanity assertions
parse fixture strings to prove the detector itself is wired correctly.

A previously-silent swallow that re-emerges (or a freshly-introduced one)
will fail this test, surfacing the regression at PR-review time rather
than after a paused session has gone missing in production.

Background: HARD-04 / CONCERNS H1 — silent broad-except handlers in
``runtime/service.py``, ``runtime/api.py``, ``runtime/orchestrator.py``
were eating asyncio teardown errors so that a misbehaving MCP transport
or checkpointer left no observable trace.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# Module-level constant so the sanity tests share the exact same threshold
# as the production walk.
_NEARBY_LINES = 3


def _is_broad_except(handler_type: str) -> bool:
    """True iff the handler catches Exception/BaseException broadly."""
    if handler_type in ("Exception", "BaseException"):
        return True
    # Bare ``except:`` — node.type is None, caller passes ``BaseException``
    # for that case; covered above.
    if handler_type.startswith("(") and "Exception" in handler_type:
        # ``except (Exception, OSError): ...`` etc.
        return True
    return False


def _body_is_silent_pass(body: list[ast.stmt]) -> bool:
    """True iff the except body is a single bare ``pass``."""
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _has_noqa_nearby(lines: list[str], handler_lineno: int) -> bool:
    """Look for ``noqa: BLE001`` within ``_NEARBY_LINES`` lines of the handler."""
    start = max(0, handler_lineno - 1 - _NEARBY_LINES)
    end = min(len(lines), handler_lineno + _NEARBY_LINES)
    blob = "\n".join(lines[start:end])
    return "noqa: BLE001" in blob or "noqa:BLE001" in blob


def find_silent_failures(source: str, filename: str = "<test>") -> list[str]:
    """Return ``"path:line"`` for each silent-pass violation in ``source``."""
    violations: list[str] = []
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        handler_type = ast.unparse(node.type) if node.type else "BaseException"
        if not _is_broad_except(handler_type):
            continue
        if not _body_is_silent_pass(node.body):
            continue
        if _has_noqa_nearby(lines, node.lineno):
            continue
        violations.append(f"{filename}:{node.lineno}")
    return violations


# ---------------------------------------------------------------------------
# Production walk — the actual ratchet
# ---------------------------------------------------------------------------

_RUNTIME_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "runtime"
)


def test_no_silent_failures_in_runtime() -> None:
    """Ratchet: no `except Exception: pass` (or equivalent) in
    ``src/runtime/`` without logging or a `noqa: BLE001` rationale.

    Adding a new silent-pass site to runtime code will fail this test;
    the fix is to either log+continue (preferred), re-raise, or document
    the deliberate ignore with a `# noqa: BLE001 — <reason>` comment.
    """
    assert _RUNTIME_ROOT.is_dir(), f"runtime root not found at {_RUNTIME_ROOT}"
    violations: list[str] = []
    for py in sorted(_RUNTIME_ROOT.rglob("*.py")):
        source = py.read_text(encoding="utf-8")
        violations.extend(find_silent_failures(source, filename=str(py)))
    assert not violations, (
        "Silent broad-except handlers found (HARD-04 regression). "
        "Add logger.warning/exception in the body, re-raise, or document "
        "with `# noqa: BLE001 — <reason>`. Sites:\n  "
        + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# Self-tests — prove the detector catches what it should and ignores
# what it should
# ---------------------------------------------------------------------------


def test_detector_flags_bare_silent_pass() -> None:
    """A bare `except Exception: pass` with no noqa is a violation."""
    src = (
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:\n"
        "        pass\n"
    )
    found = find_silent_failures(src, filename="bad.py")
    assert found == ["bad.py:4"], found


def test_detector_ignores_noqa_documented_pass() -> None:
    """A documented `# noqa: BLE001` silent pass is NOT a violation."""
    src = (
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:  # noqa: BLE001 — intentional best-effort cleanup\n"
        "        pass\n"
    )
    found = find_silent_failures(src, filename="ok.py")
    assert found == [], found


def test_detector_ignores_logged_body() -> None:
    """A non-pass body (e.g. logger call) is NOT a violation, regardless of noqa."""
    src = (
        "import logging\n"
        "_log = logging.getLogger('x')\n"
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:\n"
        "        _log.warning('boom', exc_info=True)\n"
    )
    found = find_silent_failures(src, filename="logged.py")
    assert found == [], found


def test_detector_ignores_narrow_except() -> None:
    """A narrow `except ValueError: pass` is NOT a violation — the
    ratchet only targets broad swallows."""
    src = (
        "def f():\n"
        "    try:\n"
        "        x = int('a')\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    found = find_silent_failures(src, filename="narrow.py")
    assert found == [], found


@pytest.mark.parametrize(
    "exc_clause",
    [
        "Exception",
        "BaseException",
        "(Exception, OSError)",
        "(OSError, Exception)",
    ],
)
def test_detector_flags_all_broad_variants(exc_clause: str) -> None:
    """The detector treats every common broad-except form as a candidate."""
    src = (
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        f"    except {exc_clause}:\n"
        "        pass\n"
    )
    found = find_silent_failures(src, filename="broad.py")
    assert found == ["broad.py:4"], found
