"""P8-K: ``dist/apps/code-review.py`` is a clean, parseable bundle.

The bundler at ``scripts/build_single_file.py`` produces a second
self-contained app bundle that proves the framework is generic — the
same runtime flattening that powers ``dist/apps/incident-management.py``
also produces ``dist/apps/code-review.py`` from
``examples/code_review/{config,state,mcp_server}.py``.

These tests fail loudly if the bundler regresses. They exercise the
bundle as a static artifact (``ast.parse``) — actually importing the
bundle in-process is out of scope here because that would require
fixturing every transitive runtime dependency.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _REPO_ROOT / "dist" / "apps" / "code-review.py"


@pytest.mark.skipif(not _BUNDLE.exists(),
                    reason="code-review bundle not built; run scripts/build_single_file.py")
def test_code_review_bundle_parses_cleanly():
    src = _BUNDLE.read_text()
    ast.parse(src)
    # Sanity: bundle is non-trivial (P5/P7 push it well past 200KB).
    assert len(src) > 100_000, f"bundle suspiciously small: {len(src):,} bytes"


@pytest.mark.skipif(not _BUNDLE.exists(),
                    reason="code-review bundle not built; run scripts/build_single_file.py")
def test_code_review_bundle_contains_app_symbols():
    src = _BUNDLE.read_text()
    # The CodeReviewState class body and the MCP server module-name
    # must both land in the bundle — proves the example app's source
    # was actually concatenated in.
    assert "class CodeReviewState" in src
    assert "FastMCP(\"code_review\")" in src
    # And critically: no incident-management *example app* bodies
    # leaked in. We probe via the module-marker comment the bundler
    # emits ahead of each module body.
    assert "module: examples/incident_management/" not in src
    assert "FastMCP(\"incident_management\")" not in src


def test_build_script_emits_code_review_target():
    """``scripts/build_single_file.py`` declares the new target."""
    src = (_REPO_ROOT / "scripts" / "build_single_file.py").read_text()
    assert "CODE_REVIEW_APP_MODULE_ORDER" in src
    assert "build_code_review_app" in src
    assert "dist/apps/code-review.py" in src
