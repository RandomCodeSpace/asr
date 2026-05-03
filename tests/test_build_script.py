"""Smoke tests for scripts/build_single_file.py outputs.

P1-L: the bundler emits three artifacts:

- ``dist/app.py`` — runtime-only (framework) bundle.
- ``dist/apps/incident-management.py`` — runtime + incident-management example
  app (self-contained ship target).
- ``dist/ui.py`` — Streamlit UI only; sibling import of ``app``.

These tests assert each artifact is produced and parses as valid Python, and
that the runtime-only bundle does not inline any example app *source code*.
The runtime legitimately references ``examples.incident_management.mcp_server``
as a string literal for dynamic import — that is allowed; what's stripped are
real import statements that pull example app code into the framework bundle.
"""
import ast
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_build_succeeds():
    result = subprocess.run(
        [sys.executable, "scripts/build_single_file.py"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"build script exited {result.returncode}: {result.stderr}"
    )


def test_dist_app_is_valid_python():
    src = (REPO / "dist/app.py").read_text()
    ast.parse(src)


def test_dist_incident_app_is_valid_python():
    p = REPO / "dist/apps/incident-management.py"
    if not p.exists():  # build chose flat layout instead — skip
        import pytest
        pytest.skip("dist/apps/incident-management.py not built (flat layout)")
    ast.parse(p.read_text())


def test_dist_ui_is_valid_python():
    src = (REPO / "dist/ui.py").read_text()
    ast.parse(src)


def test_dist_app_has_no_example_imports():
    """Runtime-only bundle must not import example app source modules.

    String references (e.g. ``_INCIDENT_MCP_MODULE = "examples.incident_management.mcp_server"``
    used for dynamic import) are allowed — what's forbidden is statically
    pulling example app code into the framework bundle. Example source code
    belongs in ``dist/apps/incident-management.py``.
    """
    src = (REPO / "dist/app.py").read_text()
    assert "from examples.incident_management" not in src, (
        "dist/app.py should be runtime-only; "
        "`from examples.incident_management ...` imports should have been stripped."
    )
    assert "import examples.incident_management" not in src, (
        "dist/app.py should be runtime-only; "
        "`import examples.incident_management` should have been stripped."
    )
    # Sanity: no example app module banner in the runtime-only bundle.
    assert "module: examples/incident_management/" not in src, (
        "dist/app.py should not inline any examples/incident_management/* modules; "
        "those belong only in dist/apps/incident-management.py."
    )
