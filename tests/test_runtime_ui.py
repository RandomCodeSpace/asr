"""Smoke tests for ``runtime.ui`` — generic Streamlit shell.

The framework UI must import cleanly without a live Streamlit script
context, expose its render entry point, and stay free of any
``examples.*`` imports so the framework layer remains generic.
"""
from __future__ import annotations


def test_runtime_ui_module_imports():
    import runtime.ui as ui
    assert hasattr(ui, "render_session_detail")


def test_runtime_ui_does_not_import_examples():
    """The framework UI must NOT depend on any examples.* module."""
    import runtime.ui as ui
    src = open(ui.__file__).read()
    assert "from examples." not in src
    assert "import examples." not in src
