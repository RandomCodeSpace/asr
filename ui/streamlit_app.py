"""Streamlit entry point. The UI body lives at ``runtime/ui.py`` and
is config-driven (badges, detail fields, list filters), so a single
shell renders any app whose config it loads.

Run via: ``streamlit run ui/streamlit_app.py``.

The long-lived orchestrator service boots separately via
``python -m runtime --config config/<app>.yaml``.
"""
from runtime.ui import *  # noqa: F401,F403
