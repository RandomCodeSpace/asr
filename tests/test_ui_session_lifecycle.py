"""Phase 20 (HARD-09): UI tests for session-lifecycle helpers.

Targets:
  * ``_should_poll`` (auto-refresh predicate)
  * ``_load_app_cfg`` (FrameworkAppConfig resolution: dotted-path vs YAML)
  * ``_resolve_environments`` (YAML-driven vs legacy provider fallback)
  * ``_get_service`` defensive return when no script-run context.

These are the "lifecycle wiring" helpers — they decide what the
sidebar shows, whether the detail pane keeps polling, and which
config block the rest of the UI reads. Pure functions; no Streamlit
rendering required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _should_poll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["running", "in_progress", "awaiting_input"])
def test_should_poll_true_for_inflight_statuses(status: str) -> None:
    from runtime.ui import _should_poll
    assert _should_poll(status) is True


@pytest.mark.parametrize("status", [
    "resolved", "escalated", "matched", "stopped", "deleted", "error",
    "needs_review", "new",
])
def test_should_poll_false_for_terminal_statuses(status: str) -> None:
    from runtime.ui import _should_poll
    assert _should_poll(status) is False


@pytest.mark.parametrize("status", [None, "", "  ", "totally_unknown"])
def test_should_poll_false_for_unknown_or_missing(status) -> None:
    """Unknown / falsy status → don't poll forever on bad data."""
    from runtime.ui import _should_poll
    # Strip-empty is not falsy in Python ("  " is truthy), but it's not
    # in the poll set either, so the second branch returns False.
    assert _should_poll(status) is False


# ---------------------------------------------------------------------------
# _load_app_cfg — dotted-path provider vs framework block
# ---------------------------------------------------------------------------


def test_load_app_cfg_returns_framework_block_when_no_provider() -> None:
    """Default path: read ``cfg.framework`` directly when no
    ``framework_app_config_path`` provider is configured.
    """
    from runtime.config import FrameworkAppConfig
    from runtime.ui import _load_app_cfg

    fake_cfg = MagicMock()
    fake_cfg.runtime.framework_app_config_path = None
    expected = FrameworkAppConfig(confidence_threshold=0.91)
    fake_cfg.framework = expected

    out = _load_app_cfg(fake_cfg)
    assert out is expected
    assert out.confidence_threshold == 0.91


def test_load_app_cfg_uses_dotted_path_provider_when_configured() -> None:
    """Legacy back-compat: when ``framework_app_config_path`` is set,
    delegate to ``resolve_framework_app_config`` (no fall-through to
    ``cfg.framework``).
    """
    from runtime.config import FrameworkAppConfig
    from runtime import ui as ui_mod

    fake_cfg = MagicMock()
    fake_cfg.runtime.framework_app_config_path = "fake.module:provider"

    expected = FrameworkAppConfig(confidence_threshold=0.42)
    with patch.object(ui_mod, "resolve_framework_app_config",
                      return_value=expected) as mock_resolve:
        out = ui_mod._load_app_cfg(fake_cfg)

    assert out is expected
    mock_resolve.assert_called_once_with("fake.module:provider")


# ---------------------------------------------------------------------------
# _resolve_environments — YAML-first, dotted-path fallback
# ---------------------------------------------------------------------------


def test_resolve_environments_prefers_yaml_block() -> None:
    """When ``cfg.environments`` is non-empty, return a copy and ignore
    the legacy provider path entirely.
    """
    from runtime.ui import _resolve_environments

    fake_cfg = MagicMock()
    fake_cfg.environments = ["dev", "staging", "production"]
    fake_cfg.runtime.environments_provider_path = "should.be.ignored:foo"

    out = _resolve_environments(fake_cfg)
    assert out == ["dev", "staging", "production"]
    # Caller can mutate without poisoning config — list is a fresh copy.
    out.append("new")
    assert fake_cfg.environments == ["dev", "staging", "production"]


def test_resolve_environments_returns_empty_when_no_provider_and_no_yaml() -> None:
    from runtime.ui import _resolve_environments

    fake_cfg = MagicMock()
    fake_cfg.environments = []
    fake_cfg.runtime.environments_provider_path = None

    assert _resolve_environments(fake_cfg) == []


def test_resolve_environments_returns_empty_for_malformed_dotted_path() -> None:
    """A provider string without ':' is a config bug — return empty
    rather than blowing up the sidebar.
    """
    from runtime.ui import _resolve_environments

    fake_cfg = MagicMock()
    fake_cfg.environments = []
    fake_cfg.runtime.environments_provider_path = "no_colon_here"

    assert _resolve_environments(fake_cfg) == []


# ---------------------------------------------------------------------------
# _get_service — headless return-None path
# ---------------------------------------------------------------------------


def test_get_service_returns_none_outside_script_context() -> None:
    """When ``_cached_service`` raises (e.g. cache decorator complains
    about missing script-run context), the wrapper must return ``None``
    so headless imports never crash.
    """
    from runtime import ui as ui_mod

    fake_cfg = MagicMock()
    with patch.object(ui_mod, "_cached_service",
                      side_effect=RuntimeError("no script context")):
        assert ui_mod._get_service(fake_cfg) is None
