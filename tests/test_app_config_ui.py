"""AppConfig.ui — config-driven badges/detail-fields/tags for the generic UI."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import AppConfig, UIBadge, UIConfig, UIDetailField


def _minimal_appconfig_kwargs(**overrides) -> dict:
    """AppConfig requires llm + mcp; the rest defaults. Helpers like this
    keep tests focused on the field under exercise."""
    base: dict = {
        "llm": {
            "providers": {"stub": {"kind": "stub"}},
            "models": {"stub_default": {"provider": "stub", "model": "stub-1"}},
            "default": "stub_default",
        },
        "mcp": {"servers": []},
    }
    base.update(overrides)
    return base


def test_ui_badge_has_label_and_color() -> None:
    b = UIBadge(label="SEV1", color="red")
    assert b.label == "SEV1"
    assert b.color == "red"


def test_ui_badge_is_frozen() -> None:
    b = UIBadge(label="SEV1", color="red")
    with pytest.raises(ValidationError):
        b.label = "SEV2"  # type: ignore[misc]


def test_ui_detail_field_default_section_summary() -> None:
    f = UIDetailField(key="reporter.id", label="Reporter")
    assert f.section == "summary"


def test_ui_config_full_round_trip() -> None:
    cfg = UIConfig.model_validate({
        "badges": {
            "severity": {
                "sev1": {"label": "SEV1", "color": "red"},
                "sev2": {"label": "SEV2", "color": "orange"},
            },
        },
        "detail_fields": [
            {"key": "reporter.id", "label": "Reporter"},
            {"key": "reporter.team", "label": "Team", "section": "meta"},
        ],
        "tags": {
            "prior_match_supported": "hypothesis:prior_match_supported",
        },
    })
    assert cfg.badges["severity"]["sev1"].label == "SEV1"
    assert cfg.badges["severity"]["sev1"].color == "red"
    assert cfg.detail_fields[0].key == "reporter.id"
    assert cfg.detail_fields[0].section == "summary"
    assert cfg.detail_fields[1].section == "meta"
    assert cfg.tags["prior_match_supported"] == "hypothesis:prior_match_supported"


def test_app_config_ui_default_empty() -> None:
    cfg = AppConfig.model_validate(_minimal_appconfig_kwargs())
    assert cfg.ui.badges == {}
    assert cfg.ui.detail_fields == []
    assert cfg.ui.tags == {}


def test_app_config_ui_loads_from_yaml_shaped_dict() -> None:
    cfg = AppConfig.model_validate(_minimal_appconfig_kwargs(ui={
        "badges": {"severity": {"sev1": {"label": "SEV1", "color": "red"}}},
        "detail_fields": [{"key": "reporter.id", "label": "Reporter"}],
        "tags": {"prior_match_supported": "hypothesis:prior_match_supported"},
    }))
    assert cfg.ui.badges["severity"]["sev1"].label == "SEV1"
    assert cfg.ui.detail_fields[0].label == "Reporter"
    assert cfg.ui.tags["prior_match_supported"] == "hypothesis:prior_match_supported"


def test_ui_config_rejects_unknown_field() -> None:
    """Forbid extras to catch typos in YAML."""
    with pytest.raises(ValidationError):
        UIConfig.model_validate({"badgez": {}})  # typo


def test_ui_badge_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        UIBadge.model_validate({"label": "X", "color": "red", "extra": "nope"})
