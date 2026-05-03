"""Validation tests for ``runtime.triggers.config``.

Failure modes covered:

- Required fields per variant.
- Bearer auth requires ``auth_token_env``.
- Schedule cron must be a 5-field string.
- Dotted-path validators reject malformed input.
- Discriminated union picks the right variant from a raw dict.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError, TypeAdapter

from runtime.triggers.config import (
    APITriggerConfig,
    PluginTriggerConfig,
    ScheduleTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)


def test_api_trigger_minimal():
    cfg = APITriggerConfig(name="api-default", target_app="incident_management")
    assert cfg.transport == "api"


def test_webhook_trigger_requires_payload_schema():
    with pytest.raises(ValidationError):
        WebhookTriggerConfig(
            name="pd",
            target_app="incident_management",
            transform="tests.test_triggers.conftest.transform_pagerduty",
            auth_token_env="TEST_WEBHOOK_TOKEN",
        )  # type: ignore[call-arg]


def test_webhook_trigger_requires_transform():
    with pytest.raises(ValidationError):
        WebhookTriggerConfig(
            name="pd",
            target_app="incident_management",
            payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
            auth_token_env="TEST_WEBHOOK_TOKEN",
        )


def test_webhook_bearer_requires_token_env():
    with pytest.raises(ValidationError) as exc:
        WebhookTriggerConfig(
            name="pd",
            target_app="incident_management",
            payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
            transform="tests.test_triggers.conftest.transform_pagerduty",
            auth="bearer",
        )
    assert "auth_token_env" in str(exc.value)


def test_webhook_auth_none_skips_token_env():
    cfg = WebhookTriggerConfig(
        name="pd",
        target_app="incident_management",
        payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
        transform="tests.test_triggers.conftest.transform_pagerduty",
        auth="none",
    )
    assert cfg.auth_token_env is None


def test_webhook_payload_schema_rejects_bad_dotted_path():
    with pytest.raises(ValidationError):
        WebhookTriggerConfig(
            name="pd",
            target_app="incident_management",
            payload_schema="not a dotted path",
            transform="tests.test_triggers.conftest.transform_pagerduty",
            auth_token_env="TEST_WEBHOOK_TOKEN",
        )


def test_schedule_trigger_requires_cron():
    with pytest.raises(ValidationError):
        ScheduleTriggerConfig(
            name="cron",
            target_app="incident_management",
            transform="tests.test_triggers.conftest.transform_schedule_heartbeat",
        )  # type: ignore[call-arg]


def test_schedule_trigger_rejects_six_field_cron():
    with pytest.raises(ValidationError):
        ScheduleTriggerConfig(
            name="cron",
            target_app="incident_management",
            transform="tests.test_triggers.conftest.transform_schedule_heartbeat",
            schedule="0 0 * * * *",  # 6-field — APScheduler-native, rejected
        )


def test_schedule_trigger_accepts_five_field_cron():
    cfg = ScheduleTriggerConfig(
        name="cron",
        target_app="incident_management",
        transform="tests.test_triggers.conftest.transform_schedule_heartbeat",
        schedule="*/15 * * * *",
    )
    assert cfg.schedule == "*/15 * * * *"


def test_plugin_trigger_minimal():
    cfg = PluginTriggerConfig(
        name="my-plugin",
        target_app="incident_management",
        kind="sqs",
        options={"queue_url": "sqs://q"},
    )
    assert cfg.kind == "sqs"


def test_trigger_name_alphabet():
    with pytest.raises(ValidationError):
        APITriggerConfig(name="bad name with spaces", target_app="x")


def test_discriminated_union_selects_webhook():
    adapter = TypeAdapter(TriggerConfig)
    cfg = adapter.validate_python({
        "name": "pd",
        "transport": "webhook",
        "target_app": "incident_management",
        "payload_schema": "tests.test_triggers.conftest.PagerDutyPayload",
        "transform": "tests.test_triggers.conftest.transform_pagerduty",
        "auth": "bearer",
        "auth_token_env": "TEST_WEBHOOK_TOKEN",
    })
    assert isinstance(cfg, WebhookTriggerConfig)
    assert cfg.payload_schema.endswith("PagerDutyPayload")


def test_app_config_triggers_field():
    """``triggers`` block on AppConfig coerces raw dicts to typed variants."""
    from runtime.config import AppConfig, LLMConfig, MCPConfig
    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(),
        triggers=[
            {
                "name": "pd",
                "transport": "webhook",
                "target_app": "incident_management",
                "payload_schema": "tests.test_triggers.conftest.PagerDutyPayload",
                "transform": "tests.test_triggers.conftest.transform_pagerduty",
                "auth": "bearer",
                "auth_token_env": "TEST_WEBHOOK_TOKEN",
            },
            {
                "name": "cron",
                "transport": "schedule",
                "target_app": "incident_management",
                "transform": "tests.test_triggers.conftest.transform_schedule_heartbeat",
                "schedule": "*/5 * * * *",
            },
        ],
    )
    assert len(cfg.triggers) == 2
    assert isinstance(cfg.triggers[0], WebhookTriggerConfig)
    assert isinstance(cfg.triggers[1], ScheduleTriggerConfig)
