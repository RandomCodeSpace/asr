"""Pydantic discriminated union for the ``triggers:`` block in app config.

A ``TriggerConfig`` declares ONE inbound dispatch path. The ``transport``
literal selects the concrete shape:

    - ``api``      — built-in HTTP route (back-compat with /investigate)
    - ``webhook``  — third-party POST /triggers/{name}; bearer auth
    - ``schedule`` — APScheduler in-process cron job
    - ``plugin``   — entry-point or explicitly-registered custom transport

Validation is fail-fast: bad dotted paths, missing auth env vars, and
malformed cron strings raise at config load time, never at request time.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Dotted-path regex used by ``payload_schema`` and ``transform`` fields.
# Accepts ``a.b.c`` or ``a.b:c`` (the colon form is tolerated for parity
# with entry-point syntax; ``runtime.triggers.resolve`` normalises both).
_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)+(:[A-Za-z_][\w]*)?$")

# 5-field cron, optionally allowing ``*/N`` step / ``A,B,C`` list / ``A-B``
# range tokens. Validation here is intentionally loose — APScheduler's
# ``CronTrigger.from_crontab`` is the source of truth and will reject
# semantically bad strings at scheduler-start time.
_CRON_5FIELD_RE = re.compile(
    r"^\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*$"
)


class _BaseTriggerConfig(BaseModel):
    """Shared fields for every trigger transport variant."""

    name: str = Field(..., min_length=1, max_length=128)
    target_app: str = Field(..., min_length=1)
    target_agent: str | None = None
    transform: str | None = None  # dotted path; required for webhook/schedule

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        # Webhook URLs use the name as a path segment — restrict to a safe
        # alphabet so we never have to URL-encode.
        if not re.match(r"^[A-Za-z0-9_\-]+$", v):
            raise ValueError(
                f"trigger name must match [A-Za-z0-9_-]+, got {v!r}"
            )
        return v

    @field_validator("transform")
    @classmethod
    def _validate_transform(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(f"transform must be dotted path, got {v!r}")
        return v


class APITriggerConfig(_BaseTriggerConfig):
    """Built-in HTTP route — preserves ``POST /investigate`` semantics.

    The api transport is implicitly registered for back-compat; explicitly
    listing it in ``triggers:`` is supported for symmetry but not required.
    """

    transport: Literal["api"] = "api"


class WebhookTriggerConfig(_BaseTriggerConfig):
    """Webhook trigger — third-party ``POST /triggers/{name}``.

    ``payload_schema`` is a dotted path to a Pydantic ``BaseModel``; the
    request JSON is validated against it, then ``transform(payload)`` is
    invoked to produce the keyword args for ``Orchestrator.start_session``.

    ``auth_token_env`` is the name of an environment variable holding the
    bearer token; the gateway never reads raw secrets from YAML.
    """

    transport: Literal["webhook"] = "webhook"
    payload_schema: str
    auth: Literal["bearer", "none"] = "bearer"
    auth_token_env: str | None = None
    idempotency_ttl_hours: int = Field(24, ge=1, le=24 * 30)

    @field_validator("payload_schema")
    @classmethod
    def _validate_payload_schema(cls, v: str) -> str:
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(f"payload_schema must be dotted path, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_bearer(self) -> "WebhookTriggerConfig":
        if self.auth == "bearer" and not self.auth_token_env:
            raise ValueError("auth: bearer requires auth_token_env")
        if self.transform is None:
            raise ValueError("webhook trigger requires transform: <dotted.path>")
        return self


class ScheduleTriggerConfig(_BaseTriggerConfig):
    """In-process APScheduler cron job.

    ``schedule`` is a 5-field standard cron string interpreted via
    ``CronTrigger.from_crontab``. APScheduler's native 6-field form is
    rejected here — the registry owns the cron flavour.

    ``payload`` is a static dict passed to ``transform`` on each fire;
    runtime cron jobs have no inbound payload to validate.
    """

    transport: Literal["schedule"] = "schedule"
    schedule: str
    timezone: str = "UTC"
    payload: dict = Field(default_factory=dict)

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, v: str) -> str:
        if not _CRON_5FIELD_RE.match(v):
            raise ValueError(
                f"schedule must be a 5-field cron string, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _check_transform(self) -> "ScheduleTriggerConfig":
        if self.transform is None:
            raise ValueError("schedule trigger requires transform: <dotted.path>")
        return self


class PluginTriggerConfig(_BaseTriggerConfig):
    """Plugin-defined transport, addressed by ``kind``.

    Resolution: the registry merges entry-points (group ``runtime.triggers``)
    with explicit ``plugin_transports`` passed to ``TriggerRegistry.create``;
    explicit entries win. Bad ``kind`` raises at registry init.
    """

    transport: Literal["plugin"] = "plugin"
    kind: str = Field(..., min_length=1)
    options: dict = Field(default_factory=dict)


# Discriminated union — Pydantic uses the ``transport`` literal to pick
# the right shape during validation. Field default makes the runtime
# triggers list ``list[TriggerConfig]`` resolve cleanly.
TriggerConfig = Annotated[
    Union[
        APITriggerConfig,
        WebhookTriggerConfig,
        ScheduleTriggerConfig,
        PluginTriggerConfig,
    ],
    Field(discriminator="transport"),
]
