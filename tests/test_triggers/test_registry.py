"""TriggerRegistry tests — resolution, lifecycle, dispatch, idempotency."""
from __future__ import annotations

import pytest

from runtime.triggers import TriggerRegistry
from runtime.triggers.base import TriggerInfo, TriggerTransport
from runtime.triggers.config import (
    APITriggerConfig,
    PluginTriggerConfig,
    WebhookTriggerConfig,
)
from runtime.triggers.idempotency import IdempotencyStore


class _StubTransport(TriggerTransport):
    """Plugin transport that records lifecycle calls for assertion."""

    instances: list["_StubTransport"] = []

    def __init__(self, config: PluginTriggerConfig) -> None:
        self.config = config
        self.start_calls = 0
        self.stop_calls = 0
        _StubTransport.instances.append(self)

    async def start(self, registry):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1


def _api_cfg(name: str = "api-default") -> APITriggerConfig:
    return APITriggerConfig(name=name, target_app="incident_management")


def _webhook_cfg(name: str = "pd") -> WebhookTriggerConfig:
    return WebhookTriggerConfig(
        name=name,
        target_app="incident_management",
        payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
        transform="tests.test_triggers.conftest.transform_pagerduty",
        auth="none",
    )


def test_create_resolves_dotted_paths():
    async def fake(*, trigger=None, **kw):
        return "INC-X"

    reg = TriggerRegistry.create(
        [_webhook_cfg()], start_session_fn=fake, idempotency=None
    )
    spec = reg.specs["pd"]
    assert spec.payload_schema is not None
    assert callable(spec.transform)


def test_create_fails_fast_on_bad_transform_path():
    bad = WebhookTriggerConfig(
        name="pd",
        target_app="incident_management",
        payload_schema="tests.test_triggers.conftest.PagerDutyPayload",
        transform="nonexistent.module.fn",
        auth="none",
    )

    async def fake(*, trigger=None, **kw):
        return "INC-X"

    with pytest.raises(ImportError):
        TriggerRegistry.create([bad], start_session_fn=fake)


@pytest.mark.asyncio
async def test_dispatch_calls_transform_and_start_session():
    captured = {}

    async def fake_start(*, trigger=None, **kw):
        captured["trigger"] = trigger
        captured["kwargs"] = kw
        return "INC-99"

    reg = TriggerRegistry.create(
        [_webhook_cfg()], start_session_fn=fake_start, idempotency=None
    )
    from tests.test_triggers.conftest import PagerDutyPayload
    sid = await reg.dispatch(
        "pd", PagerDutyPayload(incident_id="P-1", summary="boom")
    )
    assert sid == "INC-99"
    assert captured["kwargs"]["query"].startswith("PD P-1")
    assert isinstance(captured["trigger"], TriggerInfo)
    assert captured["trigger"].transport == "webhook"
    assert captured["trigger"].name == "pd"


@pytest.mark.asyncio
async def test_dispatch_unknown_trigger_raises_keyerror():
    async def fake(*, trigger=None, **kw):
        return "INC-X"

    reg = TriggerRegistry.create([], start_session_fn=fake)
    with pytest.raises(KeyError):
        await reg.dispatch("unknown", {})


@pytest.mark.asyncio
async def test_idempotency_returns_cached_session_id(tmp_path):
    """A duplicate Idempotency-Key must reuse the cached session id."""
    calls = []

    async def fake_start(*, trigger=None, **kw):
        calls.append(kw)
        return f"INC-{len(calls)}"

    db_url = f"sqlite:///{tmp_path / 'idem.db'}"
    store = IdempotencyStore.connect(db_url)
    reg = TriggerRegistry.create(
        [_webhook_cfg()], start_session_fn=fake_start, idempotency=store
    )
    from tests.test_triggers.conftest import PagerDutyPayload
    p = PagerDutyPayload(incident_id="P-1", summary="boom")
    s1 = await reg.dispatch("pd", p, idempotency_key="K-1")
    s2 = await reg.dispatch("pd", p, idempotency_key="K-1")
    assert s1 == s2
    assert len(calls) == 1  # only one call to start_session


@pytest.mark.asyncio
async def test_lifecycle_start_stop_idempotent():
    _StubTransport.instances.clear()

    async def fake(*, trigger=None, **kw):
        return "INC-X"

    pcfg = PluginTriggerConfig(
        name="my-stub", target_app="x", kind="stub"
    )
    reg = TriggerRegistry.create(
        [pcfg],
        start_session_fn=fake,
        plugin_transports={"stub": _StubTransport},
    )
    await reg.start_all()
    await reg.start_all()  # idempotent
    await reg.stop_all()
    await reg.stop_all()
    stub = _StubTransport.instances[-1]
    assert stub.start_calls == 1
    assert stub.stop_calls == 1


@pytest.mark.asyncio
async def test_plugin_kind_unknown_raises_at_registry_init():
    async def fake(*, trigger=None, **kw):
        return "INC-X"

    pcfg = PluginTriggerConfig(
        name="bad", target_app="x", kind="not-registered"
    )
    with pytest.raises(ImportError):
        TriggerRegistry.create([pcfg], start_session_fn=fake)
