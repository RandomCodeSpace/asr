"""Plugin transport tests — entry-points + explicit registration."""
from __future__ import annotations

import importlib.metadata
from typing import Any

import pytest

from runtime.triggers import TriggerRegistry
from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import PluginTriggerConfig


class _Recorder(TriggerTransport):
    """Plugin transport that records its lifecycle for assertion."""

    instances: list["_Recorder"] = []

    def __init__(self, config: PluginTriggerConfig) -> None:
        self.config = config
        self.events: list[str] = []
        _Recorder.instances.append(self)

    async def start(self, registry):
        self.events.append("start")

    async def stop(self):
        self.events.append("stop")


@pytest.mark.asyncio
async def test_explicit_plugin_registration():
    _Recorder.instances.clear()

    async def fake(*, trigger=None, **kw):
        return "INC-X"

    cfg = PluginTriggerConfig(
        name="rec",
        target_app="incident_management",
        kind="recorder",
        options={"k": 1},
    )
    reg = TriggerRegistry.create(
        [cfg],
        start_session_fn=fake,
        plugin_transports={"recorder": _Recorder},
    )
    await reg.start_all()
    await reg.stop_all()
    assert len(_Recorder.instances) == 1
    assert _Recorder.instances[0].events == ["start", "stop"]


def test_entry_point_loading_smoke(monkeypatch):
    """Smoke: monkeypatch importlib.metadata.entry_points to inject a fake EP."""
    class _FakeEP:
        name = "fakeep"

        def load(self):
            return _Recorder

    def fake_eps(group=None):
        if group == "runtime.triggers":
            return [_FakeEP()]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_eps)
    out = TriggerRegistry._load_entry_point_transports()
    assert "fakeep" in out
    assert out["fakeep"] is _Recorder


def test_entry_point_loading_skips_non_subclass(monkeypatch):
    class _NotATransport:
        pass

    class _FakeEP:
        name = "bogus"

        def load(self):
            return _NotATransport

    def fake_eps(group=None):
        return [_FakeEP()]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_eps)
    out = TriggerRegistry._load_entry_point_transports()
    assert "bogus" not in out


def test_entry_point_loading_skips_failed_load(monkeypatch):
    class _FakeEP:
        name = "broken"

        def load(self):
            raise RuntimeError("kaboom")

    def fake_eps(group=None):
        return [_FakeEP()]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_eps)
    out = TriggerRegistry._load_entry_point_transports()
    assert "broken" not in out
