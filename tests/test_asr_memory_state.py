"""``IncidentState.memory`` round-trips through ``SessionStore``.

The ASR memory-layer slot ``memory: MemoryLayerState`` lives on
``IncidentState`` but has no typed column on ``IncidentRow``; it rides
through the ``extra_fields`` JSON bag. These tests pin both the
default-empty shape and the populated round-trip so future schema work
doesn't silently drop sub-graph / release / playbook context off the
session.
"""
from __future__ import annotations

import pytest

from examples.incident_management.asr.memory_state import (
    L2KGContext,
    L5ReleaseContext,
    L7PlaybookSuggestion,
    MemoryLayerState,
)
from examples.incident_management.state import IncidentState, Reporter
from runtime.config import MetadataConfig
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def engine(tmp_path):
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(eng)
    return eng


# ---- Schema-shape pins -------------------------------------------------


def test_memory_layer_state_defaults_empty():
    m = MemoryLayerState()
    assert m.l2_kg is None
    assert m.l5_release is None
    assert m.l7_playbooks == []


def test_l2_kg_context_defaults():
    ctx = L2KGContext()
    assert ctx.components == []
    assert ctx.upstream == []
    assert ctx.downstream == []
    assert ctx.raw == {}


def test_l5_release_context_defaults():
    ctx = L5ReleaseContext()
    assert ctx.recent_releases == []
    assert ctx.suspect_releases == []


def test_l7_playbook_suggestion_score_bounds():
    # Scores are clamped to [0,1] by validation.
    s = L7PlaybookSuggestion(playbook_id="pb-1", score=0.5)
    assert s.matched_signals == []
    with pytest.raises(Exception):
        L7PlaybookSuggestion(playbook_id="pb-1", score=1.5)
    with pytest.raises(Exception):
        L7PlaybookSuggestion(playbook_id="pb-1", score=-0.1)


def test_incident_state_default_memory_is_empty():
    inc = IncidentState(
        id="INC-20260503-099",
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        query="boot",
        environment="staging",
        reporter=Reporter(id="u", team="t"),
    )
    assert isinstance(inc.memory, MemoryLayerState)
    assert inc.memory.l2_kg is None
    assert inc.memory.l5_release is None
    assert inc.memory.l7_playbooks == []


# ---- Round-trip through SessionStore (extra_fields) -------------------


def test_default_memory_round_trips_through_session_store(engine):
    store = SessionStore[IncidentState](engine=engine, state_cls=IncidentState)
    inc = store.create(
        query="payments slow",
        environment="production",
        reporter_id="alice",
        reporter_team="payments",
    )
    store.save(inc)

    loaded = store.load(inc.id)
    assert isinstance(loaded.memory, MemoryLayerState)
    assert loaded.memory.l2_kg is None
    assert loaded.memory.l5_release is None
    assert loaded.memory.l7_playbooks == []


def test_populated_memory_round_trips_through_session_store(engine):
    store = SessionStore[IncidentState](engine=engine, state_cls=IncidentState)
    inc = store.create(
        query="payments p99 spike",
        environment="production",
        reporter_id="alice",
        reporter_team="payments",
    )
    inc.memory = MemoryLayerState(
        l2_kg=L2KGContext(
            components=["payments"],
            upstream=["api-gateway"],
            downstream=["postgres-payments", "kafka"],
            raw={"nodes": 3, "edges": 2},
        ),
        l5_release=L5ReleaseContext(
            recent_releases=[
                {
                    "id": "rel-1",
                    "service": "payments",
                    "sha": "abc123",
                    "deployed_at": "2026-05-03T00:00:00Z",
                    "author": "bob",
                }
            ],
            suspect_releases=["rel-1"],
        ),
        l7_playbooks=[
            L7PlaybookSuggestion(
                playbook_id="pb-payments-latency",
                score=0.75,
                matched_signals=["service=payments", "metric=p99_latency"],
            )
        ],
    )
    store.save(inc)

    loaded = store.load(inc.id)
    assert isinstance(loaded.memory, MemoryLayerState)
    assert loaded.memory.l2_kg is not None
    assert loaded.memory.l2_kg.components == ["payments"]
    assert loaded.memory.l2_kg.upstream == ["api-gateway"]
    assert sorted(loaded.memory.l2_kg.downstream) == [
        "kafka",
        "postgres-payments",
    ]
    assert loaded.memory.l2_kg.raw == {"nodes": 3, "edges": 2}

    assert loaded.memory.l5_release is not None
    assert loaded.memory.l5_release.suspect_releases == ["rel-1"]
    assert loaded.memory.l5_release.recent_releases[0]["sha"] == "abc123"

    assert len(loaded.memory.l7_playbooks) == 1
    pb = loaded.memory.l7_playbooks[0]
    assert pb.playbook_id == "pb-payments-latency"
    assert pb.score == 0.75
    assert "service=payments" in pb.matched_signals
