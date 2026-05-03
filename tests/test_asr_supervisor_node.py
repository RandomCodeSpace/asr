"""P9-9h — ASR supervisor node hydration + single-active-investigation gate.

Tests for ``examples.incident_management.asr.supervisor_node``. Covers:

* ``extract_components`` — heuristic component-name extraction from a query.
* ``hydrate_and_gate`` — L2/L5/L7 hydration populates ``MemoryLayerState``.
* ``hydrate_and_gate`` — fresh investigation routes to ``triage``.
* ``hydrate_and_gate`` — duplicate investigation routes to ``__end__``
  with ``status="duplicate"`` and ``parent_session_id`` stamped.
* ``hydrate_and_gate`` — empty / unknown component query degrades
  gracefully (no L2/L5 hydration, empty L7, route stays ``triage``).
* The intake skill YAML loads cleanly under ``load_skill``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.incident_management.asr.kg_store import KGStore
from examples.incident_management.asr.memory_state import (
    L2KGContext,
    L5ReleaseContext,
    L7PlaybookSuggestion,
    MemoryLayerState,
)
from examples.incident_management.asr.playbook_store import PlaybookStore
from examples.incident_management.asr.release_store import ReleaseStore
from examples.incident_management.asr.supervisor_node import (
    extract_components,
    find_active_duplicate,
    hydrate_and_gate,
)
from runtime.skill import load_skill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def asr_dir() -> Path:
    """Bundled seed directory under examples/incident_management/asr/seeds/."""
    return Path(__file__).parent.parent / "examples" / "incident_management" / "asr" / "seeds"


@pytest.fixture
def stores(tmp_path: Path, asr_dir: Path) -> dict:
    """Store trio anchored on a tmp root that falls back to bundled seeds."""
    return {
        "kg": KGStore(tmp_path / "kg"),
        "release": ReleaseStore(tmp_path / "releases"),
        "playbook": PlaybookStore(tmp_path / "playbooks"),
    }


def _incident(*, query: str = "payments service is slow", id: str = "INC-X-1"):
    """Minimal duck-typed incident object for the helper."""
    return SimpleNamespace(id=id, query=query)


def _at() -> datetime:
    """Pinned anchor matching the L5 seed bundle's deployed_at window."""
    # Seed releases include 2026-04-30 entries; pick a time inside the
    # 24h recent window so ``recent_releases`` is non-empty.
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------


def test_extract_components_finds_known_service(stores) -> None:
    cids = extract_components("payments service p99 latency spike", stores["kg"])
    assert "payments" in cids


def test_extract_components_empty_query_returns_empty(stores) -> None:
    assert extract_components("", stores["kg"]) == []


def test_extract_components_unknown_only_returns_empty(stores) -> None:
    assert extract_components("nothing here matches", stores["kg"]) == []


def test_extract_components_matches_by_name_substring(tmp_path: Path) -> None:
    components = [
        {"id": "alpha", "name": "alpha-service", "owner": "t",
         "criticality": "tier-2", "environment": "production"},
    ]
    edges: list[dict] = []
    (tmp_path / "components.json").write_text(json.dumps(components))
    (tmp_path / "edges.json").write_text(json.dumps(edges))
    kg = KGStore(tmp_path)
    # ``alpha`` is the friendly name token; should match by name.
    assert extract_components("alpha is throwing 500s", kg) == ["alpha"]


# ---------------------------------------------------------------------------
# hydrate_and_gate — fresh path
# ---------------------------------------------------------------------------


def test_hydrate_fresh_routes_to_triage(stores) -> None:
    decision = hydrate_and_gate(
        incident=_incident(query="payments service p99 latency"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[],
        incident_at=_at(),
    )
    assert decision["route"] == "triage"
    assert decision["status"] is None
    assert decision["parent_session_id"] is None
    assert "payments" in decision["components"]


def test_hydrate_populates_l2_l5_l7(stores) -> None:
    decision = hydrate_and_gate(
        incident=_incident(query="payments p99 latency spike threshold breach"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[],
        incident_at=_at(),
    )
    memory: MemoryLayerState = decision["memory"]
    # L2: subgraph for the payments component must exist.
    assert isinstance(memory.l2_kg, L2KGContext)
    assert "payments" in memory.l2_kg.components
    # L5: release context populated for the payments service.
    assert isinstance(memory.l5_release, L5ReleaseContext)
    # L7: at least one playbook suggestion (seed bundle has pb-payments-latency).
    assert any(
        isinstance(s, L7PlaybookSuggestion) and s.playbook_id == "pb-payments-latency"
        for s in memory.l7_playbooks
    )


def test_hydrate_unknown_components_degrades_gracefully(stores) -> None:
    decision = hydrate_and_gate(
        incident=_incident(query="unrelated query that mentions nothing"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[],
        incident_at=_at(),
    )
    assert decision["route"] == "triage"
    memory: MemoryLayerState = decision["memory"]
    assert memory.l2_kg is None
    assert memory.l5_release is None
    assert memory.l7_playbooks == []


# ---------------------------------------------------------------------------
# Single-active-investigation gate
# ---------------------------------------------------------------------------


def test_dup_gate_routes_to_end_with_active_session(stores) -> None:
    """When another session is in flight against the same component, gate fires."""
    decision = hydrate_and_gate(
        incident=_incident(query="payments p99 spike"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[
            {"session_id": "INC-OTHER", "status": "in_progress",
             "started_at": "2026-05-03T09:00:00Z", "current_agent": "triage"},
        ],
        incident_at=_at(),
        component_lookup=lambda sid: ["payments"] if sid == "INC-OTHER" else [],
    )
    assert decision["route"] == "__end__"
    assert decision["status"] == "duplicate"
    assert decision["parent_session_id"] == "INC-OTHER"
    # Memory still hydrated even on the dup path so the UI can render it.
    assert decision["memory"].l2_kg is not None


def test_dup_gate_ignores_self_session(stores) -> None:
    """An active entry pointing at THIS incident must not gate it."""
    decision = hydrate_and_gate(
        incident=_incident(query="payments slow", id="INC-SELF"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[
            {"session_id": "INC-SELF", "status": "in_progress",
             "started_at": "2026-05-03T09:00:00Z", "current_agent": "triage"},
        ],
        incident_at=_at(),
        component_lookup=lambda sid: ["payments"],
    )
    assert decision["route"] == "triage"
    assert decision["parent_session_id"] is None


def test_dup_gate_ignores_terminal_sessions(stores) -> None:
    """A resolved/escalated session is not a duplicate-gate candidate."""
    decision = hydrate_and_gate(
        incident=_incident(query="payments slow"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[
            {"session_id": "INC-OLD", "status": "resolved",
             "started_at": "2026-05-03T08:00:00Z", "current_agent": None},
        ],
        incident_at=_at(),
        component_lookup=lambda sid: ["payments"],
    )
    assert decision["route"] == "triage"
    assert decision["status"] is None


def test_dup_gate_requires_component_overlap(stores) -> None:
    """Different components: no overlap, gate does NOT fire."""
    decision = hydrate_and_gate(
        incident=_incident(query="payments p99"),
        kg_store=stores["kg"],
        release_store=stores["release"],
        playbook_store=stores["playbook"],
        active_sessions=[
            {"session_id": "INC-OTHER", "status": "in_progress",
             "started_at": "2026-05-03T09:00:00Z", "current_agent": "triage"},
        ],
        incident_at=_at(),
        component_lookup=lambda sid: ["ledger"],
    )
    assert decision["route"] == "triage"
    assert decision["parent_session_id"] is None


# ---------------------------------------------------------------------------
# find_active_duplicate — direct unit
# ---------------------------------------------------------------------------


def test_find_active_duplicate_no_components_returns_none() -> None:
    assert find_active_duplicate(
        incident_id="X", components=[], active_sessions=[
            {"session_id": "Y", "status": "in_progress"},
        ],
    ) is None


def test_find_active_duplicate_permissive_fallback() -> None:
    """Without a component_lookup, any in-flight other session is a hit."""
    sid = find_active_duplicate(
        incident_id="X",
        components=["payments"],
        active_sessions=[
            {"session_id": "Y", "status": "in_progress"},
        ],
    )
    assert sid == "Y"


# ---------------------------------------------------------------------------
# Skill YAML loads
# ---------------------------------------------------------------------------


def test_intake_skill_yaml_loads(tmp_path: Path) -> None:
    """The shipped intake/config.yaml round-trips through the loader."""
    skill_dir = (
        Path(__file__).parent.parent
        / "examples" / "incident_management" / "skills" / "intake"
    )
    skill = load_skill(skill_dir)
    assert skill.name == "intake"
    assert skill.kind == "supervisor"
    assert "triage" in skill.subordinates
    assert skill.dispatch_strategy == "rule"
    assert skill.dispatch_rules
    # P9-9h: the skill must wire the default supervisor runner so the
    # framework actually invokes hydrate_and_gate at session start.
    assert skill.runner == (
        "examples.incident_management.asr.supervisor_node:default_supervisor_runner"
    )


# ---------------------------------------------------------------------------
# Framework integration: intake runs through make_supervisor_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_graph_routes_through_default_runner() -> None:
    """End-to-end: intake skill + framework supervisor node →
    runner fires, hydration populates ``IncidentState.memory``, route
    lands on triage."""
    from runtime.agents.supervisor import make_supervisor_node
    from examples.incident_management.state import IncidentState, Reporter

    skill_dir = (
        Path(__file__).parent.parent
        / "examples" / "incident_management" / "skills" / "intake"
    )
    skill = load_skill(skill_dir)
    node = make_supervisor_node(skill=skill)

    incident = IncidentState(
        id="INC-20260503-001",
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        query="payments service p99 latency spike",
        environment="prod",
        reporter=Reporter(id="op-1", team="sre"),
    )

    out = await node({"session": incident, "dispatch_depth": 0})

    # Routed to triage (no duplicate gate fires — empty active list).
    assert out["next_route"] == "triage"
    assert out["last_agent"] == "intake"
    assert out["error"] is None

    # Hydration ran: memory layers populated for the payments component.
    sess = out["session"]
    assert sess.memory.l2_kg is not None
    assert "payments" in sess.memory.l2_kg.components


@pytest.mark.asyncio
async def test_full_graph_runner_short_circuits_on_duplicate(monkeypatch) -> None:
    """When the dup gate fires the framework short-circuits to ``__end__``
    and stamps duplicate metadata before the graph terminates."""
    from runtime.agents.supervisor import make_supervisor_node
    from examples.incident_management.asr import supervisor_node as sn
    from examples.incident_management.asr.kg_store import KGStore
    from examples.incident_management.asr.playbook_store import PlaybookStore
    from examples.incident_management.asr.release_store import ReleaseStore
    from examples.incident_management.state import IncidentState, Reporter

    # Build a runner whose live active-session list contains an
    # in-flight investigation against ``payments``. Patch the module's
    # default runner singleton so the YAML-resolved hook picks it up.
    swap = sn.make_hydrate_runner(
        kg_store=KGStore(sn._DEFAULT_SEEDS / "kg"),
        release_store=ReleaseStore(sn._DEFAULT_SEEDS / "releases"),
        playbook_store=PlaybookStore(sn._DEFAULT_SEEDS / "playbooks"),
        get_active_sessions=lambda: [
            {"session_id": "INC-OTHER", "status": "in_progress"},
        ],
        component_lookup=lambda sid: ["payments"] if sid == "INC-OTHER" else [],
    )
    monkeypatch.setattr(sn, "_BUILT_DEFAULT_RUNNER", swap)

    skill_dir = (
        Path(__file__).parent.parent
        / "examples" / "incident_management" / "skills" / "intake"
    )
    skill = load_skill(skill_dir)
    node = make_supervisor_node(skill=skill)

    incident = IncidentState(
        id="INC-20260503-002",
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        query="payments p99 spike",
        environment="prod",
        reporter=Reporter(id="op-1", team="sre"),
    )

    out = await node({"session": incident, "dispatch_depth": 0})

    assert out["next_route"] == "__end__"
    sess = out["session"]
    assert sess.status == "duplicate"
    assert sess.parent_session_id == "INC-OTHER"


# ---------------------------------------------------------------------------
# Task 7: composition — default_supervisor_runner = compose_runners(
#          default_intake_runner, asr_memory_hydration)
# ---------------------------------------------------------------------------


def _make_incident() -> "IncidentState":
    from examples.incident_management.state import IncidentState, Reporter
    return IncidentState(
        id="INC-20260503-007",
        status="new",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        query="payments service p99 latency spike",
        environment="prod",
        reporter=Reporter(id="op-1", team="sre"),
    )


def test_default_supervisor_runner_calls_framework_default_first() -> None:
    """ASR runner must call default_intake_runner before ASR hydration.

    Verifies two things:
    1. Framework default ran: ``findings['prior_similar']`` is present
       (even as an empty list — HistoryStore returned no hits).
    2. ASR hydration ran: ``session.memory`` is non-None / populated,
       because the framework runner did NOT short-circuit.
    """
    from runtime.intake import IntakeContext
    from examples.incident_management.asr.supervisor_node import (
        default_supervisor_runner,
    )

    class _StubHist:
        def find_similar(self, *, query: str, filter_kwargs=None, limit: int = 3, threshold: float = 0.7):
            return []

    incident = _make_incident()
    state = {"session": incident}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=_StubHist(), dedup_pipeline=None,
    )})()

    default_supervisor_runner(state, app_cfg=app_cfg)

    # Framework default: findings['prior_similar'] populated (empty list ok).
    assert "prior_similar" in incident.findings
    # ASR hydration: memory attribute is set (MemoryLayerState, even if empty).
    assert incident.memory is not None


def test_default_supervisor_runner_skips_hydration_on_short_circuit(monkeypatch) -> None:
    """If framework runner returns next_route='__end__', ASR hydration must NOT run.

    This pins the composition contract: compose_runners stops on first
    next_route so a dedup/duplicate hit skips expensive KG lookups.
    """
    from runtime.intake import compose_runners, default_intake_runner
    from examples.incident_management.asr import supervisor_node as sn

    hydration_called = []

    def _stub_hydrate_runner(state: object, *, app_cfg: object = None):  # type: ignore[return]
        hydration_called.append(True)
        return None

    # Patch: framework default always short-circuits.
    def _short_circuit(state: object, *, app_cfg: object = None):
        sess = state.get("session") if hasattr(state, "get") else None  # type: ignore[union-attr]
        if sess is not None:
            sess.status = "duplicate"
        return {"next_route": "__end__"}

    patched = compose_runners(_short_circuit, _stub_hydrate_runner)
    monkeypatch.setattr(sn, "_BUILT_DEFAULT_RUNNER", patched)

    incident = _make_incident()
    state = {"session": incident}

    result = sn.default_supervisor_runner(state, app_cfg=None)

    assert result is not None
    assert result.get("next_route") == "__end__"
    # ASR hydration must NOT have run.
    assert hydration_called == [], "ASR hydration ran despite short-circuit — composition broken"
