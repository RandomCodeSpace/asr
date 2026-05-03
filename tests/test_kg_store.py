"""L2 Knowledge Graph filesystem backend.

Pin tests for ``KGStore``. Cover both the bundled seed fallback (used
when ``incidents/kg/`` is empty) and a tmp_path-driven custom graph
that exercises the read API in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.incident_management.asr.kg_store import KGStore
from examples.incident_management.asr.memory_state import L2KGContext


# ---- Seed-fallback path -------------------------------------------------


def test_seed_fallback_loads_when_root_empty(tmp_path: Path) -> None:
    """Empty ``incidents/kg/`` should fall back to the bundled seed."""
    store = KGStore(tmp_path)  # nothing in tmp_path
    components = store.list_components()
    assert len(components) >= 5
    ids = {c["id"] for c in components}
    assert {"payments", "api-gateway", "postgres-payments"} <= ids


def test_seed_edges_use_only_valid_kinds(tmp_path: Path) -> None:
    store = KGStore(tmp_path)
    valid = {"calls", "deploys", "reads", "writes"}
    for e in store.list_edges():
        assert e["kind"] in valid


# ---- Custom graph fixture ----------------------------------------------


@pytest.fixture
def custom_kg(tmp_path: Path) -> KGStore:
    components = [
        {"id": "a", "name": "alpha-service", "owner": "team-a",
         "criticality": "tier-2", "environment": "production"},
        {"id": "b", "name": "beta-service", "owner": "team-b",
         "criticality": "tier-1", "environment": "production"},
        {"id": "c", "name": "gamma-db", "owner": "data-team",
         "criticality": "tier-1", "environment": "production"},
        {"id": "d", "name": "delta-cache", "owner": "platform-team",
         "criticality": "tier-3", "environment": "production"},
    ]
    edges = [
        {"from": "a", "to": "b", "kind": "calls"},
        {"from": "b", "to": "c", "kind": "writes"},
        {"from": "b", "to": "d", "kind": "reads"},
        # Invalid kind — should be dropped silently.
        {"from": "a", "to": "d", "kind": "bogus"},
    ]
    (tmp_path / "components.json").write_text(json.dumps(components))
    (tmp_path / "edges.json").write_text(json.dumps(edges))
    return KGStore(tmp_path)


def test_get_component_round_trip(custom_kg: KGStore) -> None:
    assert custom_kg.get_component("a")["name"] == "alpha-service"
    assert custom_kg.get_component("missing") is None


def test_invalid_edge_kind_dropped(custom_kg: KGStore) -> None:
    assert all(e["kind"] in {"calls", "deploys", "reads", "writes"}
               for e in custom_kg.list_edges())
    assert len(custom_kg.list_edges()) == 3


def test_find_by_name_case_insensitive_substring(custom_kg: KGStore) -> None:
    matches = custom_kg.find_by_name("BETA")
    assert len(matches) == 1
    assert matches[0]["id"] == "b"
    assert custom_kg.find_by_name("does-not-exist") == []
    assert custom_kg.find_by_name("") == []


def test_neighbors_one_hop(custom_kg: KGStore) -> None:
    n = custom_kg.neighbors("b", hops=1)
    # b connects to a (in-edge calls), c (writes), d (reads)
    assert n == {"a", "c", "d"}


def test_neighbors_filtered_by_kind(custom_kg: KGStore) -> None:
    n = custom_kg.neighbors("b", kinds={"writes"}, hops=1)
    assert n == {"c"}


def test_neighbors_two_hop(custom_kg: KGStore) -> None:
    n = custom_kg.neighbors("a", hops=2)
    # a -> b at hop 1; b -> {c, d} at hop 2.
    assert n == {"b", "c", "d"}


def test_neighbors_unknown_node_returns_empty(custom_kg: KGStore) -> None:
    assert custom_kg.neighbors("zzz", hops=2) == set()


def test_neighbors_zero_hops_empty(custom_kg: KGStore) -> None:
    assert custom_kg.neighbors("a", hops=0) == set()


def test_subgraph_returns_l2kgcontext(custom_kg: KGStore) -> None:
    ctx = custom_kg.subgraph({"b"}, hops=1)
    assert isinstance(ctx, L2KGContext)
    assert ctx.components == ["b"]
    # ``a`` has an edge into ``b`` (a -> b), so a is upstream of b.
    assert ctx.upstream == ["a"]
    # ``b`` writes to c and reads from d — both downstream.
    assert sorted(ctx.downstream) == ["c", "d"]
    # raw snapshot includes b plus its neighbours.
    raw_node_ids = {n["id"] for n in ctx.raw["nodes"]}
    assert {"a", "b", "c", "d"} <= raw_node_ids


def test_subgraph_multi_seed(custom_kg: KGStore) -> None:
    ctx = custom_kg.subgraph({"b", "c"}, hops=1)
    assert sorted(ctx.components) == ["b", "c"]
    # ``a`` is upstream of b; b is in seeds so the b->c edge is internal,
    # not downstream. Only ``d`` remains downstream.
    assert ctx.upstream == ["a"]
    assert ctx.downstream == ["d"]


def test_subgraph_unknown_seed_filtered(custom_kg: KGStore) -> None:
    ctx = custom_kg.subgraph({"b", "ghost"}, hops=1)
    assert ctx.components == ["b"]
