"""L2 Knowledge Graph — filesystem backend.

Read-only thin class over two JSON files on disk:

- ``components.json`` — list of ``{id, name, owner, criticality, environment}``.
- ``edges.json``      — list of ``{from, to, kind}`` where
  ``kind in {"calls", "deploys", "reads", "writes"}``.

The store accepts a ``root: Path`` (the layer directory, conventionally
``incidents/kg/``) for testability. If that directory is missing or
empty, the store falls back to the seed bundle at
``examples/incident_management/asr/seeds/kg/`` so a freshly cloned
checkout has a working graph for tests and demos. Air-gapped friendly:
no network, no Neo4j.

Mutation is out of scope for this batch (post-MVP); the surface here
is ``get_component`` / ``find_by_name`` / ``neighbors`` / ``subgraph``.
``subgraph`` returns the assembled :class:`L2KGContext` ready to drop
onto ``IncidentState.memory.l2_kg``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from examples.incident_management.asr.memory_state import L2KGContext

_VALID_EDGE_KINDS: frozenset[str] = frozenset(
    {"calls", "deploys", "reads", "writes"}
)

_SEED_ROOT = Path(__file__).parent / "seeds" / "kg"


class KGStore:
    """Filesystem-backed L2 Knowledge Graph reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._components: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        comp_path = self._root / "components.json"
        edges_path = self._root / "edges.json"

        # Fall back to the bundled seed when the configured layer dir
        # is missing or empty. Keeps tests and fresh checkouts working
        # without forcing the operator to seed ``incidents/kg/``.
        if not comp_path.exists() or not edges_path.exists():
            comp_path = _SEED_ROOT / "components.json"
            edges_path = _SEED_ROOT / "edges.json"

        components_raw = json.loads(comp_path.read_text())
        edges_raw = json.loads(edges_path.read_text())

        for c in components_raw:
            cid = c.get("id")
            if not cid:
                continue
            self._components[cid] = dict(c)

        for e in edges_raw:
            kind = e.get("kind")
            if kind not in _VALID_EDGE_KINDS:
                # Skip silently rather than raise — air-gapped operators
                # frequently hand-edit these files. A logged warning is
                # the right move once observability lands; for now,
                # ``list_edges()`` exposes the loaded set for tests.
                continue
            if "from" not in e or "to" not in e:
                continue
            self._edges.append({
                "from": e["from"],
                "to": e["to"],
                "kind": kind,
            })

    # ----- introspection (mostly for tests) -----------------------------

    def list_components(self) -> list[dict]:
        return list(self._components.values())

    def list_edges(self) -> list[dict]:
        return list(self._edges)

    # ----- public read API ----------------------------------------------

    def get_component(self, comp_id: str) -> dict | None:
        return self._components.get(comp_id)

    def find_by_name(self, name: str) -> list[dict]:
        """Case-insensitive substring match on the ``name`` field."""
        if not name:
            return []
        needle = name.lower()
        return [
            dict(c)
            for c in self._components.values()
            if needle in (c.get("name") or "").lower()
        ]

    def neighbors(
        self,
        comp_id: str,
        *,
        kinds: set[str] | None = None,
        hops: int = 1,
    ) -> set[str]:
        """Return the set of component ids reachable from ``comp_id``.

        - ``kinds`` filters edges to a subset of the valid edge kinds.
          ``None`` means "any kind".
        - ``hops`` is the BFS depth (>= 1). The starting node itself is
          *not* included in the returned set.
        """
        if hops < 1:
            return set()
        if comp_id not in self._components:
            return set()

        kind_filter = (
            None
            if kinds is None
            else (set(kinds) & _VALID_EDGE_KINDS)
        )

        visited: set[str] = {comp_id}
        frontier: set[str] = {comp_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for edge in self._edges:
                    if kind_filter is not None and edge["kind"] not in kind_filter:
                        continue
                    # Treat edges as undirected for neighbour expansion;
                    # ``subgraph`` records direction explicitly via the
                    # upstream / downstream split.
                    if edge["from"] == node and edge["to"] not in visited:
                        next_frontier.add(edge["to"])
                    elif edge["to"] == node and edge["from"] not in visited:
                        next_frontier.add(edge["from"])
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(comp_id)
        return visited

    def subgraph(
        self,
        comp_ids: Iterable[str],
        hops: int = 1,
    ) -> L2KGContext:
        """Assemble an :class:`L2KGContext` for the given component set.

        - ``components`` — the input set (filtered to ones we know about).
        - ``upstream``   — distinct ids on the ``from`` side of any edge
          whose ``to`` is in the input set.
        - ``downstream`` — distinct ids on the ``to`` side of any edge
          whose ``from`` is in the input set.
        - ``raw``        — full subgraph snapshot
          (``{"nodes": [...], "edges": [...]}``) including ``hops``
          worth of neighbour expansion, useful for UI rendering.
        """
        seeds = {c for c in comp_ids if c in self._components}

        upstream: set[str] = set()
        downstream: set[str] = set()
        for edge in self._edges:
            if edge["to"] in seeds and edge["from"] not in seeds:
                upstream.add(edge["from"])
            if edge["from"] in seeds and edge["to"] not in seeds:
                downstream.add(edge["to"])

        # Expand to ``hops`` neighbourhood for the raw snapshot.
        expanded: set[str] = set(seeds)
        for s in seeds:
            expanded |= self.neighbors(s, hops=hops)

        raw_nodes = [
            self._components[i] for i in expanded if i in self._components
        ]
        raw_edges = [
            edge
            for edge in self._edges
            if edge["from"] in expanded and edge["to"] in expanded
        ]

        return L2KGContext(
            components=sorted(seeds),
            upstream=sorted(upstream),
            downstream=sorted(downstream),
            raw={"nodes": raw_nodes, "edges": raw_edges},
        )
