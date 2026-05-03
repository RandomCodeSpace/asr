"""ASR supervisor router — hydration + single-active-investigation gate.

The framework ships a generic ``kind: supervisor`` agent kind in
:mod:`runtime.agents.supervisor` that handles pure dispatch routing.
ASR's investigation graph needs two extra behaviours BEFORE the first
responsive agent runs:

1. **Memory-layer hydration** — fetch L2 KG / L5 Release / L7 Playbook
   context for the affected services and stamp it onto
   ``IncidentState.memory`` so downstream agents can read without an
   extra round-trip to disk.

2. **Single-active-investigation gate** — if another in-flight session
   is already investigating the same component set, the new session is
   tagged ``status="duplicate"`` with ``parent_session_id=<active>``
   and routed straight to ``__end__``. Reuses the dedup linkage
   primitives (``parent_session_id`` field on ``Session``) instead of
   rolling a new field.

The helper here is a pure async callable: given an :class:`IncidentState`
and the configured stores + an active-session lister, it returns a
shape-checked decision dict. It does NOT mutate state in place — the
caller is responsible for persistence. This keeps the helper trivially
unit-testable and reusable from both:

- the unit tests in ``tests/test_asr_supervisor_node.py``;
- a future graph-level integration once the framework grows an
  example-app extension hook.

Inputs:

* ``incident``: an :class:`IncidentState` (or any ``Session``-shaped
  pydantic model with the incident-management fields).
* ``kg_store`` / ``release_store`` / ``playbook_store``: read-only
  filesystem stores.
* ``active_sessions``: callable returning a list of in-flight session
  dicts (matches the ``OrchestratorService.list_active_sessions``
  shape — ``[{"session_id", "status", "started_at", "current_agent"}, …]``).
  Pluggable so tests don't need a live service.
* ``incident_at``: optional ``datetime`` anchor for the L5 release
  window. Defaults to ``datetime.utcnow()``.

Outputs (a ``SupervisorDecision`` dict):

* ``route``: ``"triage"`` (fresh) or ``"__end__"`` (duplicate).
* ``memory``: a fully-populated :class:`MemoryLayerState` for the
  caller to persist on ``IncidentState.memory``.
* ``status``: the terminal status to stamp — ``"duplicate"`` when the
  gate fires, otherwise ``None`` (caller leaves status untouched).
* ``parent_session_id``: the active session id when gated, else
  ``None``.
* ``components``: the component ids extracted from the query (kept on
  the decision so the caller can audit log without re-deriving).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from examples.incident_management.asr.kg_store import KGStore
from examples.incident_management.asr.memory_state import (
    L2KGContext,
    L5ReleaseContext,
    L7PlaybookSuggestion,
    MemoryLayerState,
)
from examples.incident_management.asr.playbook_store import PlaybookStore
from examples.incident_management.asr.release_store import ReleaseStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SupervisorDecision(TypedDict, total=False):
    """Shape of :func:`hydrate_and_gate`'s return value."""

    route: str
    memory: MemoryLayerState
    status: str | None
    parent_session_id: str | None
    components: list[str]


# Status values that count as "in flight" for the dup gate. A session
# with any of these is still actively running and we should refuse to
# spawn a parallel investigation against the same components. Keep this
# list in sync with the ``IncidentStatus`` non-terminal statuses; the
# terminal set (``resolved`` / ``escalated`` / ``stopped`` / ``deleted``
# / ``duplicate`` / ``matched``) is the inverse.
_NON_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "new",
    "in_progress",
    "running",
    "awaiting_input",
})


# Cheap regex: keep alnum + underscore + dash tokens of length ≥ 3. We
# lowercase before matching components by name. Good enough for the
# MVP — semantic component-extraction is a 9d-vector / future task.
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------


def extract_components(query: str, kg_store: KGStore) -> list[str]:
    """Return component ids referenced by ``query`` (heuristic, deterministic).

    Strategy:

    1. Tokenize the query into lowercase alnum tokens.
    2. For each known component, match either its ``id`` or its
       ``name`` against the token set (case-insensitive substring
       on the name, exact match on the id).
    3. Return the matched ids in deterministic order.

    No LLM, no fuzzy match — keeps the supervisor unit-testable and
    air-gapped friendly. False positives are bounded because the
    component vocabulary is small (the L2 KG seed has a handful of
    services).
    """
    if not query:
        return []
    tokens = {t.lower() for t in _TOKEN_RE.findall(query)}
    if not tokens:
        return []

    matched: list[str] = []
    for comp in kg_store.list_components():
        cid = str(comp.get("id") or "").lower()
        cname = str(comp.get("name") or "").lower()
        # Exact id token match takes precedence; fall back to substring
        # on name for the friendlier "payments service" → "payments"
        # case in user-supplied prose.
        if cid and cid in tokens:
            matched.append(comp["id"])
            continue
        if cname:
            for tok in tokens:
                if tok in cname or cname in tok:
                    matched.append(comp["id"])
                    break
    # Stable de-dup preserving first-seen order.
    seen: set[str] = set()
    out: list[str] = []
    for cid in matched:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


# ---------------------------------------------------------------------------
# Active-session matching
# ---------------------------------------------------------------------------


def find_active_duplicate(
    *,
    incident_id: str,
    components: list[str],
    active_sessions: list[dict[str, Any]],
    component_lookup: Callable[[str], list[str]] | None = None,
) -> str | None:
    """Return the session id of an in-flight investigation that overlaps.

    Two sessions are considered to overlap when:

    - The other session is non-terminal (``status`` in
      :data:`_NON_TERMINAL_STATUSES`).
    - The other session is NOT this incident itself.
    - The other session's component set has a non-empty intersection
      with ``components``.

    ``component_lookup`` is an optional ``(session_id) -> [components]``
    callable so callers can plug in a real lookup (e.g. load each row
    via ``SessionStore`` and re-run :func:`extract_components`). When
    omitted, the function falls back to a permissive rule: if any
    other session is in flight and the caller has at least one
    component, treat it as a candidate duplicate. The unit tests
    cover both modes; production wiring supplies a real lookup so the
    permissive fallback never fires.

    Returns ``None`` when no active duplicate exists.
    """
    if not components:
        return None
    needle = set(components)
    for entry in active_sessions:
        other_id = entry.get("session_id")
        if not other_id or other_id == incident_id:
            continue
        if entry.get("status") not in _NON_TERMINAL_STATUSES:
            continue
        if component_lookup is None:
            # Permissive fallback — see docstring.
            return other_id
        other_components = component_lookup(other_id) or []
        if needle & set(other_components):
            return other_id
    return None


# ---------------------------------------------------------------------------
# Hydration + gate
# ---------------------------------------------------------------------------


def _service_names_for_components(
    components: list[str],
    kg_store: KGStore,
) -> list[str]:
    """Map component ids → service names for the L5 release lookup.

    ``ReleaseStore.context`` matches on the ``service`` field of each
    release record, which is conventionally the component's KG ``id``
    (e.g. ``"payments"``) — but if a component records a distinct
    ``service`` field that's preferred. Falls back to the id.
    """
    services: list[str] = []
    seen: set[str] = set()
    for cid in components:
        comp = kg_store.get_component(cid)
        if comp is None:
            continue
        svc = str(comp.get("service") or comp.get("id") or "").strip()
        if svc and svc not in seen:
            seen.add(svc)
            services.append(svc)
    return services


def _signals_for_match(
    *,
    query: str,
    components: list[str],
    kg_store: KGStore,
) -> dict[str, Any]:
    """Build the signal dict for :meth:`PlaybookStore.match`.

    The MVP signals are intentionally small: the first matched
    component's ``service`` plus a coarse ``metric`` derived from
    keyword presence in the query. Playbook-author guidance lives in
    the seed-bundle YAML.
    """
    if not components:
        return {}
    first = kg_store.get_component(components[0]) or {}
    svc = first.get("service") or first.get("id")
    signals: dict[str, Any] = {}
    if svc:
        signals["service"] = svc

    q = (query or "").lower()
    if "p99" in q or "latency" in q or "slow" in q:
        signals["metric"] = "p99_latency"
    elif "error" in q or "5xx" in q or "failing" in q:
        signals["metric"] = "error_rate"
    elif "down" in q or "outage" in q or "unavailable" in q:
        signals["metric"] = "availability"

    if "threshold" in q or "breach" in q or "spike" in q:
        signals["threshold_breach"] = True
    return signals


def hydrate_and_gate(
    *,
    incident: Any,
    kg_store: KGStore,
    release_store: ReleaseStore,
    playbook_store: PlaybookStore,
    active_sessions: list[dict[str, Any]] | None = None,
    incident_at: datetime | None = None,
    component_lookup: Callable[[str], list[str]] | None = None,
) -> SupervisorDecision:
    """Hydrate L2/L5/L7 memory and apply the single-active-investigation gate.

    Pure (no I/O beyond what the stores do at construction) — fast and
    deterministic. The caller persists ``decision["memory"]`` onto
    ``incident.memory`` and stamps the duplicate metadata when
    ``decision["status"] == "duplicate"``.

    ``incident`` is duck-typed: any object exposing ``id`` (str),
    ``query`` (str) is sufficient. We intentionally don't import
    :class:`IncidentState` here so the helper stays domain-thin.

    ``active_sessions`` defaults to an empty list — passing ``None``
    is treated as "no other sessions running" so the gate never fires
    by accident in unit tests that don't care about it.

    ``incident_at`` defaults to ``datetime.now(UTC)``. Tests pin this
    so the L5 ``recent_releases`` window is deterministic.
    """
    active = list(active_sessions or [])
    when = incident_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    query = str(getattr(incident, "query", "") or "")
    incident_id = str(getattr(incident, "id", "") or "")

    components = extract_components(query, kg_store)

    # ------- L2 hydration ------------------------------------------------
    if components:
        l2: L2KGContext | None = kg_store.subgraph(components, hops=1)
    else:
        l2 = None

    # ------- L5 hydration ------------------------------------------------
    services = _service_names_for_components(components, kg_store)
    if services:
        l5: L5ReleaseContext | None = release_store.context(services, when)
    else:
        l5 = None

    # ------- L7 hydration ------------------------------------------------
    signals = _signals_for_match(
        query=query, components=components, kg_store=kg_store,
    )
    l7: list[L7PlaybookSuggestion] = (
        playbook_store.match(signals) if signals else []
    )

    memory = MemoryLayerState(l2_kg=l2, l5_release=l5, l7_playbooks=l7)

    # ------- Single-active-investigation gate ----------------------------
    parent = find_active_duplicate(
        incident_id=incident_id,
        components=components,
        active_sessions=active,
        component_lookup=component_lookup,
    )
    if parent is not None:
        return SupervisorDecision(
            route="__end__",
            memory=memory,
            status="duplicate",
            parent_session_id=parent,
            components=components,
        )

    return SupervisorDecision(
        route="triage",
        memory=memory,
        status=None,
        parent_session_id=None,
        components=components,
    )


__all__ = [
    "SupervisorDecision",
    "default_supervisor_runner",
    "extract_components",
    "find_active_duplicate",
    "hydrate_and_gate",
    "make_hydrate_runner",
]


# ---------------------------------------------------------------------------
# Framework supervisor-runner adapter
# ---------------------------------------------------------------------------
#
# The framework's ``runtime.agents.supervisor.make_supervisor_node`` accepts
# a ``runner`` extension point: a callable invoked at supervisor-node entry
# that may mutate ``GraphState`` and/or short-circuit to ``"__end__"``. The
# functions below adapt :func:`hydrate_and_gate` to that contract so the
# intake skill actually exercises the hydration + dup-gate logic
# inside the live graph, instead of leaving it to be called only from
# unit tests.
#
# Two flavours:
#
# * :func:`make_hydrate_runner` — closure factory; prod wiring constructs
#   one with real stores + a live ``get_active_sessions`` callable.
# * :func:`default_supervisor_runner` — module-level, seed-anchored
#   convenience hook the YAML can reference directly. Active-session
#   listing is empty (no dup gate) — apps with a real backend should
#   build their own runner via ``make_hydrate_runner``.
#
# We keep the runner returns minimal (``session`` + optional
# ``next_route``) so the framework supervisor merge logic stays simple.


_DEFAULT_SEEDS = Path(__file__).parent / "seeds"


def make_hydrate_runner(
    *,
    kg_store: KGStore,
    release_store: ReleaseStore,
    playbook_store: PlaybookStore,
    get_active_sessions: Callable[[], list[dict[str, Any]]] | None = None,
    component_lookup: Callable[[str], list[str]] | None = None,
) -> Callable[..., dict[str, Any] | None]:
    """Build a framework-compatible supervisor runner from explicit stores.

    ``get_active_sessions`` is invoked on every node entry so the dup
    gate sees the live in-flight set. Pass ``None`` (default) to disable
    the gate entirely — useful for unit-style harnesses that don't have
    an ``OrchestratorService`` running.
    """
    _list_active = get_active_sessions or (lambda: [])

    def _runner(state: Any, *, app_cfg: Any | None = None) -> dict[str, Any] | None:
        session = state.get("session") if hasattr(state, "get") else None
        if session is None:
            return None
        try:
            decision = hydrate_and_gate(
                incident=session,
                kg_store=kg_store,
                release_store=release_store,
                playbook_store=playbook_store,
                active_sessions=_list_active(),
                component_lookup=component_lookup,
            )
        except Exception:  # noqa: BLE001 — defensive, keep graph alive
            logger.exception(
                "asr supervisor runner: hydrate_and_gate raised; routing to triage",
            )
            return None

        # Stamp the hydrated memory bundle onto the live IncidentState
        # (duck-typed: the ``memory`` attribute is set when present).
        memory = decision.get("memory")
        if memory is not None and hasattr(session, "memory"):
            try:
                session.memory = memory
            except Exception:  # noqa: BLE001 — frozen / read-only field
                logger.warning(
                    "asr supervisor runner: cannot set session.memory; "
                    "downstream agents will not see hydrated context",
                )

        if decision.get("status") == "duplicate":
            parent = decision.get("parent_session_id")
            # Stamp dup metadata so the persisted row reflects the gate
            # outcome before the graph terminates.
            try:
                session.status = "duplicate"
                if parent and hasattr(session, "parent_session_id"):
                    session.parent_session_id = parent
            except Exception:  # noqa: BLE001
                logger.warning(
                    "asr supervisor runner: cannot stamp duplicate metadata "
                    "on session %s", getattr(session, "id", "?"),
                )
            return {"session": session, "next_route": "__end__"}

        return {"session": session}

    return _runner


from runtime.intake import compose_runners, default_intake_runner  # noqa: E402


def make_default_supervisor_runner(
    *,
    kg_store: KGStore,
    release_store: ReleaseStore,
    playbook_store: PlaybookStore,
    get_active_sessions: Callable[[], list[dict[str, Any]]] | None = None,
    component_lookup: Callable[[str], list[str]] | None = None,
) -> Callable[..., dict[str, Any] | None]:
    """Compose framework default_intake_runner + ASR memory hydration.

    Framework default runs first (similarity retrieval + dedup gate).
    If it short-circuits via ``next_route='__end__'`` (duplicate), the
    ASR hydration is skipped — the duplicate session ends without
    paying for KG/playbook lookups.
    """
    asr_runner = make_hydrate_runner(
        kg_store=kg_store,
        release_store=release_store,
        playbook_store=playbook_store,
        get_active_sessions=get_active_sessions,
        component_lookup=component_lookup,
    )
    return compose_runners(default_intake_runner, asr_runner)


# Build the default runner exactly once at import time so per-call
# overhead is just a closure invocation. Constructor stays cheap:
# the stores read seed JSON lazily on first access.
_BUILT_DEFAULT_RUNNER = make_default_supervisor_runner(
    kg_store=KGStore(_DEFAULT_SEEDS / "kg"),
    release_store=ReleaseStore(_DEFAULT_SEEDS / "releases"),
    playbook_store=PlaybookStore(_DEFAULT_SEEDS / "playbooks"),
    get_active_sessions=lambda: [],
)


def default_supervisor_runner(
    state: Any, *, app_cfg: Any | None = None,
) -> dict[str, Any] | None:
    """Module-level runner the YAML can wire in via dotted path.

    Anchored on the bundled seed directory (``seeds/kg``,
    ``seeds/releases``, ``seeds/playbooks``) so the intake
    skill works out of the box. Real deployments should construct
    their own runner via :func:`make_default_supervisor_runner` and
    register it with their app's skill loader.

    Composition: framework ``default_intake_runner`` (similarity
    retrieval + dedup gate) runs first; ASR memory hydration follows.
    If the framework short-circuits (``next_route='__end__'``), the
    hydration step is skipped.
    """
    return _BUILT_DEFAULT_RUNNER(state, app_cfg=app_cfg)
