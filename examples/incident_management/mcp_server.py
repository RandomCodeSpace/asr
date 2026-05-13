"""MCP tools + supervisor runner for the incident-management example app.

This module ships two things that together form the incident-management
example application's Python surface:

1. **MCP tools** — :class:`IncidentMCPServer` exposes
   ``lookup_similar_incidents``, ``create_incident`` and
   ``update_incident`` over FastMCP, backed by
   :class:`SessionStore` + :class:`HistoryStore`.
2. **Supervisor runner** — :func:`default_supervisor_runner` (and the
   :func:`make_default_supervisor_runner` factory) wires the framework's
   intake hydration + dedup gate to the per-app L2 KG / L5 Release /
   L7 Playbook stores. The intake skill YAML references the runner by
   dotted path so the live graph picks up incident-management-specific
   memory hydration.

Framework code does not import this module.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from runtime.intake import (
    compose_runners,
    default_intake_runner,
    hydrate_from_memory,
)
# Phase 16 (BUNDLER-01): use the sibling-defined ``_SEED_ROOT`` constant
# instead of an aliased module reference. The bundler's intra-import
# stripper removes ``from runtime.memory import knowledge_graph as
# _knowledge_graph_mod`` from the bundled source, leaving
# ``_knowledge_graph_mod.__file__`` as a NameError at module load. The
# import below is also stripped, but ``_SEED_ROOT`` survives module
# flattening because it's defined at module scope in knowledge_graph.py.
from runtime.memory.knowledge_graph import KnowledgeGraphStore, _SEED_ROOT
from runtime.memory.playbook_store import PlaybookStore
from runtime.memory.release_context import ReleaseContextStore
from runtime.memory.session_state import (
    L2KGContext,
    L5ReleaseContext,
    L7PlaybookSuggestion,
    MemoryLayerState,
)
from runtime.storage.history_store import HistoryStore
from runtime.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for typed terminal tools
# ---------------------------------------------------------------------------


class _TerminalPatchBase(BaseModel):
    """Common fields shared by all terminal tool requests.

    ``extra="forbid"`` is set so an LLM that types ``confidance`` (or
    any other non-allowed field) gets a ValidationError back rather
    than a silent drop.
    """
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_rationale: str = Field(min_length=1)


class ResolveRequest(_TerminalPatchBase):
    """Payload for ``mark_resolved`` — terminal close to status=resolved."""
    resolution_summary: str = Field(min_length=1)


class EscalateRequest(_TerminalPatchBase):
    """Payload for ``mark_escalated`` — terminal close to status=escalated.

    ``team`` MUST be one of the framework's configured
    ``escalation_teams``; the runtime validates that at the tool layer.
    """
    team: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class HypothesisSubmission(_TerminalPatchBase):
    """Payload for ``submit_hypothesis`` — used by the deep_investigator
    agent to record ranked hypotheses + confidence in a single typed call.

    ``findings_for`` defaults to ``"deep_investigator"``; other agents
    that submit hypotheses set it to their own name.
    """
    hypotheses: str = Field(min_length=1)
    findings_for: str = Field(default="deep_investigator")


class UpdateIncidentPatch(BaseModel):
    """Patch shape for non-terminal session updates.

    Status / resolution / escalation fields are NOT here — those move
    through the typed terminal tools (``mark_resolved`` /
    ``mark_escalated``). ``signal`` is permitted because non-terminal
    agents (triage, intake) use it to drive graph routing; terminal
    tools imply ``signal=success`` automatically and don't need to
    set it on a separate update_incident call.
    """
    model_config = ConfigDict(extra="forbid")

    severity: str | None = None
    category: str | None = None
    summary: str | None = None
    tags: list[str] | None = None
    matched_prior_inc: str | None = None
    findings: dict[str, str] | None = None
    signal: str | None = None


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


# Cheap regex: keep alnum + underscore + dash tokens of length >= 3. We
# lowercase before matching components by name. Good enough for the
# MVP — semantic component-extraction is a 9d-vector / future task.
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


_DEFAULT_SEEDS = _SEED_ROOT.parent  # parent of seeds/kg/ -> seeds/


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------


def extract_components(query: str, kg_store: KnowledgeGraphStore) -> list[str]:
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
        # on name for the friendlier "payments service" -> "payments"
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
    kg_store: KnowledgeGraphStore,
) -> list[str]:
    """Map component ids -> service names for the L5 release lookup.

    ``ReleaseContextStore.context`` matches on the ``service`` field of each
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
    kg_store: KnowledgeGraphStore,
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
    kg_store: KnowledgeGraphStore,
    release_store: ReleaseContextStore,
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


def make_hydrate_runner(
    *,
    kg_store: KnowledgeGraphStore,
    release_store: ReleaseContextStore,
    playbook_store: PlaybookStore,
    get_active_sessions: Callable[[], list[dict[str, Any]]] | None = None,
    component_lookup: Callable[[str], list[str]] | None = None,
) -> Callable[..., dict[str, Any] | None]:
    """Build a framework-compatible supervisor runner from explicit stores.

    Wraps the generic :func:`runtime.intake.hydrate_from_memory`
    runner-shape with this app's ``hydrate_and_gate`` pipeline. The
    framework helper handles the boilerplate (session extraction,
    memory-attribute stamping, duplicate-metadata + ``next_route``
    short-circuit); we only supply the app-specific hydrate + gate
    callables that operate on an :class:`IncidentState` query.

    ``get_active_sessions`` is invoked on every node entry so the dup
    gate sees the live in-flight set. Pass ``None`` (default) to
    disable the gate entirely — useful for unit-style harnesses that
    don't have an ``OrchestratorService`` running.
    """
    _list_active = get_active_sessions or (lambda: [])

    def _hydrator(
        session: Any,
        *,
        kg_store: Any = None,
        playbook_store: Any = None,
        release_store: Any = None,
    ) -> MemoryLayerState | None:
        """Run :func:`hydrate_and_gate` and return the memory bundle."""
        decision = hydrate_and_gate(
            incident=session,
            kg_store=kg_store,
            release_store=release_store,
            playbook_store=playbook_store,
            active_sessions=_list_active(),
            component_lookup=component_lookup,
        )
        # Cache decision on a hidden attribute so the gate callable
        # can reuse it without re-running the hydration.
        try:
            session.__dict__["_asr_decision"] = decision
        except Exception:  # noqa: BLE001 — frozen
            pass
        return decision.get("memory")

    def _gate(session: Any, *, kg_store: Any = None) -> str | None:
        """Return parent session id when the active-investigation gate fires."""
        decision = session.__dict__.get("_asr_decision") if hasattr(session, "__dict__") else None
        if not decision:
            return None
        if decision.get("status") != "duplicate":
            return None
        return decision.get("parent_session_id")

    def _runner(state: Any, *, app_cfg: Any | None = None) -> dict[str, Any] | None:
        return hydrate_from_memory(
            state,
            kg_store=kg_store,
            playbook_store=playbook_store,
            release_store=release_store,
            hydrator=_hydrator,
            gate=_gate,
        )

    return _runner


def make_default_supervisor_runner(
    *,
    kg_store: KnowledgeGraphStore,
    release_store: ReleaseContextStore,
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


# Phase 16 (BUNDLER-01): build the default runner LAZILY on first call.
# ``KnowledgeGraphStore.__init__`` eagerly reads ``components.json`` from
# disk, so building the runner at module-import time forced the seed
# directory to exist before ``import app`` could complete. That pattern
# broke the bundle's boot path on hosts where the seed bundle hasn't been
# laid down yet (the bundle is shipped as a 7-file copy-only payload).
# Constructing the runner on first call lets the bundle import cleanly
# and surfaces a genuine ``FileNotFoundError`` only when the runner is
# actually invoked — at which point the operator can see a configured,
# actionable error path rather than a cryptic import-time crash.
_BUILT_DEFAULT_RUNNER: Any = None


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
    global _BUILT_DEFAULT_RUNNER
    if _BUILT_DEFAULT_RUNNER is None:
        _BUILT_DEFAULT_RUNNER = make_default_supervisor_runner(
            kg_store=KnowledgeGraphStore(_DEFAULT_SEEDS / "kg"),
            release_store=ReleaseContextStore(_DEFAULT_SEEDS / "releases"),
            playbook_store=PlaybookStore(_DEFAULT_SEEDS / "playbooks"),
            get_active_sessions=lambda: [],
        )
    return _BUILT_DEFAULT_RUNNER(state, app_cfg=app_cfg)


# ---------------------------------------------------------------------------
# MCP tool helpers
# ---------------------------------------------------------------------------


def normalize_severity(
    value: str | None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentMCPServer:
    """FastMCP server bound to a single :class:`SessionStore` (+ optional :class:`HistoryStore`).

    Holds the active ``SessionStore`` and an optional ``HistoryStore``;
    only the ``lookup_similar_incidents`` tool needs the latter.
    """
    store: SessionStore | None = None
    history: HistoryStore | None = None
    # Severity aliases are injected via ``set_state``/``configure``
    # at process bootstrap; the orchestrator passes
    # ``framework_cfg.severity_aliases`` (read off
    # ``AppConfig.framework`` in the YAML). Bare default of ``{}``
    # keeps direct dataclass construction working in unit tests.
    severity_aliases: dict[str, str] = field(default_factory=dict)
    escalation_teams: list[str] = field(default_factory=list)
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)
        self.mcp.tool(name="mark_resolved")(self._tool_mark_resolved)
        self.mcp.tool(name="mark_escalated")(self._tool_mark_escalated)
        self.mcp.tool(name="submit_hypothesis")(self._tool_submit_hypothesis)

    def configure(
        self, *,
        store: SessionStore,
        history: HistoryStore | None = None,
        severity_aliases: dict[str, str] | None = None,
        escalation_teams: list[str] | None = None,
    ) -> None:
        self.store = store
        self.history = history
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases
        if escalation_teams is not None:
            self.escalation_teams = list(escalation_teams)

    def _require_store(self) -> SessionStore:
        if self.store is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.store

    def _require_history(self) -> HistoryStore:
        if self.history is None:
            raise RuntimeError(
                "incident_management server has no HistoryStore configured — "
                "pass history=... to configure() before calling lookup_similar_incidents"
            )
        return self.history

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        history = self._require_history()
        hits = history.find_similar(
            query=query,
            filter_kwargs={"environment": environment},
            limit=5,
        )
        return {"matches": [
            {"id": i.id,
             "summary": i.extra_fields.get("summary", ""),
             "resolution": i.extra_fields.get("resolution"),
             "score": round(s, 3)}
            for i, s in hits
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    submitter: dict | None = None,
                                    reporter_id: str | None = None,
                                    reporter_team: str | None = None) -> dict:
        """Create a new INC ticket and persist it.

        ``submitter`` is the canonical generic-runtime kwarg — for
        incident-management it carries ``{"id": "...", "team": "..."}``.
        ``reporter_id`` / ``reporter_team`` are deprecated; when
        supplied they are coerced into ``submitter`` and a
        ``DeprecationWarning`` is emitted. Passing both raises
        ``TypeError``.
        """
        legacy_supplied = reporter_id is not None or reporter_team is not None
        if submitter is not None and legacy_supplied:
            raise TypeError(
                "create_incident() received both submitter and "
                "reporter_id/reporter_team; pass submitter only "
                "(reporter_id/reporter_team are deprecated)"
            )
        if legacy_supplied:
            warnings.warn(
                "reporter_id and reporter_team are deprecated kwargs on "
                "create_incident(); pass submitter={'id': ..., 'team': ...} "
                "instead. The legacy kwargs will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            submitter = {
                "id": reporter_id if reporter_id is not None else "user-mock",
                "team": reporter_team if reporter_team is not None else "platform",
            }
        sub = submitter or {}
        inc = self._require_store().create(
            query=query,
            environment=environment,
            reporter_id=sub.get("id", "user-mock"),
            reporter_team=sub.get("team", "platform"),
        )
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a typed patch to an INC.

        Allowed keys are declared by ``UpdateIncidentPatch``. Unknown
        keys raise ``ValueError`` so the LLM gets a recoverable tool
        error and can retry.

        Status transitions (``resolved`` / ``escalated``), resolution
        text, and the escalated-to team are NOT writeable here — they
        flow through the typed terminal tools ``mark_resolved`` and
        ``mark_escalated``. The legacy ``findings_<agent>`` underscore
        pattern is replaced by the typed ``findings: dict[str, str]``
        field.
        """
        try:
            typed = UpdateIncidentPatch(**patch)
        except Exception as exc:  # pydantic ValidationError + others
            raise ValueError(
                f"invalid update_incident patch: {exc}. "
                f"Status/resolution/escalation use mark_resolved or mark_escalated; "
                f"per-agent findings use the typed `findings` dict."
            ) from exc

        store = self._require_store()
        inc = store.load(incident_id)
        if typed.severity is not None:
            inc.extra_fields["severity"] = normalize_severity(
                typed.severity, self.severity_aliases,
            )
        if typed.category is not None:
            inc.extra_fields["category"] = typed.category
        if typed.summary is not None:
            inc.extra_fields["summary"] = typed.summary
        if typed.tags is not None:
            inc.extra_fields["tags"] = list(typed.tags)
        if typed.matched_prior_inc is not None:
            inc.extra_fields["matched_prior_inc"] = typed.matched_prior_inc
        if typed.findings:
            for agent_name, finding in typed.findings.items():
                inc.findings[agent_name] = finding
        store.save(inc)
        return inc.model_dump()

    async def _tool_mark_resolved(
        self,
        incident_id: str,
        resolution_summary: str,
        confidence: float,
        confidence_rationale: str,
    ) -> dict:
        """Terminal close → status=resolved.

        This is the only sanctioned path to a ``resolved`` status. The
        legacy ``update_incident({"status":"resolved"})`` path no longer
        works (Task 3.5 locks down ``update_incident.patch`` to a typed
        schema that excludes ``status``).
        """
        req = ResolveRequest(
            incident_id=incident_id,
            resolution_summary=resolution_summary,
            confidence=confidence,
            confidence_rationale=confidence_rationale,
        )
        store = self._require_store()
        inc = store.load(req.incident_id)
        inc.status = "resolved"
        inc.extra_fields["resolution"] = req.resolution_summary
        store.save(inc)
        return {
            "incident_id": inc.id,
            "status": "resolved",
            "confidence": req.confidence,
            "confidence_rationale": req.confidence_rationale,
        }

    async def _tool_mark_escalated(
        self,
        incident_id: str,
        team: str,
        reason: str,
        confidence: float,
        confidence_rationale: str,
    ) -> dict:
        """Terminal close → status=escalated.

        Validates ``team`` against the configured roster (when one is
        set) so an LLM that emits a non-existent team gets a recoverable
        ToolError back instead of silently routing a page to nowhere.
        When ``escalation_teams`` is empty (e.g. test config), any
        non-empty team string is accepted.
        """
        req = EscalateRequest(
            incident_id=incident_id,
            team=team,
            reason=reason,
            confidence=confidence,
            confidence_rationale=confidence_rationale,
        )
        if self.escalation_teams and req.team not in self.escalation_teams:
            raise ValueError(
                f"team {req.team!r} not in escalation_teams "
                f"({self.escalation_teams})"
            )
        store = self._require_store()
        inc = store.load(req.incident_id)
        inc.status = "escalated"
        inc.extra_fields["escalated_to"] = req.team
        inc.extra_fields["escalation_reason"] = req.reason
        store.save(inc)
        return {
            "incident_id": inc.id,
            "status": "escalated",
            "team": req.team,
            "confidence": req.confidence,
            "confidence_rationale": req.confidence_rationale,
        }

    async def _tool_submit_hypothesis(
        self,
        incident_id: str,
        hypotheses: str,
        confidence: float,
        confidence_rationale: str,
        findings_for: str = "deep_investigator",
    ) -> dict:
        """Submit ranked hypotheses + confidence in a single typed call.

        Replaces the free-form ``update_incident({"findings_*", ...})``
        path used by the deep_investigator. ``confidence`` is required
        (Pydantic validation) so the agent cannot omit it; the graph's
        AgentRun harvester will read confidence + rationale from the
        typed return value.
        """
        req = HypothesisSubmission(
            incident_id=incident_id,
            hypotheses=hypotheses,
            confidence=confidence,
            confidence_rationale=confidence_rationale,
            findings_for=findings_for,
        )
        store = self._require_store()
        inc = store.load(req.incident_id)
        inc.findings[req.findings_for] = req.hypotheses
        store.save(inc)
        return {
            "incident_id": inc.id,
            "findings_for": req.findings_for,
            "confidence": req.confidence,
            "confidence_rationale": req.confidence_rationale,
        }


# ---------------------------------------------------------------------------
# Module-level default server.
#
# The MCP loader (``runtime.mcp_loader:137``) imports the module by name
# and reads ``getattr(mod, "mcp")`` to find the FastMCP instance to wire
# tools through. This singleton is purely a *loader-side default* —
# every concurrent orchestrator can and should construct its own fresh
# ``IncidentMCPServer()`` and ``configure(...)`` it against its own
# store. State on the class is held PER-INSTANCE; the singleton does
# not bleed into separate instances. ``tests/test_mcp_per_session_context.py``
# locks that guarantee.
#
# A future loader API (``register_in_process_server``) could let the
# orchestrator wire its own ``IncidentMCPServer`` instance instead of
# this singleton. Until then, ``set_state`` configures *the loader's
# default*, which is what the bundled example apps actually use.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, store: SessionStore,
              history: HistoryStore | None = None,
              severity_aliases: dict[str, str] | None = None) -> None:
    """Configure the loader's default IncidentMCPServer instance.

    Per-orchestrator isolation is enforced at the class level, not via
    this function. Apps that need multiple isolated servers in the
    same process should construct ``IncidentMCPServer()`` instances
    directly and configure each.
    """
    _default_server.configure(
        store=store,
        history=history,
        severity_aliases=severity_aliases,
    )


# Direct-call shims kept for tests that import these names.
async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str,
                          submitter: dict | None = None,
                          reporter_id: str | None = None,
                          reporter_team: str | None = None) -> dict:
    return await _default_server._tool_create_incident(
        query, environment,
        submitter=submitter,
        reporter_id=reporter_id,
        reporter_team=reporter_team,
    )


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)


__all__ = [
    "IncidentMCPServer",
    "SupervisorDecision",
    "create_incident",
    "default_supervisor_runner",
    "extract_components",
    "find_active_duplicate",
    "hydrate_and_gate",
    "lookup_similar_incidents",
    "make_default_supervisor_runner",
    "make_hydrate_runner",
    "mcp",
    "normalize_severity",
    "set_state",
    "update_incident",
]
