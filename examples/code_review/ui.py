"""Streamlit UI for the code-review example app.

Read-only viewer for ``CodeReviewState`` sessions. Mirrors the
incident-management UI's accordion-per-item pattern (badge-rich headers,
sharp corners) but stays deliberately minimal — code review doesn't have
intervention gating, escalation, or hypothesis trees, so the bulk of the
incident UI scaffolding doesn't apply here.

P8 ships this as read-only: agents file findings via the MCP server,
the UI just renders them. Manual finding entry is out of scope.

Lifecycle note (mirrors incident UI): the Orchestrator owns FastMCP
clients tied to a specific asyncio event loop, and Streamlit re-runs
the script on every interaction. ``OrchestratorService.get_or_create``
is wrapped in ``st.cache_resource`` so the background thread + loop
are built exactly once per Streamlit server process. The sidebar uses
``SessionStore`` directly for sync reads that need no MCP clients.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from runtime.config import load_config, AppConfig
from runtime.service import OrchestratorService
from runtime.state_resolver import resolve_state_class
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore

from examples.code_review.state import CodeReviewState


CONFIG_PATH = Path("config/config.yaml")


# ---------------------------------------------------------------------------
# Service + store wiring (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

# Code-review statuses that mean the run is still in flight; everything
# else is terminal and the detail pane renders once. Mirrors the
# incident-management ``_POLL_STATUSES`` shape so the helper stays
# unit-testable without a Streamlit runtime.
_POLL_STATUSES: frozenset[str] = frozenset({
    "fetching",
    "analyzing",
    "in_progress",
    "running",
})


def _should_poll(status: str | None) -> bool:
    """Return True iff the detail pane should auto-refresh for ``status``."""
    if not status:
        return False
    return status in _POLL_STATUSES


def _get_service(cfg: AppConfig) -> OrchestratorService | None:
    """Return the process-singleton ``OrchestratorService``, started.

    Wrapped via ``st.cache_resource`` so the loop is built once per
    Streamlit server process. Returns ``None`` when the module is
    imported outside a Streamlit run context (e.g. from tests) so a
    bare ``import`` of this module never crashes.
    """
    try:
        return _cached_service(cfg)
    except Exception:
        return None


@st.cache_resource(ttl=None, show_spinner=False)
def _cached_service(cfg: AppConfig) -> OrchestratorService:
    svc = OrchestratorService.get_or_create(cfg)
    if not svc._thread or not svc._thread.is_alive():
        svc.start()
    return svc


def _make_store(cfg: AppConfig) -> SessionStore[CodeReviewState]:
    """Build a ``SessionStore`` for ``CodeReviewState`` from the runtime config.

    Mirrors the incident UI's ``_make_repository`` pattern — a sync,
    no-MCP path used by the sidebar to list sessions cheaply.
    """
    state_cls = resolve_state_class(cfg.runtime.state_class) if cfg.runtime.state_class else CodeReviewState
    engine = build_engine(cfg.storage.url)
    Base.metadata.create_all(engine)
    return SessionStore(engine=engine, state_cls=state_cls)


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------

# Streamlit accepts these badge colors: blue/green/orange/red/violet/gray/primary.
_SEVERITY_COLOR: dict[str, str] = {
    "info": "gray",
    "warning": "orange",
    "error": "red",
    "critical": "violet",
}

_RECOMMENDATION_COLOR: dict[str, str] = {
    "approve": "green",
    "comment": "blue",
    "request_changes": "red",
}

_STATUS_COLOR: dict[str, str] = {
    "new": "gray",
    "fetching": "blue",
    "analyzing": "blue",
    "awaiting_decision": "orange",
    "approved": "green",
    "rejected": "red",
    "merged": "violet",
    "closed": "gray",
    "deleted": "gray",
}


def _badge(label: str, color: str) -> None:
    st.badge(label, color=color)


def _severity_badge(sev: str | None) -> None:
    if not sev:
        return
    _badge(sev.upper(), _SEVERITY_COLOR.get(sev, "gray"))


def _recommendation_badge(rec: str | None) -> None:
    if not rec:
        return
    _badge(rec.replace("_", " ").upper(), _RECOMMENDATION_COLOR.get(rec, "gray"))


def _status_badge(status: str | None) -> None:
    if not status:
        return
    _badge(status.upper(), _STATUS_COLOR.get(status, "gray"))


# ---------------------------------------------------------------------------
# Sidebar — list of recent code-review sessions
# ---------------------------------------------------------------------------


def _pr_label(state: CodeReviewState) -> str:
    """Compact human label for a session in the sidebar."""
    pr = state.pr
    return f"{pr.repo}#{pr.number} — {pr.title[:48]}"


def render_sidebar(store: SessionStore[CodeReviewState]) -> str | None:
    """Render the sidebar list and return the selected session id, if any."""
    st.sidebar.markdown("### Code reviews")
    try:
        sessions = store.list_recent(limit=25)
    except Exception as exc:  # noqa: BLE001 — surface storage errors in UI
        st.sidebar.error(f"Failed to load sessions: {exc}")
        return None

    if not sessions:
        st.sidebar.caption("No code-review sessions yet.")
        return None

    selected: str | None = None
    for state in sessions:
        if not isinstance(state, CodeReviewState):
            # Mixed-state DB (incidents + reviews share a row schema in
            # P8). Skip rows that don't have the code-review fields.
            continue
        button_label = _pr_label(state)
        if st.sidebar.button(button_label, key=f"sidebar-{state.id}"):
            selected = state.id
    return selected


# ---------------------------------------------------------------------------
# Detail — PR header, findings list, recommendation
# ---------------------------------------------------------------------------


def _render_pr_header(state: CodeReviewState) -> None:
    pr = state.pr
    st.markdown(f"## {pr.repo}#{pr.number} — {pr.title}")
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        _status_badge(state.status)
    with cols[1]:
        _recommendation_badge(state.overall_recommendation)
    with cols[2]:
        st.caption(f"author: {pr.author}")
    with cols[3]:
        st.caption(f"+{pr.additions} / -{pr.deletions}  ({pr.files_changed} files)")


def _render_findings_list(state: CodeReviewState) -> None:
    findings = state.review_findings
    st.markdown(f"### Findings ({len(findings)})")
    if not findings:
        st.caption("No findings filed.")
        return
    for idx, f in enumerate(findings):
        location = f"{f.file}" + (f":{f.line}" if f.line is not None else "")
        header = f"{f.severity.upper()} · {f.category} · {location}"
        with st.expander(header, expanded=(f.severity in ("error", "critical"))):
            cols = st.columns([1, 1, 6])
            with cols[0]:
                _severity_badge(f.severity)
            with cols[1]:
                st.caption(f.category)
            st.write(f.message)
            if f.suggestion:
                st.markdown("**Suggestion**")
                st.code(f.suggestion, language="text")


def _render_recommendation(state: CodeReviewState) -> None:
    st.markdown("### Recommendation")
    if state.overall_recommendation is None:
        st.caption("Pending — recommender has not run yet.")
        return
    _recommendation_badge(state.overall_recommendation)
    if state.review_summary:
        st.write(state.review_summary)


def render_detail(state: CodeReviewState) -> None:
    """Render the detail pane for a single code-review session."""
    _render_pr_header(state)
    st.divider()
    _render_findings_list(state)
    st.divider()
    _render_recommendation(state)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Code review", layout="wide")

    cfg = load_config(CONFIG_PATH)
    # Touch the service so the background loop is up before any agent
    # work needs it. Failures here are non-fatal — the read-only UI
    # still renders against a plain SessionStore.
    _get_service(cfg)
    store = _make_store(cfg)

    selected_id = render_sidebar(store)
    if selected_id is None:
        st.markdown("## Select a code-review session")
        st.caption("Pick a PR from the sidebar to inspect findings and recommendation.")
        return

    try:
        state = store.load(selected_id)
    except FileNotFoundError:
        st.error(f"Session {selected_id} not found.")
        return

    if not isinstance(state, CodeReviewState):
        st.warning(
            f"Session {selected_id} is not a code-review session "
            f"({type(state).__name__}); skipping render."
        )
        return

    render_detail(state)


# Streamlit invokes the script as ``__main__`` under ``streamlit run`` and
# also re-runs it on every interaction; in both cases ``__name__`` is
# ``"__main__"``. A bare ``import examples.code_review.ui`` (e.g. from
# tests) leaves ``main()`` uncalled so the module is side-effect free.
if __name__ == "__main__":
    main()
