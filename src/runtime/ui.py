"""Generic Streamlit UI — 2 tabs + always-on sidebar with recent sessions.

Made app-agnostic: status pills, configured-field badges, detail-pane
fields, and prior-match tags all flow from :class:`runtime.config.UIConfig`
so the shell stays domain-neutral.

Apps point at this module via their bootstrap entry (e.g.
``streamlit run -m runtime.ui`` with ``APP_CONFIG`` in env, or a thin
shim under ``examples/<app>/__main__.py`` that hands a config path
in).
"""
# Lifecycle note: the Orchestrator owns FastMCP clients tied to a specific
# asyncio event loop. Streamlit re-runs the script on every interaction and we
# use `asyncio.run(...)` per call, which creates a fresh loop each time.
# Caching an Orchestrator across reruns would leave its clients/transports
# bound to a dead loop and the first tool call would raise:
#     unable to perform operation on <TCPTransport closed=True ...>
# So: build a fresh Orchestrator inside each `asyncio.run` and `aclose` it when
# done. For pure metadata views (agents/tools) we use `_load_metadata_dicts` —
# a one-shot fetch that captures plain dicts and disposes the orchestrator.
# The sidebar uses SessionStore directly for sync list/load/delete calls
# that need no MCP clients. It builds the store from the same config the
# Orchestrator uses so both share the same SQLite DB.
from __future__ import annotations
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

from runtime.config import (
    AppConfig,
    FrameworkAppConfig,
    UIBadge,
    load_config,
    resolve_framework_app_config,
)
from runtime.orchestrator import Orchestrator
from runtime.service import OrchestratorService
from runtime.state_resolver import resolve_state_class
from runtime.storage.engine import build_engine
from runtime.storage.embeddings import build_embedder
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store


# Default config path; apps override via the ``APP_CONFIG`` env var.
CONFIG_PATH = Path(os.environ.get("APP_CONFIG", "config/config.yaml"))


def _load_app_cfg(cfg: AppConfig) -> FrameworkAppConfig:
    """Resolve the application's :class:`FrameworkAppConfig`.

    Reads ``AppConfig.framework`` directly off the YAML; falls back
    to the legacy ``framework_app_config_path`` provider for
    deployments that still wire it. Centralised here so the UI never
    imports an app-specific config module.
    """
    if cfg.runtime.framework_app_config_path is not None:
        return resolve_framework_app_config(cfg.runtime.framework_app_config_path)
    return cfg.framework


# ---------------------------------------------------------------------------
# Multi-session live state
# ---------------------------------------------------------------------------

# Statuses that mean the run is still in flight and the detail view should
# auto-refresh. Anything not in this set is terminal (resolved / escalated /
# stopped / deleted / matched / new) and the detail pane renders once.
#
# The plan-doc names ``running`` as a synonym for ``in_progress``; we accept
# both so the helper survives any future renaming on the service side
# (``_ActiveSession.status`` currently mirrors the IncidentStatus literal).
_POLL_STATUSES: frozenset[str] = frozenset({
    "running",
    "in_progress",
    "awaiting_input",
})


def _should_poll(status: str | None) -> bool:
    """Return True iff the detail pane should auto-refresh for ``status``.

    Pure function so the polling decision is unit-testable without spinning
    up a Streamlit runtime. Treat unknown / missing status as terminal —
    polling forever on bad data is worse than over-eagerly stopping.
    """
    if not status:
        return False
    return status in _POLL_STATUSES


def _get_service(cfg: AppConfig) -> OrchestratorService | None:
    """Return the process-singleton ``OrchestratorService``, started.

    Wrapped in ``st.cache_resource`` so the background thread + asyncio
    loop are built exactly once per Streamlit server process and reused
    across reruns. Returns ``None`` when running outside a Streamlit
    script-run context (e.g. plain module import from a test) so the
    module stays importable headlessly.
    """
    try:
        return _cached_service(cfg)
    except Exception:
        # Streamlit's cache decorator raises if invoked without a script
        # run context. Fall back to building the singleton directly so a
        # bare ``import`` of this module never crashes.
        return None


@st.cache_resource(ttl=None, show_spinner=False)
def _cached_service(cfg: AppConfig) -> OrchestratorService:
    svc = OrchestratorService.get_or_create(cfg)
    if not svc._thread or not svc._thread.is_alive():
        svc.start()
    return svc


def _make_repository(cfg: AppConfig) -> SessionStore:
    """Build a SessionStore from config — mirrors Orchestrator.create logic.

    Returns the active CRUD ``SessionStore`` only. The UI does not need
    similarity search; if a future view does, build a ``HistoryStore``
    alongside.
    """
    from runtime.config import MetadataConfig, resolve_framework_app_config
    default_url = MetadataConfig().url
    url = (
        cfg.storage.metadata.url if cfg.storage.metadata.url != default_url
        else f"sqlite:///{Path(cfg.paths.incidents_dir) / 'metadata.db'}"
    )
    engine = build_engine(MetadataConfig(url=url, pool_size=cfg.storage.metadata.pool_size, echo=cfg.storage.metadata.echo))
    Base.metadata.create_all(engine)
    embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
    vector_store = build_vector_store(cfg.storage.vector, embedder, engine)
    state_cls = resolve_state_class(cfg.runtime.state_class)
    # Resolve framework knobs the same way the orchestrator does:
    # dotted-path provider wins for back-compat, otherwise the YAML
    # ``framework:`` block. Only ``session_id_prefix`` is read here so
    # the UI's transient SessionStore mints ids in the same namespace.
    if cfg.runtime.framework_app_config_path is not None:
        framework_cfg = resolve_framework_app_config(
            cfg.runtime.framework_app_config_path,
        )
    else:
        framework_cfg = cfg.framework
    return SessionStore(
        engine=engine,
        state_cls=state_cls,
        embedder=embedder,
        vector_store=vector_store,
        vector_path=(cfg.storage.vector.path
                     if cfg.storage.vector.backend == "faiss" else None),
        vector_index_name=cfg.storage.vector.collection_name,
        distance_strategy=cfg.storage.vector.distance_strategy,
        id_prefix=framework_cfg.session_id_prefix,
    )


def _load_metadata_dicts(
    cfg: AppConfig,
) -> tuple[list[dict], list[dict], list[str]]:
    """Build a transient orchestrator, snapshot agents/tools/envs, then aclose.

    Per-rerun cost is dominated by FastMCP client startup (~100-200ms total for
    in-process servers); acceptable for a UI rerun.

    Environments come from the app's ``environments_provider_path`` (a
    ``module:callable`` reference on :class:`runtime.config.RuntimeConfig`).
    Apps that don't expose environments leave the field unset; the UI
    just renders an empty list.
    """
    async def _go():
        orch = await Orchestrator.create(cfg)
        try:
            return orch.list_agents(), orch.list_tools(), _resolve_environments(cfg)
        finally:
            await orch.aclose()
    return asyncio.run(_go())


def _resolve_environments(cfg: AppConfig) -> list[str]:
    """Resolve the app's environments roster.

    Prefers the YAML-driven ``AppConfig.environments`` list; falls
    back to the legacy ``RuntimeConfig.environments_provider_path``
    callable for deployments that still wire it.
    """
    if cfg.environments:
        return list(cfg.environments)
    dotted = cfg.runtime.environments_provider_path
    if not dotted:
        return []
    if ":" not in dotted:
        return []
    module_name, _, attr = dotted.partition(":")
    try:
        import importlib
        mod = importlib.import_module(module_name)
        provider = getattr(mod, attr)
        envs = provider()
    except Exception:  # pragma: no cover - defensive
        return []
    return list(envs) if envs else []


# Color palette for st.badge — Streamlit accepts: blue/green/orange/red/violet/gray/primary.
_STATUS_COLOR = {
    "new": "gray",
    "in_progress": "blue",
    "matched": "violet",
    "resolved": "green",
    "escalated": "red",
    "awaiting_input": "orange",
    "stopped": "gray",
    "deleted": "gray",
    "error": "red",
    "needs_review": "orange",
}

# Human-readable labels — awaiting_input is highlighted as the action-required state.
# Unknown statuses fall back to ``status.upper()`` via the .get() in _status_badge.
_STATUS_LABEL = {
    "new": "NEW",
    "in_progress": "IN PROGRESS",
    "matched": "MATCHED",
    "resolved": "RESOLVED",
    "escalated": "ESCALATED",
    "awaiting_input": "⚠ NEEDS INPUT",
    "stopped": "STOPPED",
    "error": "⚠ FAILED",
    "needs_review": "⚠ NEEDS REVIEW",
}

def _badge(label: str, color: str) -> None:
    """Render an inline coloured pill via st.badge.

    Centralised so the small label/colour decisions live in one place and
    the rest of the UI can call ``_status_badge(...)`` etc. without
    touching the palette dicts directly.
    """
    st.badge(label, color=color)


def _status_badge(status: str | None) -> None:
    if not status:
        return
    _badge(_STATUS_LABEL.get(status, status.upper()),
           _STATUS_COLOR.get(status, "gray"))


def _generic_badge(value: str | None,
                   badge_map: dict[str, UIBadge]) -> None:
    """Render a config-driven badge.

    ``badge_map`` is the inner dict from
    :class:`runtime.config.UIConfig.badges` for one field. Unknown
    values render as an upper-cased gray pill so an unconfigured value
    still surfaces something.
    """
    if not value:
        return
    badge = badge_map.get(value.lower())
    if badge:
        _badge(badge.label, badge.color)
    else:
        _badge(value.upper(), "gray")


def _age(ts: str) -> str:
    """Compact relative-time label: ``2m`` / ``5h`` / ``3d`` / ``2w``."""
    started = _parse_iso(ts)
    if not started:
        return ""
    delta = datetime.now(timezone.utc).replace(tzinfo=None) - started
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{max(seconds, 0)}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 604800:
        return f"{seconds // 86400}d"
    return f"{seconds // 604800}w"


def _badge_md(label: str | None, palette: dict[str, str]) -> str:
    """Inline markdown badge using Streamlit's ``:color-badge[]`` syntax."""
    if not label:
        return ""
    color = palette.get(label, "gray")
    return f":{color}-badge[{label}]"


def _palette_from_badges(badge_map: dict[str, UIBadge]) -> dict[str, str]:
    """Flatten a ``dict[str, UIBadge]`` into ``{value: color}`` for
    :func:`_badge_md`. Keeps the markdown-badge helper agnostic of the
    config layer.
    """
    return {k: v.color for k, v in badge_map.items()}


def _badge_field_slots(app_cfg: FrameworkAppConfig) -> tuple[str, str]:
    """Pick the two badge fields the row + detail panes render.

    The runtime is agnostic to which field names an app uses; we just
    take the configured ``app_cfg.ui.badges`` keys in declared order.
    Apps that configure fewer than two fields get blank slots so the
    layout stays stable.
    """
    keys = [k for k in app_cfg.ui.badges if k != "status"]
    primary = keys[0] if keys else ""
    secondary = keys[1] if len(keys) > 1 else ""
    return primary, secondary


def _field(item: dict, key: str, default: str = "") -> str:
    """Read a domain field from a session dict — top-level first, falling
    back to ``extra_fields``. Returns a string; non-string values are
    coerced via ``str``.

    Bare ``runtime.state.Session`` rounds app-specific values
    (``query``, ``environment``, ``summary``, ...) through
    ``extra_fields``. Typed state subclasses keep them at the top
    level. This helper lets the UI work against either shape without
    branching at every call site.
    """
    val = item.get(key)
    if not val:
        val = (item.get("extra_fields") or {}).get(key)
    if val is None or val == "":
        return default
    return val if isinstance(val, str) else str(val)


def _resolve_field(item: dict, dotted_key: str) -> str:
    """Resolve a dotted-path key against ``item.extra_fields`` (preferred)
    or the item dict itself.

    Returns ``""`` for missing values so callers can use the result
    directly in markdown without ``None`` checks.
    """
    parts = dotted_key.split(".")
    cur: object = item.get("extra_fields", item)
    for p in parts:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(p, "")
    return str(cur) if cur else ""


def _fmt_tokens_short(n: int) -> str:
    """Compact form for the sidebar: ``12.3k`` / ``842``."""
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n}"


def _render_session_row(sess: dict, store: SessionStore,
                        app_cfg: FrameworkAppConfig) -> None:
    """One session row as an expander.

    Header shows session id + env badge + status badge. The body holds
    configured-field badges (driven by ``app_cfg.ui.badges``), age +
    tokens, an excerpt of the summary, and Open + delete actions. The
    currently-selected session is expanded by default so its details
    are visible without an extra click.
    """
    session_id = sess["id"]
    status = sess.get("status") or ""
    env = _field(sess, "environment", "")
    age = _age(sess.get("created_at", ""))
    sev_field, cat_field = _badge_field_slots(app_cfg)
    sev = (sess.get(sev_field) or "") if sev_field else ""
    cat = (sess.get(cat_field) or "") if cat_field else ""
    summary = (_field(sess, "summary") or _field(sess, "query") or "").strip()
    toks = (sess.get("token_usage") or {}).get("total_tokens", 0)
    is_deleted = status == "deleted"
    is_selected = st.session_state.get("selected_session") == session_id

    sev_palette = _palette_from_badges(
        app_cfg.ui.badges.get(sev_field, {}) if sev_field else {}
    )
    cat_palette = _palette_from_badges(
        app_cfg.ui.badges.get(cat_field, {}) if cat_field else {}
    )

    header = (
        f"`{session_id}` "
        f"{_badge_md(env or None, _ENV_COLOR)} "
        f"{_badge_md(status, _STATUS_COLOR)} "
        f"`{_fmt_tokens_short(toks)}`"
    ).strip()

    # The delete button sits OUTSIDE the expander but on the same row,
    # which is the closest Streamlit gets to "delete in the header" —
    # ``st.expander`` labels can only be markdown text, not widgets.
    exp_col, del_col = st.columns([10, 1], gap="small")
    with del_col:
        if not is_deleted:
            if st.button("×", key=f"del_{session_id}",
                         help="Soft-delete",
                         type="tertiary",
                         use_container_width=True):
                store.delete(session_id)
                if st.session_state.get("selected_session") == session_id:
                    st.session_state.pop("selected_session", None)
                st.rerun()

    with exp_col, st.expander(header, expanded=is_selected):
        # Line 1: severity + category + age + updated-time. The time
        # carries a hover tooltip with the full timestamp and any
        # configured submitter-style attribution, so the inline row
        # stays compact.
        updated = sess.get("updated_at") or ""
        meta_bits = [b for b in (
            _badge_md(sev.lower() if sev else None, sev_palette),
            _badge_md(cat or None, cat_palette),
        ) if b]
        if age:
            meta_bits.append(f"`{age}`")
        if updated and len(updated) >= 16:
            time_str = updated[11:16]  # "HH:MM" out of "YYYY-MM-DDTHH:MM:SSZ"
            tooltip_parts = [f"updated {updated}"]
            attrib = _summary_attribution(sess, app_cfg)
            if attrib:
                tooltip_parts.append(attrib)
            tooltip = " · ".join(tooltip_parts)
            meta_bits.append(
                f'<span title="{tooltip}">`{time_str}`</span>'
            )
        if meta_bits:
            st.markdown(
                " · ".join(meta_bits),
                unsafe_allow_html=True,
            )

        if summary:
            short = summary if len(summary) <= 160 else summary[:157] + "…"
            st.markdown(f"> {short}")

        if st.button("Open", key=f"sess_{session_id}",
                     type="primary",
                     use_container_width=True):
            st.session_state["selected_session"] = session_id


def _summary_attribution(sess: dict, app_cfg: FrameworkAppConfig) -> str:
    """Build a one-line attribution string from configured summary fields.

    Reads :attr:`runtime.config.UIConfig.detail_fields` filtered to
    ``section == "summary"``. The first non-empty field becomes the
    leading ``by <value>`` clause; further fields render as
    parenthetical context.
    """
    parts: list[str] = []
    for field in app_cfg.ui.detail_fields:
        if field.section != "summary":
            continue
        v = _resolve_field(sess, field.key)
        if v:
            parts.append(v)
    if not parts:
        return ""
    head, *rest = parts
    suffix = f" ({', '.join(rest)})" if rest else ""
    return f"by {head}{suffix}"


def _render_active_row(active: dict) -> None:
    """One in-flight session as a button row.

    The row reuses the same ``selected_session`` session-state key as the
    history rows so clicking flips the detail pane to the live session.
    The status badge mirrors the same palette as history so the visual
    grammar stays consistent — ``in_progress`` reads the same colour
    whether the run is live or finished.
    """
    sid = active.get("session_id") or ""
    status = active.get("status") or ""
    current = active.get("current_agent") or ""
    age = _age(active.get("started_at", ""))

    bits = [_badge_md(status, _STATUS_COLOR)]
    if current:
        bits.append(f"`{current}`")
    if age:
        bits.append(f"`{age}`")
    label = " ".join([f"`{sid}`"] + bits).strip()

    is_selected = st.session_state.get("selected_session") == sid
    btn_type = "primary" if is_selected else "secondary"
    # Markdown badges don't render inside ``st.button`` labels, so split
    # the row into a markdown line + a compact "Open" button beneath.
    st.markdown(label, unsafe_allow_html=True)
    if st.button("Open", key=f"active_open_{sid}",
                 type=btn_type, use_container_width=True):
        st.session_state["selected_session"] = sid
        st.rerun()


def render_sidebar(store: SessionStore,
                   app_cfg: FrameworkAppConfig,
                   service: OrchestratorService | None = None) -> None:
    """Render the always-on sidebar with two sections: in-flight + history.

    The **In-flight** block reads the live registry from
    ``OrchestratorService.list_active_sessions`` so concurrent runs show
    up immediately. The **History** block is paginated sessions pulled
    from ``SessionStore.list_recent`` with a status filter.

    ``service`` is optional so this function stays callable from headless
    smoke tests / unit tests that don't spin up the orchestrator.
    """
    with st.sidebar:
        # ------------------------------------------------------------
        # In-flight section (live)
        # ------------------------------------------------------------
        active: list[dict] = []
        if service is not None:
            try:
                active = service.list_active_sessions()
            except Exception:  # pragma: no cover - defensive
                # If the loop was stopped or a snapshot timed out, fail
                # closed: hide the section rather than crash the sidebar.
                active = []
        if active:
            st.markdown("### In-flight")
            for entry in active:
                _render_active_row(entry)
            st.markdown("---")

        # ------------------------------------------------------------
        # History section (existing)
        # ------------------------------------------------------------
        head_l, head_r = st.columns([4, 1])
        with head_l:
            st.markdown("### History")
        with head_r:
            if st.button("↻", help="Refresh", type="tertiary",
                         use_container_width=True):
                st.rerun()
        show_deleted = st.checkbox("Show deleted", value=False,
                                   key="show_deleted")
        statuses = ["all", "new", "in_progress", "matched", "resolved",
                    "escalated", "awaiting_input", "needs_review",
                    "stopped", "error"]
        if show_deleted:
            statuses.append("deleted")
        status_filter = st.selectbox(
            "Filter", statuses, key="status_filter",
            label_visibility="collapsed",
        )

        recent = [i.model_dump()
                  for i in store.list_recent(50, include_deleted=show_deleted)]
        if status_filter != "all":
            recent = [i for i in recent if i["status"] == status_filter]

        if not recent:
            st.caption("No sessions.")
            return
        for sess in recent[:20]:
            _render_session_row(sess, store, app_cfg)


def _render_kv_dict_value(key: str, v: dict) -> None:
    """Render a dict value inside _render_kv_block as a nested bordered card."""
    st.markdown(f"**{key}:**")
    with st.container(border=True):
        _render_kv_block(v)


def _render_kv_list_value(key: str, v: list) -> None:
    """Render a list value inside _render_kv_block as bullets or nested cards."""
    st.markdown(f"**{key}:**")
    for item in v:
        if isinstance(item, dict):
            with st.container(border=True):
                _render_kv_block(item)
        else:
            st.markdown(f"- {item}")


def _render_kv_scalar_value(key: str, v) -> None:
    """Render a bool or plain scalar value inside _render_kv_block."""
    if isinstance(v, bool):
        st.markdown(f"**{key}:** `{str(v).lower()}`")
    else:
        st.markdown(f"**{key}:** {v}")


def _render_kv_block(d: dict) -> None:
    """Render a dict as labeled markdown lines, recursing into nested
    dicts and lists.

    Replaces ``st.json`` everywhere structured agent output bleeds into
    the main detail panel — the JSON braces look out of place between
    bordered cards and prose. The deliberately raw views (the "Args"
    block on a tool call and the bottom-of-page "Raw JSON" expander)
    keep ``st.json`` because they exist precisely to expose the wire
    shape.
    """
    for k, v in d.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        key = k.replace("_", " ").capitalize()
        if isinstance(v, dict):
            _render_kv_dict_value(key, v)
        elif isinstance(v, list):
            _render_kv_list_value(key, v)
        else:
            _render_kv_scalar_value(key, v)


def _render_value_list(v: list) -> None:
    """Render a non-empty list value inside _render_value."""
    for i, item in enumerate(v):
        if isinstance(item, dict):
            if i > 0:
                st.markdown("---")
            _render_kv_block(item)
        else:
            st.markdown(f"- {item}")


def _render_value(v) -> None:
    """Render an agent-produced value (str / dict / list / None) safely.

    Strings → ``st.write`` (markdown-aware). Dicts and lists → the labeled
    block renderer above. ``st.json`` is reserved for the explicit raw
    viewers; routing everything through it leaks JSON braces into the
    prose-y main flow.
    """
    if v is None:
        st.caption("_(none)_")
    elif isinstance(v, dict):
        _render_kv_block(v)
    elif isinstance(v, list):
        if not v:
            st.caption("_(empty)_")
            return
        _render_value_list(v)
    else:
        st.write(v)


def _parse_iso(ts: str) -> datetime | None:
    """Parse the project's ISO-like timestamp (`YYYY-MM-DDTHH:MM:SSZ`)."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def _duration_seconds(start: str, end: str) -> int:
    s, e = _parse_iso(start), _parse_iso(end)
    if not s or not e:
        return 0
    return max(0, int((e - s).total_seconds()))


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _fmt_duration(seconds: int) -> str:
    """Compact duration: ``42s``, ``3m 5s``, ``1h 12m``, ``2d 4h``.

    Sub-minute values stay as raw seconds; everything longer rolls up
    into the largest two units so the metric stays readable for any
    INC, from a 12s investigation to a multi-day awaiting-input pause.
    """
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"


def _fmt_confidence_badge(conf: float | None) -> str:
    """Inline coloured badge for an agent confidence value.

    Green ≥0.75, amber 0.5–0.75, red <0.5. Markdown only — no HTML — so the
    badge survives Streamlit's sanitizer.

    Phase 10 (FOC-03): None now indicates a structural failure (envelope
    missing) — visually flag with a red 🛑 hard-error badge, never the
    silent ⚪ fallback. The runner rejects envelope-less turns upfront;
    None here means a legacy on-disk row predating the envelope contract.
    """
    if conf is None:
        return "🛑 confidence missing"
    if conf >= 0.75:
        glyph = "🟢"
    elif conf >= 0.5:
        glyph = "🟡"
    else:
        glyph = "🔴"
    return f"{glyph} confidence {conf:.2f}"


def _render_one_hypothesis(h, label: str, idx: int) -> None:
    """Render a single hypothesis item inside a pre-existing bordered card.

    Dict entries are expanded to cause / evidence / next_steps / extra keys.
    Scalars fall back to a simple bullet line.
    """
    if isinstance(h, dict):
        st.markdown(f"**{label} {idx}:** {h.get('cause', '—')}")
        ev = h.get("evidence")
        if ev:
            st.markdown("**Evidence:**")
            for e in (ev if isinstance(ev, list) else [ev]):
                st.markdown(f"- {e}")
        ns = h.get("next_steps") or h.get("next_step") or h.get("probe")
        if ns:
            st.markdown(f"**Next steps:** {ns}")
        extra = {k: v for k, v in h.items()
                 if k not in {"cause", "evidence", "next_steps",
                              "next_step", "probe"}}
        if extra:
            _render_kv_block(extra)
    else:
        st.markdown(f"**{label} {idx}:** {h}")


def _render_hypothesis_list(items: list, label: str) -> None:
    """Render a list of hypothesis-shaped dicts (cause/evidence/next_steps)
    as bordered cards. Strings or scalar entries fall back to bullets.
    """
    for i, h in enumerate(items, 1):
        with st.container(border=True):
            _render_one_hypothesis(h, label, i)


def _render_top_badges(sess: dict, app_cfg: FrameworkAppConfig) -> None:
    """Render status + two configured-field badge columns plus the
    awaiting-input warning.

    The two extra slots read whichever fields the app configures under
    ``app_cfg.ui.badges`` (typically a primary axis and a secondary
    classifier). Status uses the framework's built-in palette so
    live/historic states stay consistent across apps.
    """
    sev_field, cat_field = _badge_field_slots(app_cfg)
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        st.caption("Status")
        _status_badge(sess.get("status"))
    with b2:
        st.caption(sev_field.title() if sev_field else "")
        if sev_field:
            _generic_badge(sess.get(sev_field),
                           app_cfg.ui.badges.get(sev_field, {}))
    with b3:
        st.caption(cat_field.title() if cat_field else "")
        if cat_field:
            _generic_badge(sess.get(cat_field),
                           app_cfg.ui.badges.get(cat_field, {}))
    if sess.get("status") == "awaiting_input":
        _pi = sess.get("pending_intervention") or {}
        _upstream = _pi.get("upstream_agent") or "the upstream agent"
        st.warning(
            f"**Human intervention required.** The intervention gate "
            f"paused this session because {_upstream} confidence was "
            f"below the configured threshold. Use the controls below to "
            f"resume with input, escalate, or stop."
        )


def _render_metrics(sess: dict) -> None:
    """Render tokens / active duration / total time metric columns.

    - **Active duration** is the sum of each agent run's wall-clock —
      reflects time the orchestrator was actually working.
    - **Total time** is ``created_at → updated_at`` — wall-clock from
      first report to the last write (covers awaiting-input pauses).
    """
    token_total = (sess.get("token_usage") or {}).get("total_tokens", 0)
    runs = sess.get("agents_run") or []
    active_s = sum(
        _duration_seconds(r.get("started_at", ""), r.get("ended_at", ""))
        for r in runs
    )
    total_s = _duration_seconds(sess.get("created_at", ""),
                                sess.get("updated_at", ""))
    if active_s == 0:
        active_s = total_s
    m1, m2, m3 = st.columns(3)
    m1.metric("Total tokens", _fmt_tokens(token_total))
    m2.metric("Active duration", _fmt_duration(active_s))
    m3.metric("Total time", _fmt_duration(total_s))


def _render_prior_match(sess: dict, app_cfg: FrameworkAppConfig) -> None:
    """Render the prior-session callout when ``matched_prior_inc`` is set.

    Tag strings are resolved from :attr:`runtime.config.UIConfig.tags`
    so apps without a prior-match concept can leave the keys unset and
    skip the panel entirely.
    """
    supported = app_cfg.ui.tags.get("prior_match_supported", "")
    rejected = app_cfg.ui.tags.get("prior_match_rejected", "")
    if not supported and not rejected:
        # App doesn't use the prior-match feature; render nothing.
        return
    tags = sess.get("tags") or []
    if supported and supported in tags:
        stance = "supported by current evidence"
        callout = st.success
    elif rejected and rejected in tags:
        stance = "rejected — fresh evidence diverges from prior cause"
        callout = st.warning
    else:
        stance = "not yet validated"
        callout = st.info
    callout(
        f"**Prior similar session (hypothesis):** "
        f"`{sess['matched_prior_inc']}` — {stance}.  \n"
        f"_Same symptom can have different root causes "
        f"(code bug vs. network vs. resource overload), so the prior "
        f"cause is one ranked hypothesis for the deep investigator — "
        f"not the answer._"
    )


def _render_summary_meta(sess: dict, app_cfg: FrameworkAppConfig) -> None:
    """Render the query, environment, tags, summary, configured detail
    fields, and prior-match callout.
    """
    query = _field(sess, "query")
    if query:
        st.markdown(f"**Query:** {query}")
    env = _field(sess, "environment")
    if env:
        st.markdown(f"**Environment:** `{env}`")
    # App-configured summary fields (e.g. submitter id / team / component).
    for field in app_cfg.ui.detail_fields:
        if field.section != "summary":
            continue
        v = _resolve_field(sess, field.key)
        if v:
            st.markdown(f"**{field.label}:** {v}")
    if sess.get("tags"):
        st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in sess["tags"]))
    summary = _field(sess, "summary")
    if summary:
        st.markdown(f"**Summary:** {summary}")
    escalated_to = _field(sess, "escalated_to")
    if escalated_to:
        st.markdown(f"**Escalated to:** `{escalated_to}`")
    extra = sess.get("extra_fields") or {}
    needs_review_reason = extra.get("needs_review_reason")
    legacy_auto_resolved = extra.get("auto_resolved")
    if needs_review_reason or legacy_auto_resolved:
        msg = needs_review_reason or "session was auto-resolved by the legacy finalizer"
        st.warning(
            "⚠ This session needs review: "
            f"{msg}. The graph completed without the agent "
            "calling a terminal tool — verify the actual outcome before "
            "closing."
        )
    if sess.get("matched_prior_inc"):
        _render_prior_match(sess, app_cfg)


def _render_agents_run_block(sess: dict) -> None:
    """Render the ### Agents run section if agents_run is non-empty."""
    agents_run = sess.get("agents_run", [])
    if not agents_run:
        return
    st.markdown("### Agents run")
    for ar in agents_run:
        a_dur = _duration_seconds(ar.get("started_at", ""),
                                  ar.get("ended_at", ""))
        a_tok = (ar.get("token_usage") or {}).get("total_tokens", 0)
        conf = ar.get("confidence")
        badge = _fmt_confidence_badge(conf)
        with st.container(border=True):
            st.markdown(
                f"**{ar['agent']}** — {_fmt_duration(a_dur)} — "
                f"{_fmt_tokens(a_tok)} tokens — {badge}"
            )
            rationale = ar.get("confidence_rationale")
            if rationale:
                st.caption(f"Why: {rationale}")
            st.write(ar.get("summary") or "_(no summary)_")


def _is_hypothesis_list(value) -> bool:
    """Return True when value is a non-empty list of dicts with a 'cause' key."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], dict)
        and "cause" in value[0]
    )


def _render_findings_block(sess: dict) -> None:
    """Render the ### Findings section for every agent that produced findings.

    Iterates over all entries in ``findings`` so custom YAML-defined agents
    are shown alongside built-in ones. Values that look like hypothesis lists
    (list of dicts with a ``cause`` key) are passed to ``_render_hypothesis_list``;
    everything else goes to ``_render_value``.
    """
    findings = sess.get("findings") or {}
    if not findings:
        return
    st.markdown("### Findings")
    for agent_name, value in findings.items():
        if value is None:
            continue
        with st.container(border=True):
            st.markdown(f"**{agent_name}**")
            if _is_hypothesis_list(value):
                _render_hypothesis_list(value, label="Hypothesis")
            else:
                _render_value(value)


def _render_resolution_block(sess: dict) -> None:
    """Render the ### Resolution section if resolution is present."""
    if sess.get("resolution") is None:
        return
    st.markdown("### Resolution")
    with st.container(border=True):
        _render_value(sess["resolution"])


def _render_tool_calls_block(sess: dict) -> None:
    """Render the ### Tool calls section if tool_calls is non-empty."""
    tool_calls = sess.get("tool_calls", [])
    if not tool_calls:
        return
    st.markdown("### Tool calls")
    for tc in tool_calls:
        with st.expander(f"`{tc['agent']}` → `{tc['tool']}`"):
            st.markdown("**Args:**")
            st.json(tc.get("args") or {})
            st.markdown("**Result:**")
            _render_value(tc.get("result"))


def _submit_approval_via_service(
    cfg: AppConfig, session_id: str, tool_call_id: str,
    decision: str, approver: str, rationale: str | None,
) -> None:
    """Resolve a pending tool approval through the OrchestratorService bridge.

    Drives ``Command(resume={...})`` on the persistent service loop so
    we reuse the live FastMCP transports + SQLAlchemy engine — same
    contract as ``POST /sessions/{sid}/approvals/{tool_call_id}``.
    Streamlit reruns will re-fetch the row and the wrap_tool audit
    (status="approved" / "rejected") will be visible.
    """
    from langgraph.types import Command

    svc = _get_service(cfg)
    if svc is None:
        st.error("Orchestrator service is not running; refresh the page.")
        return

    payload = {
        "decision": decision,
        "approver": approver,
        "rationale": rationale,
    }

    async def _drive() -> None:
        # ``_ensure_orchestrator`` is a no-op after the first call; the
        # shared Orchestrator owns the compiled graph + checkpointer.
        orch = await svc._ensure_orchestrator()
        await orch.graph.ainvoke(
            Command(resume=payload),
            config=orch._thread_config(session_id),
        )

    svc.submit_and_wait(_drive(), timeout=60.0)


def _is_hypothesis_trail(value) -> bool:
    """Return True when value is a non-empty list of dicts shaped like a
    triage hypothesis-loop trail.

    Each entry must carry ``iteration`` + ``hypothesis``; ``score`` and
    ``rationale`` are recommended but not required.
    """
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], dict)
        and "iteration" in value[0]
        and "hypothesis" in value[0]
    )


def _render_hypothesis_trail_block(sess: dict) -> None:
    """Render the ### Hypothesis Trail panel.

    Reads-only view over the triage agent's per-iteration trail. The
    triage skill writes a list of ``{iteration, hypothesis, score,
    rationale}`` dicts to ``findings.findings_triage`` (or any finding
    key whose value matches :func:`_is_hypothesis_trail`); this panel
    surfaces them as a collapsed accordion so an operator can audit
    how the hypothesis converged without scrolling through raw JSON.

    No persistent state — pulls everything from ``sess["findings"]``.
    Renders nothing when no trail is present so legacy sessions stay
    clean.
    """
    findings = sess.get("findings") or {}
    if not isinstance(findings, dict):
        return

    trails: list[tuple[str, list[dict]]] = []
    # Look under both the canonical key (``findings_triage``) and any
    # other agent finding shaped like a trail — keeps the panel
    # forward-compatible if other agents adopt the iterative pattern.
    for agent, value in findings.items():
        if _is_hypothesis_trail(value):
            trails.append((agent, value))
    if not trails:
        return

    st.markdown("### Hypothesis Trail")
    for agent, trail in trails:
        with st.expander(f"`{agent}` — {len(trail)} iteration(s)", expanded=False):
            for entry in trail:
                if not isinstance(entry, dict):
                    continue
                iteration = entry.get("iteration", "?")
                hypothesis = entry.get("hypothesis", "_(no hypothesis)_")
                score = entry.get("score")
                rationale = entry.get("rationale", "")
                with st.container(border=True):
                    score_md = (
                        f" · score `{score:.2f}`"
                        if isinstance(score, (int, float))
                        else ""
                    )
                    st.markdown(f"**Iteration {iteration}**{score_md}")
                    st.markdown(hypothesis)
                    if rationale:
                        st.caption(rationale)


def _should_render_retry_block(sess: dict) -> bool:
    """Phase 11 (FOC-04 / D-11-04) predicate.

    The retry block exists for terminally failed sessions only. A
    session in ``status='error'`` that ALSO has a ``pending_approval``
    ToolCall row is genuinely paused on a HITL gate -- the
    pending-approvals block (rendered separately) carries the
    Approve/Reject action; the retry block would be wrong-mode here.
    Returning ``False`` keeps the two blocks mutually exclusive.

    Tolerates both pydantic ``ToolCall`` objects and dict
    representations (Streamlit's ``model_dump`` on the loaded session
    yields dicts, but defensive reads from the live ``Session.tool_calls``
    return pydantic objects).
    """
    if sess.get("status") != "error":
        return False
    for tc in (sess.get("tool_calls") or []):
        status = (
            tc.get("status") if isinstance(tc, dict)
            else getattr(tc, "status", None)
        )
        if status == "pending_approval":
            return False
    return True


def _render_pending_approvals_block(sess: dict, session_id: str) -> None:
    """Render the ### Pending Approvals section for tool calls the
    framework's pure-policy gate has paused for human approval.

    Iterates ``tool_calls`` looking for entries with
    ``status="pending_approval"``. Each pending row gets a small card
    with the tool name + args, a free-text rationale input, and two
    buttons (Approve / Reject) that resolve the pending pause via the
    OrchestratorService bridge.
    """
    tool_calls = sess.get("tool_calls", [])
    pending = [
        (idx, tc) for idx, tc in enumerate(tool_calls)
        if (tc.get("status") if isinstance(tc, dict) else None) == "pending_approval"
    ]
    if not pending:
        return
    cfg = load_config(CONFIG_PATH)
    st.markdown("### Pending Approvals")
    for idx, tc in pending:
        agent = tc.get("agent", "?")
        tool = tc.get("tool", "?")
        with st.container(border=True):
            st.markdown(f"#### 🔒 `{agent}` → `{tool}` (high risk)")
            st.markdown("**Args:**")
            st.json(tc.get("args") or {})
            rationale = st.text_input(
                "Rationale (optional)",
                key=f"approval_rationale_{session_id}_{idx}",
                placeholder="Why are you approving / rejecting?",
            )
            cols = st.columns(2)
            approve = cols[0].button(
                "Approve", type="primary",
                key=f"approval_approve_{session_id}_{idx}",
            )
            reject = cols[1].button(
                "Reject",
                key=f"approval_reject_{session_id}_{idx}",
            )
            if approve or reject:
                decision = "approve" if approve else "reject"
                _submit_approval_via_service(
                    cfg, session_id, str(idx),
                    decision=decision,
                    approver="ui-user",
                    rationale=rationale.strip() or None,
                )
                st.rerun()


def render_session_detail(store: SessionStore,
                          app_cfg: FrameworkAppConfig,
                          agent_names: frozenset[str] = frozenset()) -> None:
    """Render the full detail view for the currently selected session.

    The header carries the session id, environment + status badges, and
    a compact token total. Body sections include configured top
    badges, metrics, summary meta, intervention controls (when paused),
    pending tool approvals, hypothesis trail, agents run, findings,
    resolution, and the raw JSON dump.
    """
    session_id = st.session_state.get("selected_session")
    if not session_id:
        return
    try:
        sess = store.load(session_id).model_dump()
    except FileNotFoundError:
        return
    status = sess.get("status") or ""
    env = _field(sess, "environment") or ""
    toks = (sess.get("token_usage") or {}).get("total_tokens", 0)
    header = (
        f"`{session_id}` "
        f"{_badge_md(env or None, _ENV_COLOR)} "
        f"{_badge_md(status, _STATUS_COLOR)} "
        f"`{_fmt_tokens_short(toks)}`"
    ).strip()
    with st.expander(header, expanded=True):
        _render_top_badges(sess, app_cfg)
        _render_metrics(sess)
        _render_summary_meta(sess, app_cfg)
        if sess.get("status") == "awaiting_input" and sess.get("pending_intervention"):
            _render_intervention_block(sess, session_id, app_cfg, agent_names)
        if _should_render_retry_block(sess):
            _render_retry_block(sess, session_id, agent_names)
        # Pending tool-approval cards (paused via the framework's
        # pure-policy gate; see ``runtime.policy.should_gate``).
        # Rendered above the agents/tool-calls blocks so a paused
        # approval is the first action surface the operator sees.
        _render_pending_approvals_block(sess, session_id)
        # Triage hypothesis-loop audit. Collapsed by default so the
        # agents/findings blocks stay the primary read; the trail is
        # one click away when an operator wants to audit how the triage
        # hypothesis converged.
        _render_hypothesis_trail_block(sess)
        _render_agents_run_block(sess)
        _render_findings_block(sess)
        _render_resolution_block(sess)
        _render_tool_calls_block(sess)
        with st.expander("Raw JSON"):
            st.json(sess)

    # Auto-poll while the session is in flight. The 1.5s nap is a
    # cooperative throttle: it blocks the script-runner thread, so the
    # next ``st.rerun()`` lands on a quiescent rerun cycle. Tests can
    # disable polling by setting ``st.session_state["_disable_poll"] = True``
    # before calling render to keep unit tests deterministic.
    if (
        _should_poll(status)
        and not st.session_state.get("_disable_poll")
    ):
        time.sleep(1.5)
        st.rerun()


def _format_event(ev: dict, agent_names: frozenset[str] = frozenset()) -> str | None:
    """Format a streaming orchestrator event for display in the live timeline.

    ``agent_names`` is the set of all configured agent names (derived from the
    loaded YAML at runtime). When non-empty it replaces the former hardcoded
    whitelist so custom agents appear in the log. When empty every
    on_chain_start/end node is shown (safe fallback for callers that don't have
    the agent list handy).
    """
    kind = ev.get("event")
    node = ev.get("node") or ""
    ts = ev.get("ts", "")
    if kind == "investigation_started":
        return f"[{ts}] start  id={ev.get('incident_id')}"
    if kind == "investigation_completed":
        return f"[{ts}] done   id={ev.get('incident_id')}"
    _node_visible = (not agent_names) or (node in agent_names)
    if kind == "on_chain_start" and _node_visible:
        return f"[{ts}] enter  {node}"
    if kind == "on_chain_end" and _node_visible:
        return f"[{ts}] exit   {node}"
    if kind == "on_tool_start":
        return f"[{ts}] tool   {node}"
    if kind == "on_tool_end":
        result = (ev.get("data") or {}).get("output")
        snippet = str(result)[:120] if result is not None else ""
        return f"[{ts}] tool→  {node} {snippet}"
    return None


async def _run_investigation_async(cfg: AppConfig, query: str, environment: str,
                                   log_area, lines: list[str],
                                   agent_names: frozenset[str] = frozenset()) -> None:
    """Build a fresh Orchestrator, stream events, aclose. One asyncio.run frame."""
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.stream_investigation(query=query, environment=environment):
            line = _format_event(ev, agent_names)
            if line:
                lines.append(line)
                log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()


async def _retry_async(cfg: AppConfig, session_id: str,
                       log_area, lines: list[str],
                       agent_names: frozenset[str] = frozenset()) -> dict:
    """Build a fresh Orchestrator, stream retry events, aclose.

    Returns ``{"rejected": <reason or None>}`` so the caller can render
    a warning when the orchestrator refuses the retry (e.g. session
    isn't in error state).
    """
    outcome: dict = {"rejected": None}
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.retry_session(session_id):
            kind = ev.get("event")
            ts = ev.get("ts", "")
            if kind == "retry_started":
                lines.append(f"[{ts}] retry  attempt #{ev.get('retry_count')}")
            elif kind == "retry_rejected":
                lines.append(f"[{ts}] rejected {ev.get('reason')}")
                outcome["rejected"] = ev.get("reason")
            elif kind == "retry_completed":
                lines.append(f"[{ts}] done")
            else:
                line = _format_event(ev, agent_names)
                if line:
                    lines.append(line)
            log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()
    return outcome


async def _resume_async(cfg: AppConfig, session_id: str, decision: dict,
                        log_area, lines: list[str],
                        agent_names: frozenset[str] = frozenset()) -> dict:
    """Build a fresh Orchestrator, stream resume events, aclose.

    Returns a small summary dict describing the outcome so the caller can show
    a banner: ``{"rejected": <reason or None>}``. ``rejected`` is set when the
    orchestrator emits a ``resume_rejected`` event (e.g. session no longer
    awaiting_input, invalid escalation team).
    """
    outcome: dict = {"rejected": None}
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.resume_investigation(session_id, decision):
            kind = ev.get("event")
            ts = ev.get("ts", "")
            if kind == "resume_started":
                lines.append(f"[{ts}] resume {ev.get('action')}")
            elif kind == "resume_completed":
                lines.append(f"[{ts}] done   status={ev.get('status')}")
            elif kind == "resume_rejected":
                lines.append(f"[{ts}] rejected {ev.get('reason')}")
                outcome["rejected"] = ev.get("reason")
            elif kind == "resume_failed":
                lines.append(f"[{ts}] failed {ev.get('error')}")
                outcome["rejected"] = ev.get("error")
            else:
                line = _format_event(ev, agent_names)
                if line:
                    lines.append(line)
            log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()
    return outcome


def _render_retry_block(sess: dict, session_id: str,
                        agent_names: frozenset[str] = frozenset()) -> None:
    """Render a retry control for failed sessions.

    Sessions land in ``status="error"`` when a graph node raises and
    the framework's auto-retry on transient 5xxs (see
    :data:`runtime.graph._TRANSIENT_MARKERS`) has already been
    exhausted. Surfaces the failed agent + the recorded exception so
    the operator can decide whether to retry.
    """
    cfg = load_config(CONFIG_PATH)
    failed_run = next(
        (r for r in reversed(sess.get("agents_run") or [])
         if (r.get("summary") or "").startswith("agent failed:")),
        None,
    )
    failed_agent = (failed_run or {}).get("agent", "unknown")
    failure_msg = ((failed_run or {}).get("summary") or "").removeprefix("agent failed:").strip()
    retry_count = int((sess.get("extra_fields") or {}).get("retry_count", 0))
    with st.container(border=True):
        st.markdown(f"#### 🔴 Agent failed — `{failed_agent}`")
        if failure_msg:
            st.caption(f"Last error: {failure_msg}")
        if retry_count:
            st.caption(f"Previous retry attempts: {retry_count}")
        st.caption(
            "Retry re-runs the graph from the entry node. The framework "
            "already retried transient 5xx errors automatically — this "
            "is for cases where the underlying issue may now be cleared "
            "(provider hiccup, transient network, etc.)."
        )
        if st.button("Retry", type="primary", key=f"retry_btn_{session_id}"):
            log_area = st.empty()
            lines: list[str] = []
            outcome = asyncio.run(_retry_async(
                cfg, session_id, log_area, lines, agent_names,
            ))
            if outcome.get("rejected"):
                st.warning(f"Retry rejected: {outcome['rejected']}")
                return
            st.success("Retry complete.")
            st.rerun()


def _render_intervention_block(sess: dict, session_id: str,
                               app_cfg: FrameworkAppConfig,
                               agent_names: frozenset[str] = frozenset()) -> None:
    """Render the intervention prompt above the agents_run section.

    Shows the confidence vs. threshold and a single form with an action
    selector that swaps inputs (text box / team dropdown / nothing) and a
    submit button. On submit, calls `_resume_async` and then reruns.
    """
    cfg = load_config(CONFIG_PATH)
    pi = sess.get("pending_intervention") or {}
    conf = pi.get("confidence")
    threshold = pi.get("threshold", app_cfg.confidence_threshold)
    teams = pi.get("escalation_teams") or list(app_cfg.escalation_teams)

    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
    summary = (pi.get("summary") or "").strip()
    rationale = (pi.get("rationale") or "").strip()
    with st.container(border=True):
        st.markdown(
            f"#### 🟠 Intervention required — confidence {conf_str} "
            f"< threshold {threshold:.2f}"
        )
        if summary:
            st.markdown(f"**Investigator summary** — {summary}")
        if rationale:
            st.markdown(f"**Why confidence is low** — {rationale}")
        if not summary and not rationale:
            st.caption(
                "The deep investigator's confidence is below the configured "
                "threshold. Choose how to proceed."
            )
        action = st.selectbox(
            "Action", ["resume_with_input", "escalate", "stop"],
            key=f"intervention_action_{session_id}",
        )
        decision: dict = {"action": action}
        if action == "resume_with_input":
            decision["input"] = st.text_area(
                "Add context for the investigator", height=120,
                placeholder="Anything the agent should know — recent changes, "
                            "logs you've already checked, suspected services…",
                key=f"intervention_input_{session_id}",
            )
        elif action == "escalate":
            decision["team"] = st.selectbox(
                "Escalate to team", teams, key=f"intervention_team_{session_id}",
            )

        submit = st.button("Submit", type="primary",
                           key=f"intervention_submit_{session_id}")
        if submit:
            if action == "resume_with_input" and not (decision.get("input") or "").strip():
                st.warning("Add some context before resuming.")
                return
            log_area = st.empty()
            lines: list[str] = []
            outcome = asyncio.run(_resume_async(cfg, session_id, decision, log_area, lines, agent_names))
            if outcome.get("rejected"):
                # Don't auto-rerun — let the user read the warning before the
                # form goes away. Common causes: session already closed, invalid
                # escalation team, or a sub-graph exception that restored the
                # session to awaiting_input.
                st.warning(f"Resume rejected: {outcome['rejected']}")
                return
            st.success(f"Resume complete (action: {action}).")
            st.rerun()


def _render_agents_accordion(agents: list[dict], per_row: int = 2) -> None:
    """Each agent as an expander, laid out in a ``per_row``-wide grid.

    ``st.columns`` collapses to a single stacked column on narrow
    viewports, so wide screens see a 2-up grid of expanders and mobile
    sees them stacked vertically — responsive without extra CSS.
    """
    for i in range(0, len(agents), per_row):
        cols = st.columns(per_row, gap="small")
        for j in range(per_row):
            with cols[j]:
                if i + j >= len(agents):
                    continue
                a = agents[i + j]
                label = f"{a['name']} :blue-badge[{a['model']}]"
                with st.expander(label, expanded=(i + j == 0)):
                    st.caption(a["description"])
                    if a["tools"]:
                        st.markdown(
                            "**Tools** · "
                            + " · ".join(f"`{t}`" for t in a["tools"])
                        )
                    if a["routes"]:
                        st.caption(
                            "Routes · " + ", ".join(
                                f"`{r['when']}→{r['next']}`"
                                for r in a["routes"]
                            )
                        )


_LOCALITY_COLOR = {"local": "green", "remote": "orange", "mixed": "violet"}
_ENV_COLOR = {
    "production": "red",
    "prod": "red",
    "staging": "orange",
    "stage": "orange",
    "dev": "blue",
    "local": "gray",
}


def _render_tools_by_category(tools: list[dict], per_row: int = 2) -> None:
    """Render each ``category`` as an inner expander, laid out in a
    ``per_row``-wide grid so categories sit side-by-side on wide screens
    and stack vertically on narrow ones.

    Designed to live inside the outer "Tools" expander on the registry
    tab. Streamlit 1.42+ permits one level of expander nesting.
    """
    by_cat: dict[str, list[dict]] = {}
    for t in tools:
        by_cat.setdefault(t["category"], []).append(t)
    cats = sorted(by_cat)
    for i in range(0, len(cats), per_row):
        cols = st.columns(per_row, gap="small")
        for j in range(per_row):
            with cols[j]:
                if i + j >= len(cats):
                    continue
                cat = cats[i + j]
                items = by_cat[cat]
                tag = _category_locality_tag(items)
                color = _LOCALITY_COLOR.get(tag, "gray")
                label = f"{cat} ({len(items)}) :{color}-badge[{tag}]"
                with st.expander(label, expanded=(i + j == 0)):
                    for t in items:
                        name = t.get("original_name", t["name"])
                        st.markdown(f"- **{name}** — {t['description'][:80]}")


def _category_locality_tag(items: list[dict]) -> str:
    """Classify a tool category by transport: local / remote / mixed.

    ``in_process`` is local; anything else (``http`` / ``sse`` / ``stdio``)
    is remote. Mixed categories flag both kinds being co-located.
    """
    transports = {t.get("transport", "unknown") for t in items}
    if transports == {"in_process"}:
        return "local"
    if "in_process" not in transports:
        return "remote"
    return "mixed"


_SIDEBAR_MIN_WIDTH_PX = 500


def _inject_global_css() -> None:
    """Pin the sidebar to a minimum width when expanded.

    Without the ``[aria-expanded="true"]`` predicate the min-width sticks
    around after the user collapses the sidebar, leaving a blank gap on
    the left of the main content area.

    On viewports narrower than the desktop breakpoint we (a) drop the
    min-width pin so the sidebar fits the screen, (b) cap the sidebar
    at the viewport width so the close affordance never falls off
    screen, and (c) keep streamlit's collapse control reachable above
    any sticky content the sidebar may render.
    """
    st.markdown(
        f"""
        <style>
        @media (min-width: 768px) {{
            section[data-testid="stSidebar"][aria-expanded="true"] {{
                min-width: {_SIDEBAR_MIN_WIDTH_PX}px !important;
            }}
        }}
        @media (max-width: 767px) {{
            section[data-testid="stSidebar"][aria-expanded="true"] {{
                min-width: 0 !important;
                width: 100vw !important;
                max-width: 100vw !important;
            }}
            button[data-testid="stSidebarCollapseButton"],
            button[data-testid="baseButton-headerNoPadding"][kind="headerNoPadding"] {{
                position: fixed !important;
                top: 0.5rem !important;
                right: 0.5rem !important;
                z-index: 999999 !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    _inject_global_css()
    cfg = load_config(CONFIG_PATH)
    app_cfg = _load_app_cfg(cfg)
    store = _make_repository(cfg)

    # Process-singleton background service for the in-flight session
    # list. ``_get_service`` returns ``None`` when called outside a
    # Streamlit script context (e.g. importability tests), in which
    # case the sidebar simply omits the in-flight section.
    service = _get_service(cfg)

    # One-shot snapshot of agent/tool metadata + environments. ~100-200ms per
    # rerun; acceptable, and keeps async resources strictly scoped.
    agents, tools, environments = _load_metadata_dicts(cfg)
    agent_names: frozenset[str] = frozenset(a["name"] for a in agents)

    render_sidebar(store, app_cfg, service)

    tab_investigate, tab_registry = st.tabs(["Investigate", "Agents & Tools"])

    with tab_investigate:
        st.header("Start an investigation")
        with st.form("investigate_form"):
            query = st.text_area("What's happening?", height=100, key="form_query")
            environment = st.selectbox("Impacted environment", environments,
                                       key="form_env")
            submitted = st.form_submit_button("Start investigation", type="primary")

        if submitted and query.strip():
            # Hide any previously-selected session immediately — the detail
            # panel below won't be reached this script-run, so during the
            # blocking asyncio.run the user sees only the live timeline.
            st.session_state.pop("selected_session", None)

            timeline_box = st.container()
            timeline_box.markdown("### Live timeline")
            log_area = timeline_box.empty()
            lines: list[str] = []

            try:
                asyncio.run(_run_investigation_async(cfg, query, environment, log_area, lines, agent_names))
            except Exception as _e:  # noqa: BLE001
                if _e.__class__.__name__ == "SessionBusy":
                    st.warning("Session is busy — please retry in a moment.", icon=":material/hourglass_empty:")
                    return
                raise

            # Surface the resulting session for one-click drill-in
            recent = [i.model_dump() for i in store.list_recent(1)]
            if recent:
                st.session_state["selected_session"] = recent[0]["id"]
                st.success(f"Investigation complete — {recent[0]['id']} ({recent[0]['status']})")
                st.rerun()
        else:
            render_session_detail(store, app_cfg, agent_names)

    with tab_registry:
        st.header("Agents & Tools registry")

        with st.expander("Tools", expanded=True):
            _render_tools_by_category(tools)

        st.subheader("Agents")
        _render_agents_accordion(agents)


if __name__ == "__main__":
    main()
