from __future__ import annotations
# ----- imports for ui/streamlit_app.py -----
"""Streamlit UI — 2 tabs + always-on sidebar with recent INCs."""
# Lifecycle note: the Orchestrator owns FastMCP clients tied to a specific
# asyncio event loop. Streamlit re-runs the script on every interaction and we
# use `asyncio.run(...)` per call, which creates a fresh loop each time.
# Caching an Orchestrator across reruns would leave its clients/transports
# bound to a dead loop and the first tool call would raise:
#     unable to perform operation on <TCPTransport closed=True ...>
# So: build a fresh Orchestrator inside each `asyncio.run` and `aclose` it when
# done. For pure metadata views (agents/tools) we use `_load_metadata_dicts` —
# a one-shot fetch that captures plain dicts and disposes the orchestrator.
# The sidebar uses IncidentRepository directly for sync list/load/delete calls
# that need no MCP clients. It builds the repo from the same config the
# Orchestrator uses so both share the same SQLite DB.

from app import AppConfig, Base, IncidentRepository, MetadataConfig, Orchestrator, build_embedder, build_engine, load_config
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st








# ====== module: ui/streamlit_app.py ======

CONFIG_PATH = Path("config/config.yaml")


def _make_repository(cfg: AppConfig) -> IncidentRepository:
    """Build an IncidentRepository from config — mirrors Orchestrator.create logic."""

    default_url = MetadataConfig().url
    url = (
        cfg.storage.metadata.url if cfg.storage.metadata.url != default_url
        else f"sqlite:///{Path(cfg.paths.incidents_dir) / 'incidents.db'}"
    )
    engine = build_engine(MetadataConfig(url=url, pool_size=cfg.storage.metadata.pool_size, echo=cfg.storage.metadata.echo))
    Base.metadata.create_all(engine)
    embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
    return IncidentRepository(
        engine=engine,
        embedder=embedder,
        similarity_threshold=cfg.incidents.similarity_threshold,
    )


def _load_metadata_dicts(cfg: AppConfig) -> tuple[list[dict], list[dict], list[str]]:
    """Build a transient orchestrator, snapshot agents/tools/envs, then aclose.

    Per-rerun cost is dominated by FastMCP client startup (~100-200ms total for
    in-process servers); acceptable for a UI rerun.
    """
    async def _go():
        orch = await Orchestrator.create(cfg)
        try:
            return orch.list_agents(), orch.list_tools(), list(orch.cfg.environments)
        finally:
            await orch.aclose()
    return asyncio.run(_go())


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
}

_SEVERITY_COLOR = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}

# Unknown categories fall back to "gray" via the .get() in _category_badge.
_CATEGORY_COLOR = {
    "latency": "orange",
    "availability": "red",
    "data": "violet",
    "security": "red",
    "capacity": "blue",
    "performance": "orange",
    "config": "gray",
}


def _badge(label: str, color: str) -> None:
    """Render an inline coloured pill via st.badge.

    Centralised so the small label/colour decisions live in one place and
    the rest of the UI can call ``_status_badge(inc)`` etc. without
    touching the palette dicts directly.
    """
    st.badge(label, color=color)


def _status_badge(status: str | None) -> None:
    if not status:
        return
    _badge(_STATUS_LABEL.get(status, status.upper()),
           _STATUS_COLOR.get(status, "gray"))


def _severity_badge(severity: str | None) -> None:
    if not severity:
        return
    _badge(severity.upper(), _SEVERITY_COLOR.get(severity, "gray"))


def _category_badge(category: str | None) -> None:
    if not category:
        return
    _badge(category, _CATEGORY_COLOR.get(category, "gray"))


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


def _fmt_tokens_short(n: int) -> str:
    """Compact form for the sidebar: ``12.3k`` / ``842``."""
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n}"


def _render_inc_row(inc: dict, store: IncidentRepository) -> None:
    """One INC row as an expander.

    Header shows INC id + env badge + status badge. The body holds
    severity/category badges, age + tokens, an excerpt of the summary,
    and Open + delete actions. The currently-selected INC is expanded
    by default so its details are visible without an extra click.
    """
    inc_id = inc["id"]
    status = inc.get("status") or ""
    env = inc.get("environment", "")
    age = _age(inc.get("created_at", ""))
    sev = (inc.get("severity") or "")
    cat = (inc.get("category") or "")
    summary = (inc.get("summary") or inc.get("query") or "").strip()
    toks = (inc.get("token_usage") or {}).get("total_tokens", 0)
    is_deleted = status == "deleted"
    is_selected = st.session_state.get("selected_incident") == inc_id

    header = (
        f"`{inc_id}` "
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
            if st.button("×", key=f"del_{inc_id}",
                         help="Soft-delete",
                         type="tertiary",
                         use_container_width=True):
                store.delete(inc_id)
                if st.session_state.get("selected_incident") == inc_id:
                    st.session_state.pop("selected_incident", None)
                st.rerun()

    with exp_col, st.expander(header, expanded=is_selected):
        # Line 1: severity + category + age + updated-time. The time
        # carries a hover tooltip with the full timestamp and reporter
        # attribution, so the inline row stays compact.
        updated = inc.get("updated_at") or ""
        reporter = inc.get("reporter") or {}
        rep_id = reporter.get("id") or ""
        rep_team = reporter.get("team") or ""
        meta_bits = [b for b in (
            _badge_md(sev.lower() if sev else None, _SEVERITY_COLOR),
            _badge_md(cat or None, _CATEGORY_COLOR),
        ) if b]
        if age:
            meta_bits.append(f"`{age}`")
        if updated and len(updated) >= 16:
            time_str = updated[11:16]  # "HH:MM" out of "YYYY-MM-DDTHH:MM:SSZ"
            tooltip_parts = [f"updated {updated}"]
            if rep_id:
                attrib = f"by {rep_id}" + (f" ({rep_team})" if rep_team else "")
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

        if st.button("Open", key=f"inc_{inc_id}",
                     type="primary",
                     use_container_width=True):
            st.session_state["selected_incident"] = inc_id


def render_sidebar(store: IncidentRepository) -> None:
    with st.sidebar:
        head_l, head_r = st.columns([4, 1])
        with head_l:
            st.markdown("### Recent INCs")
        with head_r:
            if st.button("↻", help="Refresh", type="tertiary",
                         use_container_width=True):
                st.rerun()
        show_deleted = st.checkbox("Show deleted", value=False,
                                   key="show_deleted")
        statuses = ["all", "new", "in_progress", "matched", "resolved",
                    "escalated", "awaiting_input", "stopped"]
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
            st.caption("No incidents.")
            return
        for inc in recent[:20]:
            _render_inc_row(inc, store)


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

    Green ≥0.75, amber 0.5–0.75, red <0.5, grey when None. Markdown only —
    no HTML — so the badge survives Streamlit's sanitizer.
    """
    if conf is None:
        return "⚪ confidence —"
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


def _render_incident_top_badges(inc: dict) -> None:
    """Render status / severity / category badge columns and the awaiting-input warning."""
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        st.caption("Status")
        _status_badge(inc.get("status"))
    with b2:
        st.caption("Severity")
        _severity_badge(inc.get("severity"))
    with b3:
        st.caption("Category")
        _category_badge(inc.get("category"))
    if inc.get("status") == "awaiting_input":
        _pi = inc.get("pending_intervention") or {}
        _upstream = _pi.get("upstream_agent") or "the upstream agent"
        st.warning(
            f"**Human intervention required.** The intervention gate "
            f"paused this INC because {_upstream} confidence was "
            f"below the configured threshold. Use the controls below to "
            f"resume with input, escalate, or stop."
        )


def _render_incident_metrics(inc: dict) -> None:
    """Render tokens / active duration / total time metric columns.

    - **Active duration** is the sum of each agent run's wall-clock —
      reflects time the orchestrator was actually working.
    - **Total time** is ``created_at → updated_at`` — wall-clock from
      first report to the last write (covers awaiting-input pauses).
    """
    token_total = (inc.get("token_usage") or {}).get("total_tokens", 0)
    runs = inc.get("agents_run") or []
    active_s = sum(
        _duration_seconds(r.get("started_at", ""), r.get("ended_at", ""))
        for r in runs
    )
    total_s = _duration_seconds(inc.get("created_at", ""),
                                inc.get("updated_at", ""))
    if active_s == 0:
        active_s = total_s
    m1, m2, m3 = st.columns(3)
    m1.metric("Total tokens", _fmt_tokens(token_total))
    m2.metric("Active duration", _fmt_duration(active_s))
    m3.metric("Total time", _fmt_duration(total_s))


def _render_incident_prior_match(inc: dict) -> None:
    """Render the prior-incident callout when matched_prior_inc is set."""
    tags = inc.get("tags") or []
    if "hypothesis:prior_match_supported" in tags:
        stance = "supported by current evidence"
        callout = st.success
    elif "hypothesis:prior_match_rejected" in tags:
        stance = "rejected — fresh evidence diverges from prior cause"
        callout = st.warning
    else:
        stance = "not yet validated"
        callout = st.info
    callout(
        f"**Prior similar incident (hypothesis):** "
        f"`{inc['matched_prior_inc']}` — {stance}.  \n"
        f"_Same symptom can have different root causes "
        f"(code bug vs. network vs. resource overload), so the prior "
        f"cause is one ranked hypothesis for the deep investigator — "
        f"not the answer._"
    )


def _render_incident_summary_meta(inc: dict) -> None:
    """Render the query, environment, tags, summary, and prior-match callout."""
    st.markdown(f"**Query:** {inc['query']}")
    st.markdown(f"**Environment:** `{inc['environment']}`")
    if inc.get("tags"):
        st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in inc["tags"]))
    if inc.get("summary"):
        st.markdown(f"**Summary:** {inc['summary']}")
    if inc.get("matched_prior_inc"):
        _render_incident_prior_match(inc)


def _render_agents_run_block(inc: dict) -> None:
    """Render the ### Agents run section if agents_run is non-empty."""
    agents_run = inc.get("agents_run", [])
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


def _render_findings_block(inc: dict) -> None:
    """Render the ### Findings section for every agent that produced findings.

    Iterates over all entries in ``findings`` so custom YAML-defined agents
    are shown alongside built-in ones. Values that look like hypothesis lists
    (list of dicts with a ``cause`` key) are passed to ``_render_hypothesis_list``;
    everything else goes to ``_render_value``.
    """
    findings = inc.get("findings") or {}
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


def _render_resolution_block(inc: dict) -> None:
    """Render the ### Resolution section if resolution is present."""
    if inc.get("resolution") is None:
        return
    st.markdown("### Resolution")
    with st.container(border=True):
        _render_value(inc["resolution"])


def _render_tool_calls_block(inc: dict) -> None:
    """Render the ### Tool calls section if tool_calls is non-empty."""
    tool_calls = inc.get("tool_calls", [])
    if not tool_calls:
        return
    st.markdown("### Tool calls")
    for tc in tool_calls:
        with st.expander(f"`{tc['agent']}` → `{tc['tool']}`"):
            st.markdown("**Args:**")
            st.json(tc.get("args") or {})
            st.markdown("**Result:**")
            _render_value(tc.get("result"))


def render_incident_detail(store: IncidentRepository,
                           agent_names: frozenset[str] = frozenset()) -> None:
    inc_id = st.session_state.get("selected_incident")
    if not inc_id:
        return
    try:
        inc = store.load(inc_id).model_dump()
    except FileNotFoundError:
        return
    status = inc.get("status") or ""
    env = inc.get("environment") or ""
    toks = (inc.get("token_usage") or {}).get("total_tokens", 0)
    header = (
        f"`{inc_id}` "
        f"{_badge_md(env or None, _ENV_COLOR)} "
        f"{_badge_md(status, _STATUS_COLOR)} "
        f"`{_fmt_tokens_short(toks)}`"
    ).strip()
    with st.expander(header, expanded=True):
        _render_incident_top_badges(inc)
        _render_incident_metrics(inc)
        _render_incident_summary_meta(inc)
        if inc.get("status") == "awaiting_input" and inc.get("pending_intervention"):
            _render_intervention_block(inc, inc_id, agent_names)
        _render_agents_run_block(inc)
        _render_findings_block(inc)
        _render_resolution_block(inc)
        _render_tool_calls_block(inc)
        with st.expander("Raw JSON"):
            st.json(inc)


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
        return f"[{ts}] start  inc={ev.get('incident_id')}"
    if kind == "investigation_completed":
        return f"[{ts}] done   inc={ev.get('incident_id')}"
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


async def _resume_async(cfg: AppConfig, inc_id: str, decision: dict,
                        log_area, lines: list[str],
                        agent_names: frozenset[str] = frozenset()) -> dict:
    """Build a fresh Orchestrator, stream resume events, aclose.

    Returns a small summary dict describing the outcome so the caller can show
    a banner: ``{"rejected": <reason or None>}``. ``rejected`` is set when the
    orchestrator emits a ``resume_rejected`` event (e.g. INC no longer
    awaiting_input, invalid escalation team).
    """
    outcome: dict = {"rejected": None}
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.resume_investigation(inc_id, decision):
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


def _render_intervention_block(inc: dict, inc_id: str,
                               agent_names: frozenset[str] = frozenset()) -> None:
    """Render the intervention prompt above the agents_run section.

    Shows the confidence vs. threshold and a single form with an action
    selector that swaps inputs (text box / team dropdown / nothing) and a
    submit button. On submit, calls `_resume_async` and then reruns.
    """
    cfg = load_config(CONFIG_PATH)
    pi = inc.get("pending_intervention") or {}
    conf = pi.get("confidence")
    threshold = pi.get("threshold", 0.75)
    teams = pi.get("escalation_teams") or list(cfg.intervention.escalation_teams)

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
            key=f"intervention_action_{inc_id}",
        )
        decision: dict = {"action": action}
        if action == "resume_with_input":
            decision["input"] = st.text_area(
                "Add context for the investigator", height=120,
                placeholder="Anything the agent should know — recent changes, "
                            "logs you've already checked, suspected services…",
                key=f"intervention_input_{inc_id}",
            )
        elif action == "escalate":
            decision["team"] = st.selectbox(
                "Escalate to team", teams, key=f"intervention_team_{inc_id}",
            )

        submit = st.button("Submit", type="primary",
                           key=f"intervention_submit_{inc_id}")
        if submit:
            if action == "resume_with_input" and not (decision.get("input") or "").strip():
                st.warning("Add some context before resuming.")
                return
            log_area = st.empty()
            lines: list[str] = []
            outcome = asyncio.run(_resume_async(cfg, inc_id, decision, log_area, lines, agent_names))
            if outcome.get("rejected"):
                # Don't auto-rerun — let the user read the warning before the
                # form goes away. Common causes: INC already closed, invalid
                # escalation team, or a sub-graph exception that restored the
                # INC to awaiting_input.
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
    """
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: {_SIDEBAR_MIN_WIDTH_PX}px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    _inject_global_css()
    cfg = load_config(CONFIG_PATH)
    store = _make_repository(cfg)

    # One-shot snapshot of agent/tool metadata + environments. ~100-200ms per
    # rerun; acceptable, and keeps async resources strictly scoped.
    agents, tools, environments = _load_metadata_dicts(cfg)
    agent_names: frozenset[str] = frozenset(a["name"] for a in agents)

    render_sidebar(store)

    tab_investigate, tab_registry = st.tabs(["Investigate", "Agents & Tools"])

    with tab_investigate:
        st.header("Start an investigation")
        with st.form("investigate_form"):
            query = st.text_area("What's happening?", height=100, key="form_query")
            environment = st.selectbox("Impacted environment", environments,
                                       key="form_env")
            submitted = st.form_submit_button("Start investigation", type="primary")

        if submitted and query.strip():
            # Hide any previously-selected INC immediately — the detail
            # panel below won't be reached this script-run, so during the
            # blocking asyncio.run the user sees only the live timeline.
            st.session_state.pop("selected_incident", None)

            timeline_box = st.container()
            timeline_box.markdown("### Live timeline")
            log_area = timeline_box.empty()
            lines: list[str] = []

            asyncio.run(_run_investigation_async(cfg, query, environment, log_area, lines, agent_names))

            # Surface the resulting INC for one-click drill-in
            recent = [i.model_dump() for i in store.list_recent(1)]
            if recent:
                st.session_state["selected_incident"] = recent[0]["id"]
                st.success(f"Investigation complete — {recent[0]['id']} ({recent[0]['status']})")
                st.rerun()
        else:
            render_incident_detail(store, agent_names)

    with tab_registry:
        st.header("Agents & Tools registry")

        with st.expander("Tools", expanded=True):
            _render_tools_by_category(tools)

        st.subheader("Agents")
        _render_agents_accordion(agents)


if __name__ == "__main__":
    main()
