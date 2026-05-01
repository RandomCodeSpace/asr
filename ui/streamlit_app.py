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
# The sidebar uses IncidentStore directly, since incident JSON I/O is sync and
# needs no MCP clients.
from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path
import streamlit as st

from orchestrator.config import load_config, AppConfig
from orchestrator.incident import IncidentStore
from orchestrator.orchestrator import Orchestrator


CONFIG_PATH = Path("config/config.yaml")


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
}

# Human-readable labels — awaiting_input is highlighted as the action-required state.
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


def render_sidebar(store: IncidentStore) -> None:
    with st.sidebar:
        st.markdown("### Recent INCs")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            statuses = ["all", "new", "in_progress", "matched", "resolved",
                        "escalated", "awaiting_input", "stopped"]
            status_filter = st.selectbox("Filter", statuses, key="status_filter",
                                         label_visibility="collapsed")
        with col_r:
            if st.button("↻", help="Refresh"):
                st.rerun()

        recent = [i.model_dump() for i in store.list_recent(50)]
        if status_filter != "all":
            recent = [i for i in recent if i["status"] == status_filter]

        if not recent:
            st.caption("No incidents.")
            return
        for inc in recent[:20]:
            with st.container(border=True):
                top_l, top_r = st.columns([3, 2])
                with top_l:
                    st.markdown(
                        f"**`{inc['id']}`** · _{inc['environment']}_"
                    )
                with top_r:
                    _status_badge(inc.get("status"))
                meta_l, meta_r = st.columns(2)
                with meta_l:
                    _severity_badge(inc.get("severity"))
                with meta_r:
                    _category_badge(inc.get("category"))
                toks = (inc.get("token_usage") or {}).get("total_tokens", 0)
                tok_str = (f"{toks/1000:.1f}k tok" if toks >= 1000
                           else f"{toks} tok")
                if st.button(f"View · {tok_str}",
                             key=f"inc_{inc['id']}",
                             use_container_width=True):
                    st.session_state["selected_incident"] = inc["id"]


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
            st.markdown(f"**{key}:**")
            with st.container(border=True):
                _render_kv_block(v)
        elif isinstance(v, list):
            st.markdown(f"**{key}:**")
            for item in v:
                if isinstance(item, dict):
                    with st.container(border=True):
                        _render_kv_block(item)
                else:
                    st.markdown(f"- {item}")
        elif isinstance(v, bool):
            st.markdown(f"**{key}:** `{str(v).lower()}`")
        else:
            st.markdown(f"**{key}:** {v}")


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
        for i, item in enumerate(v):
            if isinstance(item, dict):
                if i > 0:
                    st.markdown("---")
                _render_kv_block(item)
            else:
                st.markdown(f"- {item}")
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


def _render_hypothesis_list(items: list, label: str) -> None:
    """Render a list of hypothesis-shaped dicts (cause/evidence/next_steps)
    as bordered cards. Strings or scalar entries fall back to bullets.
    """
    for i, h in enumerate(items, 1):
        with st.container(border=True):
            if isinstance(h, dict):
                st.markdown(f"**{label} {i}:** {h.get('cause', '—')}")
                ev = h.get("evidence")
                if ev:
                    st.markdown("**Evidence:**")
                    for e in (ev if isinstance(ev, list) else [ev]):
                        st.markdown(f"- {e}")
                ns = h.get("next_steps") or h.get("next_step") or h.get("probe")
                if ns:
                    st.markdown(f"**Next steps:** {ns}")
                # Anything else in the dict that we haven't surfaced.
                extra = {k: v for k, v in h.items()
                         if k not in {"cause", "evidence", "next_steps",
                                      "next_step", "probe"}}
                if extra:
                    _render_kv_block(extra)
            else:
                st.markdown(f"**{label} {i}:** {h}")


def _render_findings_section(value, _label: str) -> None:
    """Findings can be a list of hypothesis dicts, a single dict, or free
    prose. Pick the right renderer; never silently truncate.
    """
    if isinstance(value, list):
        _render_hypothesis_list(value, label="Hypothesis")
    elif isinstance(value, dict):
        _render_kv_block(value)
    elif isinstance(value, str):
        st.write(value)
    else:
        _render_value(value)


def render_incident_detail(store: IncidentStore) -> None:
    inc_id = st.session_state.get("selected_incident")
    if not inc_id:
        return
    with st.expander(f"INC detail: {inc_id}", expanded=True):
        inc = store.load(inc_id).model_dump()

        # --- Top status / severity / category badges ----------------------
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

        # Prominent action-required call-out when the gate has paused the
        # graph. Sits above the (separate) intervention prompt block so the
        # user can't miss it on a long page.
        if inc.get("status") == "awaiting_input":
            st.warning(
                "**Human intervention required.** The intervention gate "
                "paused this INC because deep-investigator confidence was "
                "below the configured threshold. Use the controls below to "
                "resume with input, escalate, or stop."
            )

        # --- Numeric metrics ----------------------------------------------
        token_total = (inc.get("token_usage") or {}).get("total_tokens", 0)
        duration_s = _duration_seconds(inc.get("created_at", ""),
                                       inc.get("updated_at", ""))
        m1, m2 = st.columns(2)
        m1.metric("Total tokens", _fmt_tokens(token_total))
        m2.metric("Duration", f"{duration_s}s")

        # --- Header block -------------------------------------------------
        st.markdown(f"**Query:** {inc['query']}")
        st.markdown(f"**Environment:** `{inc['environment']}`")
        if inc.get("tags"):
            st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in inc["tags"]))
        if inc.get("summary"):
            st.markdown(f"**Summary:** {inc['summary']}")
        if inc.get("matched_prior_inc"):
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

        # --- Intervention prompt (only when paused on low confidence) -----
        if inc.get("status") == "awaiting_input" and inc.get("pending_intervention"):
            _render_intervention_block(inc, inc_id)

        # --- Agents run ---------------------------------------------------
        agents_run = inc.get("agents_run", [])
        if agents_run:
            st.markdown("### Agents run")
            for ar in agents_run:
                a_dur = _duration_seconds(ar.get("started_at", ""),
                                          ar.get("ended_at", ""))
                a_tok = (ar.get("token_usage") or {}).get("total_tokens", 0)
                conf = ar.get("confidence")
                badge = _fmt_confidence_badge(conf)
                with st.container(border=True):
                    st.markdown(
                        f"**{ar['agent']}** — {a_dur}s — "
                        f"{_fmt_tokens(a_tok)} tokens — {badge}"
                    )
                    rationale = ar.get("confidence_rationale")
                    if rationale:
                        st.caption(f"Why: {rationale}")
                    st.write(ar.get("summary") or "_(no summary)_")

        # --- Findings -----------------------------------------------------
        findings = inc.get("findings") or {}
        f_triage = findings.get("triage")
        f_di = findings.get("deep_investigator")
        if f_triage is not None or f_di is not None:
            st.markdown("### Findings")
        if f_triage is not None:
            with st.container(border=True):
                st.markdown("**Triage**")
                _render_findings_section(f_triage, label="Finding")
        if f_di is not None:
            with st.container(border=True):
                st.markdown("**Deep investigator**")
                _render_findings_section(f_di, label="Hypothesis")

        # --- Resolution ---------------------------------------------------
        if inc.get("resolution") is not None:
            st.markdown("### Resolution")
            with st.container(border=True):
                _render_value(inc["resolution"])

        # --- Tool calls ---------------------------------------------------
        tool_calls = inc.get("tool_calls", [])
        if tool_calls:
            st.markdown("### Tool calls")
            for idx, tc in enumerate(tool_calls):
                with st.expander(f"`{tc['agent']}` → `{tc['tool']}`"):
                    st.markdown("**Args:**")
                    st.json(tc.get("args") or {})
                    st.markdown("**Result:**")
                    _render_value(tc.get("result"))

        with st.expander("Raw JSON"):
            st.json(inc)


def _format_event(ev: dict) -> str | None:
    kind = ev.get("event")
    node = ev.get("node") or ""
    ts = ev.get("ts", "")
    if kind == "investigation_started":
        return f"[{ts}] start  inc={ev.get('incident_id')}"
    if kind == "investigation_completed":
        return f"[{ts}] done   inc={ev.get('incident_id')}"
    if kind == "on_chain_start" and node in {"intake", "triage", "deep_investigator", "resolution"}:
        return f"[{ts}] enter  {node}"
    if kind == "on_chain_end" and node in {"intake", "triage", "deep_investigator", "resolution"}:
        return f"[{ts}] exit   {node}"
    if kind == "on_tool_start":
        return f"[{ts}] tool   {node}"
    if kind == "on_tool_end":
        result = (ev.get("data") or {}).get("output")
        snippet = str(result)[:120] if result is not None else ""
        return f"[{ts}] tool→  {node} {snippet}"
    return None


async def _run_investigation_async(cfg: AppConfig, query: str, environment: str,
                                   log_area, lines: list[str]) -> None:
    """Build a fresh Orchestrator, stream events, aclose. One asyncio.run frame."""
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.stream_investigation(query=query, environment=environment):
            line = _format_event(ev)
            if line:
                lines.append(line)
                log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()


async def _resume_async(cfg: AppConfig, inc_id: str, decision: dict,
                        log_area, lines: list[str]) -> dict:
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
                line = _format_event(ev)
                if line:
                    lines.append(line)
            log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()
    return outcome


def _render_intervention_block(inc: dict, inc_id: str) -> None:
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
    with st.container(border=True):
        st.markdown(
            f"#### 🟠 Intervention required — confidence {conf_str} "
            f"< threshold {threshold:.2f}"
        )
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
            outcome = asyncio.run(_resume_async(cfg, inc_id, decision, log_area, lines))
            if outcome.get("rejected"):
                # Don't auto-rerun — let the user read the warning before the
                # form goes away. Common causes: INC already closed, invalid
                # escalation team, or a sub-graph exception that restored the
                # INC to awaiting_input.
                st.warning(f"Resume rejected: {outcome['rejected']}")
                return
            st.success(f"Resume complete (action: {action}).")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    cfg = load_config(CONFIG_PATH)
    store = IncidentStore(cfg.paths.incidents_dir)

    # One-shot snapshot of agent/tool metadata + environments. ~100-200ms per
    # rerun; acceptable, and keeps async resources strictly scoped.
    agents, tools, environments = _load_metadata_dicts(cfg)

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
            timeline_box = st.container()
            timeline_box.markdown("### Live timeline")
            log_area = timeline_box.empty()
            lines: list[str] = []

            asyncio.run(_run_investigation_async(cfg, query, environment, log_area, lines))

            # Surface the resulting INC for one-click drill-in
            recent = [i.model_dump() for i in store.list_recent(1)]
            if recent:
                st.session_state["selected_incident"] = recent[0]["id"]
                st.success(f"Investigation complete — {recent[0]['id']} ({recent[0]['status']})")
                st.rerun()

    with tab_registry:
        st.header("Agents & Tools registry")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Agents")
            for a in agents:
                with st.container(border=True):
                    st.markdown(f"**{a['name']}** — `{a['model']}`")
                    st.caption(a["description"])
                    st.markdown("Tools: " + ", ".join(f"`{t}`" for t in a["tools"]))
                    if a["routes"]:
                        st.caption("Routes: " + ", ".join(
                            f"`{r['when']}→{r['next']}`" for r in a["routes"]))

        with col_b:
            st.subheader("Tools by category")
            by_cat: dict[str, list[dict]] = {}
            for t in tools:
                by_cat.setdefault(t["category"], []).append(t)
            for cat in sorted(by_cat):
                st.markdown(f"**{cat}**")
                for t in by_cat[cat]:
                    bound = ", ".join(f"`{a}`" for a in t["bound_agents"]) or "_(unbound)_"
                    st.markdown(f"- `{t['name']}` — {t['description'][:80]}  \n  bound to: {bound}")

    render_incident_detail(store)


if __name__ == "__main__":
    main()
