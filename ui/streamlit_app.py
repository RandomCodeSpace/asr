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


def render_sidebar(store: IncidentStore) -> None:
    with st.sidebar:
        st.markdown("### Recent INCs")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            statuses = ["all", "new", "in_progress", "matched", "resolved", "escalated"]
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
            badge = {
                "new": "🟦",
                "in_progress": "🟡",
                "matched": "🟢",
                "resolved": "✅",
                "escalated": "🔴",
            }.get(inc["status"], "⚪")
            label = f"{badge} `{inc['id']}` — {inc['environment']}"
            if st.button(label, key=f"inc_{inc['id']}", use_container_width=True):
                st.session_state["selected_incident"] = inc["id"]


def _render_value(v) -> None:
    """Render a value the agents may produce (str/dict/list/None) safely.

    `st.json` parses strings as JSON; LLM-produced free-form prose like
    "The investigation found ..." breaks it. So: structured types -> st.json,
    everything else -> st.write.
    """
    if v is None:
        st.caption("_(none)_")
    elif isinstance(v, (dict, list)):
        st.json(v)
    else:
        st.write(v)


def render_incident_detail(store: IncidentStore) -> None:
    inc_id = st.session_state.get("selected_incident")
    if not inc_id:
        return
    with st.expander(f"INC detail: {inc_id}", expanded=True):
        inc = store.load(inc_id).model_dump()
        st.write(f"**Status:** {inc['status']} **Severity:** {inc.get('severity') or '—'}  "
                 f"**Category:** {inc.get('category') or '—'}")
        st.write(f"**Query:** {inc['query']}")
        st.write(f"**Environment:** {inc['environment']}")
        if inc.get("tags"):
            st.write("**Tags:** " + ", ".join(f"`{t}`" for t in inc["tags"]))
        if inc.get("summary"):
            st.write(f"**Summary:** {inc['summary']}")
        if inc.get("matched_prior_inc"):
            st.write(f"**Matched prior INC:** `{inc['matched_prior_inc']}`")

        st.markdown("**Agents run:**")
        for ar in inc.get("agents_run", []):
            st.write(f"- `{ar['agent']}` ({ar['started_at']} → {ar['ended_at']}): {ar['summary']}")

        st.markdown("**Tool calls:**")
        for tc in inc.get("tool_calls", []):
            st.write(f"- `{tc['agent']}` → `{tc['tool']}` args={tc['args']} "
                     f"result={str(tc['result'])[:200]}")

        findings = inc.get("findings") or {}
        if findings.get("triage") is not None:
            st.markdown("**Triage findings:**")
            _render_value(findings["triage"])
        if findings.get("deep_investigator") is not None:
            st.markdown("**Deep investigator findings:**")
            _render_value(findings["deep_investigator"])
        if inc.get("resolution") is not None:
            st.markdown("**Resolution:**")
            _render_value(inc["resolution"])

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
