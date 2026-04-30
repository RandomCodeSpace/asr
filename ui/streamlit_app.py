"""Streamlit UI — 2 tabs + always-on sidebar with recent INCs."""
from __future__ import annotations
import asyncio
from pathlib import Path
import streamlit as st

from orchestrator.config import load_config
from orchestrator.orchestrator import Orchestrator


CONFIG_PATH = Path("config/config.yaml")


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    """Build and cache the orchestrator across reruns (Streamlit re-runs the script per interaction)."""
    cfg = load_config(CONFIG_PATH)
    return asyncio.run(Orchestrator.create(cfg))


def render_sidebar(orch: Orchestrator) -> None:
    with st.sidebar:
        st.markdown("### Recent INCs")
        recent = orch.list_recent_incidents(limit=20)
        if not recent:
            st.caption("No incidents yet.")
            return
        for inc in recent:
            label = f"`{inc['id']}` — {inc['status']}"
            if st.button(label, key=f"inc_{inc['id']}", use_container_width=True):
                st.session_state["selected_incident"] = inc["id"]


def render_incident_detail(orch: Orchestrator) -> None:
    inc_id = st.session_state.get("selected_incident")
    if not inc_id:
        return
    with st.expander(f"INC detail: {inc_id}", expanded=True):
        inc = orch.get_incident(inc_id)
        st.write(f"**Status:** {inc['status']} **Severity:** {inc.get('severity') or '—'}  "
                 f"**Category:** {inc.get('category') or '—'}")
        st.write(f"**Query:** {inc['query']}")
        st.write(f"**Environment:** {inc['environment']}")
        st.markdown("**Agents run:**")
        for ar in inc.get("agents_run", []):
            st.write(f"- `{ar['agent']}` ({ar['started_at']} → {ar['ended_at']}): {ar['summary']}")
        st.markdown("**Tool calls:**")
        for tc in inc.get("tool_calls", []):
            st.write(f"- `{tc['agent']}` → `{tc['tool']}` args={tc['args']} result={tc['result']}")
        if inc.get("resolution"):
            st.markdown("**Resolution:**")
            st.json(inc["resolution"])
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


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    orch = get_orchestrator()

    render_sidebar(orch)

    tab_investigate, tab_registry = st.tabs(["Investigate", "Agents & Tools"])

    with tab_investigate:
        st.header("Start an investigation")
        with st.form("investigate_form"):
            query = st.text_area("What's happening?", height=100, key="form_query")
            environment = st.selectbox("Impacted environment", orch.cfg.environments,
                                       key="form_env")
            submitted = st.form_submit_button("Start investigation", type="primary")

        if submitted and query.strip():
            timeline_box = st.container()
            timeline_box.markdown("### Live timeline")
            log_area = timeline_box.empty()
            lines: list[str] = []

            async def run_and_stream():
                async for ev in orch.stream_investigation(query=query, environment=environment):
                    line = _format_event(ev)
                    if line:
                        lines.append(line)
                        log_area.code("\n".join(lines), language="text")

            asyncio.run(run_and_stream())

            # Surface the resulting INC for one-click drill-in
            recent = orch.list_recent_incidents(limit=1)
            if recent:
                st.session_state["selected_incident"] = recent[0]["id"]
                st.success(f"Investigation complete — {recent[0]['id']} ({recent[0]['status']})")
                st.rerun()

    with tab_registry:
        st.header("Agents & Tools registry")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Agents")
            for a in orch.list_agents():
                with st.container(border=True):
                    st.markdown(f"**{a['name']}** — `{a['model']}`")
                    st.caption(a["description"])
                    st.markdown("Tools: " + ", ".join(f"`{t}`" for t in a["tools"]))
                    if a["routes"]:
                        st.caption("Routes: " + ", ".join(
                            f"`{r['when']}→{r['next']}`" for r in a["routes"]))

        with col_b:
            st.subheader("Tools by category")
            tools = orch.list_tools()
            by_cat: dict[str, list[dict]] = {}
            for t in tools:
                by_cat.setdefault(t["category"], []).append(t)
            for cat in sorted(by_cat):
                st.markdown(f"**{cat}**")
                for t in by_cat[cat]:
                    bound = ", ".join(f"`{a}`" for a in t["bound_agents"]) or "_(unbound)_"
                    st.markdown(f"- `{t['name']}` — {t['description'][:80]}  \n  bound to: {bound}")

    render_incident_detail(orch)


if __name__ == "__main__":
    main()
