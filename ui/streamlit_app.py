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


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    orch = get_orchestrator()

    render_sidebar(orch)

    tab_investigate, tab_registry = st.tabs(["Investigate", "Agents & Tools"])

    with tab_investigate:
        st.header("Start an investigation")
        # Filled in by Task 24
        st.info("Investigation form — implemented in Task 24.")

    with tab_registry:
        st.header("Agents & Tools registry")
        # Filled in by Task 25
        st.info("Registry view — implemented in Task 25.")

    render_incident_detail(orch)


if __name__ == "__main__":
    main()
