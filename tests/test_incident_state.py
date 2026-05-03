"""Tests for examples/incident_management runtime surface (UI / MCP / skills).

Typed-subclass behaviour for ``IncidentState`` (pydantic field defaults,
``IncidentStatus`` literal, ``_INC_ID_RE`` regex) was removed when the
migration to ``Session.extra_fields`` rendered the typed subclass
redundant. The remaining tests pin runtime-side surfaces — UI imports,
MCP server tool inventory, skills directory layout — that survive the
typed-subclass deletion.
"""


def test_incident_mcp_server_importable_from_example():
    from examples.incident_management.mcp_server import IncidentMCPServer  # noqa: F401


def test_incident_mcp_server_has_three_tools():
    import asyncio
    from examples.incident_management.mcp_server import IncidentMCPServer
    srv = IncidentMCPServer()
    tools = asyncio.run(srv.mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "lookup_similar_incidents",
        "create_incident",
        "update_incident",
    }


def test_runtime_mcp_servers_no_incident():
    """The framework's runtime.mcp_servers package must not contain incident.py."""
    import importlib.util
    spec = importlib.util.find_spec("runtime.mcp_servers.incident")
    assert spec is None, (
        "runtime.mcp_servers.incident should not exist; "
        "the incident MCP server now lives at examples.incident_management.mcp_server"
    )


def test_example_skills_dir_exists():
    from pathlib import Path
    skills_dir = Path("examples/incident_management/skills")
    assert skills_dir.is_dir()
    agents = [
        d.name
        for d in skills_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]
    assert set(agents) >= {"intake", "triage", "deep_investigator", "resolution"}


def test_example_ui_importable():
    """The framework UI replaces the per-example ``ui.py`` files.

    After the Wave 1 strip-down, ``runtime.ui`` is the single
    Streamlit entry point shared across apps; the per-app file was
    removed so the contract is now "the framework UI imports cleanly".
    """
    import importlib.util
    spec = importlib.util.find_spec("runtime.ui")
    assert spec is not None, "runtime.ui must be importable"


def test_runtime_main_module_importable():
    """Generic runtime entry-point is the only ``__main__`` now.

    The per-app ``__main__.py`` files were dropped — apps boot via
    ``python -m runtime --config config/<app>.yaml`` against the
    framework module instead.
    """
    import importlib.util
    spec = importlib.util.find_spec("runtime.__main__")
    assert spec is not None, "runtime.__main__ must be importable"


# ---------------------------------------------------------------------------
# Sidebar in-flight list + detail-pane polling
# ---------------------------------------------------------------------------

def test_should_poll_only_for_in_flight():
    """``_should_poll`` is the pure helper that gates auto-refresh.

    Statuses inside ``{running, in_progress, awaiting_input}`` are live
    runs; everything else (terminal session statuses + unknown values)
    is treated as terminal so the detail pane stops polling.
    """
    from runtime.ui import _should_poll

    # In-flight — should poll
    assert _should_poll("running") is True
    assert _should_poll("in_progress") is True
    assert _should_poll("awaiting_input") is True

    # Terminal session statuses — should not poll
    assert _should_poll("resolved") is False
    assert _should_poll("escalated") is False
    assert _should_poll("stopped") is False
    assert _should_poll("deleted") is False
    assert _should_poll("matched") is False
    assert _should_poll("new") is False

    # Defensive — empty / unknown statuses do not poll
    assert _should_poll(None) is False
    assert _should_poll("") is False
    assert _should_poll("garbage") is False


def test_ui_module_imports_without_orchestrator():
    """The UI module must import cleanly without a Streamlit runtime.

    Importability is the contract for ``runtime.ui``: the
    in-flight-sidebar wiring (``OrchestratorService.get_or_create``,
    ``st.cache_resource`` for the service singleton) must NOT fire at
    import time, otherwise tests / docs tooling that load the module
    headlessly will explode.
    """
    import importlib
    import runtime.ui as ui_mod

    importlib.reload(ui_mod)
    # Sanity — both the new helpers and the existing entrypoint survived
    # the reload.
    assert callable(ui_mod._should_poll)
    assert callable(ui_mod.render_sidebar)
    assert callable(ui_mod.main)


# ---------------------------------------------------------------------------
# Pending tool-approval cards (risk-rated gateway)
# ---------------------------------------------------------------------------


def test_ui_module_exposes_pending_approval_helpers():
    """The UI must expose the pending-approval card renderer and the
    OrchestratorService bridge that resolves it. Importability is
    enough — the actual rendering is exercised by the Streamlit
    runtime, not by pytest.
    """
    import importlib
    import runtime.ui as ui_mod

    importlib.reload(ui_mod)
    assert callable(ui_mod._render_pending_approvals_block), (
        "regression: _render_pending_approvals_block missing"
    )
    assert callable(ui_mod._submit_approval_via_service), (
        "regression: _submit_approval_via_service missing"
    )
