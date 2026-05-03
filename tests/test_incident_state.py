"""Tests for examples/incident_management/state.py."""
import re
import typing


def test_incident_state_importable():
    from examples.incident_management.state import IncidentState  # noqa: F401


def test_incident_state_inherits_session():
    from runtime.state import Session
    from examples.incident_management.state import IncidentState

    assert issubclass(IncidentState, Session)


def test_incident_state_has_domain_fields():
    from examples.incident_management.state import IncidentState, Reporter

    inc = IncidentState(
        id="INC-20260502-001",
        status="new",
        created_at="2026-05-02T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
        query="latency spike in payments",
        environment="production",
        reporter=Reporter(id="user-1", team="platform"),
    )
    assert inc.environment == "production"
    assert inc.severity is None
    assert inc.tags == []


def test_incident_status_values():
    from examples.incident_management.state import IncidentStatus

    expected = {
        "new",
        "in_progress",
        "matched",
        "resolved",
        "escalated",
        "awaiting_input",
        "stopped",
        "deleted",
        # P7-B: dedup pipeline terminal status.
        "duplicate",
    }
    assert set(typing.get_args(IncidentStatus)) == expected


def test_id_format_validation():
    from examples.incident_management.state import _INC_ID_RE

    assert re.match(_INC_ID_RE, "INC-20260502-001")
    assert not re.match(_INC_ID_RE, "SESSION-001")


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
    import importlib.util
    spec = importlib.util.find_spec("examples.incident_management.ui")
    assert spec is not None, "examples.incident_management.ui must be importable"


def test_example_main_module_importable():
    import importlib.util
    spec = importlib.util.find_spec("examples.incident_management.__main__")
    assert spec is not None, "examples.incident_management.__main__ must be importable"


# ---------------------------------------------------------------------------
# P3-I / P3-J — sidebar in-flight list + detail-pane polling
# ---------------------------------------------------------------------------

def test_should_poll_only_for_in_flight():
    """``_should_poll`` is the pure helper that gates auto-refresh.

    Statuses inside ``{running, in_progress, awaiting_input}`` are live
    runs; everything else (terminal incident statuses + unknown values)
    is treated as terminal so the detail pane stops polling.
    """
    from examples.incident_management.ui import _should_poll

    # In-flight — should poll
    assert _should_poll("running") is True
    assert _should_poll("in_progress") is True
    assert _should_poll("awaiting_input") is True

    # Terminal incident statuses — should not poll
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

    Importability is the contract for ``examples.incident_management.ui``:
    the P3-I/J wiring (``OrchestratorService.get_or_create``,
    ``st.cache_resource`` for the service singleton) must NOT fire at
    import time, otherwise tests / docs tooling that load the module
    headlessly will explode.
    """
    import importlib
    import examples.incident_management.ui as ui_mod

    importlib.reload(ui_mod)
    # Sanity — both the new helpers and the existing entrypoint survived
    # the reload.
    assert callable(ui_mod._should_poll)
    assert callable(ui_mod.render_sidebar)
    assert callable(ui_mod.main)


# ---------------------------------------------------------------------------
# P4-H — pending tool-approval cards (risk-rated gateway)
# ---------------------------------------------------------------------------


def test_ui_module_exposes_pending_approval_helpers():
    """P4-H: the UI must expose the pending-approval card renderer and
    the OrchestratorService bridge that resolves it. Importability is
    enough — the actual rendering is exercised by the Streamlit
    runtime, not by pytest.
    """
    import importlib
    import examples.incident_management.ui as ui_mod

    importlib.reload(ui_mod)
    assert callable(ui_mod._render_pending_approvals_block), (
        "P4-H regression: _render_pending_approvals_block missing"
    )
    assert callable(ui_mod._submit_approval_via_service), (
        "P4-H regression: _submit_approval_via_service missing"
    )
