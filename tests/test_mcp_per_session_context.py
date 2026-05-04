"""Lock in the per-instance isolation guarantee for ``IncidentMCPServer``.

The module exposes a ``_default_server`` singleton and a ``mcp`` global
because the runtime's MCP loader contract requires the importable
module to expose a top-level ``mcp`` attribute (see
``runtime.mcp_loader:137``). That singleton is a *loader-side
default*, not a shared application state. Every orchestrator
constructs its own fresh ``IncidentMCPServer()`` and ``configure``s it
against its own store; this test pins that two such instances cannot
see each other's data even when both run in the same process.

If a future change accidentally moves shared state onto the class
(rather than the instance), this test fails loud.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from examples.incident_management.mcp_server import IncidentMCPServer
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.mark.asyncio
async def test_two_servers_have_isolated_state(tmp_path):
    """Two IncidentMCPServer() instances bound to different stores
    must not see each other's sessions.
    """
    e1 = create_engine(f"sqlite:///{tmp_path/'a.db'}")
    e2 = create_engine(f"sqlite:///{tmp_path/'b.db'}")
    Base.metadata.create_all(e1)
    Base.metadata.create_all(e2)
    s1, s2 = SessionStore(engine=e1), SessionStore(engine=e2)

    srv_a = IncidentMCPServer()
    srv_a.configure(store=s1)
    srv_b = IncidentMCPServer()
    srv_b.configure(store=s2)

    a = s1.create(query="A", environment="dev",
                  reporter_id="u", reporter_team="t")
    b = s2.create(query="B", environment="dev",
                  reporter_id="u", reporter_team="t")

    await srv_a._tool_mark_resolved(
        incident_id=a.id,
        resolution_summary="x",
        confidence=0.9,
        confidence_rationale="r",
    )

    assert s1.load(a.id).status == "resolved"
    assert s2.load(b.id).status == "new"


@pytest.mark.asyncio
async def test_default_server_singleton_does_not_leak_into_isolated_instance(tmp_path):
    """The module-level ``_default_server`` singleton (kept for the MCP
    loader's ``getattr(mod, 'mcp')`` contract) must not bleed state
    into freshly-constructed instances. Configuring the default does
    NOT configure a separately-constructed server.
    """
    from examples.incident_management import mcp_server as _mod

    e = create_engine(f"sqlite:///{tmp_path/'c.db'}")
    Base.metadata.create_all(e)
    s = SessionStore(engine=e)

    # Configure the module-level default with our store, but use a
    # FRESH instance for the actual call. The fresh instance has no
    # store configured — should fail with a clear error rather than
    # accidentally hitting the default's store.
    _mod.set_state(store=s)
    fresh = IncidentMCPServer()
    a = s.create(query="A", environment="dev",
                 reporter_id="u", reporter_team="t")
    with pytest.raises(Exception):
        # No store configured on `fresh` → should not silently use
        # _default_server's store. The exact exception is whatever
        # the configure-required guard raises (RuntimeError, etc.).
        await fresh._tool_mark_resolved(
            incident_id=a.id,
            resolution_summary="x",
            confidence=0.9,
            confidence_rationale="r",
        )
