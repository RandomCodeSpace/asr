"""Tests for the P4-G HITL approval endpoints.

Endpoints:
  * ``GET  /sessions/{sid}/approvals``                 — list pending.
  * ``POST /sessions/{sid}/approvals/{tool_call_id}``  — resolve.

The wrap_tool closure (P4-C) is unit-tested in
``tests/test_gateway_wrap.py`` and integration-tested in
``tests/test_gateway_integration.py``. These tests target the HTTP
surface only: schemas, status codes, routing through
``OrchestratorService``, and the legacy back-compat for unknown
session ids.

We seed the session store directly with stub ``ToolCall`` rows so we
exercise the listing endpoint without spinning up a paused graph.
The POST happy-path uses a real session id but stubs the
graph-resume step via the live ``OrchestratorService`` — the wrap
audit row is what subsequent inspection asserts on.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import AsyncClient, ASGITransport

from runtime.api import build_app
from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    Paths,
    RuntimeConfig,
)


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_api.py — self-contained for read-top-to-bottom)
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="runtime.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="runtime.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="runtime.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )


@pytest.fixture(autouse=True)
def _reset_orchestrator_service_singleton():
    yield
    try:
        from runtime.service import OrchestratorService
        OrchestratorService._reset_singleton()
    except Exception:
        pass


@asynccontextmanager
async def _client_with_lifespan(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# GET /sessions/{sid}/approvals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pending_approvals_returns_only_pending_rows(cfg):
    """The endpoint must filter ``tool_calls`` down to rows with
    ``status="pending_approval"`` — others (executed, rejected, etc.)
    are excluded."""
    from runtime.state import ToolCall

    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        # Create a session via the public API so the lifespan and store
        # are wired identically to a real run.
        start = await client.post("/sessions", json={
            "query": "test", "environment": "dev",
            "reporter_id": "u", "reporter_team": "t",
        })
        assert start.status_code == 201
        sid = start.json()["session_id"]

        # Seed three tool_calls into the row directly via the
        # orchestrator's store. Only one is pending_approval.
        orch = app.state.orchestrator
        inc = orch.store.load(sid)
        inc.tool_calls = [
            ToolCall(agent="resolution", tool="apply_fix",
                     args={"target": "payments-svc"},
                     result=None, ts="2026-05-02T00:00:00Z",
                     risk="high", status="pending_approval"),
            ToolCall(agent="intake", tool="create_incident",
                     args={"q": "x"},
                     result={"id": "INC-1"}, ts="2026-05-02T00:00:01Z",
                     status="executed"),
            ToolCall(agent="resolution", tool="apply_fix",
                     args={"target": "billing-svc"},
                     result={"rejected": True}, ts="2026-05-02T00:00:02Z",
                     risk="high", status="rejected"),
        ]
        orch.store.save(inc)

        res = await client.get(f"/sessions/{sid}/approvals")

    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) == 1
    pending = body[0]
    assert pending["tool"] == "apply_fix"
    assert pending["agent"] == "resolution"
    assert pending["args"] == {"target": "payments-svc"}
    # tool_call_id is the stable index in the audit list.
    assert pending["tool_call_id"] == "0"


@pytest.mark.asyncio
async def test_list_pending_approvals_empty_for_clean_session(cfg):
    """A session with no pending approvals must return ``[]``, not 404."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        start = await client.post("/sessions", json={
            "query": "test", "environment": "dev",
            "reporter_id": "u", "reporter_team": "t",
        })
        sid = start.json()["session_id"]

        res = await client.get(f"/sessions/{sid}/approvals")

    assert res.status_code == 200
    assert res.json() == []


@pytest.mark.asyncio
async def test_list_pending_approvals_404_for_unknown_session(cfg):
    """Unknown session id must return 404, not silently empty."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.get("/sessions/INC-DOES-NOT-EXIST/approvals")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{sid}/approvals/{tool_call_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_approval_404_for_unknown_session(cfg):
    """POST against an unknown session id must return 404."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        res = await client.post(
            "/sessions/INC-DOES-NOT-EXIST/approvals/0",
            json={"decision": "approve", "approver": "alice", "rationale": "ok"},
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_post_approval_400_on_invalid_decision(cfg):
    """The ``decision`` field is a strict Literal — any value outside
    ``{approve, reject}`` must yield 422 (FastAPI validation error)."""
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        start = await client.post("/sessions", json={
            "query": "test", "environment": "dev",
            "reporter_id": "u", "reporter_team": "t",
        })
        sid = start.json()["session_id"]
        res = await client.post(
            f"/sessions/{sid}/approvals/0",
            json={"decision": "MAYBE", "approver": "alice"},
        )
    # FastAPI surfaces validation failures as 422; either 400 or 422
    # is acceptable per the plan ("400 on bad decision") — the key
    # contract is "rejected at the boundary, not silently coerced".
    assert res.status_code in (400, 422)


@pytest.mark.asyncio
async def test_post_approval_happy_path_returns_decision_summary(cfg, monkeypatch):
    """A well-formed POST against a known session must return a 200
    summary echoing the decision payload. The actual graph resume
    behaviour is unit-tested in ``tests/test_gateway_wrap.py``; here we
    isolate the HTTP contract by stubbing ``graph.ainvoke`` so the
    endpoint exercises only request -> service.submit_and_wait -> resume
    plumbing.
    """
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        start = await client.post("/sessions", json={
            "query": "test", "environment": "dev",
            "reporter_id": "u", "reporter_team": "t",
        })
        sid = start.json()["session_id"]

        # Stub the graph.ainvoke so the resume call is a no-op — we
        # only care about the HTTP contract here. The wrap_tool +
        # interrupt + Command(resume=...) wiring is end-to-end tested
        # in tests/test_gateway_wrap.py.
        captured: dict = {}

        async def _fake_ainvoke(arg, config=None):
            captured["arg"] = arg
            captured["config"] = config
            return {}

        monkeypatch.setattr(
            app.state.orchestrator.graph,
            "ainvoke",
            _fake_ainvoke,
        )

        res = await client.post(
            f"/sessions/{sid}/approvals/0",
            json={
                "decision": "approve",
                "approver": "alice",
                "rationale": "ok",
            },
        )

    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == sid
    assert body["tool_call_id"] == "0"
    assert body["decision"] == "approve"
    assert body["approver"] == "alice"
    assert body["rationale"] == "ok"
    # The endpoint must drive the resume via Command(resume=payload).
    from langgraph.types import Command
    assert isinstance(captured.get("arg"), Command)
    assert captured["arg"].resume == {
        "decision": "approve",
        "approver": "alice",
        "rationale": "ok",
    }
    assert captured["config"] == {"configurable": {"thread_id": sid}}


@pytest.mark.asyncio
async def test_submit_approval_real_loop_no_deadlock(cfg):
    """End-to-end exercise of ``POST /sessions/{sid}/approvals/{tcid}``
    against a *real* :class:`OrchestratorService` with no patching of the
    bridge between the request handler and the service loop.

    Regression guard for the deadlock hazard: under
    ``httpx.AsyncClient + ASGITransport`` the FastAPI handler runs on the
    same event loop the ``OrchestratorService`` background thread is
    hosting. Using ``submit_and_wait`` (sync ``Future.result()``) inside
    the async handler would block that loop while the resume coroutine is
    queued onto it — the loop never gets to run the resume → permanent
    hang. The fix uses ``asyncio.wrap_future`` (via
    :meth:`OrchestratorService.submit_async`) so the handler yields
    control while the work runs.

    We only patch ``graph.ainvoke`` to count invocations and prove the
    resume coroutine actually executed; the bridge between the endpoint
    and the service loop is the code under test.

    A 5s wall-clock timeout is the deadlock detector: under the bug, the
    request would hang indefinitely.
    """
    app = build_app(cfg)
    async with _client_with_lifespan(app) as client:
        start = await client.post("/sessions", json={
            "query": "test", "environment": "dev",
            "reporter_id": "u", "reporter_team": "t",
        })
        assert start.status_code == 201
        sid = start.json()["session_id"]

        # Side-channel counter: incremented when the resume coroutine is
        # actually awaited on the service loop. If submit_and_wait
        # deadlocked the loop, this counter would never increment because
        # the request would hang before the coroutine ran.
        ainvoke_calls: list[dict] = []

        async def _fake_ainvoke(arg, config=None):
            # Yield once to prove we ran on the service loop, not in the
            # caller's stack.
            await asyncio.sleep(0)
            ainvoke_calls.append({"arg": arg, "config": config})
            return {}

        app.state.orchestrator.graph.ainvoke = _fake_ainvoke

        # 5s wall clock is the deadlock detector — under the bug this
        # would hang forever (no progress on the loop).
        res = await asyncio.wait_for(
            client.post(
                f"/sessions/{sid}/approvals/0",
                json={
                    "decision": "approve",
                    "approver": "bob",
                    "rationale": "no-deadlock-please",
                },
            ),
            timeout=5.0,
        )

    assert res.status_code == 200
    assert res.json()["session_id"] == sid
    assert res.json()["decision"] == "approve"
    # The resume coroutine actually executed on the service loop —
    # proves the bridge is working, not just that the handler returned.
    assert len(ainvoke_calls) == 1
    from langgraph.types import Command
    assert isinstance(ainvoke_calls[0]["arg"], Command)
    assert ainvoke_calls[0]["arg"].resume == {
        "decision": "approve",
        "approver": "bob",
        "rationale": "no-deadlock-please",
    }
