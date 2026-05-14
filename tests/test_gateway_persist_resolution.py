"""Regression: gateway persists pending_approval -> {approved,rejected,timeout}
status transitions to the SessionStore so the DB row reflects the actual
operator outcome.

Pre-fix the gateway only mutated ``session.tool_calls[pending_idx]`` in
memory and relied on the agent_node's later ``store.save`` to flush.
But the agent_node reloads from store at line 897 (graph.py), which
overwrites the in-memory mutation — so the persisted row stayed at
``pending_approval`` forever. The UI's
``_render_pending_approvals_block`` polls the DB and would keep
offering Approve / Reject buttons after they were already used.

These tests pin the new contract: every verdict-driven transition
calls ``store.save`` so the DB row matches the in-memory state.

The ``interrupt()`` call needs a LangGraph runnable context to receive
a synthetic verdict. We monkeypatch ``langgraph.types.interrupt`` to
return the desired verdict shape directly — the same approach the
existing telemetry test uses for the "approved" case.
"""
from __future__ import annotations

from typing import Any

import langgraph.types as lg_types
import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from runtime.config import EmbeddingConfig, GatewayConfig, MetadataConfig, ProviderConfig
from runtime.state import Session
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.tools.gateway import wrap_tool


# ---------------------------------------------------------------------------
# Fixtures


class _PingArgs(BaseModel):
    msg: str = ""


def _make_ping_tool() -> StructuredTool:
    def _impl(msg: str = "") -> dict:
        return {"echo": msg or "default"}

    return StructuredTool.from_function(
        func=_impl, name="ping", description="echo", args_schema=_PingArgs,
    )


@pytest.fixture
def store(tmp_path) -> SessionStore:
    db_path = tmp_path / "asr.db"
    engine = build_engine(MetadataConfig(url=f"sqlite:///{db_path}"))
    Base.metadata.create_all(engine)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=engine, embedder=embedder)


@pytest.fixture
def session(store: SessionStore) -> Session:
    """A fresh session row already persisted, with environment=production
    so the high-risk policy actually triggers a gate."""
    inc = store.create(
        query="ping the gate", environment="production",
        reporter_id="u", reporter_team="t",
    )
    return inc


def _gateway_cfg() -> GatewayConfig:
    return GatewayConfig(policy={"ping": "high"})


# ---------------------------------------------------------------------------
# approve / reject / timeout each persist the new status to DB


@pytest.mark.parametrize(
    "verdict_payload, expected_status, expected_result_key",
    [
        # String verdicts (legacy back-compat path).
        ("approve", "approved", None),
        ("reject", "rejected", "rejected"),
        ("timeout", "timeout", "timeout"),
        # Dict verdicts (modern UI/API contract).
        ({"decision": "approve", "approver": "alice", "rationale": "ok"},
         "approved", None),
        ({"decision": "reject", "approver": "bob", "rationale": "no"},
         "rejected", "rejected"),
        ({"decision": "timeout", "approver": None, "rationale": None},
         "timeout", "timeout"),
    ],
)
def test_gateway_persists_resolution_status_to_db(
    store, session, monkeypatch,
    verdict_payload, expected_status, expected_result_key,
):
    """Every verdict-driven transition must end with the DB row carrying
    the resolved status — never ``pending_approval``. Without the
    persist step the UI keeps offering Approve / Reject forever and
    downstream consumers (audit reports, retraining) miss the outcome.
    """
    monkeypatch.setattr(lg_types, "interrupt", lambda _payload: verdict_payload)

    wrapped = wrap_tool(
        _make_ping_tool(), session=session, gateway_cfg=_gateway_cfg(),
        agent_name="resolution", store=store,
    )

    # Drive the gated tool via sync invoke — same path the LangGraph
    # tool node uses after it resumes from the interrupt.
    out = wrapped.invoke({"msg": "danger"})
    if expected_status == "approved":
        assert out == {"echo": "danger"}
    else:
        # rejected / timeout return the marker dict, not the tool's output.
        assert out.get(expected_result_key) is True

    # Reload from store and assert the persisted row reflects the
    # transition — this is the assertion that fails without the
    # gateway's store.save on the transition branches.
    fresh = store.load(session.id)
    pending = [tc for tc in fresh.tool_calls if tc.tool == "ping"]
    assert pending, "expected one ping tool_call row in DB"
    last = pending[-1]
    assert last.status == expected_status, (
        f"expected DB row status={expected_status!r}, got {last.status!r}; "
        "gateway forgot to persist the verdict-driven transition"
    )
    assert last.risk == "high"
    if isinstance(verdict_payload, dict):
        assert last.approver == verdict_payload.get("approver")
        assert last.approval_rationale == verdict_payload.get("rationale")


# ---------------------------------------------------------------------------
# async path mirror — the gateway's _arun branch must also persist


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict_payload, expected_status",
    [
        ({"decision": "approve", "approver": "alice", "rationale": "ok"}, "approved"),
        ({"decision": "reject", "approver": "bob", "rationale": "no"}, "rejected"),
        ({"decision": "timeout", "approver": None, "rationale": None}, "timeout"),
    ],
)
async def test_gateway_async_persists_resolution_status(
    store, session, monkeypatch, verdict_payload, expected_status,
):
    """Same contract as the sync path, exercised through ``_arun`` —
    the langchain agent's tool dispatcher uses the async path."""
    monkeypatch.setattr(lg_types, "interrupt", lambda _payload: verdict_payload)

    wrapped = wrap_tool(
        _make_ping_tool(), session=session, gateway_cfg=_gateway_cfg(),
        agent_name="resolution", store=store,
    )
    await wrapped.ainvoke({"msg": "x"})

    fresh = store.load(session.id)
    last = [tc for tc in fresh.tool_calls if tc.tool == "ping"][-1]
    assert last.status == expected_status
    assert last.risk == "high"


# ---------------------------------------------------------------------------
# Edge: store=None must not crash (no-op persist branch)


def test_gateway_skips_persist_when_store_is_none(session, monkeypatch):
    """When the wrap is built without a store (legacy unit-test path),
    the verdict-driven transition still updates the in-memory row but
    does not crash on ``store.save``. ``store=None`` is a supported
    configuration."""
    monkeypatch.setattr(
        lg_types, "interrupt",
        lambda _payload: {"decision": "approve", "approver": "x", "rationale": "y"},
    )
    wrapped = wrap_tool(
        _make_ping_tool(), session=session, gateway_cfg=_gateway_cfg(),
        agent_name="resolution", store=None,
    )
    out: Any = wrapped.invoke({"msg": "go"})
    assert out == {"echo": "go"}
    # In-memory row updated even with no store.
    assert session.tool_calls[-1].status == "approved"
