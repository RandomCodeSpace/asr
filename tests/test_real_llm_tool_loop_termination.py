"""Phase 15 (LLM-COMPAT-01) — real-LLM tool-loop termination contract.

These stub-mode tests pin the behavioural contract that resolved the
``recursion_limit=25`` workaround introduced in commit ``3ba099f``:

1. ``langchain.agents.create_agent`` (the non-deprecated successor to
   ``langgraph.prebuilt.create_react_agent``) is the only agent factory
   imported in production code.
2. The agent loop terminates cleanly through the AgentTurnOutput
   envelope acting as a structured-output tool — no separate post-loop
   ``with_structured_output`` LLM call required.
3. ``_ainvoke_with_retry`` no longer caps recursion at 25 as a safety
   net; the default langgraph upper bound is back to being a true
   ceiling, not a workaround.

The tests are deterministic: they exercise the public ``make_agent_node``
factory against ``EnvelopeStubChatModel`` / ``StubChatModel`` and assert
the contract end-to-end without touching a real provider. The companion
file ``test_integration_driver_s1.py`` covers the live-provider path
under explicit env-var gates.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from runtime.agents.responsive import make_agent_node
from runtime.agents.turn_output import AgentTurnOutput
from runtime.config import EmbeddingConfig, MetadataConfig, ProviderConfig
from runtime.graph import GraphState, _ainvoke_with_retry, route_from_skill
from runtime.llm import StubChatModel
from runtime.skill import RouteRule, Skill
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore

from tests._envelope_helpers import EnvelopeStubChatModel


# ---------------------------------------------------------------------------
# Helpers


def _make_repo(tmp_path: Path) -> SessionStore:
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, embedder=embedder)


@pytest.fixture
def repo(tmp_path: Path) -> SessionStore:
    return _make_repo(tmp_path)


@pytest.fixture
def session(repo: SessionStore):
    return repo.create(
        query="exhibits stable failure mode",
        environment="dev",
        reporter_id="u",
        reporter_team="t",
    )


# ---------------------------------------------------------------------------
# T4-1 — sanity: import surface points at the non-deprecated factory


def test_create_agent_resolves_to_langchain_agents_factory():
    """Phase 15: ``langchain.agents.create_agent`` is the new home of
    the agent factory. The import must resolve from that module path,
    NOT from the deprecated ``langgraph.prebuilt.create_react_agent``.
    """
    from langchain.agents import create_agent  # noqa: PLC0415

    assert create_agent.__module__.startswith("langchain.agents")
    sig = inspect.signature(create_agent)
    # Confirms the new-API parameters present (system_prompt + middleware,
    # not the old ``prompt`` keyword).
    assert "system_prompt" in sig.parameters
    assert "response_format" in sig.parameters
    assert "middleware" in sig.parameters


# ---------------------------------------------------------------------------
# T4-2 — agent_node terminates cleanly with no tool calls


@pytest.mark.asyncio
async def test_agent_node_terminates_via_envelope_tool_call(repo, session):
    """End-to-end stub-mode contract: ``make_agent_node`` runs to
    completion against an ``EnvelopeStubChatModel`` whose
    ``tool_call_plan`` is empty, so the LLM emits zero tool calls.
    The migrated ``create_agent`` + ToolStrategy path closes the loop
    with a synthetic AgentTurnOutput tool call (recorded via
    ``_envelope_tool_name`` on the stub).
    """
    skill = Skill(
        name="intake",
        description="d",
        routes=[RouteRule(when="default", next="triage")],
        system_prompt="You are intake.",
    )
    llm = EnvelopeStubChatModel(
        role="intake",
        envelope_content="all clear",
        envelope_confidence=0.91,
        envelope_rationale="stub rationale",
        canned_responses={"intake": "all clear"},
    )
    node = make_agent_node(
        skill=skill,
        llm=llm,
        tools=[],
        decide_route=lambda inc: route_from_skill(skill, inc),
        store=repo,
    )
    state: GraphState = {"session": session, "next_route": None}
    result = await asyncio.wait_for(node(state), timeout=5.0)

    assert result["next_route"] == "triage"
    assert result.get("error") is None
    # The harvested envelope confidence flows into the agent_run row.
    inc = repo.load(session.id)
    assert inc.agents_run, "node must record an AgentRun"
    last = inc.agents_run[-1]
    assert last.confidence == pytest.approx(0.91)
    assert last.summary == "all clear"


# ---------------------------------------------------------------------------
# T4-3 — agent_node terminates after a tool round-trip


@pytest.mark.asyncio
async def test_agent_node_terminates_after_tool_round_trip(repo, session):
    """When ``tool_call_plan`` schedules one real tool call, the loop
    runs that tool, then the stub's ``_envelope_tool_name`` path emits
    the closing AgentTurnOutput. The loop terminates within the
    default langgraph recursion bound (no workaround needed).
    """

    class _PingArgs(BaseModel):
        msg: str

    def _ping(msg: str) -> str:
        return f"pong:{msg}"

    ping_tool = StructuredTool.from_function(
        func=_ping,
        name="ping",
        description="ping the system",
        args_schema=_PingArgs,
    )
    skill = Skill(
        name="intake",
        description="d",
        routes=[RouteRule(when="default", next="triage")],
        system_prompt="You are intake.",
    )
    llm = EnvelopeStubChatModel(
        role="intake",
        envelope_content="ping done",
        envelope_confidence=0.78,
        canned_responses={"intake": "ping done"},
        tool_call_plan=[{"name": "ping", "args": {"msg": "hi"}}],
    )
    node = make_agent_node(
        skill=skill,
        llm=llm,
        tools=[ping_tool],
        decide_route=lambda inc: route_from_skill(skill, inc),
        store=repo,
    )
    state: GraphState = {"session": session, "next_route": None}
    result = await asyncio.wait_for(node(state), timeout=5.0)

    assert result.get("error") is None
    inc = repo.load(session.id)
    # The real tool call landed; the closing envelope tool call is
    # NOT persisted as an actual ToolCall (it carries the structured
    # response, not a tool result).
    real_tool_calls = [tc for tc in inc.tool_calls if tc.tool == "ping"]
    assert len(real_tool_calls) == 1
    assert real_tool_calls[0].args == {"msg": "hi"}


# ---------------------------------------------------------------------------
# T4-4 — recursion_limit=25 workaround removed (regression guard)


def test_recursion_limit_workaround_removed_from_ainvoke_with_retry():
    """Source-level regression guard for Phase 15.

    Commit ``3ba099f`` introduced ``config={"recursion_limit": 25}`` as
    a safety net to surface infinite tool loops as ``GraphRecursionError``
    instead of hanging silently. The Phase 15 migration to
    ``langchain.agents.create_agent`` removes the underlying root
    cause (separate post-loop ``with_structured_output`` pass that
    Ollama models couldn't satisfy), so the workaround is gone.

    This test pins that decision: future contributors who reintroduce
    a hardcoded recursion-limit override in ``_ainvoke_with_retry``'s
    ``ainvoke`` call will fail the suite and be forced to justify the
    change in the diff. Comments mentioning the historical workaround
    are allowed (and useful for future maintainers).
    """
    src = inspect.getsource(_ainvoke_with_retry)
    # Strip hash-comment lines so we only inspect executable code.
    code_lines = [
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "recursion_limit" not in code_only, (
        "Phase 15 (LLM-COMPAT-01) removed the recursion_limit=25 safety "
        "net introduced in 3ba099f. If you need a recursion bound, "
        "either expose it via OrchestratorConfig (a deliberate decision) "
        "or use ``ModelCallLimitMiddleware`` from langchain.agents."
    )


# ---------------------------------------------------------------------------
# T4-5 — no production import of the deprecated create_react_agent


def test_no_create_react_agent_imports_in_production_runtime():
    """Source-level regression guard.

    Phase 15 migrated both call sites to
    ``langchain.agents.create_agent``. ``langgraph.prebuilt.create_react_agent``
    is officially deprecated and must not creep back into production
    code. Comments / docstrings referencing the symbol historically
    are allowed; only EXECUTABLE imports and call sites are flagged.
    """
    runtime_root = (
        Path(__file__).resolve().parent.parent / "src" / "runtime"
    )
    assert runtime_root.is_dir(), (
        f"expected src/runtime under {runtime_root.parent}; got "
        f"{runtime_root}"
    )
    offenders: list[tuple[Path, int, str]] = []
    for py in runtime_root.rglob("*.py"):
        for lineno, raw in enumerate(
            py.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            stripped = raw.lstrip()
            if stripped.startswith("#"):
                continue
            if "create_react_agent" not in raw:
                continue
            # Only treat IMPORT statements and bare call sites as
            # offenders. A docstring referencing the deprecated symbol
            # for historical context is fine — it's surrounded by
            # triple-quotes and is not executable code.
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or "create_react_agent(" in raw
            ):
                offenders.append((py, lineno, raw.strip()))
    assert not offenders, (
        "Phase 15 (LLM-COMPAT-01): langgraph.prebuilt.create_react_agent "
        "is deprecated. Use langchain.agents.create_agent instead. "
        f"Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# T4-bonus — StubChatModel.bind_tools registers the envelope tool name


def test_stub_chat_model_records_envelope_tool_name_on_bind():
    """``StubChatModel.bind_tools`` is the integration point that lets
    the new ``create_agent`` loop terminate in stub mode. This test
    locks the contract: when the bound tools include an
    ``AgentTurnOutput``-named entry, the stub records it and emits a
    closing tool call with that name on the next ``_generate``.
    """
    llm = StubChatModel(role="agent", canned_responses={"agent": "ok"})
    # Simulate what create_agent's ToolStrategy passes: a sequence of
    # tool specs where the AgentTurnOutput-named tool is the structured-
    # output sentinel.
    llm.bind_tools([AgentTurnOutput])
    assert llm._envelope_tool_name == "AgentTurnOutput"

    # Drive a single _generate and verify the closing tool call lands.
    result = llm._generate(messages=[HumanMessage(content="go")])
    msg = result.generations[0].message
    assert msg.tool_calls, "expected a closing envelope tool call"
    assert msg.tool_calls[0]["name"] == "AgentTurnOutput"
    args = msg.tool_calls[0]["args"]
    assert args["content"] == "ok"
    assert args["confidence"] == pytest.approx(0.85)
    assert "confidence_rationale" in args
