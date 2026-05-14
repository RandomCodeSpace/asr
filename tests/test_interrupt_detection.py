"""Regression: agent runner must raise GraphInterrupt when ``create_agent``
surfaces a paused HITL gate via the result dict's ``__interrupt__`` field.

Background — langgraph 1.x changed the interrupt contract: a tool that
calls ``interrupt()`` no longer raises ``GraphInterrupt`` to the caller;
instead, ``agent.invoke(...)`` (or ``ainvoke``) returns a normal result
dict with an extra ``__interrupt__`` key carrying the captured
:class:`langgraph.types.Interrupt` records. The framework's agent
runner must therefore inspect the result and re-raise ``GraphInterrupt``
itself; otherwise the outer Pregel completes the agent step, the
envelope-parsing path fires (potentially Path 6 synthesis), the session
gets finalized, and the ``pending_approval`` ToolCall row written by the
gateway just before the interrupt is left orphaned — the UI's
Approve / Reject buttons no longer drive any resume because the session
is already terminal from the framework's perspective.

These tests pin the contract at the agent-runner boundary in BOTH
implementations: ``runtime.agents.responsive.make_agent_node`` (the
generic responsive runner) and ``runtime.graph.make_agent_node`` (the
incident-shape variant exposed via ``runtime.graph``). The fix is the
same in both: after ``ainvoke``, inspect ``result["__interrupt__"]``
and raise :class:`GraphInterrupt`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt
from pydantic import BaseModel

from runtime.agents.responsive import make_agent_node as responsive_make_agent_node
from runtime.config import EmbeddingConfig, MetadataConfig, ProviderConfig
from runtime.graph import GraphState, make_agent_node as graph_make_agent_node, route_from_skill
from runtime.skill import RouteRule, Skill
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


# ---------------------------------------------------------------------------
# Fixtures (mirror test_real_llm_tool_loop_termination shape so the
# repo + session story stays consistent across the file).


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
        query="HITL gate fires mid-tool",
        environment="production",
        reporter_id="u",
        reporter_team="t",
    )


# ---------------------------------------------------------------------------
# Stubs


class _PauseArgs(BaseModel):
    why: str = "test"


def _pause_tool_impl(why: str = "test") -> str:  # noqa: D401
    """Stand-in for a high-risk tool whose gateway raises interrupt()."""
    interrupt({"reason": "approve_me", "why": why})
    return f"ran:{why}"


def _make_pause_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_pause_tool_impl,
        name="pause_tool",
        description="Calls interrupt() — emulates the gateway's HITL gate.",
        args_schema=_PauseArgs,
    )


class _OneShotToolCallLLM(BaseChatModel):
    """Stub LLM that emits exactly one tool_call for ``pause_tool``.

    Mirrors the production agent's first turn: the LLM calls a tool, the
    tool gates, and the agent should pause. After resume the tool would
    return a real value and the LLM would emit a closing markdown
    envelope — but for THIS test we only assert that the first
    invocation pauses; the runner must observe ``__interrupt__`` and
    re-raise, so a closing turn is unreachable.
    """

    @property
    def _llm_type(self) -> str:
        return "oneshot-toolcall-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "pause_tool", "args": {"why": "x"}, "id": "c1"}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


# ---------------------------------------------------------------------------
# graph.make_agent_node


@pytest.mark.asyncio
async def test_graph_make_agent_node_raises_on_interrupt_in_result(repo, session):
    """When a tool calls ``interrupt()``, langgraph 1.x returns the
    pause via ``result["__interrupt__"]`` instead of raising
    ``GraphInterrupt``. The graph.py agent runner must detect this and
    re-raise so the outer Pregel propagates the pause — otherwise the
    envelope-parsing path fires, the session is finalized, and the
    pending_approval row written by the gateway is orphaned.
    """
    skill = Skill(
        name="resolution",
        description="d",
        routes=[RouteRule(when="default", next="end")],
        system_prompt="You are resolution.",
    )
    node = graph_make_agent_node(
        skill=skill,
        llm=_OneShotToolCallLLM(),
        tools=[_make_pause_tool()],
        decide_route=lambda inc: route_from_skill(skill, inc),
        store=repo,
    )
    state: GraphState = {"session": session, "next_route": None}
    with pytest.raises(GraphInterrupt):
        await asyncio.wait_for(node(state), timeout=5.0)


# ---------------------------------------------------------------------------
# responsive.make_agent_node


@pytest.mark.asyncio
async def test_responsive_make_agent_node_raises_on_interrupt_in_result(
    repo, session,
):
    """Same contract as above, exercised through the generic responsive
    runner. Both runners share the contract: ``__interrupt__`` in the
    ainvoke result must be re-raised as ``GraphInterrupt``.
    """
    skill = Skill(
        name="resolution",
        description="d",
        routes=[RouteRule(when="default", next="end")],
        system_prompt="You are resolution.",
    )
    node = responsive_make_agent_node(
        skill=skill,
        llm=_OneShotToolCallLLM(),
        tools=[_make_pause_tool()],
        decide_route=lambda inc: route_from_skill(skill, inc),
        store=repo,
    )
    state: GraphState = {"session": session, "next_route": None}
    with pytest.raises(GraphInterrupt):
        await asyncio.wait_for(node(state), timeout=5.0)


# ---------------------------------------------------------------------------
# Resume happy path — verdict actually reaches the gated tool.
#
# The previous tests cover the FIRST turn (pause). These tests cover the
# RESUME turn: when the outer Pregel is rehydrated via
# ``Command(resume=verdict)``, the helper ``_drive_agent_with_resume``
# detects the inner agent's paused state via ``aget_state``, calls
# ``interrupt()`` at the outer level to fetch the verdict, and forwards
# it via ``Command(resume=verdict)`` to the inner agent. The inner
# agent's tool then re-enters with the verdict in hand and produces a
# real ToolMessage — which is the only proof that the gated work
# actually executed (a "silent skip" path leaves no ToolMessage
# behind).


class _LLMWithCloser(BaseChatModel):
    """LLM that emits a tool_call once, then a closing AIMessage."""

    _counter: int = 0  # pyright: ignore[reportGeneralTypeIssues] — pydantic v2 needs annotation

    @property
    def _llm_type(self) -> str:
        return "loop-toolcall-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._counter += 1
        if self._counter == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "pause_tool", "args": {"why": "x"}, "id": "c1"},
                ],
            )
        else:
            # Closing markdown envelope — Path 4 parser will pick it up.
            md = (
                "all good\n\n"
                "## Response\nall good\n\n"
                "## Confidence\n0.9 -- post-resume close\n\n"
                "## Signal\nnone\n"
            )
            msg = AIMessage(content=md)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.mark.asyncio
async def test_resume_forwards_verdict_to_inner_tool_and_completes(
    repo, session,
):
    """End-to-end pause → resume → tool actually runs with verdict.

    Drives the agent runner across two invocations sharing a
    ``langgraph.checkpoint.memory.InMemorySaver``. The first call
    raises ``GraphInterrupt`` (paused mid-tool). The second call —
    re-entered as the outer would re-enter the node body after a
    ``Command(resume=...)`` — must (a) detect the inner pause via
    ``aget_state``, (b) forward the verdict via
    ``Command(resume=verdict)``, and (c) drive the tool to completion
    so a ``ToolMessage`` appears in the final inner message list. The
    ToolMessage is the proof that the gated tool actually executed
    rather than being silently skipped.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command, interrupt

    saver = InMemorySaver()

    # Tool whose interrupt() returns the verdict and whose final
    # return value records the verdict so we can assert it ended up in
    # the message stream.
    class _Args(BaseModel):
        why: str = "x"

    captured: dict[str, str | None] = {"verdict": None}

    def _gated(why: str = "x") -> str:
        v = interrupt({"need_verdict": why})
        captured["verdict"] = v
        return f"ran:{v}"

    gated = StructuredTool.from_function(
        func=_gated, name="pause_tool", description="emulates HITL gate",
        args_schema=_Args,
    )

    # Build an inner agent the same way the framework would, but
    # against the test's saver so the pause survives across calls.
    from langchain.agents import create_agent

    llm = _LLMWithCloser()
    inner = create_agent(
        model=llm, tools=[gated], system_prompt="x", checkpointer=saver,
    )
    inner_cfg = {"configurable": {"thread_id": "T-resume-test"}}

    # The helper's ``interrupt()`` calls require a real Pregel
    # runnable context — we cannot call ``_drive_agent_with_resume``
    # directly. Exercise it inside a minimal one-node outer graph
    # that owns its own checkpointer; that's exactly how the
    # framework wires it via ``runtime.graph.build_graph``.
    from runtime.graph import _drive_agent_with_resume
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class _OuterS(TypedDict, total=False):
        result: object

    async def _wrapper(state: _OuterS) -> dict:
        out = await _drive_agent_with_resume(
            agent_executor=inner, inner_cfg=inner_cfg,
            inner_has_checkpointer=True,
            initial_input={"messages": [("human", "go")]},
        )
        return {"result": out}

    sg = StateGraph(_OuterS)
    sg.add_node("n", _wrapper)
    sg.add_edge(START, "n")
    sg.add_edge("n", END)
    outer = sg.compile(checkpointer=InMemorySaver())
    outer_cfg = {"configurable": {"thread_id": "OUTER-T"}}

    # First outer.invoke({}): the wrapper hits the inner pause and
    # then the helper's outer interrupt() pauses the outer too.
    first = await outer.ainvoke({}, config=outer_cfg)
    assert "__interrupt__" in first

    # Now resume the outer with the verdict — the helper forwards it
    # to the inner via Command(resume="approve") and the gated tool
    # actually re-enters with the verdict.
    final = await outer.ainvoke(Command(resume="approve"), config=outer_cfg)

    inner_result = final.get("result")
    assert isinstance(inner_result, dict), f"unexpected inner result: {inner_result!r}"
    msgs = inner_result.get("messages") or []
    # Assert a ToolMessage exists with the gated-tool's actual output.
    tool_msgs = [m for m in msgs if type(m).__name__ == "ToolMessage"]
    assert tool_msgs, (
        "expected a ToolMessage in the final inner messages — its "
        "absence means the gated tool was silently skipped on resume "
        f"(verdict={captured['verdict']!r}, messages={[type(m).__name__ for m in msgs]})"
    )
    assert "ran:approve" in str(tool_msgs[-1].content), (
        f"tool ran but verdict was lost: {tool_msgs[-1].content!r}"
    )
    assert captured["verdict"] == "approve", (
        f"tool's interrupt() did not receive 'approve' on resume: "
        f"{captured['verdict']!r}"
    )
