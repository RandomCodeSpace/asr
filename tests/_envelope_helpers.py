"""Test helpers for AgentTurnOutput envelope-shaped LLM stubs (Phase 10 / FOC-03).

Centralised so the 5 fixture-migration files (test_resume, test_gate,
test_build_graph, test_gateway_integration, test_injected_args) all share one
implementation. Avoids inline AIMessage(content=...) drift across tests.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from runtime.agents.turn_output import AgentTurnOutput


def envelope_stub(
    content: str = "ok",
    confidence: float = 0.85,
    rationale: str = "default rationale",
    signal: str | None = None,
) -> dict[str, Any]:
    """Return a `create_react_agent`-shaped result dict with messages + structured_response.

    Used by tests that need to fake the FULL ReAct executor return — i.e.
    tests that call `parse_envelope_from_result(...)` directly without
    actually running the executor.
    """
    return {
        "messages": [AIMessage(content=content)],
        "structured_response": AgentTurnOutput(
            content=content,
            confidence=confidence,
            confidence_rationale=rationale,
            signal=signal,
        ),
    }


class EnvelopeStubChatModel(BaseChatModel):
    """A stub chat model that emits an envelope-shaped final message AND
    answers `with_structured_output` calls with a pre-built AgentTurnOutput.

    `create_react_agent(..., response_format=AgentTurnOutput)` internally
    calls `llm.with_structured_output(AgentTurnOutput)` to produce
    `result["structured_response"]`. This stub short-circuits both the
    tool-loop AIMessage AND the structured-output pass with the same
    canned envelope so tests are deterministic.

    For tool-call chains, set `tool_call_plan` like `StubChatModel` does;
    the structured_response is the FINAL pass after the tool loop.
    """

    role: str = "default"
    envelope_content: str = "stub envelope"
    envelope_confidence: float = 0.85
    envelope_rationale: str = "stub rationale"
    envelope_signal: str | None = None
    canned_responses: dict[str, str] = Field(default_factory=dict)
    tool_call_plan: list[dict] | None = None
    _called_once: bool = False

    @property
    def _llm_type(self) -> str:
        return "envelope-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.canned_responses.get(self.role, self.envelope_content)
        tool_calls: list[dict] = []
        if self.tool_call_plan and not self._called_once:
            for tc in self.tool_call_plan:
                tool_calls.append(
                    {"name": tc["name"], "args": tc.get("args", {}), "id": str(uuid4())}
                )
            self._called_once = True
        msg = AIMessage(content=text, tool_calls=tool_calls)
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

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Return a Runnable-like object whose `invoke`/`ainvoke` returns the
        canned AgentTurnOutput. LangGraph 1.1.x calls this after the tool loop.
        """
        envelope = AgentTurnOutput(
            content=self.envelope_content,
            confidence=self.envelope_confidence,
            confidence_rationale=self.envelope_rationale,
            signal=self.envelope_signal,
        )

        class _StructuredRunnable:
            def __init__(self, env: AgentTurnOutput):
                self._env = env

            def invoke(self, *_args, **_kwargs):
                return self._env

            async def ainvoke(self, *_args, **_kwargs):
                return self._env

        return _StructuredRunnable(envelope)


def make_stub_llm_with_envelope(
    *,
    content: str = "stub envelope",
    confidence: float = 0.85,
    rationale: str = "stub rationale",
    signal: str | None = None,
    tool_call_plan: list[dict] | None = None,
    canned_responses: dict[str, str] | None = None,
    role: str = "default",
) -> EnvelopeStubChatModel:
    """Convenience factory for tests."""
    return EnvelopeStubChatModel(
        role=role,
        envelope_content=content,
        envelope_confidence=confidence,
        envelope_rationale=rationale,
        envelope_signal=signal,
        tool_call_plan=tool_call_plan,
        canned_responses=canned_responses or {},
    )


__all__ = [
    "envelope_stub",
    "EnvelopeStubChatModel",
    "make_stub_llm_with_envelope",
]
