"""Coverage tests for ``StubChatModel.with_structured_output`` (llm.py:141-160, 171-177).

The structured-output runnable was previously only exercised indirectly
via ``langchain.agents.create_agent``. These tests pin the direct
contract: the stub returns a Runnable-like that yields a valid schema
instance per ``invoke`` / ``ainvoke``, populated from the canned text and
``stub_envelope_*`` parameters.

The permissive ``model_validate`` fallback (lines 161-169) is genuinely
defensive: pydantic v2's ``model_validate`` internally calls ``__init__``
too, so any schema whose constructor raises also fails the fallback.
The fallback exists for hypothetical schemas with custom
``__pydantic_validator__`` overrides, which the framework doesn't ship
and tests can't construct without monkey-patching pydantic internals.
"""
from __future__ import annotations

import pytest

from runtime.agents.turn_output import AgentTurnOutput
from runtime.llm import StubChatModel


def _stub(*, confidence: float = 0.85, rationale: str = "stub rationale",
          signal: str | None = None, role: str = "intake",
          canned: dict[str, str] | None = None) -> StubChatModel:
    return StubChatModel(
        role=role,
        canned_responses=canned if canned is not None else {role: "stub body text"},
        stub_envelope_confidence=confidence,
        stub_envelope_rationale=rationale,
        stub_envelope_signal=signal,
    )


class TestStubStructuredOutputHappyPath:
    """Happy path: schema(...) keyword constructor succeeds."""

    def test_invoke_returns_schema_instance(self):
        runnable = _stub().with_structured_output(AgentTurnOutput)
        out = runnable.invoke("any input")
        assert isinstance(out, AgentTurnOutput)
        assert out.content == "stub body text"
        assert out.confidence == 0.85
        assert out.confidence_rationale == "stub rationale"
        assert out.signal is None

    @pytest.mark.asyncio
    async def test_ainvoke_returns_schema_instance(self):
        runnable = _stub(confidence=0.42, rationale="hedge", signal="retry").with_structured_output(AgentTurnOutput)
        out = await runnable.ainvoke("any input")
        assert isinstance(out, AgentTurnOutput)
        assert out.confidence == 0.42
        assert out.confidence_rationale == "hedge"
        assert out.signal == "retry"

    def test_canned_response_missing_uses_default_marker(self):
        runnable = _stub(role="ghost", canned={}).with_structured_output(AgentTurnOutput)
        out = runnable.invoke("x")
        assert out.content.startswith("[stub:ghost]")

    def test_include_raw_kwarg_is_accepted(self):
        # langchain passes include_raw=True/False on the call site; the stub
        # accepts the kwarg but doesn't change behaviour.
        runnable = _stub().with_structured_output(AgentTurnOutput, include_raw=True)
        assert runnable.invoke("x").content == "stub body text"

    def test_extra_kwargs_are_swallowed(self):
        runnable = _stub().with_structured_output(AgentTurnOutput, method="json_mode", strict=True)
        assert runnable.invoke("x").confidence == 0.85


