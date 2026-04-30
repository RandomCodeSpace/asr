import pytest
from langchain_core.messages import HumanMessage, AIMessage
from orchestrator.llm import StubChatModel


@pytest.mark.asyncio
async def test_stub_returns_canned_response_per_role():
    canned = {
        "intake": "Created INC and routed to triage.",
        "triage": "Severity sev3, category latency.",
    }
    llm = StubChatModel(role="intake", canned_responses=canned)
    result = await llm.ainvoke([HumanMessage(content="any")])
    assert isinstance(result, AIMessage)
    assert "Created INC" in result.content


@pytest.mark.asyncio
async def test_stub_unknown_role_returns_default():
    llm = StubChatModel(role="unknown", canned_responses={})
    result = await llm.ainvoke([HumanMessage(content="x")])
    assert "stub" in result.content.lower()


@pytest.mark.asyncio
async def test_stub_with_tools_emits_tool_call_when_configured():
    llm = StubChatModel(
        role="intake",
        canned_responses={"intake": "ok"},
        tool_call_plan=[{"name": "lookup_similar_incidents", "args": {"query": "x", "environment": "dev"}}],
    )
    result = await llm.ainvoke([HumanMessage(content="x")])
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "lookup_similar_incidents"
