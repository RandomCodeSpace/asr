import pytest
from langchain_core.messages import HumanMessage, AIMessage
from runtime.config import LLMConfig, ProviderConfig, ModelConfig
from runtime.llm import StubChatModel, get_embedding, get_llm


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


def test_get_llm_uses_default_model_when_none_passed():
    cfg = LLMConfig.stub()
    llm = get_llm(cfg)
    assert isinstance(llm, StubChatModel)


def test_get_llm_resolves_named_model():
    cfg = LLMConfig(
        default="primary",
        providers={"stub": ProviderConfig(kind="stub")},
        models={
            "primary": ModelConfig(provider="stub", model="m1"),
            "alt": ModelConfig(provider="stub", model="m2"),
        },
    )
    assert isinstance(get_llm(cfg, "alt"), StubChatModel)


def test_get_llm_unknown_name_raises():
    with pytest.raises(KeyError, match="ghost"):
        get_llm(LLMConfig.stub(), "ghost")


def test_get_embedding_raises_when_unconfigured():
    with pytest.raises(ValueError, match="llm.embedding is not configured"):
        get_embedding(LLMConfig.stub())
