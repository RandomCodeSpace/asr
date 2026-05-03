import os
import pytest
from langchain_core.messages import HumanMessage
from runtime.config import LLMConfig, ProviderConfig, ModelConfig
from runtime.llm import get_llm


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OLLAMA_API_KEY"), reason="no OLLAMA_API_KEY")
async def test_ollama_smoke():
    cfg = LLMConfig(
        default="workhorse",
        providers={
            "ollama_cloud": ProviderConfig(
                kind="ollama",
                base_url="https://ollama.com",
                api_key=os.environ["OLLAMA_API_KEY"],
            ),
        },
        models={
            "workhorse": ModelConfig(
                provider="ollama_cloud",
                model=os.environ.get("OLLAMA_TEST_MODEL", "llama3.1:8b"),
            ),
        },
    )
    llm = get_llm(cfg)
    res = await llm.ainvoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in res.content.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in ("AZURE_OPENAI_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT")),
    reason="missing Azure credentials",
)
async def test_azure_openai_smoke():
    cfg = LLMConfig(
        default="smart",
        providers={
            "azure": ProviderConfig(
                kind="azure_openai",
                endpoint=os.environ["AZURE_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_KEY"],
            ),
        },
        models={
            "smart": ModelConfig(
                provider="azure",
                model="gpt-4o",
                deployment=os.environ["AZURE_DEPLOYMENT"],
            ),
        },
    )
    llm = get_llm(cfg)
    res = await llm.ainvoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in res.content.lower()
