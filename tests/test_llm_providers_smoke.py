"""Smoke tests for the LLM provider layer.

The bulk of this file talks to live providers and is gated behind
environment variables — local runs are silent unless the relevant
credentials are present.

M8 opt-in live invocation (Ollama-via-LangChain proof point):

    OLLAMA_LIVE=1 OLLAMA_API_KEY=... \\
      pytest tests/test_llm_providers_smoke.py -k ollama -v

The two M8 tests assert that:
1. ``get_llm(cfg, "gpt_oss")`` returns a working LangChain chat
   model that round-trips a prompt against Ollama Cloud's gpt-oss:20b.
2. ``get_embedding(cfg)`` returns a working LangChain Embeddings
   instance over local Ollama's bge-m3 model (1024-dim vectors).
"""
import os
import pytest
from langchain_core.messages import HumanMessage
from runtime.config import (
    EmbeddingConfig,
    LLMConfig,
    ModelConfig,
    ProviderConfig,
)
from runtime.llm import get_embedding, get_llm


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


# ---------------------------------------------------------------------
# M8 — per-agent Ollama-via-LangChain proof point.
# Both tests are opt-in via OLLAMA_LIVE=1 so the suite stays silent
# without credentials.
# ---------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.environ.get("OLLAMA_LIVE") and os.environ.get("OLLAMA_API_KEY")),
    reason="OLLAMA_LIVE=1 + OLLAMA_API_KEY required",
)
async def test_ollama_cloud_chat_via_langchain():
    """get_llm(cfg, "gpt_oss") -> LangChain chat that invokes
    against Ollama Cloud's gpt-oss:20b and returns a non-empty
    AIMessage."""
    cfg = LLMConfig(
        default="gpt_oss",
        providers={
            "ollama_cloud": ProviderConfig(
                kind="ollama",
                base_url=os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com"),
                api_key=os.environ["OLLAMA_API_KEY"],
            ),
        },
        models={
            "gpt_oss": ModelConfig(
                provider="ollama_cloud",
                model=os.environ.get("OLLAMA_TEST_MODEL", "gpt-oss:20b"),
                temperature=0.0,
            ),
        },
    )
    llm = get_llm(cfg, "gpt_oss")
    res = llm.invoke("ping")
    assert res is not None
    text = getattr(res, "content", "")
    assert isinstance(text, str) and len(text) > 0


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_LIVE"),
    reason="OLLAMA_LIVE=1 required (assumes local Ollama at OLLAMA_LOCAL_URL "
    "with bge-m3 pulled)",
)
def test_ollama_local_embed_via_langchain():
    """get_embedding(cfg) -> LangChain Embeddings whose embed_query
    returns a 1024-dim vector against local Ollama's bge-m3 model."""
    cfg = LLMConfig(
        default="workhorse",
        providers={
            "ollama_local": ProviderConfig(
                kind="ollama",
                base_url=os.environ.get(
                    "OLLAMA_LOCAL_URL", "http://localhost:11434",
                ),
            ),
        },
        models={
            "workhorse": ModelConfig(
                provider="ollama_local", model="gpt-oss:20b",
            ),
        },
        embedding=EmbeddingConfig(
            provider="ollama_local", model="bge-m3", dim=1024,
        ),
    )
    embedder = get_embedding(cfg)
    vec = embedder.embed_query("ping")
    assert isinstance(vec, list) and len(vec) == 1024
    assert all(isinstance(x, float) for x in vec)


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
