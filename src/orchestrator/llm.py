"""LLM provider abstraction with stub/ollama/azure_openai backends.

Models are resolved by name from ``LLMConfig``. Each named entry binds a
provider (kind + connection) to a model id and optional temperature/deployment.
``get_llm(cfg, "smart")`` looks up ``cfg.models["smart"]`` and uses its
referenced ``cfg.providers[<name>]`` to build a langchain ``BaseChatModel``.
"""
from __future__ import annotations
import os
from typing import Any
from uuid import uuid4
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from orchestrator.config import LLMConfig, ModelConfig, ProviderConfig


class StubChatModel(BaseChatModel):
    """Deterministic chat model for tests/CI. Returns canned text per role.

    Optionally emits one tool call on first invocation if `tool_call_plan` is set.
    """
    role: str = "default"
    canned_responses: dict[str, str] = Field(default_factory=dict)
    tool_call_plan: list[dict] | None = None
    _called_once: bool = False

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        text = self.canned_responses.get(self.role, f"[stub:{self.role}] no canned response")
        tool_calls: list[dict] = []
        if self.tool_call_plan and not self._called_once:
            for tc in self.tool_call_plan:
                tool_calls.append({"name": tc["name"], "args": tc.get("args", {}), "id": str(uuid4())})
            self._called_once = True
        msg = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """No-op binder: stub emits tool calls only via `tool_call_plan`, not via real binding."""
        return self


def _build_ollama_chat(provider: ProviderConfig, model_id: str,
                       temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    kwargs: dict[str, Any] = {
        "base_url": provider.base_url or "https://ollama.com",
        "model": model_id,
        "temperature": temperature,
    }
    api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
    if api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return ChatOllama(**kwargs)


def _build_azure_chat(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI
    if provider.endpoint is None:
        raise ValueError("azure_openai provider requires 'endpoint'")
    if model.deployment is None:
        raise ValueError(
            f"azure_openai model {model.model!r} requires 'deployment'"
        )
    _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
    return AzureChatOpenAI(
        azure_endpoint=provider.endpoint,
        api_version=provider.api_version or "2024-08-01-preview",
        azure_deployment=model.deployment,
        api_key=SecretStr(_ak) if _ak else None,
        temperature=model.temperature,
    )


def get_llm(cfg: LLMConfig, model_name: str | None = None, *,
            role: str = "default",
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None) -> BaseChatModel:
    """Build a chat model by named entry from ``cfg.models``.

    ``model_name`` defaults to ``cfg.default``. Validation that the name
    exists is enforced by ``LLMConfig`` itself (model_validator), so a
    missing name here means caller passed a typo — raise loudly.
    """
    name = model_name or cfg.default
    model = cfg.models.get(name)
    if model is None:
        raise KeyError(
            f"llm model {name!r} not found in llm.models "
            f"(known: {sorted(cfg.models)})"
        )
    provider = cfg.providers[model.provider]  # validated at config load

    if provider.kind == "stub":
        return StubChatModel(
            role=role,
            canned_responses=stub_canned or {},
            tool_call_plan=stub_tool_plan,
        )
    if provider.kind == "ollama":
        return _build_ollama_chat(provider, model.model, model.temperature)
    if provider.kind == "azure_openai":
        return _build_azure_chat(provider, model)
    raise ValueError(f"Unknown provider kind: {provider.kind!r}")


def get_embedding(cfg: LLMConfig) -> Embeddings:
    """Build the configured embedding model. Raises if ``cfg.embedding`` is None."""
    if cfg.embedding is None:
        raise ValueError("llm.embedding is not configured")
    provider = cfg.providers[cfg.embedding.provider]
    if provider.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        kwargs: dict[str, Any] = {
            "base_url": provider.base_url or "https://ollama.com",
            "model": cfg.embedding.model,
        }
        api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return OllamaEmbeddings(**kwargs)
    if provider.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        if provider.endpoint is None:
            raise ValueError("azure_openai provider requires 'endpoint'")
        deployment = cfg.embedding.deployment or cfg.embedding.model
        _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
        return AzureOpenAIEmbeddings(
            azure_endpoint=provider.endpoint,
            api_version=provider.api_version or "2024-08-01-preview",
            azure_deployment=deployment,
            api_key=SecretStr(_ak) if _ak else None,
        )
    raise ValueError(
        f"Embedding not supported for provider kind {provider.kind!r}"
    )
