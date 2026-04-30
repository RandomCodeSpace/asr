"""LLM provider abstraction with stub/ollama/azure_openai backends."""
from __future__ import annotations
import os
from typing import Any
from uuid import uuid4
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from orchestrator.config import LLMConfig


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


def get_llm(cfg: LLMConfig, *, role: str = "default", model: str | None = None,
            temperature: float | None = None,
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None) -> BaseChatModel:
    actual_model = model or cfg.default_model
    actual_temp = temperature if temperature is not None else cfg.default_temperature

    if cfg.provider == "stub":
        return StubChatModel(
            role=role,
            canned_responses=stub_canned or {},
            tool_call_plan=stub_tool_plan,
        )
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama
        if cfg.ollama is None:
            raise ValueError("ollama provider requires llm.ollama config")
        kwargs: dict[str, Any] = {
            "base_url": cfg.ollama.base_url,
            "model": actual_model,
            "temperature": actual_temp,
        }
        api_key = cfg.ollama.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return ChatOllama(**kwargs)
    if cfg.provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        if cfg.azure_openai is None:
            raise ValueError("azure_openai provider requires llm.azure_openai config")
        return AzureChatOpenAI(
            azure_endpoint=cfg.azure_openai.endpoint,
            api_version=cfg.azure_openai.api_version,
            azure_deployment=cfg.azure_openai.deployment,
            api_key=cfg.azure_openai.api_key or os.environ.get("AZURE_OPENAI_KEY"),
            temperature=actual_temp,
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")
