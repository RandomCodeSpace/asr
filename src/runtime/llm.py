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

from runtime.config import LLMConfig, ModelConfig, ProviderConfig


class StubChatModel(BaseChatModel):
    """Deterministic chat model for tests/CI. Returns canned text per role.

    Optionally emits one tool call on first invocation if `tool_call_plan` is set.

    Phase 10 (FOC-03): also honours
    ``llm.with_structured_output(AgentTurnOutput)`` so stub-driven tests
    survive the runner's envelope contract. The structured response is
    derived from the same canned text + a default 0.85 confidence; tests
    that need a specific envelope shape can override
    ``stub_envelope_confidence`` / ``stub_envelope_rationale`` /
    ``stub_envelope_signal``.
    """
    role: str = "default"
    canned_responses: dict[str, str] = Field(default_factory=dict)
    tool_call_plan: list[dict] | None = None
    stub_envelope_confidence: float = 0.85
    stub_envelope_rationale: str = "stub envelope rationale"
    stub_envelope_signal: str | None = None
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

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Phase 10 (FOC-03): honour LangGraph's structured-output pass.

        ``create_react_agent(..., response_format=schema)`` calls this after
        the tool loop completes. We return a Runnable-like that yields a
        valid ``schema`` instance derived from the stub's canned text and
        the per-instance envelope configuration. Tests can tune
        ``stub_envelope_confidence`` etc. to drive gate / reconcile paths.
        """
        text = self.canned_responses.get(self.role, f"[stub:{self.role}] no canned response")
        confidence = self.stub_envelope_confidence
        rationale = self.stub_envelope_rationale
        signal = self.stub_envelope_signal

        class _StructuredRunnable:
            def __init__(self, schema_cls):
                self._schema = schema_cls

            def _build(self):
                # Construct an instance of whatever schema was passed.
                # Common case: AgentTurnOutput; permissive fallback handles
                # other pydantic schemas the test may pass.
                try:
                    return self._schema(
                        content=text or ".",
                        confidence=confidence,
                        confidence_rationale=rationale,
                        signal=signal,
                    )
                except Exception:
                    # Permissive fallback for unfamiliar schemas: try
                    # model_validate on a minimal dict.
                    return self._schema.model_validate({
                        "content": text or ".",
                        "confidence": confidence,
                        "confidence_rationale": rationale,
                        "signal": signal,
                    })

            def invoke(self, *_args, **_kwargs):
                return self._build()

            async def ainvoke(self, *_args, **_kwargs):
                return self._build()

        return _StructuredRunnable(schema)


def _build_ollama_chat(provider: ProviderConfig, model_id: str,
                       temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # Many Ollama models (gemma*, gpt-oss, ministral, etc.) don't support
    # native function-calling, which is langchain-ollama's default method
    # for ``with_structured_output``. Subclass to force
    # ``method='json_schema'`` (uses Ollama's structured-output API) so
    # Phase 10's ``response_format=AgentTurnOutput`` envelope actually
    # round-trips instead of failing with ``OutputParserException``
    # when the LLM emits prose. Callers that want a different method
    # may still override by passing ``method=`` explicitly.
    class _ChatOllamaJsonSchema(ChatOllama):  # type: ignore[misc, valid-type]
        def with_structured_output(self, schema, *, method=None, **kw):
            return super().with_structured_output(
                schema, method=method or "json_schema", **kw,
            )

    kwargs: dict[str, Any] = {
        "base_url": provider.base_url or "https://ollama.com",
        "model": model_id,
        "temperature": temperature,
    }
    api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
    if api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return _ChatOllamaJsonSchema(**kwargs)


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
            stub_tool_plan: list[dict] | None = None,
            stub_envelope_confidence: float | None = None,
            stub_envelope_rationale: str | None = None,
            stub_envelope_signal: str | None = None) -> BaseChatModel:
    """Build a chat model by named entry from ``cfg.models``.

    ``model_name`` defaults to ``cfg.default``. Validation that the name
    exists is enforced by ``LLMConfig`` itself (model_validator), so a
    missing name here means caller passed a typo — raise loudly.

    Phase 10 (FOC-03): stub callers can now tune the canned envelope
    (confidence / rationale / signal) so gate-trigger tests preserve their
    pre-Phase-10 semantics by emitting a low-confidence envelope.
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
        kwargs: dict[str, Any] = {
            "role": role,
            "canned_responses": stub_canned or {},
            "tool_call_plan": stub_tool_plan,
        }
        if stub_envelope_confidence is not None:
            kwargs["stub_envelope_confidence"] = stub_envelope_confidence
        if stub_envelope_rationale is not None:
            kwargs["stub_envelope_rationale"] = stub_envelope_rationale
        if stub_envelope_signal is not None:
            kwargs["stub_envelope_signal"] = stub_envelope_signal
        return StubChatModel(**kwargs)
    if provider.kind == "ollama":
        return _build_ollama_chat(provider, model.model, model.temperature)
    if provider.kind == "azure_openai":
        return _build_azure_chat(provider, model)
    if provider.kind == "openai_compat":
        return _build_openai_compat_chat(provider, model)
    raise ValueError(f"Unknown provider kind: {provider.kind!r}")


def _build_openai_compat_chat(provider: ProviderConfig,
                              model: ModelConfig) -> BaseChatModel:
    """Build a ``ChatOpenAI`` pointed at an OpenAI-compatible endpoint
    (OpenRouter, Together, vLLM, etc.). Reuses langchain-openai's
    ``ChatOpenAI`` with ``base_url=`` override and the provider's
    ``api_key`` (resolved from env via the YAML loader).
    """
    from langchain_openai import ChatOpenAI
    if provider.base_url is None:
        raise ValueError(
            "openai_compat provider requires 'base_url' "
            "(e.g. https://openrouter.ai/api/v1)"
        )
    if provider.api_key is None:
        raise ValueError("openai_compat provider requires 'api_key'")
    return ChatOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=model.model,
        temperature=model.temperature,
    )


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
