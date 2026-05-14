"""LLM provider abstraction with stub/ollama/azure_openai backends.

Models are resolved by name from ``LLMConfig``. Each named entry binds a
provider (kind + connection) to a model id and optional temperature/deployment.
``get_llm(cfg, "smart")`` looks up ``cfg.models["smart"]`` and uses its
referenced ``cfg.providers[<name>]`` to build a langchain ``BaseChatModel``.

Phase 13 (HARD-01 / HARD-05): every chat + embedding HTTP call is bounded
by an effective ``request_timeout`` resolved as
``provider.request_timeout if not None else default_llm_request_timeout``
(default 120.0s on ``OrchestratorConfig``). The native langchain timeout
knob is wired AND an ``asyncio.wait_for`` wrapper raises
``LLMTimeoutError(provider, model, elapsed_ms)`` on hang -- defence in
depth against partial-byte stalls where the httpx layer doesn't fire.
The hardcoded public-Ollama fallback is removed; ollama providers
must declare ``base_url`` (validated at config-load via
``LLMConfigError``).
"""
from __future__ import annotations
import asyncio
import os
import time
from typing import Any
from uuid import uuid4
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from runtime.config import LLMConfig, ModelConfig, ProviderConfig
from runtime.errors import LLMTimeoutError


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

    Phase 22 (D-22-05): the loop terminates on natural React END now
    that ``response_format=AgentTurnOutput`` is gone — tests drive
    markdown-shaped canned text (the framework parses it via Path 4
    in :func:`runtime.agents.turn_output.parse_envelope_from_result`).
    The Phase 15 envelope-as-callable-tool auto-emit is removed; the
    stub is back to its pre-Phase-15 simple semantics (canned text +
    optional pre-scripted ``tool_call_plan``).
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
        # Phase 22 (D-22-05): tool-call rounds emit the scripted tool
        # calls with a placeholder content body; the closing turn (no
        # remaining tool_call_plan entries) emits the canned text
        # wrapped in the D-22-03 markdown contract that
        # parse_markdown_envelope reads. The stub_envelope_*
        # parameters drive the trailing-section bodies so tests can
        # exercise specific confidence / rationale / signal paths
        # without authoring markdown by hand.
        tool_calls: list[dict] = []
        if self.tool_call_plan and not self._called_once:
            for tc in self.tool_call_plan:
                tool_calls.append(
                    {"name": tc["name"], "args": tc.get("args", {}), "id": str(uuid4())}
                )
            self._called_once = True

        body = self.canned_responses.get(
            self.role, f"[stub:{self.role}] no canned response",
        )
        # Phase 22 (D-22-05): explicit "none" when no signal so the
        # parser maps to None — preserves pre-Phase-22 behaviour where a
        # stub with no signal yielded envelope.signal=None.
        signal_str = self.stub_envelope_signal if self.stub_envelope_signal is not None else "none"
        md = (
            f"{body}\n\n"
            f"## Response\n{body}\n\n"
            f"## Confidence\n{self.stub_envelope_confidence:.4f} -- "
            f"{self.stub_envelope_rationale}\n\n"
            f"## Signal\n{signal_str}\n"
        )
        msg = AIMessage(content=md, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Phase 22 (D-22-05): no-op tool binding.

        ``langchain.agents.create_agent`` calls ``bind_tools`` to wire
        the agent's BaseTool list onto the chat model. The stub does
        not actually invoke those tools (test fixtures script tool
        calls via ``tool_call_plan`` instead), so we just return self
        unchanged. The pre-Phase-22 envelope-tool detection is gone
        because the framework no longer asks for ``response_format``.
        """
        return self

    # ``BaseChatModel.with_structured_output`` returns ``Runnable[..., dict | BaseModel]``
    # in the langchain stub; this stub override returns a deterministic
    # ``_StructuredRunnable`` so tests can drive structured outputs
    # without a live provider. Functionally a Runnable (it implements
    # ``invoke`` + ``ainvoke``); the stub mismatch is cosmetic.
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Phase 10 (FOC-03): honour the structured-output pass.

        Historically (pre-Phase-15) the deprecated
        ``langgraph.prebuilt.create_react_agent`` factory called this
        after its tool loop completed. The current
        ``langchain.agents.create_agent`` path uses a tool-strategy
        binding instead (see ``bind_tools`` above), but providers and
        test code that call ``with_structured_output`` directly still
        get a deterministic schema instance.

        We return a Runnable-like that yields a valid ``schema``
        instance derived from the stub's canned text and the
        per-instance envelope configuration. Tests can tune
        ``stub_envelope_confidence`` etc. to drive gate / reconcile
        paths.
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


def _resolve_timeout(
    provider: ProviderConfig, default: float,
) -> float:
    """Resolve effective request timeout for a provider.

    Per-provider override wins; falls back to the framework default
    (typically ``OrchestratorConfig.default_llm_request_timeout``).
    """
    if provider.request_timeout is not None:
        return provider.request_timeout
    return default


def _wrap_chat_with_timeout(
    base: BaseChatModel,
    provider_name: str,
    model_id: str,
    request_timeout: float,
) -> BaseChatModel:
    """Wrap ``base`` so every ``ainvoke`` is bounded by
    ``asyncio.wait_for(..., timeout=request_timeout)`` and raises
    ``LLMTimeoutError(provider, model, elapsed_ms)`` on hang.

    The native langchain timeout knob (``request_timeout=`` on
    openai/azure or ``client_kwargs={'timeout': ...}`` on ollama) is
    honoured at the httpx layer; this wrapper guarantees the
    framework-typed exception AND a hard ceiling even if the
    underlying client hangs in a way httpx misses (e.g., post-headers
    TCP read stall on a slow Ollama). D-13-04: subclassing
    ``TimeoutError`` means ``policy._TRANSIENT_TYPES`` auto-classifies
    the error as transient (zero edits to ``policy.py``).
    """
    base_cls = type(base)

    class _Bounded(base_cls):  # type: ignore[misc, valid-type]
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                return await asyncio.wait_for(
                    super().ainvoke(*args, **kwargs),
                    timeout=request_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError) as e:
                if isinstance(e, LLMTimeoutError):
                    # Already typed; don't double-wrap.
                    raise
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                raise LLMTimeoutError(
                    provider=provider_name,
                    model=model_id,
                    elapsed_ms=elapsed_ms,
                ) from e

    # Reuse the live pydantic instance's state without re-running
    # __init__ (which would re-init the underlying httpx clients).
    bounded = _Bounded.model_construct(**base.model_dump())
    # Some langchain client classes initialise non-pydantic attrs
    # (httpx clients, run_manager, etc.) inside __init__. Copy them
    # through so the wrapped instance shares the same network state.
    for attr_name in (
        "_client", "_async_client",
        "_async_httpx_client", "_sync_httpx_client",
        "client", "async_client",
    ):
        if hasattr(base, attr_name):
            try:
                object.__setattr__(
                    bounded, attr_name, getattr(base, attr_name),
                )
            except (AttributeError, TypeError):
                # Slot-only or read-only attrs on some langchain
                # versions -- the bounded instance will re-init on
                # first use; not a correctness issue.
                pass
    return bounded


def _build_ollama_chat(
    provider: ProviderConfig, model_id: str, temperature: float,
    *, request_timeout: float,
) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # Many Ollama models (gemma*, gpt-oss, ministral, etc.) don't support
    # native function-calling, which is langchain-ollama's default method
    # for ``with_structured_output``. Subclass to force
    # ``method='json_schema'`` (uses Ollama's structured-output API) so
    # Phase 10's ``response_format=AgentTurnOutput`` envelope actually
    # round-trips instead of failing with ``OutputParserException``
    # when the LLM emits prose.
    class _ChatOllamaJsonSchema(ChatOllama):  # type: ignore[misc, valid-type]
        def with_structured_output(self, schema, *, method=None, **kw):
            return super().with_structured_output(
                schema, method=method or "json_schema", **kw,
            )

    # Phase 13 (HARD-01): ChatOllama has NO native ``request_timeout``
    # field; the canonical incantation is ``client_kwargs={"timeout": ...}``,
    # which propagates to the underlying httpx.AsyncClient.
    client_kwargs: dict[str, Any] = {"timeout": request_timeout}
    api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
    if api_key:
        client_kwargs["headers"] = {
            "Authorization": f"Bearer {api_key}",
        }
    # Phase 13 (HARD-05): base_url is now config-load-validated by
    # ProviderConfig._validate_required_fields. NO fallback to a
    # public Ollama URL (air-gap rule violation).
    kwargs: dict[str, Any] = {
        "base_url": provider.base_url,
        "model": model_id,
        "temperature": temperature,
        "client_kwargs": client_kwargs,
    }
    base = _ChatOllamaJsonSchema(**kwargs)
    return _wrap_chat_with_timeout(
        base, "ollama", model_id, request_timeout,
    )


def _build_azure_chat(
    provider: ProviderConfig, model: ModelConfig,
    *, request_timeout: float,
) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI
    if provider.endpoint is None:
        raise ValueError("azure_openai provider requires 'endpoint'")
    if model.deployment is None:
        raise ValueError(
            f"azure_openai model {model.model!r} requires 'deployment'"
        )
    _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
    # ``request_timeout`` is a runtime alias for ``timeout`` on
    # AzureChatOpenAI (langchain-openai > 0.3 declares it via Pydantic
    # ``Field(alias="timeout")``); the langchain stubs only expose
    # ``timeout``, hence the stub gap.
    base = AzureChatOpenAI(
        azure_endpoint=provider.endpoint,
        api_version=provider.api_version or "2024-08-01-preview",
        azure_deployment=model.deployment,
        api_key=SecretStr(_ak) if _ak else None,
        temperature=model.temperature,
        request_timeout=request_timeout,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
    )
    return _wrap_chat_with_timeout(
        base, "azure_openai", model.model, request_timeout,
    )


def get_llm(cfg: LLMConfig, model_name: str | None = None, *,
            role: str = "default",
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None,
            stub_envelope_confidence: float | None = None,
            stub_envelope_rationale: str | None = None,
            stub_envelope_signal: str | None = None,
            default_llm_request_timeout: float = 120.0,
            ) -> BaseChatModel:
    """Build a chat model by named entry from ``cfg.models``.

    ``model_name`` defaults to ``cfg.default``. Validation that the name
    exists is enforced by ``LLMConfig`` itself (model_validator), so a
    missing name here means caller passed a typo -- raise loudly.

    Phase 10 (FOC-03): stub callers can now tune the canned envelope
    (confidence / rationale / signal) so gate-trigger tests preserve their
    pre-Phase-10 semantics by emitting a low-confidence envelope.

    Phase 13 (HARD-01): non-stub builds are bounded by an effective
    ``request_timeout`` resolved as ``provider.request_timeout`` (per-
    provider override) -> ``default_llm_request_timeout`` (framework
    default; callers pass ``cfg.orchestrator.default_llm_request_timeout``).
    The default keyword value (120.0) matches OrchestratorConfig's default
    so test paths that build LLMs without an OrchestratorConfig in scope
    still get a sane bound.
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

    effective = _resolve_timeout(provider, default_llm_request_timeout)

    if provider.kind == "ollama":
        return _build_ollama_chat(
            provider, model.model, model.temperature,
            request_timeout=effective,
        )
    if provider.kind == "azure_openai":
        return _build_azure_chat(
            provider, model, request_timeout=effective,
        )
    if provider.kind == "openai_compat":
        return _build_openai_compat_chat(
            provider, model, request_timeout=effective,
        )
    raise ValueError(f"Unknown provider kind: {provider.kind!r}")


def _build_openai_compat_chat(
    provider: ProviderConfig, model: ModelConfig,
    *, request_timeout: float,
) -> BaseChatModel:
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
    # See AzureChatOpenAI block above: ``request_timeout`` is a runtime
    # alias for ``timeout`` not in the langchain stubs.
    base = ChatOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=model.model,
        temperature=model.temperature,
        request_timeout=request_timeout,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
    )
    return _wrap_chat_with_timeout(
        base, "openai_compat", model.model, request_timeout,
    )


def get_embedding(
    cfg: LLMConfig, *, default_llm_request_timeout: float = 120.0,
) -> Embeddings:
    """Build the configured embedding model. Raises if ``cfg.embedding`` is None.

    Phase 13 (HARD-01): same per-provider override -> framework default
    timeout resolution as ``get_llm``. Embeddings traffic shares the
    request_timeout knob with chat (see CONTEXT.md "Deferred Ideas" --
    splitting embedding timeout from chat is a future refinement).

    Note (Phase 13 review WR-01): unlike the chat builders -- which apply a
    defence-in-depth ``asyncio.wait_for`` wrapper (``_wrap_chat_with_timeout``)
    that guarantees a structured ``LLMTimeoutError`` with ``elapsed_ms`` even
    on partial-byte stalls -- embeddings rely SOLELY on the underlying
    httpx-layer timeout configured above (``client_kwargs={"timeout": ...}``
    for Ollama, ``request_timeout=`` for Azure). This asymmetry is a
    deliberate scope choice tied to Phase 13 CONTEXT.md "Deferred Ideas" #4
    (splitting embeddings timeout from chat timeout). If embeddings need
    stricter bounds than chat -- or if the httpx-layer timeout proves
    insufficient against post-headers TCP read stalls on the embeddings
    path the same way it can on chat -- a future phase can mirror
    ``_wrap_chat_with_timeout`` for the embeddings public surface
    (``aembed_query`` / ``aembed_documents``).
    """
    if cfg.embedding is None:
        raise ValueError("llm.embedding is not configured")
    provider = cfg.providers[cfg.embedding.provider]
    effective = _resolve_timeout(provider, default_llm_request_timeout)
    if provider.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        # Phase 13 (HARD-01): OllamaEmbeddings has NO native
        # ``request_timeout`` field; canonical incantation is
        # ``client_kwargs={"timeout": ...}`` (same as ChatOllama).
        client_kwargs: dict[str, Any] = {"timeout": effective}
        api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            client_kwargs["headers"] = {
                "Authorization": f"Bearer {api_key}",
            }
        # Phase 13 (HARD-05): base_url config-load-validated; NO public fallback.
        return OllamaEmbeddings(
            base_url=provider.base_url,
            model=cfg.embedding.model,
            client_kwargs=client_kwargs,
        )
    if provider.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        if provider.endpoint is None:
            raise ValueError("azure_openai provider requires 'endpoint'")
        deployment = cfg.embedding.deployment or cfg.embedding.model
        _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
        # See chat builders above: ``request_timeout`` is a runtime
        # alias for ``timeout`` not surfaced in the langchain-openai stub.
        return AzureOpenAIEmbeddings(
            azure_endpoint=provider.endpoint,
            api_version=provider.api_version or "2024-08-01-preview",
            azure_deployment=deployment,
            api_key=SecretStr(_ak) if _ak else None,
            request_timeout=effective,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
        )
    raise ValueError(
        f"Embedding not supported for provider kind {provider.kind!r}"
    )
