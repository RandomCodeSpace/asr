"""Phase 13 -- LLM Provider Hardening (HARD-01 timeouts + HARD-05 fallback removal).

Acceptance tests for:
- ROADMAP success-criteria #1: bounded request_timeout on every provider HTTP call
- ROADMAP success-criteria #2: typed LLMConfigError at config-load for ollama
- ROADMAP success-criteria #3: typed LLMTimeoutError with provider/model/elapsed_ms
- ROADMAP success-criteria #4: covered separately by acceptance grep (Task 8)
- D-13-04: LLMTimeoutError classified transient via policy._TRANSIENT_TYPES
- D-13-05/06: LLMConfigError contract; ollama-only validation in scope
- Hidden contract: LLMTimeoutError.__str__ contains "timed out" so existing
  graph.py / orchestrator.py string-matchers catch it.
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from runtime.config import (
    LLMConfig, ModelConfig, OrchestratorConfig, ProviderConfig,
)
from runtime.errors import LLMConfigError, LLMTimeoutError


# ---------------------------------------------------------------------------
# OrchestratorConfig.default_llm_request_timeout (D-13-02)
# ---------------------------------------------------------------------------

def test_orchestrator_config_default_timeout_120s() -> None:
    cfg = OrchestratorConfig()
    assert cfg.default_llm_request_timeout == 120.0


def test_orchestrator_config_timeout_field_bounded() -> None:
    # gt=0
    with pytest.raises(ValidationError):
        OrchestratorConfig(default_llm_request_timeout=0)
    with pytest.raises(ValidationError):
        OrchestratorConfig(default_llm_request_timeout=-1)
    # le=600
    with pytest.raises(ValidationError):
        OrchestratorConfig(default_llm_request_timeout=601)
    # accepted bounds
    OrchestratorConfig(default_llm_request_timeout=0.001)
    OrchestratorConfig(default_llm_request_timeout=600)


# ---------------------------------------------------------------------------
# ProviderConfig.request_timeout (D-13-01) + ollama validator (D-13-06)
# ---------------------------------------------------------------------------

def test_provider_request_timeout_override_resolves() -> None:
    p = ProviderConfig(
        kind="ollama", base_url="http://localhost:11434",
        request_timeout=300,
    )
    assert p.request_timeout == 300.0


def test_provider_request_timeout_default_is_none() -> None:
    p = ProviderConfig(kind="ollama", base_url="http://x")
    assert p.request_timeout is None


def test_provider_request_timeout_field_bounded() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(
            kind="ollama", base_url="http://x", request_timeout=0,
        )
    with pytest.raises(ValidationError):
        ProviderConfig(
            kind="ollama", base_url="http://x", request_timeout=-5,
        )
    with pytest.raises(ValidationError):
        ProviderConfig(
            kind="ollama", base_url="http://x", request_timeout=601,
        )


def test_ollama_provider_missing_base_url_raises_at_config_load() -> None:
    """D-13-06 + ROADMAP #2: pydantic validator fires before any HTTP call."""
    with pytest.raises(ValidationError) as excinfo:
        ProviderConfig(kind="ollama")  # base_url omitted
    causes = [
        err.get("ctx", {}).get("error") for err in excinfo.value.errors()
    ]
    matched = [c for c in causes if isinstance(c, LLMConfigError)]
    assert matched, f"expected LLMConfigError in causes, got: {causes!r}"
    assert matched[0].missing_field == "base_url"
    assert matched[0].provider == "ollama"


def test_ollama_provider_empty_base_url_raises_at_config_load() -> None:
    """Empty string base_url is still 'missing' -- the validator uses 'not base_url'."""
    with pytest.raises(ValidationError):
        ProviderConfig(kind="ollama", base_url="")


def test_ollama_provider_present_base_url_validates() -> None:
    p = ProviderConfig(kind="ollama", base_url="http://localhost:11434")
    assert p.base_url == "http://localhost:11434"


def test_other_providers_unaffected_by_ollama_validator() -> None:
    """D-13-06: only ollama is promoted to config-load validation in Phase 13.

    azure_openai (`endpoint`) and openai_compat (`base_url` + `api_key`) keep
    their existing first-request ValueError raises in `_build_*_chat`.
    """
    ProviderConfig(kind="azure_openai")  # no endpoint required at load
    ProviderConfig(kind="openai_compat")  # no base_url/api_key required at load
    ProviderConfig(kind="stub")           # no fields required at all


# ---------------------------------------------------------------------------
# LLMConfigError contract (D-13-05)
# ---------------------------------------------------------------------------

def test_llm_config_error_subclass_of_value_error() -> None:
    e = LLMConfigError(provider="ollama", missing_field="base_url")
    assert isinstance(e, ValueError)
    assert e.provider == "ollama"
    assert e.missing_field == "base_url"
    assert "ollama" in str(e)
    assert "base_url" in str(e)


# ---------------------------------------------------------------------------
# LLMTimeoutError contract + policy classification (D-13-04)
# ---------------------------------------------------------------------------

def test_llm_timeout_error_subclass_of_timeout_error() -> None:
    e = LLMTimeoutError(provider="x", model="y", elapsed_ms=42)
    assert isinstance(e, TimeoutError)
    assert e.provider == "x"
    assert e.model == "y"
    assert e.elapsed_ms == 42


def test_llm_timeout_error_str_contains_timed_out() -> None:
    """Hidden contract: graph.py:_TRANSIENT_MARKERS and orchestrator.py:809
    string-match on 'timed out'. If the message wording changes the markers
    silently miss the new error -- see CONTEXT.md 'specifics' note.
    """
    e = LLMTimeoutError(provider="ollama", model="llama3.1:8b", elapsed_ms=1500)
    assert "timed out" in str(e)
    assert "ollama" in str(e)
    assert "llama3.1:8b" in str(e)
    assert "1500" in str(e)


def test_llm_timeout_error_classified_transient_in_policy() -> None:
    """D-13-04: subclass of TimeoutError -> auto-classified by
    policy._TRANSIENT_TYPES via isinstance. Zero edits to policy.py.
    """
    from runtime.policy import _is_transient_error
    err = LLMTimeoutError(provider="x", model="y", elapsed_ms=100)
    assert _is_transient_error(err) is True


# ---------------------------------------------------------------------------
# get_llm signature + threading (Task 4 contract)
# ---------------------------------------------------------------------------

def test_get_llm_signature_has_default_llm_request_timeout() -> None:
    import inspect
    from runtime.llm import get_llm
    sig = inspect.signature(get_llm)
    assert "default_llm_request_timeout" in sig.parameters
    p = sig.parameters["default_llm_request_timeout"]
    assert p.default == 120.0
    assert p.kind == inspect.Parameter.KEYWORD_ONLY


def test_get_embedding_signature_has_default_llm_request_timeout() -> None:
    import inspect
    from runtime.llm import get_embedding
    sig = inspect.signature(get_embedding)
    assert "default_llm_request_timeout" in sig.parameters
    p = sig.parameters["default_llm_request_timeout"]
    assert p.default == 120.0


def test_get_llm_stub_path_ignores_timeout() -> None:
    """Stub LLMs are in-process -- the timeout knob has no effect.

    Verifies (a) stub still works, (b) the new keyword is accepted on
    the signature (regression guard for Task 3 edits).
    """
    from runtime.llm import get_llm
    cfg = LLMConfig.stub()
    llm = get_llm(cfg, default_llm_request_timeout=42.0)
    # Stub model -- no _wrap_chat_with_timeout applied.
    from runtime.llm import StubChatModel
    assert isinstance(llm, StubChatModel)


# ---------------------------------------------------------------------------
# Timeout fires (HARD-01 / ROADMAP #3) -- monkey-patch ChatOllama.ainvoke
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_timeout_fires_with_structured_error(monkeypatch) -> None:
    """Slow upstream -> LLMTimeoutError with provider/model/elapsed_ms.

    Strategy (RESEARCH.md Q3): monkey-patch the parent ChatOllama.ainvoke
    to await asyncio.sleep(1.0); set request_timeout=0.05; the
    _Bounded.ainvoke wrapper's asyncio.wait_for fires first and converts
    asyncio.TimeoutError -> LLMTimeoutError. No new test deps.
    """
    cfg = LLMConfig(
        default="m",
        providers={
            "ollama_local": ProviderConfig(
                kind="ollama",
                base_url="http://localhost:11434",
                request_timeout=0.05,  # 50ms -- way under the sleep below
            ),
        },
        models={
            "m": ModelConfig(
                provider="ollama_local", model="llama3.1:8b",
            ),
        },
    )
    from runtime.llm import get_llm
    # default_llm_request_timeout doesn't matter -- per-provider
    # request_timeout=0.05 wins via _resolve_timeout.
    llm = get_llm(cfg, default_llm_request_timeout=120.0)

    from langchain_ollama import ChatOllama

    async def _slow_ainvoke(self, *_args, **_kwargs):
        await asyncio.sleep(1.0)
        raise AssertionError("should have timed out before this")

    monkeypatch.setattr(ChatOllama, "ainvoke", _slow_ainvoke)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await llm.ainvoke([HumanMessage(content="hi")])
    err = excinfo.value
    # provider name is the provider KIND ("ollama"), not the YAML key.
    # _wrap_chat_with_timeout in src/runtime/llm.py is called with the
    # literal kind so structured logs aggregate by upstream-provider type.
    assert err.provider == "ollama"
    assert err.model == "llama3.1:8b"
    assert err.elapsed_ms >= 40  # rough lower bound (50ms timeout)
    assert err.elapsed_ms < 1000  # didn't actually wait the full 1s
    assert "timed out" in str(err)


@pytest.mark.asyncio
async def test_llm_timeout_uses_default_when_provider_unset(monkeypatch) -> None:
    """If ProviderConfig.request_timeout is None, get_llm uses
    default_llm_request_timeout (D-13-02 resolution order).
    """
    cfg = LLMConfig(
        default="m",
        providers={
            "ollama_local": ProviderConfig(
                kind="ollama",
                base_url="http://localhost:11434",
                # request_timeout NOT set -- falls back to default
            ),
        },
        models={
            "m": ModelConfig(
                provider="ollama_local", model="llama3.1:8b",
            ),
        },
    )
    from runtime.llm import get_llm
    llm = get_llm(cfg, default_llm_request_timeout=0.05)

    from langchain_ollama import ChatOllama

    async def _slow_ainvoke(self, *_args, **_kwargs):
        await asyncio.sleep(1.0)
        raise AssertionError("should have timed out before this")

    monkeypatch.setattr(ChatOllama, "ainvoke", _slow_ainvoke)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await llm.ainvoke([HumanMessage(content="hi")])
    err = excinfo.value
    assert err.elapsed_ms < 1000
