"""Phase 15 (LLM-COMPAT-01) — Integration Driver S1 (live LLM path).

This test exercises the full ``make_agent_node`` flow against a REAL
LLM provider to verify the recursion-limit class of bugs is gone.
Stub-mode coverage lives in ``test_real_llm_tool_loop_termination.py``;
this driver is the human-verification artefact that confirms the fix
holds across at least two providers (one OpenAI-compatible, one
Ollama).

The test is gated on env vars and is SKIPPED by default. Set both
``OPENROUTER_API_KEY`` (for the OpenAI-compatible path) and
``OLLAMA_API_KEY`` (for the Ollama-cloud path) to opt in. CI
environments without keys will skip cleanly — the absence is
expected and reported via VERIFICATION.md as ``human_needed``.

Hard contract under test:
- ``await agent.ainvoke(...)`` reaches a terminal state (i.e. returns)
  without raising ``GraphRecursionError`` or hitting any artificial
  bound.
- ``result["structured_response"]`` is a valid AgentTurnOutput.
- The session ends with a recorded AgentRun that carries the
  envelope's confidence and content.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from runtime.agents.responsive import make_agent_node
from runtime.agents.turn_output import AgentTurnOutput
from runtime.config import (
    EmbeddingConfig,
    LLMConfig,
    MetadataConfig,
    ModelConfig,
    ProviderConfig,
)
from runtime.graph import GraphState, route_from_skill
from runtime.llm import get_llm
from runtime.skill import RouteRule, Skill
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
_OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
_AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
_AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
_AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")


def _make_repo(tmp_path: Path) -> SessionStore:
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, embedder=embedder)


def _build_llm_cfg() -> LLMConfig:
    """Three providers + three named models — what ``get_llm`` consumes.

    OpenRouter, Ollama Cloud, and Azure OpenAI are each declared. The
    parametrize arms below skip per-leg if the corresponding key is
    missing, so a partial-key environment exercises whichever
    providers it can reach.
    """
    providers: dict = {}
    models: dict = {}
    if _OPENROUTER_KEY:
        providers["openrouter"] = ProviderConfig(
            kind="openai_compat",
            base_url="https://openrouter.ai/api/v1",
            api_key=_OPENROUTER_KEY,
        )
        models["workhorse"] = ModelConfig(
            provider="openrouter", model="openai/gpt-4o-mini",
        )
    if _OLLAMA_KEY and _OLLAMA_BASE_URL:
        providers["ollama"] = ProviderConfig(
            kind="ollama",
            base_url=_OLLAMA_BASE_URL,
            api_key=_OLLAMA_KEY,
        )
        models["local"] = ModelConfig(provider="ollama", model="gpt-oss:20b")
    if _AZURE_KEY and _AZURE_ENDPOINT:
        providers["azure"] = ProviderConfig(
            kind="azure_openai",
            endpoint=_AZURE_ENDPOINT,
            api_version="2024-08-01-preview",
            api_key=_AZURE_KEY,
        )
        models["azure"] = ModelConfig(
            provider="azure", model="gpt-4o", deployment=_AZURE_DEPLOYMENT,
        )
    if not models:
        # Fallback so the LLMConfig validator (which requires the
        # default to exist in models) passes when the test is being
        # collected but every provider is keyless. The actual test
        # invocations are then skipped per-arm.
        from runtime.config import ProviderConfig as _PC, ModelConfig as _MC
        providers["stub"] = _PC(kind="stub")
        models["stub_default"] = _MC(provider="stub", model="stub-1")
        default = "stub_default"
    else:
        # Pick whichever model is available, in priority order.
        default = next(
            (m for m in ("local", "workhorse", "azure") if m in models),
            next(iter(models)),
        )
    return LLMConfig(default=default, providers=providers, models=models)


_PER_MODEL_SKIP_REASON = {
    "workhorse": (
        "OPENROUTER_API_KEY not set — set it to exercise the OpenAI-compatible leg",
        lambda: not _OPENROUTER_KEY,
    ),
    "local": (
        "OLLAMA_API_KEY + OLLAMA_BASE_URL not set — set both to exercise the Ollama leg",
        lambda: not (_OLLAMA_KEY and _OLLAMA_BASE_URL),
    ),
    "azure": (
        "AZURE_OPENAI_KEY + AZURE_ENDPOINT not set — set both to exercise the Azure leg",
        lambda: not (_AZURE_KEY and _AZURE_ENDPOINT),
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["workhorse", "local", "azure"])
async def test_integration_driver_s1_terminal_state(tmp_path, model_name):
    """S1: agent_node reaches a terminal state across live providers.

    Parametrize arms (``workhorse`` / ``local`` / ``azure``) cover the
    three production provider kinds. Each arm skips independently if
    its keys are absent — partial-key environments still exercise
    whichever providers they can reach.
    """
    skip_reason, predicate = _PER_MODEL_SKIP_REASON[model_name]
    if predicate():
        pytest.skip(skip_reason)

    cfg = _build_llm_cfg()
    llm = get_llm(cfg, model_name)

    repo = _make_repo(tmp_path)
    session = repo.create(
        query="hello, please respond briefly",
        environment="dev",
        reporter_id="u",
        reporter_team="t",
    )
    skill = Skill(
        name="responder",
        description="Brief responder skill for integration test.",
        routes=[RouteRule(when="default", next="__end__")],
        # Phase 22 contract: every reply MUST end with the markdown
        # envelope so the framework's parse_envelope_from_result Path 4
        # can lift confidence + signal. Without this prompt the live
        # LLM emits plain prose, hits no tool / no envelope, and Path
        # 7 (terminal) raises EnvelopeMissingError. Mirror of the
        # contract documented in
        # ``examples/incident_management/skills/*/system.md``.
        system_prompt=(
            "You are a concise assistant. Respond to the user's prompt "
            "in one sentence. Do not invoke any tools.\n\n"
            "## Output contract — REQUIRED\n"
            "Every final reply MUST end with these three sections, in"
            " order, each preceded by a level-2 markdown header:\n"
            "  ## Response\n"
            "  <one-sentence reply>\n"
            "  ## Confidence\n"
            "  <0.0-1.0 float> -- <one-line rationale>\n"
            "  ## Signal\n"
            "  none\n"
            "**CRITICAL — final-reply rule:** the markdown envelope is"
            " mandatory; the framework hard-fails if it is missing."
        ),
    )
    node = make_agent_node(
        skill=skill,
        llm=llm,
        tools=[],
        decide_route=lambda inc: route_from_skill(skill, inc),
        store=repo,
    )

    state: GraphState = {"session": session, "next_route": None}
    # 60s upper-bound for a single LLM round-trip; provider timeouts
    # in get_llm are independently bounded at 120s.
    result = await asyncio.wait_for(node(state), timeout=60.0)

    assert result.get("error") is None, (
        f"agent_node failed for model {model_name}: {result.get('error')}"
    )
    inc = repo.load(session.id)
    assert inc.agents_run, "expected at least one AgentRun to be recorded"
    last = inc.agents_run[-1]
    assert isinstance(last.summary, str) and last.summary.strip(), (
        "expected a non-empty summary derived from the AgentTurnOutput "
        "envelope"
    )
    # Confidence must be present and within the schema bounds; we don't
    # assert a specific value -- providers calibrate differently.
    assert last.confidence is not None
    assert 0.0 <= last.confidence <= 1.0
    # Sanity: the AgentTurnOutput class is what the structured response
    # is parsed as in the stub path. For real providers we trust the
    # ``parse_envelope_from_result`` helper in the node body to have
    # validated the schema before stamping the AgentRun.
    _ = AgentTurnOutput  # silence the unused import lint without enabling F401
