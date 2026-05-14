"""v1.5-C (M8 proof point): per-agent LLM dispatch contract.

Pins the contract that ``runtime.graph._build_agent_nodes`` resolves
``skill.model`` per-skill, so apps can route different agents through
different providers without touching the framework.

The live demonstration of this contract — intake on Ollama Cloud
gpt-oss while downstream agents follow ``llm.default`` — lives in
``examples/incident_management/skills/intake/config.yaml`` (the
``model: gpt_oss_cheap`` line) and is exercised by
``tests/test_integration_driver_s1.py`` when the appropriate API
keys are set.

These tests run without keys: they intercept ``runtime.graph.get_llm``
and assert the model name passed for each skill matches the skill's
``model`` field (or the ``LLMConfig.default`` fallback when ``model``
is None).
"""
from __future__ import annotations

from unittest.mock import patch

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    OrchestratorConfig,
    Paths,
    RuntimeConfig,
)
from runtime.mcp_loader import ToolRegistry
from runtime.skill import RouteRule, Skill


def _stub_app_cfg() -> AppConfig:
    """Minimal AppConfig with two named models — the framework picks
    between them by ``skill.model`` only."""
    llm_cfg = LLMConfig.stub()
    return AppConfig(
        llm=llm_cfg,
        mcp=MCPConfig(servers=[]),
        paths=Paths(skills_dir="config/skills", incidents_dir="/tmp"),
        runtime=RuntimeConfig(state_class=None),
        orchestrator=OrchestratorConfig(),
    )


def test_build_agent_nodes_passes_skill_model_to_get_llm():
    """The framework must call ``get_llm(cfg.llm, skill.model, ...)`` for
    every responsive skill. Without this, per-agent provider swaps
    silently collapse to the default model.

    We fully mock ``get_llm`` to capture the (role, model_name) tuple
    per skill — this isolates the test from the LLMConfig.models
    registry shape, which is what the production code resolves the
    name through downstream.
    """
    from runtime.graph import _build_agent_nodes
    from runtime.llm import StubChatModel

    skills = {
        "intake": Skill(
            name="intake",
            description="d",
            kind="responsive",
            model="gpt_oss_cheap",
            routes=[RouteRule(when="default", next="triage")],
            system_prompt="x",
        ),
        "triage": Skill(
            name="triage",
            description="d",
            kind="responsive",
            model=None,  # falls back to llm.default downstream
            routes=[RouteRule(when="default", next="__end__")],
            system_prompt="x",
        ),
    }

    captured: list[tuple[str, str | None]] = []

    def _fake_get_llm(cfg, model_name, *, role, **kwargs):
        captured.append((role, model_name))
        return StubChatModel(role=role)

    cfg = _stub_app_cfg()
    with patch("runtime.graph.get_llm", side_effect=_fake_get_llm):
        nodes = _build_agent_nodes(
            cfg=cfg,
            skills=skills,
            store=None,  # type: ignore[arg-type] — _build_agent_nodes
            # only forwards ``store`` to make_agent_node, never reads it
            # itself; tests of the dispatch contract leave it None.
            registry=ToolRegistry(entries={}),
        )

    # Both skills produced a node.
    assert set(nodes.keys()) == {"intake", "triage"}

    # Per-skill model resolution: intake got its override, triage got
    # None (which get_llm resolves to llm.default downstream).
    by_role = dict(captured)
    assert by_role.get("intake") == "gpt_oss_cheap", (
        f"intake should resolve to its skill.model override; got {by_role!r}"
    )
    assert by_role.get("triage") is None, (
        f"triage skill.model was None; should pass None through so "
        f"get_llm falls back to llm.default; got {by_role!r}"
    )


def test_intake_skill_yaml_has_per_agent_override_uncommented():
    """The intake skill config must carry ``model: gpt_oss_cheap`` — the
    v1.5-C deliverable. A human flipping this back to a comment is
    intentional (e.g. forcing all-default for a benchmark run); the
    test fails if that happens silently in a refactor.
    """
    import yaml
    from pathlib import Path

    cfg_path = Path(
        "examples/incident_management/skills/intake/config.yaml"
    )
    parsed = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert parsed.get("model") == "gpt_oss_cheap", (
        f"intake skill must declare ``model: gpt_oss_cheap`` per the "
        f"v1.5-C M8 proof point; got {parsed.get('model')!r}. If "
        f"intentionally rolling back, remove this test guard too."
    )
