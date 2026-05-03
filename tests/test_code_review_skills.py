"""Skill-loader tests for ``examples/code_review/skills/``.

Verifies P8-D scaffolding: the three skill directories load via the
framework's :func:`runtime.skill.load_all_skills`, and the ``_common/``
fragment is shared across all three agent prompts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.skill import Skill, load_all_skills


SKILLS_DIR = Path(__file__).resolve().parents[1] / "examples" / "code_review" / "skills"


def test_skills_dir_has_three_agents() -> None:
    """The example app must ship intake / analyzer / recommender + ``_common``.

    Pure filesystem assertion — does not exercise the loader. Catches
    accidental directory deletions before the more expensive load step.
    """
    assert SKILLS_DIR.is_dir(), f"missing skills dir: {SKILLS_DIR}"
    children = {p.name for p in SKILLS_DIR.iterdir() if p.is_dir()}
    expected = {"intake", "analyzer", "recommender", "_common"}
    missing = expected - children
    assert not missing, f"missing skill dirs: {sorted(missing)}"

    # Every agent dir must have config.yaml + at least one .md.
    for agent in ("intake", "analyzer", "recommender"):
        d = SKILLS_DIR / agent
        assert (d / "config.yaml").exists(), f"{agent}/config.yaml missing"
        mds = list(d.glob("*.md"))
        assert mds, f"{agent}/ has no .md files"


def test_skills_load_via_runtime_skill_loader() -> None:
    """``load_all_skills`` returns the three agents indexed by name."""
    skills = load_all_skills(SKILLS_DIR)
    assert set(skills.keys()) == {"intake", "analyzer", "recommender"}, (
        f"unexpected skills: {sorted(skills.keys())}"
    )
    for name, skill in skills.items():
        assert isinstance(skill, Skill)
        assert skill.name == name
        assert skill.description, f"{name!r} has empty description"
        assert skill.system_prompt, f"{name!r} has empty system_prompt"

    # The shared ``_common/style.md`` content must be appended to every
    # agent's prompt. This guards against the loader silently dropping
    # the common fragment for a renamed sentinel file.
    common_marker = "Be specific."
    for name, skill in skills.items():
        assert common_marker in skill.system_prompt, (
            f"{name!r} prompt missing _common/style.md content"
        )


def test_skills_routes_form_a_pipeline() -> None:
    """Routes must chain intake -> analyzer -> recommender -> END.

    Not strictly required by the loader, but a regression test against
    accidentally rewiring the example pipeline. Failures here flag a
    YAML edit that left the routing graph disconnected.
    """
    skills = load_all_skills(SKILLS_DIR)
    intake_routes = {r.when: r.next for r in skills["intake"].routes}
    analyzer_routes = {r.when: r.next for r in skills["analyzer"].routes}
    recommender_routes = {r.when: r.next for r in skills["recommender"].routes}

    assert intake_routes.get("success") == "analyzer"
    assert analyzer_routes.get("success") == "recommender"
    # Recommender is terminal — its success edge must end the graph.
    assert recommender_routes.get("success") in {"__end__", "END"}


@pytest.mark.parametrize("agent_name", ["intake", "analyzer", "recommender"])
def test_skills_declare_local_tools(agent_name: str) -> None:
    """Each skill declares at least one tool under ``tools.local``.

    Code-review skills don't use external MCP servers in P8 — every tool
    is local to ``examples/code_review/mcp_server.py``. A skill with an
    empty tools map is a misconfiguration, not a valid no-op.
    """
    skills = load_all_skills(SKILLS_DIR)
    skill = skills[agent_name]
    assert "local" in skill.tools, f"{agent_name!r} declares no local tools"
    assert skill.tools["local"], f"{agent_name!r} local tools list is empty"
