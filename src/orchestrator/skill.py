"""Skill loader.

Each agent lives in its own subdirectory under ``config/skills/``::

    config/skills/
      intake/
        config.yaml         # name, description, tools, routes, temperature, model
        system.md           # the system prompt (markdown body — format is free)
        confidence.md       # OPTIONAL extra prompt sections; every *.md in the
        output.md           # directory is concatenated in alphabetical order

Adding a directory adds an agent. Splitting prompts across multiple ``.md``
files is purely organisational — they are joined with double newlines into
one ``system_prompt`` string. Structured config is validated through the
:class:`Skill` / :class:`RouteRule` Pydantic models; markdown content is
loaded verbatim.
"""
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator


class RouteRule(BaseModel):
    when: str
    next: str


class Skill(BaseModel):
    name: str
    description: str
    model: str | None = None
    temperature: float | None = None
    tools: list[str] = Field(default_factory=list)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        return v.strip()


def load_skill(agent_dir: str | Path) -> Skill:
    """Load one agent from its directory.

    Reads ``config.yaml`` for structured metadata and concatenates every
    ``*.md`` file (sorted alphabetically) into ``system_prompt``.
    """
    base = Path(agent_dir)
    config_path = base / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.yaml in skill dir: {base}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    md_files = sorted(base.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"no .md prompt files in skill dir: {base}")
    cfg["system_prompt"] = "\n\n".join(p.read_text().strip() for p in md_files)
    return Skill(**cfg)


def load_all_skills(skills_dir: str | Path) -> dict[str, Skill]:
    base = Path(skills_dir)
    if not base.exists():
        raise FileNotFoundError(f"skills dir not found: {base}")
    skills: dict[str, Skill] = {}
    for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if not (agent_dir / "config.yaml").exists():
            continue
        skill = load_skill(agent_dir)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills
