"""Skill loader.

A *skill* is one agent's spec: structured metadata (name, description,
temperature, tools, routes) plus a system prompt. All four pipeline
skills live in ``config/skills.yaml`` as a list under the ``skills:`` key;
the loader validates each entry through :class:`Skill` and returns them
keyed by name.
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


class _SkillsFile(BaseModel):
    """Top-level shape of ``config/skills.yaml``."""
    skills: list[Skill]


def load_skill(data: dict) -> Skill:
    """Validate one skill dict (programmatic / test callers)."""
    return Skill(**data)


def load_all_skills(skills_file: str | Path) -> dict[str, Skill]:
    path = Path(skills_file)
    if not path.exists():
        raise FileNotFoundError(f"skills file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    parsed = _SkillsFile(**raw)
    skills: dict[str, Skill] = {}
    for s in parsed.skills:
        if s.name in skills:
            raise ValueError(f"Duplicate skill name: {s.name}")
        skills[s.name] = s
    return skills
