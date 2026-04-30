"""Parser for skill.md files."""
from __future__ import annotations
from pathlib import Path
import frontmatter
from pydantic import BaseModel, Field


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


def load_skill(path: str | Path) -> Skill:
    post = frontmatter.load(str(path))
    meta = dict(post.metadata)
    if "name" not in meta:
        raise ValueError(f"Skill at {path} missing required field: name")
    if "description" not in meta:
        raise ValueError(f"Skill at {path} missing required field: description")
    return Skill(
        name=meta["name"],
        description=meta["description"],
        model=meta.get("model"),
        temperature=meta.get("temperature"),
        tools=meta.get("tools", []),
        routes=[RouteRule(**r) for r in meta.get("routes", [])],
        system_prompt=post.content.strip(),
    )


def load_all_skills(skills_dir: str | Path) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for path in sorted(Path(skills_dir).glob("*.md")):
        skill = load_skill(path)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills
