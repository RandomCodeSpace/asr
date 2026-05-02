"""Skill loader.

Each agent lives in its own subdirectory under ``config/skills/``::

    config/skills/
      _common/                # OPTIONAL: prompt fragments shared by all agents
        confidence.md         # appended to every agent's system_prompt, in
        output.md             # alphabetical order, joined with blank lines
      intake/
        config.yaml           # description, tools, routes, model
        system.md             # the agent's specialty (markdown body — format is free)
        guidelines.md         # OPTIONAL extra fragments; every *.md in the
        ...                   # directory is concatenated in alphabetical order

Adding a directory under ``config/skills/`` (with a ``config.yaml`` and at
least one ``.md`` file) adds an agent. Directories whose name starts with
``_`` are reserved for shared content and never become agents.

The final ``system_prompt`` for each agent is::

    <concatenated *.md from agent_dir>
    \\n\\n
    <concatenated *.md from _common/, if present>

Structured config is validated through the :class:`Skill` /
:class:`RouteRule` Pydantic models; markdown content is loaded verbatim.
"""
from __future__ import annotations
import re
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator


_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _validate_agent_name(name: str, *, source: str) -> None:
    """Reject directory names that can't safely become agent identifiers.

    Agent names appear in route targets (``next: triage``), LangGraph node
    IDs, and INC tags. Restricting them to ``[a-z][a-z0-9_]{0,63}`` keeps
    them grep-able, shell-safe, and free of case-sensitivity surprises
    across filesystems.
    """
    if not _AGENT_NAME_RE.match(name):
        raise ValueError(
            f"invalid agent name {name!r} (from {source}): must match "
            f"[a-z][a-z0-9_]{{0,63}} — lowercase, start with a letter, "
            f"alphanumerics + underscore only, max 64 chars"
        )


class RouteRule(BaseModel):
    when: str
    next: str
    gate: str | None = None


class Skill(BaseModel):
    name: str
    description: str
    model: str | None = None
    tools: dict[str, list[str]] = Field(default_factory=dict)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for server, names in v.items():
            if not names:
                raise ValueError(
                    f"empty tool list for server {server!r}; "
                    f"remove the key or use ['*']"
                )
            if "*" in names and len(names) != 1:
                raise ValueError(
                    f"'*' must be the sole entry for server "
                    f"{server!r}; got {names!r}"
                )
        return v

    @field_validator("system_prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        return v.strip()


def _concat_md(md_files: list[Path]) -> str:
    return "\n\n".join(p.read_text().strip() for p in md_files)


def _load_common_prompt(skills_dir: Path) -> str:
    """Read every ``*.md`` under ``<skills_dir>/_common/`` (if present) and
    return them concatenated in alphabetical order.

    Returns the empty string when no ``_common/`` directory exists or it
    contains no markdown files — ``load_all_skills`` then leaves agent
    prompts unchanged.
    """
    common_dir = skills_dir / "_common"
    if not common_dir.is_dir():
        return ""
    return _concat_md(sorted(common_dir.glob("*.md")))


def load_skill(agent_dir: str | Path, *, common: str = "") -> Skill:
    """Load one agent from its directory.

    The directory name is the agent's ``name`` (single source of truth).
    Reads ``config.yaml`` for the rest of the structured metadata and
    concatenates every ``*.md`` file (sorted alphabetically) into
    ``system_prompt``. If ``common`` is non-empty, it is appended after
    the agent's own prompt so shared sections (Confidence, Output) only
    need to be authored once.

    Raises ``ValueError`` if ``config.yaml`` declares its own ``name`` —
    the duplication used to drift silently when a directory was renamed
    without updating the config.
    """
    base = Path(agent_dir)
    config_path = base / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.yaml in skill dir: {base}")
    _validate_agent_name(base.name, source=f"directory {base}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    if "name" in cfg:
        raise ValueError(
            f"config.yaml at {config_path} must not declare 'name' — the "
            f"agent name is taken from the directory ({base.name!r})"
        )
    cfg["name"] = base.name
    md_files = sorted(base.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"no .md prompt files in skill dir: {base}")
    agent_prompt = _concat_md(md_files)
    cfg["system_prompt"] = (
        f"{agent_prompt}\n\n{common}".strip() if common else agent_prompt
    )
    return Skill(**cfg)


def load_all_skills(skills_dir: str | Path) -> dict[str, Skill]:
    base = Path(skills_dir)
    if not base.exists():
        raise FileNotFoundError(f"skills dir not found: {base}")
    common = _load_common_prompt(base)
    skills: dict[str, Skill] = {}
    for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # Reserve the leading underscore for shared content (_common,
        # _drafts, etc.) — never treat those as agents.
        if agent_dir.name.startswith("_"):
            continue
        if not (agent_dir / "config.yaml").exists():
            continue
        skill = load_skill(agent_dir, common=common)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills
