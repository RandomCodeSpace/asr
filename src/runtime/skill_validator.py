"""Load-time validation of skill YAML against the live MCP registry.

Catches:
  * tools.local entries that reference a non-existent (server, tool)
    pair (typically typos that would silently make the tool invisible).
  * routes that omit ``when: default`` (would cause graph hangs at
    __end__ when no signal matches).
"""
from __future__ import annotations


class SkillValidationError(RuntimeError):
    """Raised when skill YAML references a tool or route that does not
    exist or is malformed. Refuses to start the orchestrator."""


def _build_bare_to_full_map(registered_tools: set[str]) -> dict[str, list[str]]:
    """Map bare tool name → list of fully-qualified ``<server>:<tool>``."""
    bare_to_full: dict[str, list[str]] = {}
    for full in registered_tools:
        bare = full.split(":", 1)[1] if ":" in full else full
        bare_to_full.setdefault(bare, []).append(full)
    return bare_to_full


def _check_tool_ref(
    skill_name: str,
    tool_ref: str,
    registered_tools: set[str],
    bare_to_full: dict[str, list[str]],
) -> None:
    """Raise SkillValidationError if ``tool_ref`` doesn't resolve to a
    registered tool, or resolves ambiguously across multiple servers."""
    if tool_ref in registered_tools:
        return
    resolutions = bare_to_full.get(tool_ref)
    if resolutions is None:
        raise SkillValidationError(
            f"skill {skill_name!r} references tool {tool_ref!r} which "
            f"is not registered. Known tools: {sorted(registered_tools)[:10]}..."
        )
    if len(resolutions) > 1:
        raise SkillValidationError(
            f"skill {skill_name!r} uses bare tool ref {tool_ref!r} but "
            f"it is exposed by multiple servers: {sorted(resolutions)}. "
            f"Use the prefixed form to disambiguate."
        )


def validate_skill_tool_references(
    skills: dict, registered_tools: set[str],
) -> None:
    """Assert every ``tools.local`` entry in every skill resolves to a
    registered MCP tool.

    ``registered_tools`` is the set of fully-qualified ``<server>:<tool>``
    names from the MCP loader. We accept either bare or prefixed forms
    in skill YAML (the LLM-facing call uses prefixed; YAML can use
    either for ergonomics).
    """
    bare_to_full = _build_bare_to_full_map(registered_tools)
    for skill_name, skill in skills.items():
        local = (skill.get("tools") or {}).get("local") or []
        for tool_ref in local:
            _check_tool_ref(skill_name, tool_ref, registered_tools, bare_to_full)


def validate_skill_routes(skills: dict) -> None:
    """Assert every skill has a ``when: default`` route entry.

    Skipped for ``kind: supervisor`` skills — supervisors dispatch via
    ``dispatch_rules`` to subordinates and do not use the ``routes``
    table at all.
    """
    for skill_name, skill in skills.items():
        if skill.get("kind") == "supervisor":
            continue
        routes = skill.get("routes") or []
        if not any((r.get("when") == "default") for r in routes):
            raise SkillValidationError(
                f"skill {skill_name!r} has no ``when: default`` route — "
                f"agents whose signal doesn't match a rule will hang."
            )
