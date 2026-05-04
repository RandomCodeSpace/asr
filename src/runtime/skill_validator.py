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
    bare_to_full: dict[str, list[str]] = {}
    for full in registered_tools:
        if ":" in full:
            bare = full.split(":", 1)[1]
            bare_to_full.setdefault(bare, []).append(full)
        else:
            bare_to_full.setdefault(full, []).append(full)

    for skill_name, skill in skills.items():
        local = (skill.get("tools") or {}).get("local") or []
        for tool_ref in local:
            if tool_ref in registered_tools:
                continue
            if tool_ref in bare_to_full:
                continue
            raise SkillValidationError(
                f"skill {skill_name!r} references tool {tool_ref!r} which "
                f"is not registered. Known tools: {sorted(registered_tools)[:10]}..."
            )


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
