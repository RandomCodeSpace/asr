"""Resolution agent helpers — playbook → tool-call translation.

The resolution agent matches the L7 PlaybookStore for the incident's
signals and produces a list of suggested tool calls. The framework's
risk-rated gateway (``runtime.tools.gateway``) decides whether each
call runs auto / notify-soft / require-approval based on its policy
and the prod-environment override.

This module is a thin, deterministic translator. The agent's prompt
(``skills/resolution/system.md``) describes the reasoning; the helpers
here are pure functions exercised by unit tests.

Surface:

* :class:`ToolCallSpec` — typed dict for a single suggested tool call.
* :func:`playbook_to_tool_calls` — given a playbook dict (the YAML
  shape :class:`PlaybookStore` already validates), return the list of
  :class:`ToolCallSpec` entries the agent should issue.
* :func:`top_playbook` — given the full
  ``IncidentState.memory.l7_playbooks`` suggestion list, return the
  highest-scoring playbook id (None if empty).
"""
from __future__ import annotations

from typing import Any, TypedDict

from examples.incident_management.asr.memory_state import L7PlaybookSuggestion


class ToolCallSpec(TypedDict):
    """A single suggested tool call sourced from a playbook step."""

    tool: str
    args: dict[str, Any]
    requires_approval: bool


def playbook_to_tool_calls(playbook: dict) -> list[ToolCallSpec]:
    """Translate a playbook dict's ``remediation`` block into tool calls.

    Each ``remediation`` entry is expected to have:

    - ``tool`` (str, required) — the tool name as known to the gateway.
    - ``args`` (dict, optional) — keyword args; defaults to ``{}``.

    The playbook's top-level ``required_approval`` flag (see
    ``examples/incident_management/asr/seeds/playbooks/*.yaml``) is
    propagated to every emitted spec — it represents the playbook
    author's stated risk posture. The gateway's risk policy still has
    final say at execution time; this flag is purely advisory metadata
    for the agent's prompt and the UI.

    Returns an empty list when ``playbook`` is None / lacks
    ``remediation`` / has malformed entries — the caller can branch on
    "no suggestion" without a try/except.
    """
    if not playbook or not isinstance(playbook, dict):
        return []
    remediation = playbook.get("remediation") or []
    if not isinstance(remediation, list):
        return []

    requires_approval = bool(playbook.get("required_approval"))
    out: list[ToolCallSpec] = []
    for entry in remediation:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if not tool or not isinstance(tool, str):
            continue
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCallSpec(
            tool=tool,
            args=dict(args),
            requires_approval=requires_approval,
        ))
    return out


def top_playbook(
    suggestions: list[L7PlaybookSuggestion],
) -> str | None:
    """Return the playbook_id of the highest-scoring suggestion, or None.

    ``suggestions`` is the list ``PlaybookStore.match`` already sorts
    by descending score; we still pick by ``max`` so callers that pass
    a re-ordered list (e.g. tests) still get a deterministic answer.
    """
    if not suggestions:
        return None
    best = max(suggestions, key=lambda s: (s.score, -hash(s.playbook_id)))
    return best.playbook_id


__all__ = [
    "ToolCallSpec",
    "playbook_to_tool_calls",
    "top_playbook",
]
