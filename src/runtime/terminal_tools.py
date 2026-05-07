"""Generic terminal-tool registry types.

Apps register their terminal-tool rules and status vocabulary via
``OrchestratorConfig.terminal_tools`` / ``OrchestratorConfig.statuses``;
the framework reads these models without knowing app-specific tool
or status names. Cf. .planning/phases/06-generic-terminal-tool-registry/
06-CONTEXT.md (D-06-01, D-06-02, D-06-05).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TerminalToolRule(BaseModel):
    """Maps a terminal tool name to the session status it produces.

    ``tool_name`` matches both bare (``set_recommendation``) and prefixed
    (``<server>:set_recommendation``) MCP tool-call names — the framework
    does the suffix check.

    ``status`` must reference a name declared in the same
    ``OrchestratorConfig.statuses`` map; ``OrchestratorConfig``'s
    cross-field validator enforces this at config-load.

    ``extract_fields`` declares per-rule extra-metadata pulls. Each
    key is the destination field name on the session
    (``Session.extra_fields[<key>]``); each value is an ordered list
    of ``args.X`` / ``result.X`` lookup hints. The framework picks
    the first non-falsy match. Empty dict (default) means "no extra
    metadata to capture". Generalises the v1.0
    ``_extract_team(tc, team_keys)`` path; the same lookup syntax is
    preserved (D-06-02).

    ``match_args`` is an optional argument-value discriminator. When
    non-empty, the rule matches a tool call only if EVERY ``(key,
    value)`` pair in ``match_args`` matches ``tool_call.args[key]``
    exactly. Lets one tool name route to multiple statuses based on
    a discriminator argument (e.g. ``set_recommendation`` with
    ``recommendation=approve`` vs ``recommendation=request_changes``).
    Empty default = no arg dispatch; preserves the v1.0 single-rule
    shape (DECOUPLE-07 / D-08-03).
    """

    model_config = {"extra": "forbid"}

    tool_name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    extract_fields: dict[str, list[str]] = Field(default_factory=dict)
    match_args: dict[str, str] = Field(default_factory=dict)


StatusKind = Literal[
    "success",       # e.g. set_recommendation(approve) -> approved
    "failure",       # e.g. set_recommendation(request_changes) -> changes_requested
    "escalation",    # app-defined escalation terminal (e.g. <terminal_tool>)
    "needs_review",  # finalize fired with no rule match
    "pending",       # session in flight
]


class StatusDef(BaseModel):
    """Pydantic record of one app status.

    Framework reads ``terminal`` to decide finalize-vs-pending and
    ``kind`` to dispatch the needs_review fallback path / let UIs
    group statuses without owning their own taxonomy. ``color`` and
    other presentation fields stay in ``UIConfig.badges`` (D-06-05
    rejected alternative — presentation leak).
    """

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1)
    terminal: bool
    kind: StatusKind
