"""Generic session model — the framework's unit of work.

A ``Session`` is the in-progress (or archived) record of one agent run.
Applications extend this via subclassing::

    class IncidentState(Session):
        environment: str
        reporter: Reporter
        ...

``Session`` deliberately contains *no* domain-specific fields. Adding one
here is a framework regression — domain fields belong in the example
app's ``state.py``.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


# Per-call audit metadata for the risk-rated tool gateway.
ToolRisk = Literal["low", "medium", "high"]
ToolStatus = Literal[
    "executed",                 # auto / legacy default
    "executed_with_notify",     # medium-risk: ran + soft-notify
    "pending_approval",         # high-risk: graph paused on interrupt()
    "approved",                 # high-risk: operator approved, then ran
    "rejected",                 # high-risk: operator rejected, did not run
    "timeout",                  # high-risk: approval window expired
]


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str
    # Audit fields for the risk-rated gateway. All optional and
    # default-permissive so legacy rows in the JSON column hydrate with
    # ``status="executed"`` and the rest of the fields ``None`` —
    # preserving back-compat with older sessions.
    risk: ToolRisk | None = None
    status: ToolStatus = "executed"
    approver: str | None = None
    approved_at: str | None = None
    approval_rationale: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_rationale: str | None = None
    signal: str | None = None


class Session(BaseModel):
    """Framework base session. Lifecycle + telemetry fields only.

    Applications subclass this and add domain fields. The framework only
    reads/writes the fields declared here.
    """

    id: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    findings: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
    # Dedup linkage. NULL by default; set when this session is
    # confirmed as a duplicate of a prior closed session. The value is
    # the prior session's id; the link is non-destructive — both
    # sessions remain queryable. See ``runtime.dedup``.
    parent_session_id: str | None = None
    # Stage-2 LLM rationale for the dedup decision. Stored on the
    # session row so the UI can render "why was this marked duplicate?"
    # without needing a separate join.
    dedup_rationale: str | None = None
    # Bag for app-specific session data the framework doesn't touch.
    # Apps that previously subclassed Session to add typed fields now
    # store them here. The storage layer round-trips this via the
    # matching ``IncidentRow.extra_fields`` JSON column.
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    # Optimistic concurrency token. Incremented on every successful
    # ``SessionStore.save``; reads observe the value at load time. Saves
    # with a stale version raise ``StaleVersionError`` so the caller can
    # reload + retry.
    version: int = 1
    # Phase 11 (FOC-04): transient per-turn confidence hint set by the
    # agent runner (graph.py / responsive.py) AFTER each
    # _harvest_tool_calls_and_patches call so the gateway's should_gate
    # boundary can apply low_confidence gating using whatever
    # confidence the agent has emitted so far. Reset to ``None`` at
    # turn start; never persisted (``Field(exclude=True)``). The
    # framework treats ``None`` as "no signal yet" and does NOT fire a
    # low_confidence gate -- this avoids a false-positive gate on the
    # very first tool call of a turn before any envelope/tool-arg
    # carrying confidence has surfaced.
    turn_confidence_hint: float | None = Field(default=None, exclude=True)

    # ------------------------------------------------------------------
    # App-overridable agent-input formatter hook.
    # ------------------------------------------------------------------
    def to_agent_input(self) -> str:
        """Return the human-message preamble each agent receives.

        Apps subclass ``Session`` and override this to surface the
        domain shape (``Incident X / Environment Y / Query Z`` for the
        incident-management app, ``PR title / repo / diff stats`` for
        code review, etc.). The framework default keeps the prompt
        framework-agnostic — id + status only — so that any app that
        has not overridden the hook still gets a usable preamble.

        Findings, prior agent output, and operator-supplied user input
        are appended at the end so the surface stays useful even when
        the subclass keeps the default prefix.
        """
        base = (
            f"Session {self.id}\n"
            f"Status: {self.status}\n"
        )
        for agent_key, finding in self.findings.items():
            base += f"Findings ({agent_key}): {finding}\n"
        if self.user_inputs:
            bullets = "\n".join(f"- {ui}" for ui in self.user_inputs)
            base += (
                "\nUser-provided context (appended via intervention):\n"
                f"{bullets}\n"
            )
        return base

    # ------------------------------------------------------------------
    # App-overridable session id minting hook.
    # ------------------------------------------------------------------
    @classmethod
    def id_format(cls, *, seq: int, prefix: str = "SES") -> str:
        """Return the canonical session id for the given sequence number.

        ``prefix`` is supplied by ``SessionStore._next_id`` from
        ``FrameworkAppConfig.session_id_prefix`` so each app picks its
        own namespace via plain config (e.g. ``INC`` for incident
        management, ``REVIEW`` for code review, ``HR`` for HR cases,
        ...). Apps with truly bespoke id shapes can still override this
        classmethod on their ``Session`` subclass and ignore ``prefix``.

        ``seq`` is the per-day monotonic sequence supplied by
        ``SessionStore._next_id``; it lets the default format produce
        the expected zero-padded suffix without each subclass
        re-implementing the SQL scan.
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{prefix}-{today}-{seq:03d}"
