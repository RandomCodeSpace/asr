"""Risk-rated tool gateway: pure resolver + ``BaseTool`` HITL wrapper.

The gateway sits between the ReAct agent and each tool the orchestrator
configures. It enforces the *hybrid* HITL policy resolved by
``effective_action``:

  ``auto``    -> call the underlying tool directly (no plumbing)
  ``notify``  -> call the tool, then persist a soft-notify audit entry
  ``approve`` -> raise ``langgraph.types.interrupt(...)`` BEFORE calling
                 the tool; on resume re-invoke

The resolver is a plain function with no I/O so it can be unit-tested
exhaustively without spinning up Pydantic Sessions, MCP servers, or a
LangGraph runtime. The wrapper is a closure factory deliberately built
inside ``make_agent_node`` so the closure captures the live ``Session``
per agent invocation (mitigation R2 in the Phase-4 plan).
"""
from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool

from runtime.config import GatewayConfig
from runtime.state import Session, ToolCall

if TYPE_CHECKING:
    from runtime.storage.session_store import SessionStore

GatewayAction = Literal["auto", "notify", "approve"]

_RISK_TO_ACTION: dict[str, GatewayAction] = {
    "low": "auto",
    "medium": "notify",
    "high": "approve",
}

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def effective_action(
    tool_name: str,
    *,
    env: str | None,
    gateway_cfg: GatewayConfig | None,
) -> GatewayAction:
    """Resolve the effective gateway action for a tool invocation.

    Order of evaluation (the prod-override predicate runs FIRST so it can
    only TIGHTEN the action — never relax it):

      1. ``gateway_cfg is None`` -> ``"auto"`` (gateway disabled).
      2. Prod override: if ``cfg.prod_overrides`` is configured AND
         ``env`` is in ``prod_environments`` AND ``tool_name`` matches
         one of the ``resolution_trigger_tools`` globs -> ``"approve"``.
      3. Risk-tier lookup: ``cfg.policy.get(tool_name)`` mapped via
         ``low->auto``, ``medium->notify``, ``high->approve``.
      4. No policy entry -> ``"auto"`` (safe default).

    Tool-name lookups try both the fully-qualified name (``<server>:<tool>``,
    as registered by ``runtime.mcp_loader``) AND the bare original name
    (``<tool>``). This lets app config use the bare names that match the
    MCP tool's source declaration without having to know the server
    prefix. Globs in ``resolution_trigger_tools`` are matched against
    both forms for the same reason.

    The function is pure: same inputs always yield the same output and
    no argument is mutated.
    """
    if gateway_cfg is None:
        return "auto"

    # Build the lookup-name list: prefixed first (most specific), then
    # the bare suffix (so config can be server-agnostic).
    candidates = [tool_name]
    if ":" in tool_name:
        candidates.append(tool_name.split(":", 1)[1])

    overrides = gateway_cfg.prod_overrides
    if overrides is not None and env:
        if env in overrides.prod_environments:
            for pattern in overrides.resolution_trigger_tools:
                if any(fnmatchcase(c, pattern) for c in candidates):
                    return "approve"

    for c in candidates:
        risk = gateway_cfg.policy.get(c)
        if risk is not None:
            return _RISK_TO_ACTION[risk]
    return "auto"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_UTC_TS_FMT)


def _find_pending_index(
    tool_calls: list,
    tool_name: str,
    ts: str,
) -> int | None:
    """Locate the index of the ``pending_approval`` ToolCall row that
    matches ``tool_name`` and ``ts``.

    Used by the wrap_tool resume path to update the in-place audit row
    rather than appending a duplicate. The watchdog may have replaced
    the row with a ``timeout`` entry while the graph was paused — in
    that case we return ``None`` and the resume path leaves the audit
    list unchanged (the watchdog already wrote the canonical record).

    Searches from the end of the list because the pending row is
    almost always the most recent ToolCall.
    """
    for idx in range(len(tool_calls) - 1, -1, -1):
        tc = tool_calls[idx]
        if (getattr(tc, "tool", None) == tool_name
                and getattr(tc, "ts", None) == ts
                and getattr(tc, "status", None) == "pending_approval"):
            return idx
    return None


def _find_existing_pending_index(
    tool_calls: list,
    tool_name: str,
) -> int | None:
    """Find the most recent ``pending_approval`` row for ``tool_name``.

    LangGraph's interrupt/resume model re-runs the gated node from the
    top after ``Command(resume=...)``; we re-use the existing pending
    row rather than appending a duplicate every time the closure
    re-enters the approve branch.
    """
    for idx in range(len(tool_calls) - 1, -1, -1):
        tc = tool_calls[idx]
        if (getattr(tc, "tool", None) == tool_name
                and getattr(tc, "status", None) == "pending_approval"):
            return idx
    return None


class _GatedToolMarker(BaseTool):
    """Marker base class so ``isinstance(t, _GatedToolMarker)`` identifies
    a tool that has already been wrapped by :func:`wrap_tool`. Used to
    short-circuit ``wrap_tool(wrap_tool(t))`` and avoid wrapper recursion.

    Not instantiated directly — every ``_GatedTool`` defined inside
    :func:`wrap_tool` inherits from this.
    """

    name: str = "_gated_marker"
    description: str = "internal — never invoked"

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("marker base — _GatedTool overrides this")


def wrap_tool(
    base_tool: BaseTool,
    *,
    session: Session,
    gateway_cfg: GatewayConfig | None,
    agent_name: str = "",
    store: "SessionStore | None" = None,
) -> BaseTool:
    """Wrap ``base_tool`` so every invocation passes through the gateway.

    The factory closes over ``session`` and ``gateway_cfg`` so the live
    audit log (``session.tool_calls``) is the same instance the rest of
    the orchestrator reads — no detour through a separate audit table.

    Returned object is a ``BaseTool`` subclass instance whose ``name``
    and ``description`` mirror the underlying tool, so LangGraph's ReAct
    prompt builder still sees the right tool surface.

    Idempotent: wrapping an already-gated tool returns it unchanged so a
    second ``wrap_tool(wrap_tool(t))`` does not nest wrappers (which would
    cause unbounded recursion when ``_run`` calls ``inner.invoke`` and
    that dispatches back into another ``_GatedTool._run``).
    """
    if isinstance(base_tool, _GatedToolMarker):
        return base_tool

    env = getattr(session, "environment", None)
    inner = base_tool

    def _sync_invoke_inner(payload: Any) -> Any:
        """Sync-invoke the inner tool, translating BaseTool's
        default-``_run`` ``NotImplementedError`` into a clearer message
        for native-async-only tools. Without this, callers see a vague
        ``NotImplementedError`` from langchain core with no hint that
        the right path is ``ainvoke``."""
        try:
            return inner.invoke(payload)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Tool {inner.name!r} appears to be async-only "
                f"(``_run`` not implemented). Use ``ainvoke`` / ``_arun`` "
                f"for this tool instead of the sync invoke path."
            ) from exc

    class _GatedTool(_GatedToolMarker):
        name: str = inner.name
        description: str = inner.description
        # The wrapper does its own arg coercion via the inner tool's schema,
        # so no need to copy it here. Keep ``args_schema`` aligned.
        args_schema: Any = inner.args_schema  # type: ignore[assignment]

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            action = effective_action(inner.name, env=env, gateway_cfg=gateway_cfg)
            if action == "approve":
                from langgraph.types import interrupt

                # Persist a ``pending_approval`` ToolCall row BEFORE
                # raising GraphInterrupt so the approval-timeout watchdog
                # has a record to scan. ``ts`` is the moment the human
                # approval window opened. Stored args mirror the post-
                # decision rows so the audit history reads consistently.
                #
                # On resume, LangGraph re-enters this node and runs us
                # again from the top — so we must re-use the existing
                # pending row instead of appending a duplicate. The most
                # recent ``pending_approval`` row for this tool wins.
                pending_args = dict(kwargs) if kwargs else {"args": list(args)}
                existing_idx = _find_existing_pending_index(
                    session.tool_calls, inner.name,
                )
                if existing_idx is not None:
                    pending_ts = session.tool_calls[existing_idx].ts
                else:
                    pending_ts = _now_iso()
                    session.tool_calls.append(
                        ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result=None,
                            ts=pending_ts,
                            risk="high",
                            status="pending_approval",
                        )
                    )
                    # CRITICAL: persist the pending_approval row BEFORE
                    # raising interrupt() so the approval-timeout
                    # watchdog (which reads from the DB) and the
                    # /approvals UI can see the pending state. Without
                    # this save the in-memory mutation is invisible to
                    # any out-of-process observer.
                    if store is not None:
                        store.save(session)
                payload = {
                    "kind": "tool_approval",
                    "tool": inner.name,
                    "args": kwargs or args,
                    "tool_call_id": kwargs.get("tool_call_id"),
                }
                # First execution: raises GraphInterrupt, checkpointer pauses.
                # Resume: returns whatever Command(resume=...) supplied.
                decision = interrupt(payload)
                # Decision payload may be a string ("approve" / "reject" /
                # "timeout") or a dict {decision, approver, rationale}.
                if isinstance(decision, dict):
                    verdict = decision.get("decision", "approve")
                    approver = decision.get("approver")
                    rationale = decision.get("rationale")
                else:
                    verdict = decision or "approve"
                    approver = None
                    rationale = None
                # Update the pending_approval row in place rather than
                # appending a second audit entry. The watchdog and the
                # /approvals UI both reason about a single audit row per
                # high-risk call.
                pending_idx = _find_pending_index(
                    session.tool_calls, inner.name, pending_ts,
                )
                verdict_str = str(verdict).lower()
                if verdict_str == "reject":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"rejected": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="rejected",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    return {"rejected": True, "rationale": rationale}
                if verdict_str == "timeout":
                    # The approval window expired. Do NOT run the tool;
                    # mark the audit row ``status="timeout"`` so
                    # downstream consumers (UI, retraining) can
                    # distinguish operator-initiated rejections from
                    # automatic timeouts.
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"timeout": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="timeout",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    return {"timeout": True, "rationale": rationale}
                # Approved -> run the tool, then update the audit row.
                result = _sync_invoke_inner(kwargs if kwargs else args[0] if args else {})
                if pending_idx is not None:
                    session.tool_calls[pending_idx] = ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=pending_args,
                        result=result,
                        ts=pending_ts,
                        risk="high",
                        status="approved",
                        approver=approver,
                        approved_at=_now_iso(),
                        approval_rationale=rationale,
                    )
                return result

            # auto / notify both run the tool now.
            result = _sync_invoke_inner(kwargs if kwargs else args[0] if args else {})

            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=dict(kwargs) if kwargs else {"args": list(args)},
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
            return result

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            action = effective_action(inner.name, env=env, gateway_cfg=gateway_cfg)
            if action == "approve":
                from langgraph.types import interrupt

                # Persist a ``pending_approval`` audit row BEFORE the
                # GraphInterrupt fires so the watchdog can spot stale
                # approvals. See the sync ``_run`` mirror for details.
                pending_args = dict(kwargs) if kwargs else {"args": list(args)}
                existing_idx = _find_existing_pending_index(
                    session.tool_calls, inner.name,
                )
                if existing_idx is not None:
                    pending_ts = session.tool_calls[existing_idx].ts
                else:
                    pending_ts = _now_iso()
                    session.tool_calls.append(
                        ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result=None,
                            ts=pending_ts,
                            risk="high",
                            status="pending_approval",
                        )
                    )
                    # CRITICAL: persist the pending_approval row BEFORE
                    # raising interrupt() so the approval-timeout
                    # watchdog (which reads from the DB) and the
                    # /approvals UI can see the pending state.
                    if store is not None:
                        store.save(session)
                payload = {
                    "kind": "tool_approval",
                    "tool": inner.name,
                    "args": kwargs or args,
                    "tool_call_id": kwargs.get("tool_call_id"),
                }
                decision = interrupt(payload)
                if isinstance(decision, dict):
                    verdict = decision.get("decision", "approve")
                    approver = decision.get("approver")
                    rationale = decision.get("rationale")
                else:
                    verdict = decision or "approve"
                    approver = None
                    rationale = None
                pending_idx = _find_pending_index(
                    session.tool_calls, inner.name, pending_ts,
                )
                verdict_str = str(verdict).lower()
                if verdict_str == "reject":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"rejected": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="rejected",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    return {"rejected": True, "rationale": rationale}
                if verdict_str == "timeout":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"timeout": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="timeout",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    return {"timeout": True, "rationale": rationale}
                result = await inner.ainvoke(kwargs if kwargs else args[0] if args else {})
                if pending_idx is not None:
                    session.tool_calls[pending_idx] = ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=pending_args,
                        result=result,
                        ts=pending_ts,
                        risk="high",
                        status="approved",
                        approver=approver,
                        approved_at=_now_iso(),
                        approval_rationale=rationale,
                    )
                return result

            result = await inner.ainvoke(kwargs if kwargs else args[0] if args else {})

            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=dict(kwargs) if kwargs else {"args": list(args)},
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
            return result

    return _GatedTool()
