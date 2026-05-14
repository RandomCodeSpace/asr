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

import json
import logging
import time
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool

from runtime.config import GatePolicy, GatewayConfig
from runtime.state import Session, ToolCall

# ``GateDecision`` is imported lazily inside ``_evaluate_gate`` (function
# body) to avoid a runtime cycle (policy.py imports gateway types). The
# type-only import below lets pyright resolve the string-literal return
# annotation on ``_evaluate_gate`` without forming a real cycle.
if TYPE_CHECKING:
    from runtime.policy import GateDecision  # noqa: F401
    from runtime.storage.event_log import EventLog
    from runtime.storage.session_store import SessionStore

_log = logging.getLogger("runtime.tools.gateway")

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

    Tool-name lookups try the fully-qualified name (``<server>:<tool>``,
    as registered by ``runtime.mcp_loader``) FIRST, then the bare
    suffix as a fallback. This lets app config use bare names without
    knowing the server prefix while keeping prefixed-form policy keys
    deterministically more specific. Globs in
    ``resolution_trigger_tools`` are matched against both forms for
    the same reason, prefixed first.

    The function is pure: same inputs always yield the same output and
    no argument is mutated.
    """
    if gateway_cfg is None:
        return "auto"

    bare = tool_name.split(":", 1)[1] if ":" in tool_name else None

    overrides = gateway_cfg.prod_overrides
    if overrides is not None and env and env in overrides.prod_environments:
        for pattern in overrides.resolution_trigger_tools:
            if fnmatchcase(tool_name, pattern):
                return "approve"
            if bare is not None and fnmatchcase(bare, pattern):
                return "approve"

    risk = gateway_cfg.policy.get(tool_name)
    if risk is not None:
        return _RISK_TO_ACTION[risk]
    if bare is not None:
        risk = gateway_cfg.policy.get(bare)
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


def _evaluate_gate(
    *,
    session: Session,
    tool_name: str,
    gate_policy: GatePolicy | None,
    gateway_cfg: GatewayConfig | None,
) -> "GateDecision":
    """Phase 11 (FOC-04) bridge: invoke ``should_gate`` from the wrap.

    Constructs a minimal ``ToolCall`` shape for the pure-function
    boundary, and a temporary ``OrchestratorConfig`` shim with the
    in-flight ``gate_policy`` + ``gateway`` so the pure function sees
    a single config object (its declared signature).

    When ``gate_policy`` is ``None`` -- the legacy callers that have
    not yet been threaded -- a default ``GatePolicy()`` is used so
    Phase-11 behaviour applies uniformly. The default mirrors v1.0
    HITL behaviour (``gated_risk_actions={"approve"}``), so existing
    pre-Phase-11 tests keep passing.
    """
    # Local imports (avoid cycle on policy.py importing gateway).
    # ``GateDecision`` is type-only here -- the lazy import sits in the
    # TYPE_CHECKING block at module top.
    from runtime.policy import should_gate
    from runtime.config import OrchestratorConfig

    effective_policy = gate_policy if gate_policy is not None else GatePolicy()
    # OrchestratorConfig has model_config={"extra": "forbid"} so we
    # cannot stash gateway as a top-level field. We thread gateway via
    # the cfg.gateway lookup that should_gate already performs via
    # ``getattr(cfg, "gateway", None)``. Building a transient cfg with
    # gate_policy and a stashed gateway attr is the smallest-diff
    # pathway -- avoids changing should_gate's signature.
    cfg = OrchestratorConfig(gate_policy=effective_policy)
    object.__setattr__(cfg, "gateway", gateway_cfg)

    minimal_tc = ToolCall(
        agent="",
        tool=tool_name,
        args={},
        result=None,
        ts=_now_iso(),
        risk="low",
        status="executed",
    )
    confidence = getattr(session, "turn_confidence_hint", None)
    decision: GateDecision = should_gate(
        session=session, tool_call=minimal_tc, confidence=confidence, cfg=cfg,
    )
    return decision


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
    injected_args: dict[str, str] | None = None,
    gate_policy: GatePolicy | None = None,
    event_log: "EventLog | None" = None,
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

    Phase 9 (D-09-01 / D-09-03): when ``injected_args`` is supplied, the
    gateway expands ``kwargs`` with session-derived values BEFORE
    ``effective_action`` is consulted — so the gateway's risk-rating
    sees the canonical ``environment`` (avoiding T-09-05: gateway
    misclassifies prod as auto because env was missing from the LLM
    args).
    """
    if isinstance(base_tool, _GatedToolMarker):
        return base_tool

    env = getattr(session, "environment", None)
    inner = base_tool
    inject_cfg = injected_args or {}

    # Phase 9 (D-09-01): the LLM-visible args_schema on the wrapper must
    # exclude every injected key — otherwise BaseTool's input validator
    # rejects the call when the LLM omits a "required" arg the framework
    # is about to supply. The inner tool keeps its full schema so the
    # downstream invoke still sees every kwarg.
    if inject_cfg:
        from runtime.tools.arg_injection import strip_injected_params
        _llm_visible_schema = strip_injected_params(
            inner, frozenset(inject_cfg.keys()),
        ).args_schema
    else:
        _llm_visible_schema = inner.args_schema

    # Phase 9 follow-up: compute the set of param names the inner tool
    # actually accepts so injection skips keys the target tool doesn't
    # declare. Without this filter, a config-wide ``injected_args``
    # entry like ``session_id: session.id`` is unconditionally written
    # to every tool's kwargs — tools that don't accept ``session_id``
    # then raise pydantic ``unexpected_keyword`` errors at the FastMCP
    # validation boundary. ``accepted_params_for_tool`` handles both
    # pydantic-model and JSON-Schema-dict ``args_schema`` shapes.
    from runtime.tools.arg_injection import accepted_params_for_tool
    _accepted_params: frozenset[str] | None = accepted_params_for_tool(inner)

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

    # Tool-naming regex differs across LLM providers — Ollama allows
    # ``[a-zA-Z0-9_.\-]{1,256}``, OpenAI is stricter at
    # ``^[a-zA-Z0-9_-]+$`` (no dots). The framework's internal naming
    # uses ``<server>:<tool>`` for PVC-08 prefixed-form policy lookups,
    # but the LLM only sees the *wrapper*'s ``.name``. Use ``__``
    # (double underscore) as the LLM-visible separator: it satisfies
    # both providers' regexes and is unambiguous (no real tool name
    # contains a double underscore). ``inner.name`` keeps the colon
    # form so ``effective_action`` / ``should_gate`` policy lookups
    # stay PVC-08-compliant.
    _llm_visible_name = inner.name.replace(":", "__")

    # M3 (per-step telemetry): emit `tool_invoked` and `gate_fired` events
    # through the optional EventLog. Telemetry failures never break a
    # tool call — they are logged at DEBUG and dropped.

    def _cap_args(args_dict: Any) -> Any:
        """Cap args payload at 4 KB of JSON; oversized payloads become
        a small ``{"_truncated": True, "preview": ...}`` marker."""
        try:
            blob = json.dumps(args_dict, default=str)
        except (TypeError, ValueError):
            return {"_unencodable": True}
        if len(blob) <= 4096:
            return args_dict
        return {"_truncated": True, "preview": blob[:4096]}

    def _emit_invoked(
        *,
        status: str,
        risk: str,
        args_dict: Any,
        result: Any,
        latency_ms: float,
    ) -> None:
        if event_log is None:
            return
        try:
            event_log.record(
                session.id,
                "tool_invoked",
                tool=inner.name,
                agent=agent_name,
                args=_cap_args(args_dict),
                result_kind=type(result).__name__,
                latency_ms=round(latency_ms, 3),
                risk=risk,
                status=status,
            )
        except Exception:  # noqa: BLE001 — telemetry must not break a tool call
            _log.debug(
                "event_log.record(tool_invoked) failed", exc_info=True,
            )

    def _emit_gate(*, reason: str) -> None:
        if event_log is None:
            return
        try:
            event_log.record(
                session.id,
                "gate_fired",
                tool=inner.name,
                agent=agent_name,
                reason=reason,
            )
        except Exception:  # noqa: BLE001
            _log.debug(
                "event_log.record(gate_fired) failed", exc_info=True,
            )

    class _GatedTool(_GatedToolMarker):
        name: str = _llm_visible_name
        description: str = inner.description
        # The wrapper does its own arg coercion via the inner tool's schema,
        # so no need to copy it here. Keep ``args_schema`` aligned with the
        # LLM-visible (post-strip) schema so BaseTool's input validator
        # accepts the post-strip kwargs the LLM emits. Phase 9 strips
        # injected keys here; pre-Phase-9 callers see the full schema.
        args_schema: Any = _llm_visible_schema  # type: ignore[assignment]

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            # M3 (per-step telemetry): start the latency clock for every
            # tool invocation. _emit_invoked computes ``(now - t0) * 1000``.
            t0 = time.monotonic()
            # Phase 9 (D-09-01 / T-09-05): inject session-derived args
            # BEFORE the gateway risk lookup so risk-rating sees the
            # post-injection environment value. Pure no-op when
            # ``injected_args`` is empty.
            if inject_cfg:
                from runtime.tools.arg_injection import inject_injected_args
                kwargs = inject_injected_args(
                    kwargs,
                    session=session,
                    injected_args_cfg=inject_cfg,
                    tool_name=inner.name,
                    accepted_params=_accepted_params or None,
                )
            # Phase 11 (FOC-04): pure-policy gating boundary. Call
            # should_gate to decide whether to pause for HITL approval;
            # also call effective_action so the notify-audit branch
            # below still fires for medium-risk tools that should NOT
            # gate but should record an audit row.
            action = effective_action(
                inner.name, env=env, gateway_cfg=gateway_cfg,
            )
            decision = _evaluate_gate(
                session=session,
                tool_name=inner.name,
                gate_policy=gate_policy,
                gateway_cfg=gateway_cfg,
            )
            if decision.gate:
                from langgraph.types import interrupt

                # M3: emit gate_fired BEFORE the interrupt fires so the
                # event ordering in the log matches the runtime causality
                # (gate decision precedes tool execution / pause).
                _emit_gate(reason=decision.reason)

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
                        # Persist the status transition. Without this,
                        # the DB row stays at ``pending_approval`` and
                        # the UI keeps offering the buttons forever.
                        if store is not None:
                            store.save(session)
                    rejected_result = {"rejected": True, "rationale": rationale}
                    _emit_invoked(
                        status="rejected", risk="high",
                        args_dict=pending_args, result=rejected_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return rejected_result
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
                        if store is not None:
                            store.save(session)
                    timeout_result = {"timeout": True, "rationale": rationale}
                    _emit_invoked(
                        status="timeout", risk="high",
                        args_dict=pending_args, result=timeout_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return timeout_result
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
                    if store is not None:
                        store.save(session)
                _emit_invoked(
                    status="approved", risk="high",
                    args_dict=pending_args, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return result

            # auto / notify both run the tool now.
            result = _sync_invoke_inner(kwargs if kwargs else args[0] if args else {})

            _args_dict = dict(kwargs) if kwargs else {"args": list(args)}
            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=_args_dict,
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
                _emit_invoked(
                    status="executed_with_notify", risk="medium",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            else:
                _emit_invoked(
                    status="executed", risk="low",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return result

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            # M3: start latency clock; mirror of sync ``_run``.
            t0 = time.monotonic()
            # Phase 9 (D-09-01 / T-09-05): inject session-derived args
            # BEFORE the gateway risk lookup. Mirror of the sync ``_run``.
            if inject_cfg:
                from runtime.tools.arg_injection import inject_injected_args
                kwargs = inject_injected_args(
                    kwargs,
                    session=session,
                    injected_args_cfg=inject_cfg,
                    tool_name=inner.name,
                    accepted_params=_accepted_params or None,
                )
            # Phase 11 (FOC-04): pure-policy gating boundary. Mirror of
            # the sync ``_run`` -- consult should_gate via
            # ``_evaluate_gate``; still call ``effective_action`` to
            # keep the notify-audit branch for medium-risk tools.
            action = effective_action(
                inner.name, env=env, gateway_cfg=gateway_cfg,
            )
            decision = _evaluate_gate(
                session=session,
                tool_name=inner.name,
                gate_policy=gate_policy,
                gateway_cfg=gateway_cfg,
            )
            if decision.gate:
                from langgraph.types import interrupt

                # M3: emit gate_fired BEFORE interrupt.
                _emit_gate(reason=decision.reason)

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
                        # Persist the status transition (mirror of the
                        # sync path) so the DB row reflects the actual
                        # outcome instead of staying at pending_approval.
                        if store is not None:
                            store.save(session)
                    rejected_result = {"rejected": True, "rationale": rationale}
                    _emit_invoked(
                        status="rejected", risk="high",
                        args_dict=pending_args, result=rejected_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return rejected_result
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
                        if store is not None:
                            store.save(session)
                    timeout_result = {"timeout": True, "rationale": rationale}
                    _emit_invoked(
                        status="timeout", risk="high",
                        args_dict=pending_args, result=timeout_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return timeout_result
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
                    if store is not None:
                        store.save(session)
                _emit_invoked(
                    status="approved", risk="high",
                    args_dict=pending_args, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return result

            result = await inner.ainvoke(kwargs if kwargs else args[0] if args else {})

            _args_dict = dict(kwargs) if kwargs else {"args": list(args)}
            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=_args_dict,
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
                _emit_invoked(
                    status="executed_with_notify", risk="medium",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            else:
                _emit_invoked(
                    status="executed", risk="low",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return result

    return _GatedTool()
