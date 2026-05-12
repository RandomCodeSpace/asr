"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict, Callable, Awaitable
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END

from runtime.state import Session, ToolCall, AgentRun, TokenUsage, _UTC_TS_FMT
from runtime.skill import Skill
from runtime.config import (
    AppConfig,
    FrameworkAppConfig,
    GatePolicy,
    GatewayConfig,
    resolve_framework_app_config,
)
from runtime.llm import get_llm
from runtime.mcp_loader import ToolRegistry
from runtime.storage.session_store import SessionStore
from runtime.tools.gateway import wrap_tool

if TYPE_CHECKING:
    from runtime.storage.event_log import EventLog
# Phase 11 (FOC-04 / D-11-04): GraphInterrupt is the LangGraph
# pending-approval pause signal. It is NOT an error and must NOT route
# through _handle_agent_failure -- the orchestrator's interrupt-aware
# bridge handles the resume protocol via the checkpointer.
from langgraph.errors import GraphInterrupt
from runtime.agents.turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
    reconcile_confidence,
)

logger = logging.getLogger(__name__)


_CONFIDENCE_LABELS: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


def _coerce_confidence(raw) -> float | None:
    """Coerce a raw confidence value emitted by an LLM to a clamped float in
    [0.0, 1.0], or None when the value cannot be interpreted.

    Order matters: bool **must** be rejected before float because Python treats
    ``True``/``False`` as instances of ``int`` (and therefore acceptable to
    ``float()``). Strings are matched against the canonical {high, medium, low}
    labels case-insensitively; any other string emits a warning and yields
    None. Floats outside [0.0, 1.0] are clamped (with a warning) rather than
    dropped — clamping is more forgiving when an LLM is on the right track but
    miscalibrated.
    """
    if isinstance(raw, bool):
        logger.warning("confidence value is bool (%r); rejecting", raw)
        return None
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _CONFIDENCE_LABELS:
            mapped = _CONFIDENCE_LABELS[key]
            logger.warning("coerced string confidence %r -> %s", raw, mapped)
            return mapped
        logger.warning("unknown confidence string %r; treating as None", raw)
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning("uncoercible confidence value %r (%s); treating as None",
                       raw, type(raw).__name__)
        return None
    clamped = max(0.0, min(1.0, val))
    if clamped != val:
        logger.warning("clamped out-of-range confidence %s -> %s", val, clamped)
    return clamped


def _coerce_rationale(raw) -> str | None:
    """Coerce a confidence_rationale value to a stripped string, or None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning("confidence_rationale is bool (%r); rejecting", raw)
        return None
    try:
        return str(raw)
    except Exception:  # noqa: BLE001 — defensive; any object should be str-able
        logger.warning("uncoercible confidence_rationale %r; dropping", raw)
        return None


# Fallback signal vocabulary used by tests and any caller that doesn't
# thread ``cfg.orchestrator.signals`` through. Production code passes the
# configured set down via ``make_agent_node``.
_DEFAULT_SIGNALS: frozenset[str] = frozenset({"success", "failed", "needs_input"})


def _coerce_signal(raw, valid_signals: frozenset[str] | None = None) -> str | None:
    """Coerce a raw signal value emitted by an LLM to a canonical lowercase
    string, or None when the value cannot be interpreted.

    The accepted vocabulary comes from ``cfg.orchestrator.signals`` (passed
    in as ``valid_signals``); when the caller omits it, the historical
    ``{success, failed, needs_input}`` default is used. Any value outside
    the set emits a warning and yields None — the route lookup then falls
    back to ``when: default``. ``bool`` is rejected explicitly because
    Python treats it as ``int`` and string-coerces to ``"True"``/``"False"``.
    """
    allowed = valid_signals if valid_signals is not None else _DEFAULT_SIGNALS
    if isinstance(raw, bool):
        logger.warning("signal value is bool (%r); rejecting", raw)
        return None
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning("non-string signal %r (%s); rejecting",
                       raw, type(raw).__name__)
        return None
    key = raw.strip().lower()
    if key in allowed:
        return key
    logger.warning("unknown signal %r; treating as None (will fall through to default)", raw)
    return None


class GraphState(TypedDict, total=False):
    session: Session
    next_route: str | None
    last_agent: str | None
    gated_target: str | None  # set by gate node; the downstream target if gate passes
    # Depth counter for supervisor recursion. The supervisor node bumps
    # it on entry and aborts at ``skill.max_dispatch_depth``. Carrying
    # it on graph state — rather than stashing it on the session —
    # keeps audit fields off the session and the depth check cheap
    # (no store reload).
    dispatch_depth: int | None
    error: str | None


def route_from_skill(skill: Skill, signal: str) -> str:
    if not skill.routes:
        raise ValueError(f"Skill '{skill.name}' has no routes defined")
    for rule in skill.routes:
        if rule.when == signal:
            return rule.next
    for rule in skill.routes:
        if rule.when == "default":
            return rule.next
    return skill.routes[0].next


class AgentRunRecorder:
    """Helper to capture an agent's run + tool calls into the incident."""

    def __init__(self, *, agent: str, incident: Session):
        self.agent = agent
        self.incident = incident
        self._started_at: str | None = None

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

    def record_tool_call(self, tool: str, args: dict, result) -> None:
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
        self.incident.tool_calls.append(
            ToolCall(agent=self.agent, tool=tool, args=args, result=result, ts=ts)
        )

    def finish(self, *, summary: str) -> None:
        ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
        self.incident.agents_run.append(AgentRun(
            agent=self.agent,
            started_at=self._started_at or ended_at,
            ended_at=ended_at,
            summary=summary,
        ))


_TRANSIENT_MARKERS = (
    "internal server error",
    "status code: -1",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "remoteprotocolerror",
    "incomplete chunked read",
    "connection reset",
)


async def _ainvoke_with_retry(executor, input_, *, max_attempts: int = 3,
                              base_delay: float = 1.5):
    """Wrap a LangGraph agent invocation with retry on transient cloud errors.

    Retries on common Ollama Cloud / streaming hiccups (500, status -1, etc.).
    Non-transient exceptions (4xx, validation, etc.) propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            # Phase 15 (LLM-COMPAT-01): the recursion_limit=25 workaround
            # introduced in 3ba099f as a safety net is gone — the
            # ``langchain.agents.create_agent`` migration replaces the
            # old two-call structure (loop + separate
            # ``with_structured_output`` pass) with a single tool-loop
            # whose terminal signal is the AgentTurnOutput tool call
            # itself (AutoStrategy → ToolStrategy fallback for non-
            # function-calling Ollama models). The default langgraph
            # recursion bound is now a true upper bound, not a workaround.
            return await executor.ainvoke(input_)
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): never retry a HITL pause.
            # GraphInterrupt is a checkpointed pending_approval signal,
            # not a transient error.
            raise
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = any(m in msg for m in _TRANSIENT_MARKERS)
            if not transient or attempt == max_attempts - 1:
                raise
            last_exc = exc
            await asyncio.sleep(base_delay * (attempt + 1))
    raise last_exc or RuntimeError("retry exhausted with no attempts")  # pragma: no cover


def _format_agent_input(incident: Session) -> str:
    """Build the human-message preamble each agent receives.

    Delegates to ``Session.to_agent_input`` so each app subclass owns the
    domain-shape of its prompt. The framework default surfaces only the
    session id + status; ``IncidentState`` and ``CodeReviewState``
    override with their respective shapes.
    """
    return incident.to_agent_input()


def _merge_patch_metadata(
    patch: dict,
    confidence: float | None,
    rationale: str | None,
    signal: str | None,
    valid_signals: frozenset[str] | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Update the (confidence, rationale, signal) trio with whatever the
    patch carries; preserve the prior value when the patch is silent on a
    field. Centralises the per-key conditional that used to nest 3 deep
    inside ``_harvest_tool_calls_and_patches``.
    """
    new_conf = _coerce_confidence(patch["confidence"]) if "confidence" in patch else confidence
    new_rationale = (
        _coerce_rationale(patch["confidence_rationale"])
        if "confidence_rationale" in patch else rationale
    )
    new_signal = (
        _coerce_signal(patch["signal"], valid_signals)
        if "signal" in patch else signal
    )
    return new_conf, new_rationale, new_signal


def _harvest_typed_terminal(
    tc_args: dict,
    state: tuple[float | None, str | None, str | None],
    valid_signals: frozenset[str] | None,
) -> tuple[float | None, str | None, str | None]:
    """Apply a typed-terminal tool call's args to the harvest state."""
    conf, rat, sig = state
    new_conf = _coerce_confidence(tc_args.get("confidence"))
    if new_conf is not None:
        conf = new_conf
    new_rat = _coerce_rationale(tc_args.get("confidence_rationale"))
    if new_rat is not None:
        rat = new_rat
    terminal = _coerce_signal("success", valid_signals)
    if terminal is not None:
        sig = terminal
    return conf, rat, sig


def _harvest_patch_tool(
    tc_args: dict,
    state: tuple[float | None, str | None, str | None],
    terminal_locked: bool,
    valid_signals: frozenset[str] | None,
) -> tuple[float | None, str | None, str | None]:
    """Apply a configured patch-tool's ``args.patch`` blob to the
    harvest state.

    When ``terminal_locked`` is True (a typed-terminal call already
    fired this session), confidence/rationale are pinned; only signal
    can flow through. Generalises the v1.0 single-tool path —
    apps register patch-tool names via
    ``OrchestratorConfig.patch_tools``.
    """
    conf, rat, sig = state
    patch = tc_args.get("patch") or {}
    merged_conf, merged_rat, merged_sig = _merge_patch_metadata(
        patch, conf, rat, sig, valid_signals,
    )
    if not terminal_locked:
        conf, rat = merged_conf, merged_rat
    return conf, rat, merged_sig


def _harvest_tool_calls_and_patches(
    messages: list,
    skill_name: str,
    incident: Session,
    ts: str,
    valid_signals: frozenset[str] | None = None,
    terminal_tool_names: frozenset[str] = frozenset(),
    patch_tool_names: frozenset[str] = frozenset(),
) -> tuple[float | None, str | None, str | None]:
    """Iterate agent messages, record ToolCall entries on the session, and
    harvest confidence / confidence_rationale / signal from typed terminal
    tools or app-declared patch tools.

    Typed terminal tools (those whose bare name is in
    ``terminal_tool_names``, supplied by the caller from
    ``OrchestratorConfig.terminal_tools``) carry confidence and rationale
    as flat kwargs; they imply ``signal=success`` since invoking a
    terminal tool is the agent's declaration that *its stage* completed
    cleanly — not that the session itself was successfully resolved.
    The session-level outcome is inferred separately from tool_calls
    history by ``_finalize_session_status``. Non-terminal agents emit
    routing signal via patch-tool args.

    ``patch_tool_names`` lists the bare tool names that ship a
    ``patch:`` arg the harvester should merge. Empty default means
    "no patch tools" so unconfigured apps pay nothing. Generalises
    the v1.0 single-tool path; apps register patch-tool names via
    ``OrchestratorConfig.patch_tools``.

    Once a typed terminal tool has fired, its confidence/rationale are
    authoritative — a same-message patch must not override them.
    Signal still flows from later patches so triage-style routing
    remains expressive.

    Returns ``(agent_confidence, agent_rationale, agent_signal)``.
    """
    state: tuple[float | None, str | None, str | None] = (None, None, None)
    terminal_locked = False
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            tc_name = tc.get("name", "unknown")
            tc_args = tc.get("args", {}) or {}
            # MCP tools follow ``<server>:<tool>`` with exactly one
            # colon; rsplit on the rightmost colon recovers the bare
            # tool name for both prefixed and unprefixed forms.
            tc_original = tc_name.rsplit(":", 1)[-1]
            incident.tool_calls.append(ToolCall(
                agent=skill_name, tool=tc_name, args=tc_args,
                result=None, ts=ts,
            ))
            if tc_original in terminal_tool_names:
                state = _harvest_typed_terminal(tc_args, state, valid_signals)
                terminal_locked = True
            elif tc_original in patch_tool_names:
                state = _harvest_patch_tool(
                    tc_args, state, terminal_locked, valid_signals,
                )
    return state


def _pair_tool_responses(messages: list, incident: Session) -> None:
    """Match ToolMessage responses back to their corresponding ToolCall entries."""
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage":
            for entry in reversed(incident.tool_calls):
                if entry.tool == getattr(msg, "name", None) and entry.result is None:
                    entry.result = getattr(msg, "content", None)
                    break


def _extract_final_text(messages: list) -> str:
    """Return the text content of the last non-empty AIMessage, or empty string."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            return str(msg.content)
    return ""


def _first_terminal_tool_called_this_turn(
    messages: list,
    terminal_tool_names: frozenset[str],
) -> str | None:
    """Return the bare name of the first typed-terminal tool called this turn.

    Phase 10 (FOC-03 / D-10-03): used to label the reconciliation log so
    operators can correlate envelope-vs-tool-arg confidence divergences
    against a specific tool. Tool names may be MCP-prefixed
    (``<server>:<tool>``); we rsplit on the rightmost colon to recover the
    bare name and match against the configured ``terminal_tool_names``.
    Returns None when no terminal tool fired this turn.
    """
    if not terminal_tool_names:
        return None
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name", "")
            bare = name.rsplit(":", 1)[-1]
            if bare in terminal_tool_names:
                return bare
    return None


def _sum_token_usage(messages: list) -> TokenUsage:
    """Sum input/output token counts across all messages that report usage_metadata."""
    agent_in = agent_out = 0
    for msg in messages:
        um = getattr(msg, "usage_metadata", None) or {}
        agent_in += int(um.get("input_tokens") or 0)
        agent_out += int(um.get("output_tokens") or 0)
    return TokenUsage(
        input_tokens=agent_in,
        output_tokens=agent_out,
        total_tokens=agent_in + agent_out,
    )


def _try_recover_envelope_from_raw(raw: str) -> AgentTurnOutput | None:
    """Attempt to extract an :class:`AgentTurnOutput` from a raw LLM
    string when LangGraph's structured-output pass raised
    ``OutputParserException``.

    Strategy:
    1. Parse the whole string as JSON.
    2. If that fails, scan for the first balanced ``{...}`` substring
       and try parsing that (handles markdown-fenced JSON or trailing
       chatter).
    3. Validate the parsed dict against :class:`AgentTurnOutput`.

    Returns the parsed envelope on success, ``None`` on any failure.
    """
    if not raw or not raw.strip():
        return None
    candidates: list[str] = [raw]
    # Markdown-fenced JSON: ```json\n{...}\n```
    if "```" in raw:
        for chunk in raw.split("```"):
            stripped = chunk.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
            if stripped.startswith("{"):
                candidates.append(stripped)
    # Greedy: first '{' through last '}'
    first = raw.find("{")
    last = raw.rfind("}")
    if 0 <= first < last:
        candidates.append(raw[first:last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            return AgentTurnOutput.model_validate(payload)
        except Exception:  # noqa: BLE001
            continue
    return None


def _handle_agent_failure(
    *,
    skill_name: str,
    started_at: str,
    exc: Exception,
    inc_id: str,
    store: "SessionStore",
    fallback: "Session",
) -> dict:
    """Reload incident (absorbing partial tool writes), stamp a failure AgentRun,
    persist, and return the error state dict for the LangGraph node.

    ``fallback`` is the in-memory incident from the caller; we use it only
    when the on-disk state has gone missing (FileNotFoundError on reload).
    """
    try:
        incident = store.load(inc_id)
    except FileNotFoundError:
        incident = fallback
    ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
    incident.agents_run.append(AgentRun(
        agent=skill_name, started_at=started_at, ended_at=ended_at,
        summary=f"agent failed: {exc}",
        token_usage=TokenUsage(),
    ))
    # Mark the session as terminally failed so the UI can render a
    # retry control. The retry path (``Orchestrator.retry_session``)
    # is the only documented way to move out of this state.
    incident.status = "error"
    store.save(incident)
    return {"session": incident, "next_route": None,
            "last_agent": skill_name, "error": str(exc)}


def _record_success_run(
    *,
    incident: "Session",
    skill_name: str,
    started_at: str,
    final_text: str,
    usage: "TokenUsage",
    confidence: float | None,
    rationale: str | None,
    signal: str | None,
    store: "SessionStore",
) -> None:
    """Append the success-path AgentRun, update the incident's running token
    totals, and persist. Mutates ``incident`` in place."""
    ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
    incident.agents_run.append(AgentRun(
        agent=skill_name, started_at=started_at, ended_at=ended_at,
        summary=final_text or f"{skill_name} completed",
        token_usage=usage,
        confidence=confidence,
        confidence_rationale=rationale,
        signal=signal,
    ))
    incident.token_usage.input_tokens += usage.input_tokens
    incident.token_usage.output_tokens += usage.output_tokens
    incident.token_usage.total_tokens += usage.total_tokens
    store.save(incident)


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Session], str],
    store: SessionStore,
    valid_signals: frozenset[str] | None = None,
    gateway_cfg: GatewayConfig | None = None,
    terminal_tool_names: frozenset[str] = frozenset(),
    patch_tool_names: frozenset[str] = frozenset(),
    injected_args: dict[str, str] | None = None,
    gate_policy: "GatePolicy | None" = None,
    event_log: "EventLog | None" = None,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route.

    ``valid_signals`` is the orchestrator-wide accepted signal vocabulary
    (``cfg.orchestrator.signals``). When omitted, the legacy
    ``{success, failed, needs_input}`` default is used so older callers and
    tests keep working.

    ``gateway_cfg`` is the optional risk-rated tool gateway config.
    When supplied, every ``BaseTool`` in ``tools`` is wrapped via
    :func:`runtime.tools.gateway.wrap_tool` *inside the node body* so
    the closure captures the live ``Session`` per agent invocation.
    When ``None``, tools are passed through untouched.

    ``terminal_tool_names`` and ``patch_tool_names`` are the bare tool
    names the harvester should treat as typed-terminal /
    patch-emitting (sourced from ``OrchestratorConfig.terminal_tools``
    union ``OrchestratorConfig.harvest_terminal_tools`` /
    ``OrchestratorConfig.patch_tools``). Empty defaults preserve the
    "no harvester recognition" behavior for legacy callers.

    ``injected_args`` (Phase 9 / D-09-01) is the orchestrator-wide
    map of ``arg_name -> dotted_path`` declared in
    :attr:`OrchestratorConfig.injected_args`. Every entry is stripped
    from each tool's LLM-visible signature (so the LLM cannot emit a
    value for it) and re-supplied at invocation time from session
    state. When ``None`` or empty, tools pass through to the LLM
    unchanged — preserves legacy callers and the framework default.
    """

    async def node(state: GraphState) -> dict:
        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # M3 (per-step telemetry): emit agent_started.
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "agent_started",
                    agent=skill.name, started_at=started_at,
                )
            except Exception:  # noqa: BLE001 — telemetry must not break the agent
                logger.debug(
                    "event_log.record(agent_started) failed", exc_info=True,
                )

        # Phase 9 (D-09-01): strip injected-arg keys from every tool's
        # LLM-visible signature BEFORE create_react_agent serialises the
        # tool surface — so the LLM literally cannot emit values for
        # those params. The framework re-supplies them at invocation
        # time inside the gateway (or an inject-only wrapper) below.
        from runtime.tools.arg_injection import (
            inject_injected_args as _inject_args,
            strip_injected_params,
        )
        injected_keys = frozenset((injected_args or {}).keys())
        if injected_keys:
            visible_tools = [
                strip_injected_params(t, injected_keys) for t in tools
            ]
        else:
            visible_tools = tools

        # Wrap tools per-invocation so each wrap closes over the live
        # ``Session`` for this run. When the gateway is unconfigured,
        # the original tools pass through untouched and
        # ``create_react_agent`` sees the same surface as before.
        if gateway_cfg is not None:
            # Pass ORIGINAL tools (pre-strip) to wrap_tool — the gateway
            # wrapper strips internally for the LLM-visible schema while
            # keeping ``inner.args_schema`` intact so
            # ``accepted_params_for_tool`` correctly recognises injected
            # keys (e.g. ``environment``) as accepted by the underlying
            # tool. Stripping twice (here AND in wrap_tool) hides those
            # keys from ``accepted_params``, the inject step skips them,
            # and FastMCP rejects the call as missing required arg.
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store,
                          injected_args=injected_args or {},
                          gate_policy=gate_policy,
                          event_log=event_log)
                for t in tools
            ]
        elif injected_keys:
            # No gateway, but injected_args is configured — wrap each
            # tool in an inject-only ``StructuredTool`` so the LLM-visible
            # sig matches ``visible_tools`` while the underlying call
            # still receives the framework-supplied values.
            from langchain_core.tools import StructuredTool

            _inject_cfg = injected_args or {}

            def _make_inject_only_wrapper(
                base: BaseTool, llm_visible: BaseTool, sess: Session,
            ) -> BaseTool:
                async def _arun(**kwargs: Any) -> Any:
                    new_kwargs = _inject_args(
                        kwargs,
                        session=sess,
                        injected_args_cfg=_inject_cfg,
                        tool_name=base.name,
                    )
                    return await base.ainvoke(new_kwargs)

                def _run(**kwargs: Any) -> Any:
                    new_kwargs = _inject_args(
                        kwargs,
                        session=sess,
                        injected_args_cfg=_inject_cfg,
                        tool_name=base.name,
                    )
                    return base.invoke(new_kwargs)

                return StructuredTool.from_function(
                    func=_run,
                    coroutine=_arun,
                    name=base.name,
                    description=base.description,
                    args_schema=llm_visible.args_schema,
                )

            run_tools = [
                _make_inject_only_wrapper(orig, vis, incident)
                for orig, vis in zip(tools, visible_tools)
            ]
        else:
            run_tools = visible_tools
        # Phase 10 (FOC-03 / D-10-02) + Phase 15 (LLM-COMPAT-01): every
        # responsive agent invocation is wrapped in an AgentTurnOutput
        # envelope. ``langchain.agents.create_agent`` (the non-deprecated
        # successor to ``langgraph.prebuilt.create_react_agent``) accepts a
        # bare schema as ``response_format`` and, by default, wraps it in
        # ``AutoStrategy`` — ProviderStrategy for models with native
        # structured-output (OpenAI-class), falling back to ToolStrategy
        # otherwise (Ollama). ToolStrategy injects AgentTurnOutput as a
        # callable tool: when the LLM ``calls`` it, the loop terminates on
        # the same turn with ``result["structured_response"]`` populated.
        # Eliminates the old two-call structure (loop + separate
        # ``with_structured_output`` pass) that hit recursion_limit=25 on
        # Ollama models without true function-calling.
        agent_executor = create_agent(
            model=llm,
            tools=run_tools,
            system_prompt=skill.system_prompt,
            response_format=AgentTurnOutput,
        )

        # Phase 11 (FOC-04): reset per-turn confidence hint. The hint
        # is updated below after _harvest_tool_calls_and_patches; on
        # re-entry from a HITL pause the hint resets cleanly so a new
        # turn starts from "no signal yet" (None).
        try:
            incident.turn_confidence_hint = None
        except (AttributeError, ValueError):
            pass

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): HITL pause is NOT an error.
            # Re-raise so LangGraph's checkpointer captures the paused
            # state. Session.status is left to the orchestrator's
            # interrupt-aware bridge, NOT _handle_agent_failure.
            raise
        except Exception as exc:  # noqa: BLE001
            # Phase 10 follow-up: when LangGraph's structured-output pass
            # raises ``OutputParserException`` (Ollama / non-OpenAI
            # providers don't always honor ``response_format`` cleanly),
            # try to recover by parsing the raw LLM output ourselves.
            # The exception's ``llm_output`` carries the model's reply
            # verbatim; if it contains JSON matching the envelope schema,
            # build a synthetic ``result`` and continue. On unrecoverable
            # failure, log the raw output for diagnosis and fall through
            # to ``_handle_agent_failure``.
            try:
                from langchain_core.exceptions import OutputParserException
            except ImportError:  # pragma: no cover — langchain always present
                OutputParserException = ()  # type: ignore[assignment]
            if isinstance(exc, OutputParserException):
                raw = getattr(exc, "llm_output", "") or ""
                logger.warning(
                    "agent.structured_output_parse_failure agent=%s "
                    "raw_len=%d raw_preview=%r",
                    skill.name, len(raw), raw[:500],
                )
                recovered = _try_recover_envelope_from_raw(raw)
                if recovered is not None:
                    logger.info(
                        "agent.structured_output_recovered agent=%s",
                        skill.name,
                    )
                    result = {
                        "messages": [],
                        "structured_response": recovered,
                    }
                else:
                    return _handle_agent_failure(
                        skill_name=skill.name, started_at=started_at, exc=exc,
                        inc_id=inc_id, store=store, fallback=incident,
                    )
            else:
                return _handle_agent_failure(
                    skill_name=skill.name, started_at=started_at, exc=exc,
                    inc_id=inc_id, store=store, fallback=incident,
                )

        # Tools (e.g. registered patch tools) write straight to disk.
        # Reload so the node's own append of agent_run + tool_calls
        # happens against the tool-mutated state — otherwise saving
        # the stale in-memory object clobbers the tools' writes.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Record tool calls and harvest confidence/signal from configured
        # patch / typed-terminal tools.
        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
        )
        # Phase 11 (FOC-04): update hint so any subsequent in-turn
        # tool call sees the harvested confidence at the gateway.
        if agent_confidence is not None:
            try:
                incident.turn_confidence_hint = agent_confidence
            except (AttributeError, ValueError):
                pass

        # Pair tool responses with their tool calls.
        _pair_tool_responses(messages, incident)

        # Phase 10 (FOC-03 / D-10-03): parse the structural envelope and
        # reconcile its confidence against any typed-terminal-tool arg
        # confidence harvested above. Envelope failure is a hard error —
        # mark the agent_run failed with structured cause.
        try:
            envelope = parse_envelope_from_result(result, agent=skill.name)
        except EnvelopeMissingError as exc:
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        terminal_tool_for_log = _first_terminal_tool_called_this_turn(
            messages, terminal_tool_names,
        )
        final_confidence = reconcile_confidence(
            envelope.confidence,
            agent_confidence,
            agent=skill.name,
            session_id=inc_id,
            tool_name=terminal_tool_for_log,
        )
        final_rationale = agent_rationale or envelope.confidence_rationale
        final_signal = agent_signal if agent_signal is not None else envelope.signal

        # Final summary text and token usage.
        # Envelope content takes precedence over last AIMessage scrape.
        final_text = envelope.content or _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        # M3: emit confidence_emitted after reconcile lands.
        if event_log is not None and final_confidence is not None:
            try:
                event_log.record(
                    inc_id, "confidence_emitted",
                    agent=skill.name,
                    value=float(final_confidence),
                    rationale=final_rationale or "",
                    signal=final_signal,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(confidence_emitted) failed", exc_info=True,
                )

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=final_confidence, rationale=final_rationale, signal=final_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)

        # M3: emit route_decided + agent_finished (carrying token_usage).
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "route_decided",
                    agent=skill.name,
                    signal=next_route_signal,
                    next_node=next_node,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(route_decided) failed", exc_info=True,
                )
            try:
                event_log.record(
                    inc_id, "agent_finished",
                    agent=skill.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(agent_finished) failed", exc_info=True,
                )

        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Session) -> str:
    """Return the latest agent's emitted signal, or "default" if absent.

    Agents emit one of {success, failed, needs_input} via the ``signal``
    key of a configured patch tool (see ``_coerce_signal``).
    The node harvests it onto ``AgentRun.signal``; this decider then reads
    the *most recent* run (which is the one that just finished, since the
    node has already appended it). If no signal is present we return
    "default" so ``route_from_skill`` picks the fallback rule.
    """
    if not inc.agents_run:
        return "default"
    return inc.agents_run[-1].signal or "default"


_DEFAULT_STUB_CANNED: dict[str, str] = {
    # Back-compat defaults for the four canonical agents.  Any YAML-defined
    # agent can override or extend this via ``skill.stub_response``; new agents
    # without an entry here fall through to StubChatModel's generic placeholder.
    "intake": "Created INC, no prior matches. Routing to triage.",
    "triage": "Severity medium, category latency. No recent deploys correlate.",
    "deep_investigator": "Hypothesis: upstream payments timeout. Evidence: log line 'upstream_timeout target=payments'.",
    "resolution": "Proposed fix: restart api service. Auto-applied. INC resolved.",
}

# Phase 10 (FOC-03): per-agent default envelope confidence for the stub
# LLM. Pre-Phase-10 the deep_investigator stub emitted no confidence at
# all, so the gate (threshold 0.75) always interrupted on the first
# call. Post-Phase-10 every agent must emit a confidence value — drive
# DI's stub envelope below threshold to preserve gate-pause behavior in
# existing tests. Other agents default to 0.85 (above threshold).
_DEFAULT_STUB_ENVELOPE_CONFIDENCE: dict[str, float] = {
    "deep_investigator": 0.30,
}


def _latest_run_for(incident: Session, agent_name: str | None):
    """Return the most recent ``AgentRun`` for ``agent_name``, or None.

    ``agent_name`` is whichever agent ran immediately before the gate,
    derived at runtime from ``state["last_agent"]`` — so the gate is
    config-driven and works for any YAML-defined upstream, not just
    ``deep_investigator``.
    """
    if not agent_name:
        return None
    for run in reversed(incident.agents_run):
        if run.agent == agent_name:
            return run
    return None


def make_gate_node(
    *,
    cfg: AppConfig,
    store: SessionStore,
    threshold: float | None = None,
    teams: list[str] | None = None,
):
    """Build the intervention gate node placed before a gated downstream.

    The gate evaluates the confidence of whichever agent ran immediately
    before it (``state["last_agent"]``). If that confidence is below the
    configured threshold (or absent), the gate persists a
    ``pending_intervention`` payload on the Session row **and** raises
    ``langgraph.types.interrupt(payload)``. The checkpointer captures
    the suspended state; ``Orchestrator.resume_session`` resumes the
    graph via ``Command(resume=user_input)``. On resume the node body
    re-executes from the start, ``interrupt()`` returns the resume
    value (instead of raising), the user input is appended to
    ``session.user_inputs``, ``pending_intervention`` is cleared, and
    execution falls through to the gated downstream.

    The dual-write order is intentional: persist the row **before**
    calling ``interrupt()`` so the Streamlit UI (which polls
    ``Session.pending_intervention``) sees the pending state even on a
    cold restart. Reversing the order would leave the UI blind to a
    pending intervention captured only inside the LangGraph checkpoint.

    Implemented as a plain async coroutine (not via ``make_agent_node``)
    so it does not invoke an LLM — but it IS a real graph node, so
    streamed events surface ``enter gate`` / ``exit gate``.

    The gate is fully agent-agnostic: any YAML route carrying
    ``gate: confidence`` causes the gate to evaluate the upstream agent
    named on that route, regardless of agent identity.

    ``threshold`` and ``teams`` are the cross-cutting confidence cutoff
    and escalation roster the gate exposes to the operator on
    intervention. They are supplied by ``build_graph`` from the
    orchestrator's resolved :class:`FrameworkAppConfig` so the gate is
    completely free of any app-specific config import. When ``None``
    (legacy callers / unit tests that build the gate directly), the
    framework defaults are used (0.75 / empty roster).
    """
    if threshold is None:
        threshold = FrameworkAppConfig().confidence_threshold
    if teams is None:
        teams = []
    teams = list(teams)

    async def gate(state: GraphState) -> dict:
        # Imported lazily so unit tests that import graph.py without a
        # checkpointer don't pay the import cost. ``interrupt`` raises
        # ``GraphInterrupt`` on first execution and returns the resume
        # value on subsequent executions of the same node.
        from langgraph.types import interrupt

        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
        upstream = state.get("last_agent")
        # Capture the intended downstream target before we overwrite next_route.
        # The upstream agent set next_route to the gated target; we stash it in
        # gated_target so _make_gate_to can route correctly for multi-target graphs.
        intended_target = state.get("next_route")
        # Reload from disk in case earlier nodes wrote tool-driven patches.
        try:
            incident = store.load(incident.id)
        except FileNotFoundError:
            pass
        upstream_run = _latest_run_for(incident, upstream)
        upstream_conf = upstream_run.confidence if upstream_run else None
        if upstream_conf is None or upstream_conf < threshold:
            incident.status = "awaiting_input"
            # Surface the upstream agent's own summary + rationale so the
            # human reviewer can decide what input to give without scrolling
            # through every step of the agents-run log.
            payload = {
                "reason": "low_confidence",
                "confidence": upstream_conf,
                "threshold": threshold,
                "upstream_agent": upstream,
                "summary": upstream_run.summary if upstream_run else "",
                "rationale": upstream_run.confidence_rationale if upstream_run else "",
                "options": ["resume_with_input", "escalate", "stop"],
                "escalation_teams": teams,
                "intended_target": intended_target,
            }
            incident.pending_intervention = payload
            # CRITICAL ORDERING: persist the Session row BEFORE calling
            # ``interrupt()``. ``interrupt()`` raises ``GraphInterrupt`` on
            # first execution; if we reversed the order the UI (which
            # polls Session.pending_intervention) would never see the
            # pending state. See plan R4 / "Streamlit hand-off".
            store.save(incident)
            # First execution: this raises GraphInterrupt and the
            # checkpointer captures the paused state.
            # Resume: this returns the value supplied via
            # ``Command(resume=...)``.
            decision = interrupt(payload)
            # Post-resume continuation. ``decision`` is whatever the
            # caller passed to ``Command(resume=...)`` — typically the
            # operator's free-text note (str) or a dict with an
            # ``input`` key. Empty/None decisions are treated as a
            # no-op (the gate still clears and falls through to the
            # gated target — the orchestrator wraps non-text actions
            # like ``stop``/``escalate`` outside the graph).
            user_text: str | None = None
            if isinstance(decision, str):
                stripped = decision.strip()
                if stripped:
                    user_text = stripped
            elif isinstance(decision, dict):
                raw = decision.get("input")
                if isinstance(raw, str) and raw.strip():
                    user_text = raw.strip()
            if user_text is not None:
                incident.user_inputs.append(user_text)
            incident.pending_intervention = None
            incident.status = "in_progress"
            store.save(incident)
            return {"session": incident, "next_route": "default",
                    "gated_target": intended_target, "last_agent": "gate", "error": None}
        # Confidence met threshold — clear any stale intervention payload.
        if incident.pending_intervention is not None:
            incident.pending_intervention = None
            store.save(incident)
        return {"session": incident, "next_route": "default",
                "gated_target": intended_target, "last_agent": "gate", "error": None}

    return gate


def _build_agent_nodes(*, cfg: AppConfig, skills: dict, store: SessionStore,
                       registry: ToolRegistry,
                       event_log: "EventLog | None" = None) -> dict:
    """Materialize agent nodes from skills + registry. Reused by main + resume graphs.

    Dispatches on ``skill.kind``:

    * ``responsive`` — builds a ReAct LLM node via :func:`make_agent_node`
      (today's path).
    * ``supervisor`` — builds a no-LLM router node via
      :func:`runtime.agents.supervisor.make_supervisor_node`. Supervisor
      skills with ``dispatch_strategy=llm`` get a small LLM bound; rule
      strategy uses ``None``.
    * ``monitor``   — **skipped**: monitor skills run out-of-band under
      :class:`runtime.agents.monitor.MonitorRunner`, not inside the
      session graph.
    """
    # Local import — agents package depends on this module's helpers.
    from runtime.agents.supervisor import make_supervisor_node

    valid_signals = frozenset(cfg.orchestrator.signals)
    gateway_cfg = getattr(cfg.runtime, "gateway", None)
    # Phase 11 (FOC-04): thread the orchestrator's gate_policy down to
    # wrap_tool so should_gate can apply the configured per-app
    # confidence threshold + gated environments / risk actions.
    gate_policy = getattr(cfg.orchestrator, "gate_policy", None)
    # Build the harvester's tool-name sets once per graph-build. The
    # union of ``terminal_tools`` (status-transitioning) and
    # ``harvest_terminal_tools`` (harvest-only) gives the full
    # typed-terminal recognition surface for confidence/rationale
    # capture; ``patch_tools`` carries the patch-blob harvester path.
    terminal_tool_names = frozenset(
        r.tool_name for r in cfg.orchestrator.terminal_tools
    ) | frozenset(cfg.orchestrator.harvest_terminal_tools)
    patch_tool_names = frozenset(cfg.orchestrator.patch_tools)
    nodes: dict = {}
    for agent_name, skill in skills.items():
        kind = getattr(skill, "kind", "responsive")
        if kind == "monitor":
            # Monitors are not graph nodes; skip silently.
            continue
        if kind == "supervisor":
            llm = None
            if skill.dispatch_strategy == "llm":
                llm = get_llm(
                    cfg.llm, skill.model, role=agent_name,
                    default_llm_request_timeout=cfg.orchestrator.default_llm_request_timeout,
                )
            nodes[agent_name] = make_supervisor_node(skill=skill, llm=llm)
            continue
        # Default / "responsive" path.
        if skill.stub_response is not None:
            stub_canned: dict[str, str] | None = {skill.name: skill.stub_response}
        elif agent_name in _DEFAULT_STUB_CANNED:
            stub_canned = {agent_name: _DEFAULT_STUB_CANNED[agent_name]}
        else:
            stub_canned = None
        # Phase 10 (FOC-03): wire a per-agent default envelope confidence
        # into the stub so pre-Phase-10 gate-pause-on-DI tests still pass.
        stub_env_conf = _DEFAULT_STUB_ENVELOPE_CONFIDENCE.get(agent_name)
        llm = get_llm(
            cfg.llm,
            skill.model,
            role=agent_name,
            stub_canned=stub_canned,
            stub_envelope_confidence=stub_env_conf,
            default_llm_request_timeout=cfg.orchestrator.default_llm_request_timeout,
        )
        tools = registry.resolve(skill.tools, cfg.mcp)
        decide = _decide_from_signal
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
            valid_signals=valid_signals,
            gateway_cfg=gateway_cfg,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
            injected_args=cfg.orchestrator.injected_args,
            gate_policy=gate_policy,
            event_log=event_log,
        )
    return nodes


def _make_router(gated_edges: dict[tuple[str, str], str]):
    """Build a state router that intercepts gated edges into the gate node.

    Used by ``build_graph`` to route an agent's outbound edges through
    the gate when the corresponding ``RouteRule`` carries ``gate=...``.
    """
    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        la = state.get("last_agent")
        if (la, nr) in gated_edges:
            return "gate"
        return nr
    return _router


def _make_gate_to(gate_targets: set[str]):
    """Build the gate's outbound router.

    On a low-confidence fail the gate sets ``next_route="__end__"`` and
    we terminate. On a pass the gate sets ``next_route="default"`` and also
    stamps ``gated_target`` with the intended downstream node (captured from
    the incoming ``next_route`` before the gate ran). We read ``gated_target``
    here so this router handles any number of gated targets without needing
    a hardcoded closure over a single target name.
    """
    def _gate_to(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        gt = state.get("gated_target")
        if gt in gate_targets:
            return gt
        return END
    return _gate_to


def _collect_gated_edges(skills: dict) -> dict[tuple[str, str], str]:
    """Return ``{(from_agent, to_node): gate_type}`` for every route rule
    whose ``gate`` is set. Today only ``gate: confidence`` is recognised."""
    edges: dict[tuple[str, str], str] = {}
    for agent_name, skill in skills.items():
        for rule in skill.routes:
            if rule.gate:
                edges[(agent_name, rule.next)] = rule.gate
    return edges


async def build_graph(*, cfg: AppConfig, skills: dict, store: SessionStore,
                      registry: ToolRegistry,
                      checkpointer=None,
                      framework_cfg: FrameworkAppConfig | None = None,
                      event_log: "EventLog | None" = None):
    """Compile the main LangGraph from configured skills and routes.

    The entry agent is read from ``cfg.orchestrator.entry_agent``. Gate
    insertions are derived from each skill's route rules: a rule with
    ``gate: confidence`` causes the router to redirect ``(this_agent, next)``
    through the ``gate`` node.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.

    ``checkpointer`` is an optional :class:`BaseCheckpointSaver` that
    LangGraph uses for durable per-thread state. When ``None``, the
    graph compiles without one (back-compat for the few callers that
    build a graph outside the orchestrator, e.g. unit tests).

    ``framework_cfg`` carries the cross-cutting confidence/escalation
    knobs the gate node reads. When ``None`` (legacy / unit-test
    callers that compile a graph without an orchestrator), the
    runtime falls back to ``resolve_framework_app_config(None)`` which
    yields a bare ``FrameworkAppConfig()``.
    """
    entry = cfg.orchestrator.entry_agent
    if entry not in skills:
        raise ValueError(
            f"orchestrator.entry_agent={entry!r} is not a known skill "
            f"(known: {sorted(skills.keys())})"
        )
    if framework_cfg is None:
        # Prefer the YAML-driven ``AppConfig.framework`` field; fall
        # back to the legacy provider-callable path for backward
        # compatibility with deployments that still wire it.
        if getattr(cfg.runtime, "framework_app_config_path", None) is not None:
            framework_cfg = resolve_framework_app_config(
                cfg.runtime.framework_app_config_path,
            )
        else:
            framework_cfg = getattr(cfg, "framework", None) or resolve_framework_app_config(None)
    # ``resolve_framework_app_config(None)`` always returns a bare
    # ``FrameworkAppConfig`` (never None), so the chain above is
    # exhaustive — assert for pyright's flow narrowing.
    assert framework_cfg is not None
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store,
                                registry=registry, event_log=event_log)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    sg.add_node("gate", make_gate_node(
        cfg=cfg, store=store,
        threshold=framework_cfg.confidence_threshold,
        teams=list(framework_cfg.escalation_teams),
    ))

    sg.set_entry_point(entry)

    _router = _make_router(gated_edges)

    for agent_name, _skill in skills.items():
        possible_targets = {s.name for s in skills.values()} | {END, "gate"}
        # Exclude targets that are intercepted via a gated edge for this agent:
        # the router redirects (agent_name, gated_target) -> "gate", so the
        # gated_target must NOT appear in this agent's target_map. Leaving it
        # in would cause LangGraph to register a visible direct edge in the
        # compiled graph, defeating the structural assertion in the test (and
        # misleading graph visualisations).
        gated_targets_for_agent = {to for (frm, to) in gated_edges.keys() if frm == agent_name}
        target_map = {
            name: name
            for name in possible_targets
            if name != END and name not in gated_targets_for_agent
        }
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel

    # Build the gate's outbound target map from all distinct gated downstream
    # nodes. _make_gate_to reads next_route from state (set by the upstream
    # agent) so it routes to the correct target regardless of how many gated
    # edges exist — the multi-target restriction is no longer needed.
    gate_targets = {to for (_from, to) in gated_edges.keys()}
    if gate_targets:
        _gate_to = _make_gate_to(gate_targets)
        gate_target_map: dict = {t: t for t in gate_targets}
        gate_target_map[END] = END
        sg.add_conditional_edges("gate", _gate_to, gate_target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel
    else:
        sg.add_edge("gate", END)

    return sg.compile(checkpointer=checkpointer) if checkpointer is not None else sg.compile()
