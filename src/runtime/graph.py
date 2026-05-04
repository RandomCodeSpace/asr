"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
import asyncio
import logging
from typing import TypedDict, Callable, Awaitable
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from runtime.state import Session, ToolCall, AgentRun, TokenUsage, _UTC_TS_FMT
from runtime.skill import Skill
from runtime.config import (
    AppConfig,
    FrameworkAppConfig,
    GatewayConfig,
    resolve_framework_app_config,
)
from runtime.llm import get_llm
from runtime.mcp_loader import ToolRegistry
from runtime.storage.session_store import SessionStore
from runtime.tools.gateway import wrap_tool

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
            return await executor.ainvoke(input_)
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


def _harvest_tool_calls_and_patches(
    messages: list,
    skill_name: str,
    incident: Session,
    ts: str,
    valid_signals: frozenset[str] | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Iterate agent messages, record ToolCall entries on the incident, and
    harvest any confidence / confidence_rationale / signal from update_incident
    patches.

    Returns ``(agent_confidence, agent_rationale, agent_signal)``.
    """
    agent_confidence: float | None = None
    agent_rationale: str | None = None
    agent_signal: str | None = None
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            tc_name = tc.get("name", "unknown")
            tc_args = tc.get("args", {}) or {}
            # Tool names are now namespaced as ``<server>:<original>``;
            # match on the un-prefixed suffix so the bare and prefixed
            # forms both harvest confidence/signal patches.
            tc_original = tc_name.rsplit(":", 1)[-1]
            incident.tool_calls.append(ToolCall(
                agent=skill_name,
                tool=tc_name,
                args=tc_args,
                result=None,
                ts=ts,
            ))
            if tc_original == "update_incident":
                patch = tc_args.get("patch") or {}
                agent_confidence, agent_rationale, agent_signal = _merge_patch_metadata(
                    patch, agent_confidence, agent_rationale, agent_signal,
                    valid_signals,
                )
    return agent_confidence, agent_rationale, agent_signal


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
    """

    async def node(state: GraphState) -> dict:
        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Wrap tools per-invocation so each wrap closes over the live
        # ``Session`` for this run. When the gateway is unconfigured,
        # the original tools pass through untouched and
        # ``create_react_agent`` sees the same surface as before.
        if gateway_cfg is not None:
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store)
                for t in tools
            ]
        else:
            run_tools = tools
        agent_executor = create_react_agent(
            llm, run_tools, prompt=skill.system_prompt,
        )

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except Exception as exc:  # noqa: BLE001
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        # Tools (e.g. update_incident) write straight to disk. Reload so the
        # node's own append of agent_run + tool_calls happens against the
        # tool-mutated state — otherwise saving the stale in-memory object
        # clobbers the tools' writes.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Record tool calls and harvest confidence/signal from update_incident patches.
        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
        )

        # Pair tool responses with their tool calls.
        _pair_tool_responses(messages, incident)

        # Final summary text and token usage.
        final_text = _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=agent_confidence, rationale=agent_rationale, signal=agent_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Session) -> str:
    """Return the latest agent's emitted signal, or "default" if absent.

    Agents emit one of {success, failed, needs_input} via the ``signal``
    key of their final ``update_incident`` patch (see ``_coerce_signal``).
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
                       registry: ToolRegistry) -> dict:
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
    nodes: dict = {}
    for agent_name, skill in skills.items():
        kind = getattr(skill, "kind", "responsive")
        if kind == "monitor":
            # Monitors are not graph nodes; skip silently.
            continue
        if kind == "supervisor":
            llm = None
            if skill.dispatch_strategy == "llm":
                llm = get_llm(cfg.llm, skill.model, role=agent_name)
            nodes[agent_name] = make_supervisor_node(skill=skill, llm=llm)
            continue
        # Default / "responsive" path.
        if skill.stub_response is not None:
            stub_canned: dict[str, str] | None = {skill.name: skill.stub_response}
        elif agent_name in _DEFAULT_STUB_CANNED:
            stub_canned = {agent_name: _DEFAULT_STUB_CANNED[agent_name]}
        else:
            stub_canned = None
        llm = get_llm(
            cfg.llm,
            skill.model,
            role=agent_name,
            stub_canned=stub_canned,
        )
        tools = registry.resolve(skill.tools, cfg.mcp)
        decide = _decide_from_signal
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
            valid_signals=valid_signals,
            gateway_cfg=gateway_cfg,
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
                      framework_cfg: FrameworkAppConfig | None = None):
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
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
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
