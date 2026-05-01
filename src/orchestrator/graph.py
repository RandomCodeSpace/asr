"""LangGraph state, routing helpers, and node runner."""
from __future__ import annotations
import asyncio
import logging
from typing import TypedDict, Callable, Awaitable
from datetime import datetime, timezone

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


_VALID_SIGNALS: frozenset[str] = frozenset({"success", "failed", "needs_input"})


def _coerce_signal(raw) -> str | None:
    """Coerce a raw signal value emitted by an LLM to a canonical lowercase
    string, or None when the value cannot be interpreted.

    Recognised signals are ``success``, ``failed``, and ``needs_input``. Any
    other string emits a warning and yields None — the route lookup then
    falls back to ``when: default``. ``bool`` is rejected explicitly because
    Python treats it as ``int`` and string-coerces to ``"True"``/``"False"``.
    """
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
    if key in _VALID_SIGNALS:
        return key
    logger.warning("unknown signal %r; treating as None (will fall through to default)", raw)
    return None


from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from orchestrator.incident import Incident, ToolCall, AgentRun, IncidentStore, TokenUsage
from orchestrator.skill import Skill
from orchestrator.config import AppConfig
from orchestrator.llm import get_llm
from orchestrator.mcp_loader import ToolRegistry


class GraphState(TypedDict, total=False):
    incident: Incident
    next_route: str | None
    last_agent: str | None
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

    def __init__(self, *, agent: str, incident: Incident):
        self.agent = agent
        self.incident = incident
        self._started_at: str | None = None

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record_tool_call(self, tool: str, args: dict, result) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.incident.tool_calls.append(
            ToolCall(agent=self.agent, tool=tool, args=args, result=result, ts=ts)
        )

    def finish(self, *, summary: str) -> None:
        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    raise last_exc  # pragma: no cover  (unreachable)


def _format_agent_input(incident: Incident) -> str:
    base = (
        f"Incident {incident.id}\n"
        f"Environment: {incident.environment}\n"
        f"Query: {incident.query}\n"
        f"Status: {incident.status}\n"
        f"Findings (triage): {incident.findings.triage}\n"
        f"Findings (deep_investigator): {incident.findings.deep_investigator}\n"
    )
    if incident.user_inputs:
        bullets = "\n".join(f"- {ui}" for ui in incident.user_inputs)
        base += (
            "\nUser-provided context (appended via intervention):\n"
            f"{bullets}\n"
        )
    return base


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Incident], str],
    store: IncidentStore,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route."""
    agent_executor = create_react_agent(llm, tools, prompt=skill.system_prompt)

    async def node(state: GraphState) -> dict:
        incident = state["incident"]
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except Exception as exc:  # noqa: BLE001
            # Reload to absorb any partial writes from tools that ran before the failure.
            try:
                incident = store.load(inc_id)
            except FileNotFoundError:
                pass
            ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            incident.agents_run.append(AgentRun(
                agent=skill.name, started_at=started_at, ended_at=ended_at,
                summary=f"agent failed: {exc}",
                token_usage=TokenUsage(),
            ))
            store.save(incident)
            return {"incident": incident, "next_route": None,
                    "last_agent": skill.name, "error": str(exc)}

        # Tools (e.g. update_incident) write straight to disk. Reload so the
        # node's own append of agent_run + tool_calls happens against the
        # tool-mutated state — otherwise saving the stale in-memory object
        # clobbers the tools' writes.
        incident = store.load(inc_id)

        # Record tool calls from the agent's message trace. While iterating,
        # also harvest the latest `confidence` / `confidence_rationale` carried
        # in any `update_incident` patch — those keys are stamped on the
        # AgentRun (the MCP tool itself silently ignores extra patch keys).
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        agent_confidence: float | None = None
        agent_rationale: str | None = None
        agent_signal: str | None = None
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("args", {}) or {}
                incident.tool_calls.append(ToolCall(
                    agent=skill.name,
                    tool=tc_name,
                    args=tc_args,
                    result=None,
                    ts=ts,
                ))
                if tc_name == "update_incident":
                    patch = tc_args.get("patch") or {}
                    if "confidence" in patch:
                        agent_confidence = _coerce_confidence(patch["confidence"])
                    if "confidence_rationale" in patch:
                        agent_rationale = _coerce_rationale(patch["confidence_rationale"])
                    if "signal" in patch:
                        agent_signal = _coerce_signal(patch["signal"])

        # Pair tool responses with their tool calls.
        for msg in result.get("messages", []):
            if msg.__class__.__name__ == "ToolMessage":
                for entry in reversed(incident.tool_calls):
                    if entry.tool == getattr(msg, "name", None) and entry.result is None:
                        entry.result = getattr(msg, "content", None)
                        break

        # Final summary text from the agent's last AIMessage. Persist it
        # verbatim — concision is enforced via the skill prompts (each one
        # instructs the agent to keep the final reply ≤150 words). Storing
        # the full message preserves the audit trail; mid-fence truncation
        # used to corrupt downstream markdown rendering.
        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and msg.content:
                final_text = str(msg.content)
                break

        # Sum token usage across every message that reports it. langchain-ollama
        # populates `usage_metadata` on AIMessages from Ollama's
        # prompt_eval_count / eval_count fields. Stub/test models leave it
        # absent — those simply contribute zero.
        agent_in = agent_out = 0
        for msg in result.get("messages", []):
            um = getattr(msg, "usage_metadata", None) or {}
            agent_in += int(um.get("input_tokens") or 0)
            agent_out += int(um.get("output_tokens") or 0)
        agent_total = agent_in + agent_out

        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        incident.agents_run.append(AgentRun(
            agent=skill.name, started_at=started_at, ended_at=ended_at,
            summary=final_text or f"{skill.name} completed",
            token_usage=TokenUsage(
                input_tokens=agent_in,
                output_tokens=agent_out,
                total_tokens=agent_total,
            ),
            confidence=agent_confidence,
            confidence_rationale=agent_rationale,
            signal=agent_signal,
        ))
        incident.token_usage.input_tokens += agent_in
        incident.token_usage.output_tokens += agent_out
        incident.token_usage.total_tokens += agent_total

        next_route_signal = decide_route(incident)
        store.save(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"incident": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Incident) -> str:
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


_STUB_CANNED = {
    "intake": "Created INC, no prior matches. Routing to triage.",
    "triage": "Severity medium, category latency. No recent deploys correlate.",
    "deep_investigator": "Hypothesis: upstream payments timeout. Evidence: log line 'upstream_timeout target=payments'.",
    "resolution": "Proposed fix: restart api service. Auto-applied. INC resolved.",
}


def _latest_di_confidence(incident: Incident) -> float | None:
    """Return the most recent deep_investigator AgentRun confidence, or None."""
    for run in reversed(incident.agents_run):
        if run.agent == "deep_investigator":
            return run.confidence
    return None


def make_gate_node(*, cfg: AppConfig, store: IncidentStore):
    """Build the intervention gate node placed between DI and resolution.

    If the latest deep_investigator confidence is below the configured
    threshold (or absent), the gate marks the incident as `awaiting_input`,
    populates `pending_intervention`, and routes to END. Otherwise it routes
    to `resolution`.

    Implemented as a plain async coroutine (not via ``make_agent_node``) so
    it does not invoke an LLM — but it IS a real graph node, so streamed
    events surface ``enter gate`` / ``exit gate``.

    .. note::
       The gate's *structural* placement is config-driven (see
       :func:`_collect_gated_edges`), but its *semantic* check is still
       pinned to ``deep_investigator``'s confidence via
       :func:`_latest_di_confidence`. Moving the gate marker to a
       different agent pair via ``gate: confidence`` on another route
       will compile, but the gate will silently evaluate the wrong (or
       absent) confidence value. Generalising this is tracked as
       follow-up work — pass the upstream agent name as a parameter so
       the lookup is config-driven.
    """
    threshold = cfg.intervention.confidence_threshold
    teams = list(cfg.intervention.escalation_teams)

    async def gate(state: GraphState) -> dict:
        incident = state["incident"]
        # Reload from disk in case earlier nodes wrote tool-driven patches.
        try:
            incident = store.load(incident.id)
        except FileNotFoundError:
            pass
        di_conf = _latest_di_confidence(incident)
        if di_conf is None or di_conf < threshold:
            incident.status = "awaiting_input"
            incident.pending_intervention = {
                "reason": "low_confidence",
                "confidence": di_conf,
                "threshold": threshold,
                "options": ["resume_with_input", "escalate", "stop"],
                "escalation_teams": teams,
            }
            store.save(incident)
            return {"incident": incident, "next_route": "__end__",
                    "last_agent": "gate", "error": None}
        # Confidence met threshold — clear any stale intervention payload.
        if incident.pending_intervention is not None:
            incident.pending_intervention = None
            store.save(incident)
        return {"incident": incident, "next_route": "default",
                "last_agent": "gate", "error": None}

    return gate


def _build_agent_nodes(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                       registry: ToolRegistry) -> dict:
    """Materialize agent nodes from skills + registry. Reused by main + resume graphs."""
    nodes: dict = {}
    for agent_name, skill in skills.items():
        llm = get_llm(
            cfg.llm,
            role=agent_name,
            model=skill.model,
            temperature=skill.temperature,
            stub_canned=_STUB_CANNED,
        )
        tools = registry.get(skill.tools)
        decide = _decide_from_signal
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
        )
    return nodes


def _make_router(gated_edges: dict[tuple[str, str], str]):
    """Build a state router that intercepts gated edges into the gate node.

    Used by both ``build_graph`` and ``build_resume_graph`` — they share
    the same routing semantics, so this factory eliminates duplication.
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


def _make_gate_to(gate_target: str):
    """Build the gate's outbound router. On a "default" pass, the gate
    forwards to its single downstream target; on "__end__", it terminates.
    """
    def _gate_to(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        return gate_target
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


async def build_graph(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                      registry: ToolRegistry):
    """Compile the main LangGraph from configured skills and routes.

    The entry agent is read from ``cfg.orchestrator.entry_agent``. Gate
    insertions are derived from each skill's route rules: a rule with
    ``gate: confidence`` causes the router to redirect ``(this_agent, next)``
    through the ``gate`` node.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.
    """
    entry = cfg.orchestrator.entry_agent
    if entry not in skills:
        raise ValueError(
            f"orchestrator.entry_agent={entry!r} is not a known skill "
            f"(known: {sorted(skills.keys())})"
        )
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))

    sg.set_entry_point(entry)

    _router = _make_router(gated_edges)

    for agent_name in skills.keys():
        possible_targets = {s.name for s in skills.values()} | {END, "gate"}
        # Exclude targets that are intercepted via a gated edge for this agent:
        # the router redirects (agent_name, gated_target) -> "gate", so the
        # gated_target must NOT appear in this agent's target_map. Leaving it
        # in would cause LangGraph to register a visible direct edge in the
        # compiled graph, defeating the structural assertion in the test (and
        # misleading graph visualisations).
        gated_targets_for_agent = {to for (frm, to) in gated_edges if frm == agent_name}
        target_map = {
            name: name
            for name in possible_targets
            if name != END and name not in gated_targets_for_agent
        }
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)

    # Determine where the gate forwards on a "default" pass — there is at
    # most one downstream agent per gated edge; with one gate type today
    # we collapse to a single target. If the user defines multiple gated
    # edges with different downstream agents in the future, the closure
    # below will need a state-aware lookup; for now we assert "exactly one"
    # and error loudly otherwise.
    gate_targets = {to for (_from, to) in gated_edges}
    if len(gate_targets) > 1:
        raise ValueError(
            f"multiple gated downstream targets {sorted(gate_targets)} not "
            f"yet supported; only one gated edge per graph"
        )
    gate_target = next(iter(gate_targets), None)
    if gate_target is not None:
        _gate_to = _make_gate_to(gate_target)
        sg.add_conditional_edges("gate", _gate_to, {
            gate_target: gate_target, END: END,
        })
    else:
        sg.add_edge("gate", END)

    return sg.compile()


async def build_resume_graph(*, cfg: AppConfig, skills: dict,
                             store: IncidentStore, registry: ToolRegistry):
    """Compile a sub-graph that re-runs from the *upstream* end of the
    (single) gated edge through to the gate's downstream target.

    Used by ``Orchestrator.resume_investigation`` after the user supplies
    new context: agents before the gated edge already ran, so we resume
    from the gated agent with the updated incident. Same gate semantics —
    if the new run is still low-confidence, we'll pause again.
    """
    gated_edges = _collect_gated_edges(skills)
    if not gated_edges:
        raise ValueError(
            "build_resume_graph requires at least one route with gate set; "
            "no gated edges were found in the configured skills — add "
            "'gate: confidence' to the relevant agent's route in "
            "config/skills/<agent>/config.yaml"
        )
    if len(gated_edges) > 1:
        raise ValueError(
            f"multiple gated edges {sorted(gated_edges.keys())} not yet "
            f"supported; resume entry is ambiguous"
        )
    resume_from, gate_target = next(iter(gated_edges.keys()))

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name in (resume_from, gate_target):
        if agent_name in nodes:
            sg.add_node(agent_name, nodes[agent_name])
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))
    sg.set_entry_point(resume_from)

    _router = _make_router(gated_edges)

    for agent_name in (resume_from, gate_target):
        sg.add_conditional_edges(agent_name, _router, {
            resume_from: resume_from,
            gate_target: gate_target,
            "gate": "gate",
            END: END,
        })

    _gate_to = _make_gate_to(gate_target)
    sg.add_conditional_edges("gate", _gate_to, {
        gate_target: gate_target, END: END,
    })
    return sg.compile()
