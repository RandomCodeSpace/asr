"""Responsive agent kind — the today-default LLM agent.

A responsive skill is a LangGraph node that:

1. Builds a ReAct executor over the skill's ``tools`` and ``model``.
2. Invokes the executor with the live ``Session`` payload as a human
   message preamble.
3. Records ``ToolCall`` and ``AgentRun`` rows on the session, harvests
   the agent's confidence / signal / rationale, and decides the next
   route from ``skill.routes``.

This module owns only the node-factory entrypoint
(``make_agent_node``); the implementation reuses helpers in
:mod:`runtime.graph` so existing call sites and the gate node continue
to work unchanged. Supervisor and monitor factories live alongside it
under :mod:`runtime.agents` rather than piling more kinds into
``graph.py``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from langgraph.errors import GraphInterrupt

from runtime.config import GatePolicy, GatewayConfig
from runtime.skill import Skill
from runtime.state import Session, _UTC_TS_FMT
from runtime.storage.session_store import SessionStore
from runtime.tools.gateway import wrap_tool
from runtime.agents.turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
    reconcile_confidence,
)

if TYPE_CHECKING:
    from runtime.storage.event_log import EventLog

logger = logging.getLogger(__name__)


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
    gate_policy: "GatePolicy | None" = None,
    event_log: "EventLog | None" = None,
):
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route.

    ``valid_signals`` is the orchestrator-wide accepted signal vocabulary
    (``cfg.orchestrator.signals``). When omitted, the legacy
    ``{success, failed, needs_input}`` default is used so older callers and
    tests keep working.

    ``gateway_cfg`` is the optional risk-rated tool gateway config.
    When supplied, every ``BaseTool`` in ``tools`` is wrapped via
    :func:`runtime.tools.gateway.wrap_tool` *inside the node body* so the
    closure captures the live ``Session`` per agent invocation. When
    ``None``, tools are passed through untouched.
    """
    # Imported lazily to avoid an import cycle: ``runtime.graph`` depends
    # on this module via ``_build_agent_nodes``, but the helpers used
    # inside the node body live in ``graph`` so we keep a single
    # implementation for the responsive path. The cycle is benign at
    # call time — both modules are fully imported before ``node()`` runs.
    from runtime.graph import (
        GraphState,
        _ainvoke_with_retry,
        _format_agent_input,
        _handle_agent_failure,
        _harvest_tool_calls_and_patches,
        _pair_tool_responses,
        _extract_final_text,
        _first_terminal_tool_called_this_turn,
        _sum_token_usage,
        _record_success_run,
        route_from_skill,
    )

    async def node(state: GraphState) -> dict:
        incident: Session = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # M3: emit agent_started telemetry before any work happens.
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

        # Wrap tools per-invocation so each wrap closes over the
        # live ``Session`` for this run.
        if gateway_cfg is not None:
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store,
                          gate_policy=gate_policy,
                          event_log=event_log)
                for t in tools
            ]
        else:
            run_tools = tools
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

        # Phase 11 (FOC-04): reset per-turn confidence hint at the
        # start of each agent step so the gateway treats the first
        # tool call of the turn as "no signal yet".
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
            # Phase 11 (FOC-04 / D-11-04): HITL pause -- propagate up.
            raise
        except Exception as exc:  # noqa: BLE001
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        # Tools (e.g. registered patch tools) write straight to disk.
        # Reload so the node's own append of agent_run + tool_calls
        # happens against the tool-mutated state.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
        )
        # Phase 11 (FOC-04): update hint so any subsequent in-turn
        # tool call sees the harvested confidence.
        if agent_confidence is not None:
            try:
                incident.turn_confidence_hint = agent_confidence
            except (AttributeError, ValueError):
                pass
        _pair_tool_responses(messages, incident)

        # Phase 10 (FOC-03 / D-10-03): parse envelope; reconcile against
        # any typed-terminal-tool-arg confidence. Envelope failure is a
        # structured agent_run error.
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

        final_text = envelope.content or _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        # M3: emit confidence_emitted after reconcile_confidence + signal
        # harvest land, before _record_success_run persists the agent_run.
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
            confidence=final_confidence, rationale=final_rationale,
            signal=final_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)

        # M3: emit route_decided + agent_finished. agent_finished carries
        # the token_usage harvested by _sum_token_usage above so the
        # session-level telemetry has per-step counts.
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


__all__ = ["make_agent_node"]
