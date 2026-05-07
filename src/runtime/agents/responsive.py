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
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from runtime.config import GatewayConfig
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

        # Wrap tools per-invocation so each wrap closes over the
        # live ``Session`` for this run.
        if gateway_cfg is not None:
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store)
                for t in tools
            ]
        else:
            run_tools = tools
        # Phase 10 (FOC-03 / D-10-02): every responsive agent invocation
        # is wrapped in an AgentTurnOutput envelope. LangGraph internally
        # calls llm.with_structured_output(AgentTurnOutput) on a final pass
        # after the tool loop, populating result["structured_response"].
        agent_executor = create_react_agent(
            llm, run_tools, prompt=skill.system_prompt,
            response_format=AgentTurnOutput,
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

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=final_confidence, rationale=final_rationale,
            signal=final_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


__all__ = ["make_agent_node"]
