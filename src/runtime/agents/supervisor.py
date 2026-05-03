"""Supervisor agent kind — no-LLM router (P6-D).

A supervisor skill is a LangGraph node that:

1. Reads the live ``Session`` plus the current dispatch depth.
2. Picks one or more subordinate agents per ``dispatch_strategy``:
   ``rule`` (deterministic, evaluated via the same safe-eval AST that
   gates monitor expressions) or ``llm`` (one short LLM call against
   ``dispatch_prompt``).
3. Emits a structured ``supervisor_dispatch`` log entry (no
   ``AgentRun`` row — supervisors are bookkeeping, not token-burning
   agents).
4. Returns ``next_route`` set to the chosen subordinate (or to
   ``__end__`` when the depth limit is hit).

The recursion depth is tracked in :class:`runtime.graph.GraphState`'s
``dispatch_depth`` field; if a supervisor would exceed
``skill.max_dispatch_depth`` the node aborts with a clean error
instead of recursing forever (R1).

This is **not** a fan-out implementation; we always pick a single
target. Multi-target ``Send()`` is left for a future phase per the
non-goal in §5 of the plan.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from runtime.skill import Skill, _resolve_dotted_callable
from runtime.state import Session, _UTC_TS_FMT

logger = logging.getLogger(__name__)


def _safe_eval(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a pre-validated safe-eval expression against ``ctx``.

    The expression must already have passed
    :func:`runtime.skill._validate_safe_expr` — that's enforced at
    skill-load time. We re-parse here (cheap) and walk the tree
    against the same allowlist; any non-whitelisted node is treated
    as evaluating to ``False`` so a malformed runtime expression can
    never escalate to arbitrary code execution.
    """
    from runtime.skill import _validate_safe_expr
    _validate_safe_expr(expr, source="supervisor.dispatch_rule")
    # ``compile`` + ``eval`` over a built-in-stripped namespace is the
    # cheapest correct evaluator once the AST is whitelisted. The
    # ``__builtins__`` removal blocks ``__import__`` etc. should the
    # AST checker miss something.
    code = compile(expr, "<safe-eval>", "eval")
    return eval(code, {"__builtins__": {}}, ctx)  # noqa: S307 — AST-whitelisted


def _ctx_for_session(incident: Session) -> dict[str, Any]:
    """Build the variable namespace dispatch-rule expressions see.

    Exposes the live session payload as ``session`` plus a few
    ergonomic top-level aliases for fields operators reach for most
    often. Adding new top-level names is a one-liner; the safe-eval
    AST checker already restricts the language so we don't need to
    sandbox the namespace any further.
    """
    payload = incident.model_dump()
    return {
        "session": payload,
        "status": payload.get("status"),
        "agents_run": payload.get("agents_run") or [],
        "tool_calls": payload.get("tool_calls") or [],
    }


def log_supervisor_dispatch(
    *,
    session: Session,
    supervisor: str,
    strategy: str,
    depth: int,
    targets: list[str],
    rule_matched: str | None,
    payload_size: int,
) -> None:
    """Emit one structured ``supervisor_dispatch`` log entry (P6-H).

    Operators wanting an end-to-end audit join ``agent_runs`` and the
    log stream by ``incident_id``. The audit trail is deliberately a
    different stream from ``agent_runs`` because supervisors don't burn
    tokens — bloating ``agents_run`` with router rows is a known trap
    we explicitly avoid (R3).
    """
    record = {
        "event": "supervisor_dispatch",
        "ts": datetime.now(timezone.utc).strftime(_UTC_TS_FMT),
        "incident_id": session.id,
        "session_id": session.id,
        "supervisor": supervisor,
        "strategy": strategy,
        "depth": depth,
        "targets": targets,
        "rule_matched": rule_matched,
        "dispatch_payload_size": payload_size,
    }
    logger.info("supervisor_dispatch %s", json.dumps(record))


def _llm_pick_target(
    *,
    skill: Skill,
    llm: BaseChatModel,
    incident: Session,
) -> str:
    """One-shot LLM dispatch: ask the model to choose a subordinate.

    The model is asked to reply with **only** the name of one
    subordinate. We accept the first matching name in the response
    (case-insensitive substring match) and fall back to the first
    subordinate when the response is unparseable — keeping the graph
    moving rather than failing outright.
    """
    prompt = (
        f"{skill.dispatch_prompt}\n\n"
        f"Choose ONE of: {', '.join(skill.subordinates)}.\n"
        f"Reply with only the agent name."
    )
    payload = json.dumps(incident.model_dump(), default=str)
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content=payload),
    ]
    try:
        result = llm.invoke(msgs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "supervisor %s: LLM dispatch failed (%s); falling back to %s",
            skill.name, exc, skill.subordinates[0],
        )
        return skill.subordinates[0]
    text = (getattr(result, "content", "") or "").strip().lower()
    for name in skill.subordinates:
        if name.lower() in text:
            return name
    logger.warning(
        "supervisor %s: LLM reply %r did not name a subordinate; "
        "falling back to %s", skill.name, text, skill.subordinates[0],
    )
    return skill.subordinates[0]


def _rule_pick_target(
    *,
    skill: Skill,
    incident: Session,
) -> tuple[str, str | None]:
    """Walk dispatch_rules in order; return (target, matched_when).

    Falls back to the first subordinate when no rule matches; the
    fallback case carries ``matched_when=None`` so the audit log can
    distinguish "default" from "rule X matched".
    """
    ctx = _ctx_for_session(incident)
    for rule in skill.dispatch_rules:
        try:
            if bool(_safe_eval(rule.when, ctx)):
                return rule.target, rule.when
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor %s: dispatch_rule %r raised %s; skipping",
                skill.name, rule.when, exc,
            )
    return skill.subordinates[0], None


def _normalize_runner_route(value: Any) -> str:
    """Map runner-supplied route aliases to the canonical graph end token.

    Apps writing runners reach for ``"END"`` / ``"end"`` / ``"__end__"``
    interchangeably; LangGraph's conditional edges only recognise
    ``"__end__"``. Normalising here keeps the runner contract permissive
    without spreading the alias check across the graph layer.
    """
    if isinstance(value, str) and value.strip().lower() in {"end", "__end__"}:
        return "__end__"
    return value


def make_supervisor_node(
    *,
    skill: Skill,
    llm: BaseChatModel | None = None,
    framework_cfg: Any | None = None,
):
    """Build the supervisor LangGraph node (P6-D, P9-9h).

    Pure routing: no ``AgentRun`` row, no tool execution, no token
    accounting beyond what the optional LLM call itself reports. The
    node sets ``state["next_route"]`` to a subordinate name and returns;
    LangGraph's conditional edges fan out to that node from there.

    The optional ``llm`` is only used when ``skill.dispatch_strategy``
    is ``"llm"``. Callers using ``"rule"`` may pass ``None``.

    P9-9h — when ``skill.runner`` is set, the dotted-path callable is
    resolved at build time and invoked at the start of each node call
    BEFORE the routing dispatch. The runner gets the live ``GraphState``
    and the optional ``framework_cfg`` and may return ``None`` (continue
    with the routing table) or a dict patch that gets merged into state.
    A patch carrying ``"next_route"`` short-circuits the routing table
    entirely (use ``"__end__"`` to terminate the graph).
    """
    # Local import to avoid the circular runtime.graph -> runtime.agents
    # cycle at module-load time.
    from runtime.graph import GraphState

    if skill.kind != "supervisor":
        raise ValueError(
            f"make_supervisor_node called with non-supervisor skill "
            f"{skill.name!r} (kind={skill.kind!r})"
        )

    runner: Callable[..., Any] | None = None
    if skill.runner is not None:
        if callable(skill.runner):
            # Test stubs and composed runners may supply a live callable
            # directly rather than a dotted-path string. Access via the
            # class __dict__ to avoid Python binding it as an instance
            # method when the skill is a plain object (not a Pydantic model).
            raw = vars(type(skill)).get("runner", skill.runner)
            runner = raw if callable(raw) else skill.runner
        else:
            # Resolved a second time here so a runner that fails to import
            # at graph-build time still surfaces a clear error. The skill
            # validator catches most issues at YAML load; this is belt-and-
            # braces and also gives us the live callable to invoke.
            runner = _resolve_dotted_callable(
                skill.runner, source=f"supervisor {skill.name!r} runner"
            )

    async def node(state: GraphState) -> dict:
        sess: Session = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        # ``dispatch_depth`` is an extension field on GraphState
        # (Phase-6); start at 0 and increment per supervisor entry.
        depth = int(state.get("dispatch_depth") or 0) + 1
        if depth > skill.max_dispatch_depth:
            logger.warning(
                "supervisor %s: dispatch depth %d exceeds limit %d; aborting",
                skill.name, depth, skill.max_dispatch_depth,
            )
            return {
                "session": sess,
                "next_route": "__end__",
                "last_agent": skill.name,
                "dispatch_depth": depth,
                "error": (
                    f"supervisor {skill.name!r}: max_dispatch_depth "
                    f"{skill.max_dispatch_depth} exceeded"
                ),
            }

        # ----- P9-9h: app-supplied runner hook ------------------------
        runner_patch: dict[str, Any] = {}
        if runner is not None:
            # Build a thin proxy so the runner can reach intake_context
            # (and any other framework_cfg attributes) without needing
            # framework_cfg to be mutable. The proxy exposes intake_context
            # directly and falls back to framework_cfg for all other attrs.
            _app_cfg_proxy = type("_RunnerAppCfg", (), {
                "intake_context": getattr(framework_cfg, "intake_context", None),
                "__getattr__": lambda self, name: getattr(framework_cfg, name),
            })()
            try:
                result = runner(state, app_cfg=_app_cfg_proxy)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "supervisor %s: runner %s raised; aborting to __end__",
                    skill.name, skill.runner,
                )
                return {
                    "session": sess,
                    "next_route": "__end__",
                    "last_agent": skill.name,
                    "dispatch_depth": depth,
                    "error": (
                        f"supervisor {skill.name!r}: runner failed: {exc}"
                    ),
                }
            if isinstance(result, dict):
                runner_patch = dict(result)
            elif result is not None:
                logger.warning(
                    "supervisor %s: runner returned %s (expected dict|None); "
                    "ignoring", skill.name, type(result).__name__,
                )
            override = runner_patch.pop("next_route", None)
            if override is not None:
                # Short-circuit: skip the routing table entirely. Audit
                # log still fires so operators can trace the decision.
                target = _normalize_runner_route(override)
                # Pick up any fresh reference the runner returned.
                sess = runner_patch.get("session", sess)
                try:
                    payload_size = len(
                        json.dumps(sess.model_dump(), default=str)
                    )
                except Exception:  # noqa: BLE001 — defensive
                    payload_size = 0
                log_supervisor_dispatch(
                    session=sess,
                    supervisor=skill.name,
                    strategy=f"runner:{skill.runner}",
                    depth=depth,
                    targets=[target],
                    rule_matched=None,
                    payload_size=payload_size,
                )
                out: dict[str, Any] = {
                    "session": sess,
                    "next_route": target,
                    "last_agent": skill.name,
                    "dispatch_depth": depth,
                    "error": None,
                }
                # Merge any non-route keys the runner returned (e.g.
                # extra GraphState fields apps want to carry forward).
                for k, v in runner_patch.items():
                    if k not in out:
                        out[k] = v
                return out
            # No override: fold any payload mutation back so the
            # routing table sees the up-to-date object.
            if "session" in runner_patch:
                sess = runner_patch["session"]

        rule_matched: str | None = None
        if skill.dispatch_strategy == "rule":
            target, rule_matched = _rule_pick_target(skill=skill, incident=sess)
        else:  # "llm"
            if llm is None:
                logger.warning(
                    "supervisor %s: strategy=llm but no llm provided; "
                    "falling back to first subordinate", skill.name,
                )
                target = skill.subordinates[0]
            else:
                target = _llm_pick_target(skill=skill, llm=llm, incident=sess)

        # Audit: one structured log entry per dispatch (R3).
        try:
            payload_size = len(json.dumps(sess.model_dump(), default=str))
        except Exception:  # noqa: BLE001 — defensive; size is a hint
            payload_size = 0
        log_supervisor_dispatch(
            session=sess,
            supervisor=skill.name,
            strategy=skill.dispatch_strategy,
            depth=depth,
            targets=[target],
            rule_matched=rule_matched,
            payload_size=payload_size,
        )

        out: dict[str, Any] = {
            "session": sess,
            "next_route": target,
            "last_agent": skill.name,
            "dispatch_depth": depth,
            "error": None,
        }
        # Carry through any extra keys the runner emitted that the
        # framework didn't consume itself (e.g. memory snapshots).
        for k, v in runner_patch.items():
            if k not in out:
                out[k] = v
        return out

    return node


__all__ = ["make_supervisor_node", "log_supervisor_dispatch"]
