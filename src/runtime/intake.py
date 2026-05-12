"""Framework default intake runner.

Every app's first step is a ``kind: supervisor`` skill named
``intake``. The skill's ``runner`` field defaults to
``runtime.intake:default_intake_runner`` when not overridden. The
default runner does two generic things, both opt-in:

1. **Prior-similar retrieval** — if ``app_cfg.intake_context.history_store``
   is wired, call ``HistoryStore.find_similar`` with the new session's
   ``to_agent_input()`` text and stash the top-K matches as a small
   list of dicts on ``state.findings['prior_similar']``. Downstream
   agents read this as a hypothesis surface, not a verdict.
2. **Dedup short-circuit** — if ``app_cfg.intake_context.dedup_pipeline``
   is wired and reports a duplicate, the runner stamps
   ``parent_session_id`` and ``status='duplicate'`` on the session and
   returns ``next_route='__end__'`` to skip the rest of the graph.

When neither is wired the runner returns ``None`` and the supervisor
falls through to its dispatch table.

Apps that need additional preparation (memory hydration, plan
loading, etc.) compose the framework default with their own runner
via :func:`compose_runners`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from runtime.state import Session

_log = logging.getLogger("runtime.intake")


@dataclass
class IntakeContext:
    """Optional store handles passed through ``app_cfg.intake_context``.

    The graph builder attaches one of these to the ``app_cfg`` argument
    before invoking the runner so that stateless module-level runners
    can still reach the live stores.
    """

    history_store: Any = None       # Optional[HistoryStore[StateT]]
    dedup_pipeline: Any = None      # Optional[DedupPipeline[StateT]]
    event_log: Any = None           # Optional[EventLog] — M1 telemetry sink
    top_k: int = 3
    similarity_threshold: float = 0.7


def _project_prior(session: Session) -> dict[str, Any]:
    """Compact representation suitable for stashing on findings."""
    return {"id": session.id, "status": session.status}


def default_intake_runner(
    state: Any,
    *,
    app_cfg: Any | None = None,
) -> dict[str, Any] | None:
    """Generic similarity retrieval + dedup short-circuit.

    Returns ``None`` when nothing changed, a dict patch otherwise.
    The patch is merged by ``runtime.agents.supervisor`` and may
    include ``next_route='__end__'`` to short-circuit the graph.

    Called synchronously from the supervisor node. When a running event
    loop is detected (e.g. inside a LangGraph async node), the dedup
    step is skipped with a warning rather than raising ``RuntimeError``.
    """
    import asyncio

    session: Session | None = (
        state.get("session") if hasattr(state, "get") else None
    )
    if session is None:
        return None
    ctx: IntakeContext | None = getattr(app_cfg, "intake_context", None)
    if ctx is None:
        return None

    patch: dict[str, Any] = {}
    text = (session.to_agent_input() or "").strip()

    if ctx.history_store is not None and text:
        hits = ctx.history_store.find_similar(
            query=text,
            filter_kwargs=None,
            limit=ctx.top_k,
            threshold=ctx.similarity_threshold,
        )
        # hits is list[tuple[Session, float]]
        session.findings["prior_similar"] = [_project_prior(h) for h, _ in hits]
        patch["session"] = session

    if ctx.dedup_pipeline is not None:
        try:
            result = asyncio.run(
                ctx.dedup_pipeline.run(
                    session=session,
                    history_store=ctx.history_store,
                )
            )
        except RuntimeError:
            # Already inside a running event loop (e.g. LangGraph async
            # node). Fall through without dedup — the caller must use
            # the async API directly in that context.
            _log.warning(
                "default_intake_runner: asyncio.run() called from a running "
                "event loop; dedup short-circuit skipped for session %s",
                session.id,
            )
            result = None

        if result is not None and getattr(result, "matched", False):
            session.parent_session_id = result.parent_session_id
            session.status = "duplicate"
            rationale = None
            if result.decision is not None:
                rationale = getattr(result.decision, "rationale", None)
            if rationale:
                session.dedup_rationale = rationale
            patch["session"] = session
            patch["next_route"] = "__end__"

    return patch or None


def compose_runners(
    *runners: Callable[..., dict[str, Any] | None],
) -> Callable[..., dict[str, Any] | None]:
    """Chain multiple runners; first one to return ``next_route`` wins.

    Each runner is called in order with the same ``(state, *, app_cfg)``
    signature. Non-route patches are merged left-to-right (later
    runners can overwrite earlier keys, except ``next_route`` which is
    sticky once set).
    """

    def _composed(state: Any, *, app_cfg: Any | None = None) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        for r in runners:
            out = r(state, app_cfg=app_cfg)
            if not out:
                continue
            merged.update({k: v for k, v in out.items() if k != "next_route"})
            if "next_route" in out and "next_route" not in merged:
                merged["next_route"] = out["next_route"]
                # Short-circuit: subsequent runners do not run.
                return merged
        return merged or None

    return _composed


def hydrate_from_memory(
    state: Any,
    *,
    kg_store: Any = None,
    playbook_store: Any = None,
    release_store: Any = None,
    hydrator: Callable[..., Any] | None = None,
    gate: Callable[..., str | None] | None = None,
) -> dict[str, Any] | None:
    """Generic memory-hydration runner shell.

    Apps that wire L2 / L5 / L7 stores via :mod:`runtime.memory` plug
    them in here. The framework supplies the runner-shape contract
    (``state.session`` access, ``next_route='__end__'`` short-circuit,
    duplicate-metadata stamping) so per-app supervisors collapse to:

        from runtime.intake import compose_runners, default_intake_runner
        from runtime.intake import hydrate_from_memory

        def app_hydration(state, *, app_cfg=None):
            return hydrate_from_memory(
                state,
                kg_store=...,
                playbook_store=...,
                release_store=...,
                hydrator=app_specific_hydrate_callable,
                gate=app_specific_gate_callable,  # optional
            )

        default_supervisor_runner = compose_runners(
            default_intake_runner, app_hydration,
        )

    ``hydrator`` signature: ``(session, *, kg_store, playbook_store,
    release_store) -> Any`` where the returned object is stamped on
    ``session.memory`` if that attribute is settable. If ``hydrator``
    is ``None`` the function is a no-op (returns ``None``).

    ``gate`` signature: ``(session, *, kg_store) -> str | None`` —
    return a parent session id to mark the new session as a duplicate
    (caller stamps ``status='duplicate'`` and emits
    ``next_route='__end__'``). When ``None`` is returned (or no gate
    supplied) the session proceeds normally.

    Returns ``None`` when no hydration occurred, otherwise a runner
    patch suitable for merging by ``runtime.agents.supervisor``.
    """
    session = state.get("session") if hasattr(state, "get") else None
    if session is None:
        return None
    if hydrator is None and gate is None:
        return None

    patch: dict[str, Any] = {}

    if hydrator is not None:
        try:
            memory = hydrator(
                session,
                kg_store=kg_store,
                playbook_store=playbook_store,
                release_store=release_store,
            )
        except Exception:  # noqa: BLE001 — defensive, keep graph alive
            _log.exception(
                "hydrate_from_memory: hydrator raised; routing through",
            )
            memory = None

        if memory is not None and hasattr(session, "memory"):
            try:
                session.memory = memory
            except Exception:  # noqa: BLE001 — frozen / read-only field
                _log.warning(
                    "hydrate_from_memory: cannot set session.memory; "
                    "downstream agents will not see hydrated context",
                )
        patch["session"] = session

    if gate is not None:
        try:
            parent = gate(session, kg_store=kg_store)
        except Exception:  # noqa: BLE001 — defensive
            _log.exception(
                "hydrate_from_memory: gate raised; routing through",
            )
            parent = None

        if parent is not None:
            try:
                session.status = "duplicate"
                if hasattr(session, "parent_session_id"):
                    session.parent_session_id = parent
            except Exception:  # noqa: BLE001
                _log.warning(
                    "hydrate_from_memory: cannot stamp duplicate metadata "
                    "on session %s", getattr(session, "id", "?"),
                )
            patch["session"] = session
            patch["next_route"] = "__end__"

    return patch or None
