"""Bootstrap endpoint for the React UI's single view-model.

GET /api/v1/sessions/{id}/full returns everything the UI needs to render
the session in one round-trip — replaces the old pattern of multiple GETs.
The same shape is then patched in place by SSE delta events.

Registered only via :func:`runtime.api.build_app` (requires
``app.state.orchestrator``); unlike :mod:`runtime.api_dedup`, this module
is NOT suitable for lightweight test fixtures that construct a bare
``FastAPI()`` app — use ``build_app(cfg)`` for tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request


def add_routes(api_v1: APIRouter) -> None:
    """Mount the /sessions/{id}/full handler on the api_v1 router."""

    @api_v1.get("/sessions/{session_id}/full")
    async def get_session_full(
        session_id: str, request: Request,
    ) -> dict[str, Any]:
        orch = request.app.state.orchestrator
        try:
            inc = orch.store.load(session_id)
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            # ``ValueError`` covers the SessionStore id-format guard
            # (``Invalid session id ...``); semantically a 404 at the
            # API boundary — same convention as other /sessions/* GETs.
            raise HTTPException(
                status_code=404, detail="session not found",
            ) from e

        # Replay the EventLog backlog. ``vm_seq`` is the high-water mark
        # the UI uses to ?since=N when it later opens the SSE stream, so
        # delta events stitch onto the same view-model without gap or
        # overlap.
        event_log = getattr(orch, "event_log", None)
        events: list[dict[str, Any]] = []
        vm_seq = 0
        if event_log is not None:
            for ev in event_log.iter_for(session_id, since=0):
                events.append({
                    "seq": ev.seq,
                    "kind": ev.kind,
                    "payload": ev.payload,
                    "ts": ev.ts,
                })
                if ev.seq > vm_seq:
                    vm_seq = ev.seq

        # Agent definitions: skill metadata the UI needs to render the
        # graph diagram + per-agent header chips. ``orch.skills`` is a
        # ``dict[str, Skill]`` keyed by name. ``Skill.tools`` is itself
        # a ``dict[str, list[str]]`` (server -> tool list) — expose the
        # server keys as the ref strings; ``Skill.routes`` is a
        # ``list[RouteRule]`` (when/next/gate) — flatten to the
        # signal->next mapping the UI consumes.
        agent_definitions: dict[str, dict[str, Any]] = {}
        for name, skill in orch.skills.items():
            agent_definitions[name] = {
                "name": skill.name,
                "kind": skill.kind,
                "model": skill.model or orch.cfg.llm.default,
                "tools": list(skill.tools or {}),
                "routes": {r.when: r.next for r in skill.routes},
                "system_prompt_excerpt": (skill.system_prompt or "")[:500],
            }

        return {
            "session": inc.model_dump(mode="json"),
            "agents_run": [r.model_dump(mode="json") for r in inc.agents_run],
            "tool_calls": [tc.model_dump(mode="json") for tc in inc.tool_calls],
            "events": events,
            "agent_definitions": agent_definitions,
            "vm_seq": vm_seq,
        }
