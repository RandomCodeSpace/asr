"""Framework default intake runner — generic similarity retrieval and dedup gate.

These tests pin the contract that ANY app gets when it declares a
``kind: supervisor`` skill without an explicit ``runner:``: prior
similar closed sessions are surfaced under
``state.findings['prior_similar']`` so downstream agents can reason
about them.
"""
from __future__ import annotations

import asyncio
from typing import Any


from runtime.intake import IntakeContext, default_intake_runner, compose_runners
from runtime.state import Session


def _mk_session(sid: str = "S-001") -> Session:
    return Session(
        id=sid,
        status="in_progress",
        created_at="2026-05-03T10:00:00Z",
        updated_at="2026-05-03T10:00:00Z",
    )


class _StubHistoryStore:
    """Stub matching HistoryStore.find_similar(*, query, filter_kwargs, limit, threshold)."""

    def __init__(self, hits: list[Session]) -> None:
        self._hits = hits
        self.calls: list[dict[str, Any]] = []

    def find_similar(
        self,
        *,
        query: str,
        filter_kwargs: dict | None = None,
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[Session, float]]:
        self.calls.append(
            {"query": query, "filter_kwargs": filter_kwargs,
             "limit": limit, "threshold": threshold}
        )
        return [(h, 0.9) for h in self._hits]


# ---------------------------------------------------------------------------
# Task 1: happy path — populate findings["prior_similar"]
# ---------------------------------------------------------------------------

def test_default_intake_runner_populates_prior_similar() -> None:
    prior = _mk_session("S-PRIOR")
    history = _StubHistoryStore(hits=[prior])
    state = {"session": _mk_session("S-NEW")}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=None,
        top_k=3, similarity_threshold=0.7,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)

    assert patch is not None
    assert "session" in patch
    assert patch["session"].findings["prior_similar"] == [
        {"id": "S-PRIOR", "status": "in_progress"}
    ]
    # Runner forwarded the configured top_k / threshold.
    assert history.calls[0]["limit"] == 3
    assert history.calls[0]["threshold"] == 0.7


# ---------------------------------------------------------------------------
# Task 2: no-op without context
# ---------------------------------------------------------------------------

def test_default_intake_runner_noop_without_context() -> None:
    """Missing app_cfg or missing intake_context => no-op (returns None)."""
    state = {"session": _mk_session()}

    assert default_intake_runner(state, app_cfg=None) is None

    class _NoCtx: ...

    assert default_intake_runner(state, app_cfg=_NoCtx()) is None


def test_default_intake_runner_noop_when_history_store_none() -> None:
    """intake_context with history_store=None => no patch, no crash."""
    state = {"session": _mk_session()}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=None, dedup_pipeline=None,
    )})()

    assert default_intake_runner(state, app_cfg=app_cfg) is None


# ---------------------------------------------------------------------------
# Task 3: dedup short-circuit
# ---------------------------------------------------------------------------

class _StubDedupPipeline:
    """Stub matching DedupPipeline.run(*, session, history_store) -> DedupResult."""

    def __init__(self, *, parent_session_id: str | None, rationale: str | None) -> None:
        self._parent = parent_session_id
        self._rationale = rationale

    async def run(self, *, session: Session, history_store: Any) -> Any:
        from runtime.dedup import DedupResult, DedupDecision
        decision = None
        if self._parent is not None and self._rationale:
            decision = DedupDecision(
                is_duplicate=True,
                confidence=0.95,
                rationale=self._rationale,
            )
        return DedupResult(
            matched=self._parent is not None,
            parent_session_id=self._parent,
            decision=decision,
        )


def test_default_intake_runner_short_circuits_on_dedup_hit() -> None:
    new_session = _mk_session("S-NEW")
    history = _StubHistoryStore(hits=[])
    pipeline = _StubDedupPipeline(parent_session_id="S-PRIOR", rationale="same outage")
    state = {"session": new_session}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=pipeline,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)

    assert patch is not None
    assert patch["next_route"] == "__end__"
    assert patch["session"].parent_session_id == "S-PRIOR"
    assert patch["session"].status == "duplicate"
    assert patch["session"].dedup_rationale == "same outage"


def test_default_intake_runner_no_short_circuit_when_not_duplicate() -> None:
    new_session = _mk_session("S-NEW")
    history = _StubHistoryStore(hits=[])
    pipeline = _StubDedupPipeline(parent_session_id=None, rationale=None)
    state = {"session": new_session}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=pipeline,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)

    # No history hits, no dedup hit => nothing to write back.
    assert patch is None or "next_route" not in patch
    assert new_session.parent_session_id is None
    assert new_session.status == "in_progress"


def test_default_intake_runner_falls_through_under_running_loop() -> None:
    """When called from within a running event loop, dedup falls through
    gracefully (no asyncio.run RuntimeError propagates to caller)."""
    new_session = _mk_session("S-LOOP")
    history = _StubHistoryStore(hits=[])
    pipeline = _StubDedupPipeline(parent_session_id="S-PRIOR", rationale="same outage")
    state = {"session": new_session}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=pipeline,
    )})()

    # Run the sync runner from inside a running event loop by using a
    # thread to host a fresh loop, then call the runner there.
    import threading

    result: dict[str, Any] = {}
    error: list[Exception] = []

    def _run_in_new_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _inner():
            # At this point there IS a running event loop.
            # asyncio.run() inside default_intake_runner must not raise.
            return default_intake_runner(state, app_cfg=app_cfg)

        try:
            result["patch"] = loop.run_until_complete(_inner())
        except Exception as exc:
            error.append(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_run_in_new_loop)
    t.start()
    t.join()

    assert not error, f"Unexpected error: {error[0]}"
    # Under a running loop the dedup short-circuit is skipped; patch may
    # be None (no history hits, dedup skipped) or a partial patch.
    patch = result["patch"]
    assert patch is None or "next_route" not in patch


# ---------------------------------------------------------------------------
# Task 4: compose_runners semantics
# ---------------------------------------------------------------------------

def test_compose_runners_merges_non_route_patches() -> None:
    def r1(state, *, app_cfg=None):
        return {"a": 1}

    def r2(state, *, app_cfg=None):
        return {"b": 2}

    composed = compose_runners(r1, r2)
    assert composed({}) == {"a": 1, "b": 2}


def test_compose_runners_short_circuits_on_first_route() -> None:
    seen = []

    def r1(state, *, app_cfg=None):
        seen.append("r1")
        return {"x": 1, "next_route": "__end__"}

    def r2(state, *, app_cfg=None):
        seen.append("r2")
        return {"y": 2}

    composed = compose_runners(r1, r2)
    out = composed({})
    assert out == {"x": 1, "next_route": "__end__"}
    assert seen == ["r1"]  # r2 never ran


def test_compose_runners_returns_none_when_all_runners_no_op() -> None:
    composed = compose_runners(lambda *a, **k: None, lambda *a, **k: None)
    assert composed({}) is None


# ---------------------------------------------------------------------------
# Task 6: supervisor node passes intake_context to runner via app_cfg
# ---------------------------------------------------------------------------


from runtime.agents.supervisor import make_supervisor_node  # noqa: E402


def test_supervisor_node_passes_intake_context_to_runner() -> None:
    """The supervisor node must read intake_context off framework_cfg
    and forward it to the runner via app_cfg, so module-level runners
    like default_intake_runner can reach the live stores."""
    history = _StubHistoryStore(hits=[])
    captured: dict[str, Any] = {}

    def _runner(state, *, app_cfg=None):
        captured["app_cfg"] = app_cfg
        return None

    skill = type("Skill", (), {
        "name": "intake",
        "kind": "supervisor",
        "subordinates": ["triage"],
        "dispatch_strategy": "rule",
        "dispatch_rules": [],
        "runner": _runner,
        "max_dispatch_depth": 2,
    })()
    framework_cfg = type("FwkCfg", (), {
        "intake_context": IntakeContext(history_store=history, dedup_pipeline=None),
    })()

    node = make_supervisor_node(
        skill=skill,
        framework_cfg=framework_cfg,
    )
    asyncio.run(node({"session": _mk_session(), "dispatch_depth": 0}))

    assert captured["app_cfg"].intake_context.history_store is history


# ---------------------------------------------------------------------------
# M6: default_intake_runner reads from LessonStore alongside HistoryStore
# ---------------------------------------------------------------------------

class _StubLessonRow:
    """Quack-typed SessionLessonRow stand-in for the test."""

    def __init__(
        self,
        *,
        id: str,
        outcome_summary: str,
        tools: list[str],
        source_session_id: str = "SES-PRIOR",
    ) -> None:
        self.id = id
        self.outcome_summary = outcome_summary
        self.tool_sequence = [{"tool": t} for t in tools]
        # M9: intake filters lessons whose source row is soft-deleted.
        # The default value here points at a non-existent SQL row, so
        # the in-memory engine returns "live" via the fallback path.
        self.source_session_id = source_session_id


class _StubLessonStore:
    """Stub matching LessonStore.find_similar(query, limit, threshold)."""

    def __init__(self, hits: list[_StubLessonRow]) -> None:
        self._hits = hits
        self.calls: list[dict[str, Any]] = []

    def find_similar(
        self, *, query: str, limit: int = 3, threshold: float | None = None,
    ) -> list[tuple[_StubLessonRow, float]]:
        self.calls.append({"query": query, "limit": limit, "threshold": threshold})
        return [(h, 0.87) for h in self._hits]


def test_default_intake_runner_populates_lessons() -> None:
    """M6: when lesson_store is wired, the runner stamps findings["lessons"]
    with {id, summary, tools} for every hit. prior_similar continues to
    populate from history_store; the two surfaces coexist."""
    prior = _mk_session("S-PRIOR")
    history = _StubHistoryStore(hits=[prior])
    lessons = _StubLessonStore(hits=[
        _StubLessonRow(
            id="L-1", outcome_summary="rolled back bad deploy",
            tools=["get_logs", "rollback_deploy"],
        ),
        _StubLessonRow(
            id="L-2", outcome_summary="restarted unhealthy pod",
            tools=["restart_pod"],
        ),
    ])
    state = {"session": _mk_session("S-NEW")}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=None,
        lesson_store=lessons,
        top_k=3, similarity_threshold=0.7,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)

    assert patch is not None
    sess = patch["session"]
    # prior_similar still populated.
    assert sess.findings["prior_similar"] == [
        {"id": "S-PRIOR", "status": "in_progress"}
    ]
    # lessons stamped with the expected shape and ordering.
    assert sess.findings["lessons"] == [
        {"id": "L-1", "summary": "rolled back bad deploy",
         "tools": ["get_logs", "rollback_deploy"]},
        {"id": "L-2", "summary": "restarted unhealthy pod",
         "tools": ["restart_pod"]},
    ]
    # find_similar received the configured top_k / threshold.
    assert lessons.calls[0]["limit"] == 3
    assert lessons.calls[0]["threshold"] == 0.7


def test_default_intake_runner_skips_lessons_when_store_absent() -> None:
    """No lesson_store -> no findings["lessons"] key. prior_similar
    still populates."""
    history = _StubHistoryStore(hits=[_mk_session("S-PRIOR")])
    state = {"session": _mk_session("S-NEW")}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=None,
        lesson_store=None,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)
    assert patch is not None
    assert "lessons" not in patch["session"].findings
    assert "prior_similar" in patch["session"].findings


def test_default_intake_runner_dedup_short_circuits_with_lessons() -> None:
    """When both lesson_store + dedup_pipeline are wired and dedup
    fires, the dedup short-circuit still wins — but lessons (and
    prior_similar) get populated first as side-effects, so the
    operator UI showing the duplicate can still surface them."""
    new_session = _mk_session("S-NEW")
    history = _StubHistoryStore(hits=[_mk_session("S-PRIOR")])
    lessons = _StubLessonStore(hits=[
        _StubLessonRow(id="L-9", outcome_summary="ok", tools=["t"]),
    ])
    pipeline = _StubDedupPipeline(
        parent_session_id="S-PRIOR", rationale="same outage",
    )
    state = {"session": new_session}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=pipeline,
        lesson_store=lessons,
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)
    assert patch is not None
    # Dedup wins.
    assert patch["next_route"] == "__end__"
    assert patch["session"].status == "duplicate"
    # Lessons + prior_similar were populated before the short-circuit.
    assert patch["session"].findings.get("lessons") == [
        {"id": "L-9", "summary": "ok", "tools": ["t"]},
    ]
    assert "prior_similar" in patch["session"].findings


def test_default_intake_runner_lesson_failure_is_non_fatal() -> None:
    """A raising lesson_store doesn't break the intake runner —
    findings["lessons"] is set to []."""
    class _RaisingLessonStore:
        def find_similar(self, **kwargs):
            raise RuntimeError("vector backend down")

    history = _StubHistoryStore(hits=[])
    state = {"session": _mk_session("S-NEW")}
    app_cfg = type("AC", (), {"intake_context": IntakeContext(
        history_store=history, dedup_pipeline=None,
        lesson_store=_RaisingLessonStore(),
    )})()

    patch = default_intake_runner(state, app_cfg=app_cfg)
    assert patch is not None
    assert patch["session"].findings["lessons"] == []
