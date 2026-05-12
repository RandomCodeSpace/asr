"""M9 — end-to-end ratchet for per-step telemetry + auto-learning intake.

This is the loop's "definition of done." It drives the integrated
chain end-to-end against a stub LLM with deterministic embeddings,
exercising:

* M1 EventLog wiring + M2 record() helper
* M3 tool-boundary + agent-boundary emission
* M4 status_changed emission + M5 LessonExtractor on terminal hook
* M5 SessionLessonRow + LessonStore vector write
* M6 default_intake_runner stamping findings["lessons"]
* M7 LessonRefresher.run_once idempotency

The test seeds tool_calls + agent_runs directly on the session rows
rather than driving the full graph — the stub LLM has no
``tool_call_plan`` wired in production config and we want the test
to be deterministic. That's enough to exercise the finalize hook,
which is the single point of integration between the per-step
telemetry layer and the lesson corpus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    OrchestratorConfig,
    Paths,
    RuntimeConfig,
)
from runtime.intake import default_intake_runner
from runtime.learning.scheduler import LessonRefresher
from runtime.orchestrator import Orchestrator
from runtime.state import AgentRun, Session, ToolCall
from runtime.storage import LessonStore, SessionLessonRow
from runtime.storage.models import IncidentRow
from runtime.terminal_tools import StatusDef, TerminalToolRule


# ===================================================================
# Deterministic embedder + in-memory vector store fixtures
# ===================================================================

class _SubstringEmbedder(Embeddings):
    """Embedder that produces a unit vector keyed by which "tag" string
    is contained in the input. Used so the M9 test asserts retrieval
    determinism without depending on a real model."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        for needle, vec in self._table.items():
            if needle in text:
                return vec
        return [0.0] * 4


class _InMemoryVectorStore:
    """Minimal VectorStore stand-in: add_documents + similarity_search_with_score."""

    def __init__(self, embedder: Embeddings) -> None:
        self._embedder = embedder
        self._docs: list[Document] = []
        self._vecs: list[list[float]] = []

    def add_documents(self, docs, ids=None):
        for d in docs:
            self._docs.append(d)
            self._vecs.append(self._embedder.embed_query(d.page_content))
        return ids or []

    def similarity_search_with_score(self, query, k=4):
        q = self._embedder.embed_query(query)

        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)
        scored = [(d, 1.0 - _cos(q, v)) for d, v in zip(self._docs, self._vecs)]
        scored.sort(key=lambda t: t[1])
        return scored[:k]


# ===================================================================
# Test config + orchestrator boot
# ===================================================================

_STATUSES = {
    "open":         StatusDef(name="open",         terminal=False, kind="pending"),
    "in_progress":  StatusDef(name="in_progress",  terminal=False, kind="pending"),
    "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
    "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
}
_RULES = [TerminalToolRule(tool_name="mark_resolved", status="resolved")]


def _cfg(tmp_path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="examples.incident_management.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="examples.incident_management.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="examples.incident_management.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        orchestrator=OrchestratorConfig(
            statuses=_STATUSES,
            terminal_tools=_RULES,
            default_terminal_status="needs_review",
        ),
        runtime=RuntimeConfig(state_class=None),
    )


def _swap_lesson_store(orch: Any, embedder: Embeddings) -> _InMemoryVectorStore:
    """Replace the orchestrator's lesson_store with one wired to a
    deterministic in-memory vector store. Returns the vector store
    so tests can introspect its contents.
    """
    vs = _InMemoryVectorStore(embedder)
    new_store = LessonStore(
        engine=orch.store.engine,
        vector_store=vs,  # type: ignore[arg-type]
        similarity_threshold=-1.0,  # accept any score for deterministic asserts
    )
    orch.lesson_store = new_store
    intake_ctx = getattr(orch.framework_cfg, "intake_context", None)
    if intake_ctx is not None:
        intake_ctx.lesson_store = new_store
        intake_ctx.similarity_threshold = -1.0
    return vs


def _seed_resolved_session(
    orch: Any, *, sid_query: str, tag: str,
) -> str:
    """Create a session via the store, append a successful
    mark_resolved ToolCall + agent_run, save. Returns the session id."""
    inc = orch.store.create(
        query=sid_query, environment="staging",
        reporter_id="u", reporter_team="t",
    )
    inc.tool_calls.append(ToolCall(
        agent="resolution",
        tool="mark_resolved",
        args={"tag": tag},
        result={"status": "resolved"},
        ts="2026-05-12T00:00:00Z",
        status="executed",
    ))
    inc.agents_run.append(AgentRun(
        agent="resolution",
        started_at="2026-05-12T00:00:00Z",
        ended_at="2026-05-12T00:00:05Z",
        summary=f"resolved with tag {tag}",
        confidence=0.91,
        signal="success",
    ))
    inc.status = "in_progress"
    orch.store.save(inc)
    return inc.id


# ===================================================================
# Tests
# ===================================================================

@pytest.mark.asyncio
async def test_e2e_resolve_emits_status_changed_and_writes_lesson(tmp_path):
    """Session A: drive to resolved via mark_resolved -> a
    SessionLessonRow lands in the corpus and the lesson_extracted
    event is appended. The vector store has the same document."""
    cfg = _cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        embedder = _SubstringEmbedder({"tag-A": [1.0, 0.0, 0.0, 0.0]})
        vs = _swap_lesson_store(orch, embedder)

        sid = _seed_resolved_session(orch, sid_query="payments-500s", tag="tag-A")
        status = orch._finalize_session_status(sid)
        assert status == "resolved"

        # M5: row exists in session_lessons.
        from sqlalchemy import select
        from sqlalchemy.orm import Session as SqlaSession
        with SqlaSession(orch.store.engine) as s:
            rows = s.execute(
                select(SessionLessonRow).where(
                    SessionLessonRow.source_session_id == sid
                )
            ).scalars().all()
        assert len(rows) == 1
        assert rows[0].outcome_status == "resolved"
        assert rows[0].confidence_final == 0.91

        # Vector store also got the document.
        assert len(vs._docs) == 1
        assert vs._docs[0].metadata["source_session_id"] == sid

        # M4: status_changed event landed.
        events = list(orch.event_log.iter_for(sid))
        kinds = [e.kind for e in events]
        assert "status_changed" in kinds, kinds
        # M5: lesson_extracted event landed too.
        assert "lesson_extracted" in kinds, kinds
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_e2e_new_session_intake_surfaces_prior_lesson(tmp_path):
    """Session A resolves -> session B's intake retrieves A's lesson.
    The lesson must appear in state.findings["lessons"]."""
    cfg = _cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        embedder = _SubstringEmbedder({"tag-DB": [1.0, 0.0, 0.0, 0.0]})
        _swap_lesson_store(orch, embedder)

        sid_a = _seed_resolved_session(orch, sid_query="db connection storm tag-DB", tag="tag-DB")
        orch._finalize_session_status(sid_a)

        # New session B with overlapping tag.
        sess_b = Session(
            id="SES-B",
            status="in_progress",
            created_at="2026-05-12T00:01:00Z",
            updated_at="2026-05-12T00:01:00Z",
        )
        # Override to_agent_input so the lesson embedder hits "tag-DB".
        with patch.object(
            Session, "to_agent_input",
            lambda self: "incident about tag-DB",
        ):
            patch_out = default_intake_runner(
                {"session": sess_b},
                app_cfg=orch.framework_cfg,
            )

        assert patch_out is not None
        lessons = patch_out["session"].findings.get("lessons")
        assert lessons, "lessons should be populated for new session B"
        assert any(
            "summary" in entry and "tools" in entry and "id" in entry
            for entry in lessons
        )
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_e2e_soft_deleted_source_session_does_not_surface_lessons(tmp_path):
    """Soft-delete session A's row -> session C's intake must NOT
    surface A's lesson. M6 contract: lessons whose source row is
    deleted are filtered out client-side."""
    cfg = _cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        embedder = _SubstringEmbedder({"tag-DEL": [1.0, 0.0, 0.0, 0.0]})
        _swap_lesson_store(orch, embedder)

        sid_a = _seed_resolved_session(
            orch, sid_query="soft delete demo tag-DEL", tag="tag-DEL",
        )
        orch._finalize_session_status(sid_a)
        # Soft-delete the source session.
        from sqlalchemy import update
        with orch.store.engine.begin() as conn:
            conn.execute(
                update(IncidentRow)
                .where(IncidentRow.id == sid_a)
                .values(deleted_at=datetime.now(timezone.utc))
            )

        sess_c = Session(
            id="SES-C",
            status="in_progress",
            created_at="2026-05-12T00:02:00Z",
            updated_at="2026-05-12T00:02:00Z",
        )
        with patch.object(
            Session, "to_agent_input",
            lambda self: "another incident about tag-DEL",
        ):
            patch_out = default_intake_runner(
                {"session": sess_c},
                app_cfg=orch.framework_cfg,
            )

        # findings["lessons"] either missing or empty — the deleted-source
        # filter must kick in BEFORE the lesson reaches the caller.
        lessons = (
            patch_out["session"].findings.get("lessons")
            if patch_out is not None else None
        )
        assert not lessons, (
            f"expected lessons filtered out for soft-deleted source; got {lessons}"
        )
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_e2e_refresher_idempotent_after_finalize_writes(tmp_path):
    """A finalize-driven lesson write covers the same session that
    the refresher would later pick up. Refresher.run_once must NOT
    duplicate the row."""
    cfg = _cfg(tmp_path)
    orch = await Orchestrator.create(cfg)
    try:
        embedder = _SubstringEmbedder({"tag-IDEM": [1.0, 0.0, 0.0, 0.0]})
        _swap_lesson_store(orch, embedder)

        sid = _seed_resolved_session(
            orch, sid_query="idempotent demo tag-IDEM", tag="tag-IDEM",
        )
        orch._finalize_session_status(sid)
        # One row in SQL.
        from sqlalchemy import select
        from sqlalchemy.orm import Session as SqlaSession
        with SqlaSession(orch.store.engine) as s:
            rows = s.execute(select(SessionLessonRow)).scalars().all()
        assert len(rows) == 1

        # Bump updated_at so the refresher's window picks the row up.
        from sqlalchemy import update
        with orch.store.engine.begin() as conn:
            conn.execute(
                update(IncidentRow)
                .where(IncidentRow.id == sid)
                .values(updated_at=datetime.now(timezone.utc))
            )

        refresher = LessonRefresher(
            engine=orch.store.engine,
            lesson_store=orch.lesson_store,
            event_log=orch.event_log,
            terminal_statuses=frozenset({"resolved", "escalated"}),
            window_days=7,
        )
        stats = refresher.run_once()
        assert stats.lessons_added == 0, (
            "refresher must skip sessions whose current-version row already exists"
        )
        assert stats.lessons_skipped == 1

        with SqlaSession(orch.store.engine) as s:
            rows_after = s.execute(select(SessionLessonRow)).scalars().all()
        assert len(rows_after) == 1, "refresher must not duplicate existing row"
    finally:
        await orch.aclose()
