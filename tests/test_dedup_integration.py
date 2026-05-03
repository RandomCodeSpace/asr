"""End-to-end integration test for the P7 dedup pipeline.

Drives a real ``Orchestrator`` and a fake history+LLM to prove the
locked-in P7 contract:

  * The new session is marked ``status="duplicate"``.
  * ``parent_session_id`` points at the prior session.
  * The agent graph is short-circuited (downstream agents never run).
  * ``list_recent`` excludes duplicates by default; opting in surfaces
    them.
  * ``list_children(parent)`` returns the duplicate.
"""
from __future__ import annotations

import pytest

from runtime.config import (
    AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths, RuntimeConfig,
)
from runtime.orchestrator import Orchestrator
from runtime.dedup import (
    DedupConfig, DedupDecision, DedupResult,
)


@pytest.fixture
def cfg(tmp_path):
    # The runtime no longer reaches into examples.incident_management
    # for dedup config — RuntimeConfig.dedup_config_path is left unset
    # (None) so Orchestrator.create skips the pipeline by default. We
    # wire a scripted pipeline directly post-create below.
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="examples.incident_management.mcp_server",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="runtime.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="runtime.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="runtime.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )


class _ScriptedPipeline:
    config = DedupConfig(enabled=True)

    def __init__(self, parent_id: str) -> None:
        self.parent_id = parent_id
        self.calls: list[str] = []

    async def run(self, *, session, history_store):  # noqa: ARG002
        self.calls.append(getattr(session, "id", "?"))
        return DedupResult(
            matched=True,
            parent_session_id=self.parent_id,
            candidate_id=self.parent_id,
            decision=DedupDecision(
                is_duplicate=True, confidence=0.92,
                rationale="payments-latency repeat",
            ),
            stage1_score=0.95,
        )


@pytest.mark.asyncio
async def test_dedup_e2e_marks_duplicate_and_skips_graph(cfg):
    """The full plan §9.3 contract — minus the embedding step which is
    exercised by the unit tests; here we focus on the lifecycle."""
    orch = await Orchestrator.create(cfg)
    try:
        # Session 1: prime a closed parent.
        s1 = await orch.start_session(
            query="payments latency", environment="production",
        )
        prior = orch.store.load(s1)
        prior.status = "resolved"
        orch.store.save(prior)

        # Inject a scripted pipeline that confirms the duplicate.
        scripted = _ScriptedPipeline(parent_id=s1)
        orch.dedup_pipeline = scripted

        # Spy on the graph so we can assert it was bypassed.
        graph_invocations: list[None] = []
        original = orch.graph.ainvoke

        async def _spy(*args, **kwargs):
            graph_invocations.append(None)
            return await original(*args, **kwargs)

        orch.graph.ainvoke = _spy  # type: ignore[assignment]

        # Session 2: should be flagged + linked + graph-skipped.
        s2 = await orch.start_session(
            query="payments latency", environment="production",
        )
        loaded2 = orch.store.load(s2)
        assert loaded2.status == "duplicate"
        assert loaded2.parent_session_id == s1
        assert loaded2.dedup_rationale == "payments-latency repeat"
        # Pipeline got called once for s2 (s1 ran with no pipeline).
        assert scripted.calls == [s2]
        # The graph never ran for s2.
        assert graph_invocations == []
        # No agent_runs because the graph never executed.
        assert loaded2.agents_run == []

        # list_recent default excludes the duplicate.
        recent_default = [i.id for i in orch.store.list_recent(50)]
        assert s1 in recent_default
        assert s2 not in recent_default

        # list_recent(include_duplicates=True) surfaces both.
        recent_all = [
            i.id for i in orch.store.list_recent(50, include_duplicates=True)
        ]
        assert s1 in recent_all
        assert s2 in recent_all

        # list_children(s1) returns the duplicate child.
        children = [i.id for i in orch.store.list_children(s1)]
        assert children == [s2]
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_dedup_e2e_full_flow_with_retraction_round_trip(cfg, tmp_path):
    """End-to-end: dedup → retraction → audit row → list_recent reflects
    the retraction (the formerly-duplicate session shows up again)."""
    orch = await Orchestrator.create(cfg)
    try:
        s1 = await orch.start_session(query="db pool", environment="production")
        prior = orch.store.load(s1)
        prior.status = "resolved"
        orch.store.save(prior)

        orch.dedup_pipeline = _ScriptedPipeline(parent_id=s1)
        s2 = await orch.start_session(query="db pool", environment="production")
        assert orch.store.load(s2).status == "duplicate"
        # Hidden by default.
        assert s2 not in [i.id for i in orch.store.list_recent(50)]

        # Retract.
        retracted = orch.store.un_duplicate(s2, retracted_by="ops",
                                            note="actually different")
        assert retracted.status == "new"
        assert retracted.parent_session_id is None
        # Now visible in default list_recent.
        assert s2 in [i.id for i in orch.store.list_recent(50)]
        # Audit row written.
        from sqlalchemy import select
        from sqlalchemy.orm import Session as SqlSession
        from runtime.storage.models import DedupRetractionRow
        with SqlSession(orch.store.engine) as session:
            rows = session.execute(select(DedupRetractionRow)).scalars().all()
        assert len(rows) == 1
        assert rows[0].original_match_id == s1
        assert rows[0].retracted_by == "ops"
    finally:
        await orch.aclose()
