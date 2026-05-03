"""Integration tests for the P7-F dedup_check lifecycle hook.

The orchestrator wires ``DedupPipeline.run`` into ``start_session`` /
``stream_session`` so a confirmed-duplicate session is marked + the
agent graph is skipped entirely. Tests inject a fake pipeline through
``orch.dedup_pipeline`` so they exercise the lifecycle without spinning
up a real LLM.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from orchestrator.config import (
    AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths, RuntimeConfig,
)
from orchestrator.orchestrator import Orchestrator
from runtime.dedup import DedupConfig, DedupDecision, DedupPipeline, DedupResult


@pytest.fixture
def cfg(tmp_path):
    # Match the rest of the test suite — point at the framework MCP
    # servers the skills actually consume so build_graph can resolve
    # every required tool.
    #
    # The runtime no longer reaches into examples.incident_management to
    # discover the dedup config — RuntimeConfig.dedup_config_path is
    # left unset (None) so Orchestrator.create skips the pipeline by
    # default. Tests that need a pipeline inject one directly via
    # ``orch.dedup_pipeline``.
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="local_inc", transport="in_process",
                            module="orchestrator.mcp_servers.incident",
                            category="incident_management"),
            MCPServerConfig(name="local_obs", transport="in_process",
                            module="orchestrator.mcp_servers.observability",
                            category="observability"),
            MCPServerConfig(name="local_rem", transport="in_process",
                            module="orchestrator.mcp_servers.remediation",
                            category="remediation"),
            MCPServerConfig(name="local_user", transport="in_process",
                            module="orchestrator.mcp_servers.user_context",
                            category="user_context"),
        ]),
        paths=Paths(skills_dir="config/skills",
                    incidents_dir=str(tmp_path)),
        runtime=RuntimeConfig(
            state_class="examples.incident_management.state.IncidentState",
        ),
    )


class _ScriptedPipeline:
    """Drop-in stand-in for ``DedupPipeline`` used by lifecycle tests.

    Captures every ``run`` call so tests can assert that it was (or was
    not) invoked, and returns a pre-baked ``DedupResult``.
    """

    def __init__(self, result: DedupResult) -> None:
        self.result = result
        self.calls: list[str] = []
        # Echo the real pipeline's surface so attribute access in the
        # orchestrator (.run, .config) doesn't blow up if extended later.
        self.config = DedupConfig(enabled=True)

    async def run(self, *, session, history_store):  # noqa: ARG002
        self.calls.append(getattr(session, "id", "?"))
        return self.result


@pytest.mark.asyncio
async def test_start_session_skips_graph_on_confirmed_duplicate(cfg):
    """When dedup matches, the graph must NOT be invoked and the new
    session must land in ``status="duplicate"`` with the parent linked."""
    orch = await Orchestrator.create(cfg)
    try:
        # First a "prior" closed session so the parent id is real.
        prior_id = await orch.start_session(
            query="payments timeout production", environment="production",
        )
        prior = orch.store.load(prior_id)
        prior.status = "resolved"
        orch.store.save(prior)

        # Inject the scripted pipeline AFTER the priming run so the
        # parent isn't itself flagged as a duplicate.
        orch.dedup_pipeline = _ScriptedPipeline(DedupResult(
            matched=True,
            parent_session_id=prior_id,
            candidate_id=prior_id,
            decision=DedupDecision(
                is_duplicate=True, confidence=0.95,
                rationale="same root cause",
            ),
            stage1_score=0.88,
        ))

        # Patch graph.ainvoke so we can prove it was bypassed.
        graph_calls = []
        original_ainvoke = orch.graph.ainvoke

        async def _spy(*args, **kwargs):
            graph_calls.append((args, kwargs))
            return await original_ainvoke(*args, **kwargs)

        orch.graph.ainvoke = _spy  # type: ignore[assignment]

        new_id = await orch.start_session(
            query="payments timeout production", environment="production",
        )
        assert graph_calls == [], (
            "P7-F: graph must not run for confirmed duplicates"
        )
        loaded = orch.store.load(new_id)
        assert loaded.status == "duplicate"
        assert loaded.parent_session_id == prior_id
        assert loaded.dedup_rationale == "same root cause"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_start_session_runs_graph_when_no_match(cfg):
    """A pipeline that returns matched=False must NOT alter the session
    and the agent graph must run normally."""
    orch = await Orchestrator.create(cfg)
    try:
        orch.dedup_pipeline = _ScriptedPipeline(DedupResult(matched=False))
        sid = await orch.start_session(
            query="something fresh", environment="production",
        )
        loaded = orch.store.load(sid)
        # Without dedup, the stub graph would normally produce a non-new
        # status (downstream agents run). The exact terminal status
        # depends on confidence stub behaviour — assert only that we
        # did NOT land in duplicate.
        assert loaded.status != "duplicate"
        assert loaded.parent_session_id is None
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_start_session_disabled_dedup_is_passthrough(cfg):
    """When ``dedup_pipeline is None`` (framework default), start_session
    behaves identically to pre-P7."""
    orch = await Orchestrator.create(cfg)
    try:
        # Force the pipeline off so the test is independent of the
        # bundled YAML (which opts in for incident_management).
        orch.dedup_pipeline = None
        sid = await orch.start_session(
            query="anything", environment="production",
        )
        loaded = orch.store.load(sid)
        assert loaded.status != "duplicate"
        assert loaded.parent_session_id is None
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_run_dedup_check_idempotent_on_already_marked(cfg):
    """Calling the hook a second time on a session that's already
    ``status="duplicate"`` must be a no-op — the pipeline must not be
    invoked again."""
    orch = await Orchestrator.create(cfg)
    try:
        sid = await orch.start_session(
            query="x", environment="production",
        )
        inc = orch.store.load(sid)
        inc.status = "duplicate"
        inc.parent_session_id = "INC-19000101-001"
        orch.store.save(inc)

        scripted = _ScriptedPipeline(DedupResult(matched=True))
        orch.dedup_pipeline = scripted
        result = await orch._run_dedup_check(inc)
        assert result is True  # already a duplicate — fast-path
        assert scripted.calls == []  # pipeline was NOT invoked
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_run_dedup_check_swallows_pipeline_errors(cfg):
    """A misbehaving dedup pipeline must NOT crash intake."""
    orch = await Orchestrator.create(cfg)
    try:
        class _Boom:
            config = DedupConfig(enabled=True)
            async def run(self, **kwargs):
                raise RuntimeError("network down")
        orch.dedup_pipeline = _Boom()  # type: ignore[assignment]
        # If the helper raises, this call fails. We assert it does NOT.
        sid = await orch.start_session(
            query="latency", environment="production",
        )
        loaded = orch.store.load(sid)
        assert loaded.status != "duplicate"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_stream_session_emits_dedup_matched_event(cfg):
    """The streaming entry point must emit a ``dedup_matched`` event
    when the pipeline confirms a duplicate, and skip downstream agent
    events entirely."""
    orch = await Orchestrator.create(cfg)
    try:
        prior_id = await orch.start_session(
            query="prior", environment="production",
        )
        prior = orch.store.load(prior_id)
        prior.status = "resolved"
        orch.store.save(prior)

        orch.dedup_pipeline = _ScriptedPipeline(DedupResult(
            matched=True, parent_session_id=prior_id, candidate_id=prior_id,
            decision=DedupDecision(is_duplicate=True, confidence=0.9,
                                   rationale="same"),
            stage1_score=0.9,
        ))
        events = []
        async for ev in orch.stream_session(query="prior",
                                            environment="production"):
            events.append(ev["event"])
        assert "dedup_matched" in events
        assert "investigation_completed" in events
        # Agent-node events use names like "intake"/"triage" — none of
        # those should be emitted because the graph never ran.
        agent_events = [e for e in events
                        if e in {"intake", "triage", "deep_investigator",
                                 "resolution"}]
        assert agent_events == []
    finally:
        await orch.aclose()
