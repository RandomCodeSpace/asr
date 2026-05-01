"""Tests for Orchestrator.resume_investigation across all three actions."""
import pytest

from orchestrator.config import (
    AppConfig, InterventionConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths,
)
from orchestrator.incident import AgentRun
from orchestrator.orchestrator import Orchestrator


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
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
        paths=Paths(skills_dir="config/skills", incidents_dir=str(tmp_path)),
        intervention=InterventionConfig(confidence_threshold=0.75),
    )


def _seed_paused_incident(orch: Orchestrator, *, di_confidence: float | None):
    """Create an INC that already ran intake/triage/DI and is paused at the gate."""
    inc = orch.store.create(query="db latency", environment="production",
                            reporter_id="u", reporter_team="t")
    inc.status = "awaiting_input"
    inc.agents_run.append(AgentRun(
        agent="intake", started_at="t1", ended_at="t2", summary="ok",
    ))
    inc.agents_run.append(AgentRun(
        agent="triage", started_at="t3", ended_at="t4", summary="ok",
    ))
    inc.agents_run.append(AgentRun(
        agent="deep_investigator",
        started_at="t5", ended_at="t6", summary="weak hypothesis",
        confidence=di_confidence,
        confidence_rationale="thin signal",
    ))
    inc.pending_intervention = {
        "reason": "low_confidence",
        "confidence": di_confidence,
        "threshold": 0.75,
        "options": ["resume_with_input", "escalate", "stop"],
        "escalation_teams": ["platform-oncall"],
    }
    orch.store.save(inc)
    return inc.id


@pytest.mark.asyncio
async def test_resume_stop_sets_status_stopped(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.42)
        events = []
        async for ev in orch.resume_investigation(inc_id, {"action": "stop"}):
            events.append(ev)
        inc = orch.store.load(inc_id)
        assert inc.status == "stopped"
        assert inc.pending_intervention is None
        assert any(e["event"] == "resume_started" for e in events)
        completed = [e for e in events if e["event"] == "resume_completed"]
        assert completed and completed[-1]["status"] == "stopped"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_resume_escalate_pages_oncall_and_marks_escalated(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.40)
        events = []
        async for ev in orch.resume_investigation(
            inc_id, {"action": "escalate", "team": "data-oncall"},
        ):
            events.append(ev)
        inc = orch.store.load(inc_id)
        assert inc.status == "escalated"
        assert inc.pending_intervention is None
        # A notify_oncall ToolCall must have been appended by the orchestrator.
        notifies = [tc for tc in inc.tool_calls if tc.tool == "notify_oncall"]
        assert notifies, "expected a notify_oncall tool call after escalate"
        assert notifies[-1].agent == "orchestrator"
        assert notifies[-1].args["incident_id"] == inc_id
        completed = [e for e in events if e["event"] == "resume_completed"]
        assert completed and completed[-1]["status"] == "escalated"
        assert completed[-1]["team"] == "data-oncall"
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_resume_rejects_when_not_awaiting_input(cfg):
    """An INC that has already moved on (e.g. resolved) must NOT be resumable.

    The orchestrator must yield a `resume_rejected` event citing the actual
    status, and must NOT advance to escalate / stop / re-run side effects.
    """
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.42)
        # Move INC out of awaiting_input
        inc = orch.store.load(inc_id)
        inc.status = "resolved"
        orch.store.save(inc)

        events = []
        async for ev in orch.resume_investigation(inc_id, {"action": "stop"}):
            events.append(ev)

        rejected = [e for e in events if e["event"] == "resume_rejected"]
        assert rejected, f"expected resume_rejected, got {events}"
        assert "not_awaiting_input" in rejected[-1]["reason"]
        assert "resolved" in rejected[-1]["reason"]
        # Status must be unchanged (still resolved, not stopped).
        assert orch.store.load(inc_id).status == "resolved"
        # No resume_completed should have been emitted.
        assert not any(e["event"] == "resume_completed" for e in events)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_resume_rejects_invalid_team(cfg):
    """Escalating to a team not in escalation_teams must be rejected.

    Otherwise an attacker / careless caller could fire ``notify_oncall`` with
    an arbitrary team string, and the INC would be marked escalated against a
    team that doesn't exist in the configured roster.
    """
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.40)
        # Default escalation_teams = ["platform-oncall", "data-oncall", "security-oncall"]
        events = []
        async for ev in orch.resume_investigation(
            inc_id, {"action": "escalate", "team": "marketing"},
        ):
            events.append(ev)

        rejected = [e for e in events if e["event"] == "resume_rejected"]
        assert rejected, f"expected resume_rejected, got {events}"
        assert "marketing" in rejected[-1]["reason"]
        assert "escalation_teams" in rejected[-1]["reason"]

        # INC must NOT have been escalated — still awaiting_input, no
        # notify_oncall tool call recorded.
        inc = orch.store.load(inc_id)
        assert inc.status == "awaiting_input"
        assert not any(tc.tool == "notify_oncall" for tc in inc.tool_calls)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_resume_handles_subgraph_exception(cfg, monkeypatch):
    """If the resume sub-graph raises, the INC must be restored to
    awaiting_input with its original pending_intervention payload — not left
    stuck at in_progress with a cleared intervention.

    The orchestrator should yield a ``resume_failed`` event and NOT re-raise.
    """
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.42)
        before = orch.store.load(inc_id)
        original_pi = dict(before.pending_intervention)

        # Force the sub-graph to blow up partway through.
        async def _boom(*_args, **_kwargs):
            # Yield nothing and then raise — makes this a proper async generator
            # that raises on first iteration, simulating a sub-graph failure.
            if False:  # noqa: SIM210
                yield  # makes this function an async generator
            raise RuntimeError("apply_fix exploded mid-stream")

        monkeypatch.setattr(orch.resume_graph, "astream_events", _boom)

        events = []
        async for ev in orch.resume_investigation(
            inc_id,
            {"action": "resume_with_input", "input": "operator note"},
        ):
            events.append(ev)

        # Restored state.
        inc = orch.store.load(inc_id)
        assert inc.status == "awaiting_input", (
            f"INC must be restored to awaiting_input, got {inc.status}"
        )
        assert inc.pending_intervention == original_pi, (
            "pending_intervention must be restored to its pre-resume value"
        )

        # Events.
        failed = [e for e in events if e["event"] == "resume_failed"]
        assert failed, f"expected resume_failed event, got {events}"
        assert "apply_fix exploded" in str(failed[-1].get("error", ""))
        # No resume_completed must have been emitted (failure short-circuits).
        assert not any(e["event"] == "resume_completed" for e in events)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_resume_with_input_reruns_di_and_resolution(cfg):
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.35)
        before = orch.store.load(inc_id)
        n_runs_before = len(before.agents_run)

        events = []
        async for ev in orch.resume_investigation(
            inc_id,
            {"action": "resume_with_input",
             "input": "Operator note: DB pool exhausted at 14:30."},
        ):
            events.append(ev)

        inc = orch.store.load(inc_id)
        # The user's text was appended to user_inputs and intervention cleared.
        assert "DB pool exhausted at 14:30." in inc.user_inputs[-1]
        # At least one new agent_run was appended for the resume (DI re-run).
        assert len(inc.agents_run) > n_runs_before
        new_runs = inc.agents_run[n_runs_before:]
        assert any(r.agent == "deep_investigator" for r in new_runs)
        # pending_intervention is either cleared (gate passed) or refreshed
        # (gate paused again). Either way, status reflects the gate's verdict.
        assert inc.status in {
            "in_progress", "resolved", "escalated", "awaiting_input",
        }
        completed = [e for e in events if e["event"] == "resume_completed"]
        assert completed
    finally:
        await orch.aclose()
