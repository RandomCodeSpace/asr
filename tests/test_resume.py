"""Tests for Orchestrator.resume_investigation across all three actions."""
import pytest

from runtime.config import (
    AppConfig, FrameworkAppConfig, LLMConfig, MCPConfig, MCPServerConfig,
    OrchestratorConfig, Paths, RuntimeConfig,
)
from runtime.state import AgentRun
from runtime.orchestrator import Orchestrator
from runtime.terminal_tools import StatusDef, TerminalToolRule


@pytest.fixture
def cfg(tmp_path):
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
        # confidence_threshold + escalation_teams come from
        # AppConfig.framework — read directly off the YAML at runtime;
        # tests construct an in-memory FrameworkAppConfig with the
        # incident roster so the resume_session escalate action can
        # validate against it.
        framework=FrameworkAppConfig(
            confidence_threshold=0.75,
            escalation_teams=["platform-oncall", "data-oncall", "security-oncall"],
        ),
        # Phase 6 (DECOUPLE-02 / Resolution A): the escalate path is
        # parameterised on the OrchestratorConfig fields; the test
        # fixture mirrors the incident_management.yaml registration.
        orchestrator=OrchestratorConfig(
            statuses={
                "open":         StatusDef(name="open",         terminal=False, kind="pending"),
                "escalated":    StatusDef(name="escalated",    terminal=True,  kind="escalation"),
                "resolved":     StatusDef(name="resolved",     terminal=True,  kind="success"),
                "needs_review": StatusDef(name="needs_review", terminal=True,  kind="needs_review"),
            },
            terminal_tools=[
                TerminalToolRule(tool_name="mark_resolved", status="resolved"),
                TerminalToolRule(
                    tool_name="mark_escalated", status="escalated",
                    extract_fields={"team": ["args.team", "result.team"]},
                ),
                TerminalToolRule(
                    tool_name="notify_oncall", status="escalated",
                    extract_fields={"team": ["args.team"]},
                ),
            ],
            patch_tools=["update_incident"],
            harvest_terminal_tools=["submit_hypothesis"],
            # Phase 7 (DECOUPLE-04 / D-07-02): orchestrator imports and
            # binds these app-MCP-server modules at create()-time. Each
            # exposes ``register(mcp_app, cfg)`` which closes over the
            # fixture's ``framework.escalation_teams`` /
            # ``environments`` rosters so the per-tool guards line up
            # with what the test asserts on.
            mcp_servers=[
                "examples.incident_management.mcp_servers.observability",
                "examples.incident_management.mcp_servers.remediation",
                "examples.incident_management.mcp_servers.user_context",
            ],
            default_terminal_status="needs_review",
            escalate_action_tool_name="notify_oncall",
            escalate_action_default_team="platform-oncall",
        ),
        runtime=RuntimeConfig(state_class=None),
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
    """If the resume invocation on the main graph raises, the INC must be
    restored to awaiting_input with its original pending_intervention
    payload — not left stuck at in_progress with a cleared intervention.

    The orchestrator should yield a ``resume_failed`` event and NOT re-raise.
    """
    orch = await Orchestrator.create(cfg)
    try:
        inc_id = _seed_paused_incident(orch, di_confidence=0.42)
        before = orch.store.load(inc_id)
        original_pi = dict(before.pending_intervention)

        # Force the resume invocation (Command(resume=...) on the main
        # graph) to blow up partway through.
        async def _boom(*_args, **_kwargs):
            # Yield nothing and then raise — makes this a proper async generator
            # that raises on first iteration, simulating a graph failure.
            if False:  # noqa: SIM210
                yield  # makes this function an async generator
            raise RuntimeError("apply_fix exploded mid-stream")

        # Resume goes through the same compiled graph; patch
        # orch.graph.astream_events.
        monkeypatch.setattr(orch.graph, "astream_events", _boom)

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


async def _drive_to_interrupt(orch: Orchestrator) -> str:
    """Start a session and drive it through the graph until the gate's
    ``interrupt()`` pauses execution. Returns the session id.

    With stub LLMs the deep_investigator never emits a confidence value,
    so the gate always interrupts on the first call.
    """
    sid = await orch.start_investigation(
        query="payments slow", environment="production",
    )
    # The gate dual-write leaves the row in awaiting_input.
    sess = orch.store.load(sid)
    assert sess.status == "awaiting_input", (
        f"expected gate to interrupt; got status={sess.status!r}"
    )
    assert sess.pending_intervention is not None
    return sid


@pytest.mark.asyncio
async def test_resume_uses_command_resume(cfg):
    """Resume invokes the same graph via Command(resume=...).

    Drives a fresh investigation to the gate's interrupt, then resumes
    with operator input. The session should advance past the gate and
    end no longer awaiting_input.
    """
    async with await Orchestrator.create(cfg) as orch:
        sid = await _drive_to_interrupt(orch)
        before = orch.store.load(sid)
        n_runs_before = len(before.agents_run)

        events = []
        async for ev in orch.resume_investigation(
            sid,
            {"action": "resume_with_input",
             "input": "Operator note: DB pool exhausted at 14:30."},
        ):
            events.append(ev)

        sess2 = orch.store.load(sid)
        # Gate's post-resume continuation appended the user input and
        # cleared pending_intervention.
        assert "DB pool exhausted at 14:30." in sess2.user_inputs[-1]
        assert sess2.pending_intervention is None
        assert sess2.status != "awaiting_input"
        # At least one downstream agent ran after the gate cleared.
        assert len(sess2.agents_run) >= n_runs_before
        completed = [e for e in events if e["event"] == "resume_completed"]
        assert completed


def test_orchestrator_has_no_resume_graph_attr():
    """The bespoke resume graph is gone from the source."""
    import runtime.orchestrator as mod
    src = open(mod.__file__).read()
    assert "build_resume_graph" not in src
    assert "self.resume_graph" not in src


def test_graph_module_has_no_build_resume_graph():
    """build_resume_graph is deleted from the graph module."""
    import runtime.graph as mod
    src = open(mod.__file__).read()
    assert "build_resume_graph" not in src
    assert not hasattr(mod, "build_resume_graph")


@pytest.mark.asyncio
async def test_cold_restart_resume(tmp_path):
    """Cold-restart resume — checkpointer state
    survives orchestrator teardown.

    Process 1: start a session, drive to the gate's interrupt, tear down.
    Process 2: cold-instantiate against the same DB and resume via
    Command(resume=...). The graph should advance past the gate.
    """
    cfg_one = AppConfig(
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
        runtime=RuntimeConfig(
            state_class=None,
        ),
    )

    # --- Process 1: drive into interrupt, then tear down ---
    async with await Orchestrator.create(cfg_one) as orch1:
        sid = await _drive_to_interrupt(orch1)
        # The DB on disk has the row + the langgraph checkpoint.

    # --- Process 2: cold-instantiate against the same DB ---
    async with await Orchestrator.create(cfg_one) as orch2:
        # The checkpointer rehydrates the paused thread on resume.
        events = []
        async for ev in orch2.resume_investigation(
            sid,
            {"action": "resume_with_input", "input": "post-restart note"},
        ):
            events.append(ev)

        sess = orch2.store.load(sid)
        assert "post-restart note" in sess.user_inputs[-1], (
            "gate must have appended the resume input on the cold-restart side"
        )
        assert sess.pending_intervention is None
        assert sess.status != "awaiting_input"
        completed = [e for e in events if e["event"] == "resume_completed"]
        assert completed
