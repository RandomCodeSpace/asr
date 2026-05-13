"""M1 wiring test — EventLog is instantiated on orchestrator boot
and shared with the intake_context.

Per `.plan/per-step-telemetry-auto-learning-intake.md`:
- `orch.event_log` must be an `EventLog` instance
- `framework_cfg.intake_context.event_log` must be the SAME ref
"""
import pytest
from runtime.config import AppConfig, LLMConfig, MCPConfig, MCPServerConfig, Paths, RuntimeConfig
from runtime.orchestrator import Orchestrator
from runtime.storage import EventLog


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
        runtime=RuntimeConfig(state_class=None),
    )


@pytest.mark.asyncio
async def test_orchestrator_creates_event_log(cfg):
    """orch.event_log is an EventLog instance after Orchestrator.create."""
    orch = await Orchestrator.create(cfg)
    try:
        assert isinstance(orch.event_log, EventLog)
    finally:
        await orch.aclose()


@pytest.mark.asyncio
async def test_intake_context_shares_event_log(cfg):
    """The intake_context.event_log is the same ref as orch.event_log
    so module-level supervisor runners can emit through one handle."""
    orch = await Orchestrator.create(cfg)
    try:
        # ``intake_context`` is attached via object.__setattr__ in
        # Orchestrator.create; pyright doesn't see the dynamic attr,
        # so go via getattr.
        intake_ctx = getattr(orch.framework_cfg, "intake_context")
        assert intake_ctx.event_log is orch.event_log
    finally:
        await orch.aclose()
