"""Test helpers for Phase 11 should_gate matrix."""
from __future__ import annotations

from runtime.config import GatePolicy, GatewayConfig, OrchestratorConfig
from runtime.state import Session, ToolCall


def make_orch_cfg(
    *,
    policy: dict[str, str] | None = None,
    confidence_threshold: float = 0.7,
    gated_environments: set[str] | None = None,
    gated_risk_actions: set[str] | None = None,
) -> OrchestratorConfig:
    """Construct an OrchestratorConfig with a populated GatePolicy.

    The fields the test matrix exercises are the gate_policy block plus
    a sibling GatewayConfig.policy dict so that effective_action's
    PVC-08 prefixed-form lookup is exercised honestly. All other
    OrchestratorConfig defaults are used.

    Returns
    -------
    OrchestratorConfig
        A pydantic-validated OrchestratorConfig with a populated
        ``gate_policy`` field and a sibling ``gateway`` block. The
        OrchestratorConfig itself does not own the gateway field at the
        framework default — callers thread it independently — so we
        attach the gateway as an attribute the should_gate boundary
        will read via ``cfg.gateway`` if exposed, or directly via the
        sibling ``GatewayConfig`` argument the runtime wires today.
    """
    cfg = OrchestratorConfig(
        gate_policy=GatePolicy(
            confidence_threshold=confidence_threshold,
            gated_environments=gated_environments or {"production"},
            gated_risk_actions=gated_risk_actions or {"approve"},
        ),
    )
    # Stash the GatewayConfig on the cfg under a known attribute. The
    # production code threads gateway separately (via runtime.gateway)
    # but should_gate's signature accepts an OrchestratorConfig and
    # delegates to effective_action, which reads its own gateway_cfg
    # parameter. The pure-function tests pass cfg.gateway through.
    cfg.__dict__["gateway"] = GatewayConfig(policy=policy or {})  # type: ignore[index]
    return cfg


def make_session(env: str = "dev") -> Session:
    """Construct a minimal pydantic-validated Session for matrix tests."""
    return Session(
        id="t-session",
        status="open",
        created_at="2026-05-07T00:00:00Z",
        updated_at="2026-05-07T00:00:00Z",
    )._with_env(env) if hasattr(Session, "_with_env") else Session(
        id="t-session",
        status="open",
        created_at="2026-05-07T00:00:00Z",
        updated_at="2026-05-07T00:00:00Z",
    )


def make_tool_call(name: str) -> ToolCall:
    """Construct a minimal ToolCall row for matrix tests."""
    return ToolCall(
        agent="t",
        tool=name,
        args={},
        result=None,
        ts="2026-05-07T00:00:00Z",
        risk="low",
        status="executed",
    )


# Session subclass for environment threading -- the framework's base
# Session has no ``environment`` field; that's an app-level extension.
# For these pure-function tests we want a Session-shaped object with a
# settable ``environment`` attribute so should_gate can read it.
class _EnvSession:
    """Minimal Session-shaped stand-in carrying ``environment``.

    The pure should_gate function reads ``session.environment`` only.
    The OrchestratorConfig and ToolCall are fully pydantic-validated;
    the Session role here is just to surface the environment string
    + a place for the transient confidence hint. Using a plain class
    avoids forcing the framework's domain-free Session base to gain
    an ``environment`` field.
    """

    def __init__(self, env: str = "dev") -> None:
        self.environment: str = env
        self._turn_confidence_hint: float | None = None
        self.id = "t-session"
        self.status = "open"
        self.tool_calls: list[ToolCall] = []


def make_env_session(env: str = "dev") -> _EnvSession:
    return _EnvSession(env=env)
