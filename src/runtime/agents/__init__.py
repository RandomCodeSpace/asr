"""Per-kind agent factories.

One module per :attr:`runtime.skill.Skill.kind`:

* :mod:`runtime.agents.responsive`  — today's LLM agent (a LangGraph node).
* :mod:`runtime.agents.supervisor`  — no-LLM router (a LangGraph node).
* :mod:`runtime.agents.monitor`     — out-of-band scheduled observer
  (NOT a graph node; runs under :class:`MonitorRunner`).

``runtime.graph._build_agent_nodes`` dispatches on ``skill.kind`` to pick
the right factory.
"""
from __future__ import annotations

from .responsive import make_agent_node
from .supervisor import make_supervisor_node, log_supervisor_dispatch
from .monitor import (
    MonitorRunner,
    SafeEvalError,
    make_monitor_callable,
    safe_eval,
)
from .turn_output import (
    AgentTurnOutput,
    EnvelopeMissingError,
    parse_envelope_from_result,
    reconcile_confidence,
)

__all__ = [
    "make_agent_node",
    "make_supervisor_node",
    "log_supervisor_dispatch",
    "MonitorRunner",
    "SafeEvalError",
    "make_monitor_callable",
    "safe_eval",
    "AgentTurnOutput",
    "EnvelopeMissingError",
    "parse_envelope_from_result",
    "reconcile_confidence",
]
