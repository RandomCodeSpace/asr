"""Risk-rated tool gateway (Phase 4).

Public surface:

  * ``effective_action``      -- pure resolver (P4-B)
  * ``wrap_tool``             -- BaseTool wrapper factory (P4-C)
  * ``GatewayAction``         -- ``Literal["auto", "notify", "approve"]``
"""
from runtime.tools.gateway import GatewayAction, effective_action, wrap_tool

__all__ = ["GatewayAction", "effective_action", "wrap_tool"]
