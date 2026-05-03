"""Risk-rated tool gateway.

Public surface:

  * ``effective_action``      -- pure resolver
  * ``wrap_tool``             -- BaseTool wrapper factory
  * ``GatewayAction``         -- ``Literal["auto", "notify", "approve"]``
"""
from runtime.tools.gateway import GatewayAction, effective_action, wrap_tool

__all__ = ["GatewayAction", "effective_action", "wrap_tool"]
