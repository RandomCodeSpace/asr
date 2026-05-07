"""FastMCP server: user_context mock tool."""
from __future__ import annotations
from fastmcp import FastMCP

mcp = FastMCP("user_context")


def register(mcp_app, cfg) -> None:
    """App-MCP-server discovery contract: no-op binder.

    This server returns static mock data and has no config-derived
    state. The `register` callable exists so the orchestrator's generic
    discovery loop (`for module_path in cfg.orchestrator.mcp_servers:
    importlib.import_module(module_path).register(mcp_app, cfg)`)
    treats every entry uniformly. Idempotent by definition.
    """
    return None


@mcp.tool()
async def get_user_context(user_id: str) -> dict:
    """Return canned user metadata."""
    return {
        "user_id": user_id,
        "team": "platform",
        "role": "engineer",
        "manager": "manager-mock",
        "timezone": "UTC",
    }
