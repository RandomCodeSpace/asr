"""FastMCP server: user_context mock tool."""
from __future__ import annotations
from fastmcp import FastMCP

mcp = FastMCP("user_context")


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
