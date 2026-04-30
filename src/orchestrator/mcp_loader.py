"""Load MCP servers (in_process / stdio / http / sse) and build a tool registry."""
from __future__ import annotations
import importlib
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools

from orchestrator.config import MCPConfig, MCPServerConfig


@dataclass
class ToolEntry:
    name: str
    description: str
    server: str
    category: str
    tool: BaseTool


@dataclass
class ToolRegistry:
    entries: dict[str, ToolEntry] = field(default_factory=dict)

    def add(self, entry: ToolEntry) -> None:
        if entry.name in self.entries:
            raise ValueError(f"Duplicate tool name in registry: {entry.name}")
        self.entries[entry.name] = entry

    def get(self, names: list[str]) -> list[BaseTool]:
        out: list[BaseTool] = []
        for n in names:
            if n not in self.entries:
                raise KeyError(f"Tool not in registry: {n}")
            out.append(self.entries[n].tool)
        return out

    def by_category(self) -> dict[str, list[ToolEntry]]:
        out: dict[str, list[ToolEntry]] = {}
        for e in self.entries.values():
            out.setdefault(e.category, []).append(e)
        return out


async def _load_in_process(server_cfg: MCPServerConfig) -> list[BaseTool]:
    if server_cfg.module is None:
        raise ValueError(f"in_process server '{server_cfg.name}' missing 'module'")
    mod = importlib.import_module(server_cfg.module)
    fmcp = getattr(mod, "mcp", None)
    if fmcp is None:
        raise ValueError(f"Module {server_cfg.module} has no 'mcp' (FastMCP instance)")
    # FastMCP exposes tools as functions; convert to langchain tools via adapter.
    # We use the in-memory client transport.
    from fastmcp import Client
    client = Client(fmcp)
    async with client:
        return await load_mcp_tools(client.session)


async def _load_remote(server_cfg: MCPServerConfig) -> list[BaseTool]:
    from fastmcp import Client
    if server_cfg.transport in ("http", "sse"):
        if not server_cfg.url:
            raise ValueError(f"remote server '{server_cfg.name}' missing 'url'")
        client = Client(server_cfg.url, headers=server_cfg.headers or None)
    elif server_cfg.transport == "stdio":
        if not server_cfg.command:
            raise ValueError(f"stdio server '{server_cfg.name}' missing 'command'")
        client = Client({"command": server_cfg.command[0], "args": server_cfg.command[1:]})
    else:
        raise ValueError(f"Unknown transport: {server_cfg.transport}")
    async with client:
        return await load_mcp_tools(client.session)


async def load_tools(cfg: MCPConfig) -> ToolRegistry:
    registry = ToolRegistry()
    for server_cfg in cfg.servers:
        if not server_cfg.enabled:
            continue
        if server_cfg.transport == "in_process":
            tools = await _load_in_process(server_cfg)
        else:
            tools = await _load_remote(server_cfg)
        for t in tools:
            registry.add(ToolEntry(
                name=t.name, description=t.description or "",
                server=server_cfg.name, category=server_cfg.category, tool=t,
            ))
    return registry
