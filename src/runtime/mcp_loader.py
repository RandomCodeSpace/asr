"""Load MCP servers (in_process / stdio / http / sse) and build a tool registry.

Each tool is registered by ``(server_name, original_tool_name)`` and its
LangChain ``.name`` is rewritten to ``<server_name>:<original_tool_name>`` so
the LLM sees disambiguated names when two MCP servers both expose a tool with
the same base name.

The caller resolves a skill's ``tools`` map (``dict[str, list[str]]``) into a
flat list of :class:`~langchain_core.tools.BaseTool` via
:meth:`ToolRegistry.resolve`.
"""
# The FastMCP Client instances loaded here MUST stay open for the entire
# lifetime of the returned ToolRegistry, otherwise the LangChain tool wrappers
# hold references to a closed transport and the FIRST tool invocation raises:
#     unable to perform operation on <TCPTransport closed=True ...>
# To make the lifetime explicit, the caller passes an already-entered
# contextlib.AsyncExitStack; each FastMCP client is registered into it via
# `await stack.enter_async_context(client)`. The caller controls teardown by
# calling `await stack.aclose()`.
from __future__ import annotations
import importlib
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools

from runtime.config import MCPConfig, MCPServerConfig


@dataclass
class ToolEntry:
    name: str          # original tool name as exposed by the server
    description: str
    server: str        # server name as declared in cfg.mcp.servers
    category: str
    tool: BaseTool     # LangChain tool with .name = "<server>:<name>"


@dataclass
class ToolRegistry:
    entries: dict[tuple[str, str], ToolEntry] = field(default_factory=dict)

    def add(self, entry: ToolEntry) -> None:
        key = (entry.server, entry.name)
        if key in self.entries:
            raise ValueError(
                f"Duplicate tool {entry.name!r} on server {entry.server!r}"
            )
        self.entries[key] = entry

    def tools_for_server(self, server: str) -> list[ToolEntry]:
        """Return all entries belonging to ``server``."""
        return [e for e in self.entries.values() if e.server == server]

    def tools_in_process(self, in_process_server_names: set[str]) -> list[ToolEntry]:
        """Return all entries whose server is in ``in_process_server_names``."""
        return [e for e in self.entries.values()
                if e.server in in_process_server_names]

    def resolve(self, spec: dict[str, list[str]], cfg: "MCPConfig") -> list[BaseTool]:
        """Resolve a skill's ``tools`` map to a flat, deduplicated list of
        LangChain tools.

        Keys are server names declared in ``cfg.mcp.servers`` or the special
        key ``"local"`` which aggregates every server with
        ``transport=in_process``. Values are explicit tool-name lists or
        ``["*"]`` for "all tools from this server".

        Raises :class:`ValueError` on unknown server key or unknown tool name.
        """
        in_process_servers = {
            s.name for s in cfg.servers if s.transport == "in_process"
        }
        declared_servers = {s.name for s in cfg.servers}
        out: list[BaseTool] = []
        seen: set[tuple[str, str]] = set()

        def _add(entry: ToolEntry) -> None:
            key = (entry.server, entry.name)
            if key in seen:
                return
            seen.add(key)
            out.append(entry.tool)

        for key, names in spec.items():
            if key == "local":
                available = [e for e in self.entries.values()
                             if e.server in in_process_servers]
            elif key in declared_servers:
                available = [e for e in self.entries.values()
                             if e.server == key]
            else:
                raise ValueError(
                    f"unknown server {key!r} in skill tools; known: "
                    f"{sorted(declared_servers | {'local'})}"
                )

            if names == ["*"]:
                for e in available:
                    _add(e)
            else:
                for n in names:
                    matched = [e for e in available if e.name == n]
                    if not matched:
                        raise ValueError(
                            f"tool {n!r} not found in server {key!r} "
                            f"(available: {sorted(e.name for e in available)})"
                        )
                    for e in matched:
                        _add(e)

        return out

    def by_category(self) -> dict[str, list[ToolEntry]]:
        out: dict[str, list[ToolEntry]] = {}
        for e in self.entries.values():
            out.setdefault(e.category, []).append(e)
        return out


def build_fastmcp_client(server_cfg: MCPServerConfig):
    """Build an un-entered FastMCP ``Client`` for ``server_cfg``.

    Returned client is not yet attached to any exit stack — the caller is
    responsible for ``await stack.enter_async_context(client)``. Used by
    :class:`runtime.service.OrchestratorService` to populate the
    process-singleton MCP client pool; the legacy per-orchestrator
    loaders below stay as-is.
    """
    from fastmcp import Client
    if server_cfg.transport == "in_process":
        if server_cfg.module is None:
            raise ValueError(
                f"in_process server '{server_cfg.name}' missing 'module'"
            )
        mod = importlib.import_module(server_cfg.module)
        fmcp = getattr(mod, "mcp", None)
        if fmcp is None:
            raise ValueError(
                f"Module {server_cfg.module} has no 'mcp' (FastMCP instance)"
            )
        return Client(fmcp)
    if server_cfg.transport in ("http", "sse"):
        if not server_cfg.url:
            raise ValueError(f"remote server '{server_cfg.name}' missing 'url'")
        return Client(server_cfg.url, headers=server_cfg.headers or None)
    if server_cfg.transport == "stdio":
        if not server_cfg.command:
            raise ValueError(
                f"stdio server '{server_cfg.name}' missing 'command'"
            )
        return Client(
            {"command": server_cfg.command[0], "args": server_cfg.command[1:]}
        )
    raise ValueError(f"Unknown transport: {server_cfg.transport}")


async def _load_in_process(server_cfg: MCPServerConfig,
                           stack: AsyncExitStack) -> list[BaseTool]:
    if server_cfg.module is None:
        raise ValueError(f"in_process server '{server_cfg.name}' missing 'module'")
    mod = importlib.import_module(server_cfg.module)
    fmcp = getattr(mod, "mcp", None)
    if fmcp is None:
        raise ValueError(f"Module {server_cfg.module} has no 'mcp' (FastMCP instance)")
    # FastMCP exposes tools as functions; convert to langchain tools via adapter.
    # We use the in-memory client transport. The client is registered into the
    # caller's exit stack so its session/transport stays open while the loaded
    # tools are in use.
    from fastmcp import Client
    client = Client(fmcp)
    await stack.enter_async_context(client)
    tools = await load_mcp_tools(client.session)
    # Rewrite each tool's .name to "<server>:<original>" for LLM disambiguation.
    for t in tools:
        original_name = t.name
        t.name = f"{server_cfg.name}:{original_name}"
        t._original_mcp_name = original_name  # type: ignore[attr-defined]
    return tools


async def _load_remote(server_cfg: MCPServerConfig,
                       stack: AsyncExitStack) -> list[BaseTool]:
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
    await stack.enter_async_context(client)
    tools = await load_mcp_tools(client.session)
    # Rewrite each tool's .name to "<server>:<original>" for LLM disambiguation.
    for t in tools:
        original_name = t.name
        t.name = f"{server_cfg.name}:{original_name}"
        t._original_mcp_name = original_name  # type: ignore[attr-defined]
    return tools


async def load_tools(cfg: MCPConfig, stack: AsyncExitStack) -> ToolRegistry:
    """Load all enabled MCP servers and return a :class:`ToolRegistry`.

    The caller MUST pass an already-entered :class:`AsyncExitStack`. Each
    FastMCP ``Client`` is registered into it; the caller controls lifetime via
    ``await stack.aclose()``.
    """
    registry = ToolRegistry()
    for server_cfg in cfg.servers:
        if not server_cfg.enabled:
            continue
        if server_cfg.transport == "in_process":
            tools = await _load_in_process(server_cfg, stack)
        else:
            tools = await _load_remote(server_cfg, stack)
        for t in tools:
            original = getattr(t, "_original_mcp_name", t.name)
            registry.add(ToolEntry(
                name=original, description=t.description or "",
                server=server_cfg.name, category=server_cfg.category, tool=t,
            ))
    return registry
