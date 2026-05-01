from __future__ import annotations
# ----- imports for config.py -----
"""Config schemas for the orchestrator."""

from typing import Literal
from pydantic import BaseModel, Field


# ----- imports for incident.py -----
"""Incident domain model."""

from typing import Any, Literal


# ----- imports for similarity.py -----
"""Similarity scoring for incident matching."""

from typing import Protocol
import re

# ----- imports for skill.py -----
"""Skill loader.

Each agent lives in its own subdirectory under ``config/skills/``::

    config/skills/
      _common/                # OPTIONAL: prompt fragments shared by all agents
        confidence.md         # appended to every agent's system_prompt, in
        output.md             # alphabetical order, joined with blank lines
      intake/
        config.yaml           # name, description, tools, routes, temperature, model
        system.md             # the agent's specialty (markdown body — format is free)
        guidelines.md         # OPTIONAL extra fragments; every *.md in the
        ...                   # directory is concatenated in alphabetical order

Adding a directory under ``config/skills/`` (with a ``config.yaml`` and at
least one ``.md`` file) adds an agent. Directories whose name starts with
``_`` are reserved for shared content and never become agents.

The final ``system_prompt`` for each agent is::

    <concatenated *.md from agent_dir>
    \\n\\n
    <concatenated *.md from _common/, if present>

Structured config is validated through the :class:`Skill` /
:class:`RouteRule` Pydantic models; markdown content is loaded verbatim.
"""

from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator


# ----- imports for llm.py -----
"""LLM provider abstraction with stub/ollama/azure_openai backends."""

import os
from typing import Any
from uuid import uuid4
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field



# ----- imports for mcp_servers/incident.py -----
"""FastMCP server: incident_management mock tools.

State scoping
-------------
Earlier revisions used a module-level ``_state`` dict, which made the FastMCP
instance and its store reference process-global. Two concurrent
:class:`Orchestrator` instances (or two pytest workers) would clobber each
other's stores. Now state is held on a per-instance :class:`IncidentMCPServer`,
and a fresh ``mcp`` (FastMCP) instance is built for every server.

The module-level ``mcp`` symbol (imported by the MCP loader) is constructed
lazily — the first call to :func:`get_or_create_default_server` (or the
back-compat :func:`set_state`) builds it. ``set_state`` remains as a thin shim
that mutates the default server's state, so existing call-sites and tests keep
working without churn.
"""

from dataclasses import dataclass, field
from fastmcp import FastMCP




# ----- imports for mcp_servers/observability.py -----
"""FastMCP server: observability mock tools."""

import hashlib
from datetime import datetime, timezone, timedelta

# ----- imports for mcp_servers/remediation.py -----
"""FastMCP server: remediation mock tools."""

from datetime import datetime, timezone

# ----- imports for mcp_servers/user_context.py -----
"""FastMCP server: user_context mock tool."""


# ----- imports for mcp_loader.py -----
"""Load MCP servers (in_process / stdio / http / sse) and build a tool registry."""
# The FastMCP Client instances loaded here MUST stay open for the entire
# lifetime of the returned ToolRegistry, otherwise the LangChain tool wrappers
# hold references to a closed transport and the FIRST tool invocation raises:
#     unable to perform operation on <TCPTransport closed=True ...>
# To make the lifetime explicit, the caller passes an already-entered
# contextlib.AsyncExitStack; each FastMCP client is registered into it via
# `await stack.enter_async_context(client)`. The caller controls teardown by
# calling `await stack.aclose()`.

import importlib
from contextlib import AsyncExitStack
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools



# ----- imports for graph.py -----
"""LangGraph state, routing helpers, and node runner."""

import asyncio
import logging
from typing import TypedDict, Callable, Awaitable

# ----- imports for orchestrator.py -----
"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""

from typing import AsyncIterator








# ----- imports for api.py -----
"""FastAPI app — health, listings, and (future) external orchestrator endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel




# ----- imports for ui/streamlit_app.py -----
"""Streamlit UI — 2 tabs + always-on sidebar with recent INCs."""
# Lifecycle note: the Orchestrator owns FastMCP clients tied to a specific
# asyncio event loop. Streamlit re-runs the script on every interaction and we
# use `asyncio.run(...)` per call, which creates a fresh loop each time.
# Caching an Orchestrator across reruns would leave its clients/transports
# bound to a dead loop and the first tool call would raise:
#     unable to perform operation on <TCPTransport closed=True ...>
# So: build a fresh Orchestrator inside each `asyncio.run` and `aclose` it when
# done. For pure metadata views (agents/tools) we use `_load_metadata_dicts` —
# a one-shot fetch that captures plain dicts and disposes the orchestrator.
# The sidebar uses IncidentStore directly, since incident JSON I/O is sync and
# needs no MCP clients.

from datetime import datetime
import streamlit as st







# ====== module: orchestrator/config.py ======

class OllamaConfig(BaseModel):
    base_url: str = "https://ollama.com"
    api_key: str | None = None


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    api_version: str = "2024-08-01-preview"
    api_key: str | None = None
    deployment: str


class StubConfig(BaseModel):
    pass


class LLMConfig(BaseModel):
    provider: Literal["ollama", "azure_openai", "stub"] = "stub"
    default_model: str = "stub-1"
    default_temperature: float = 0.0
    ollama: OllamaConfig | None = None
    azure_openai: AzureOpenAIConfig | None = None
    stub: StubConfig = Field(default_factory=StubConfig)


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["in_process", "stdio", "http", "sse"]
    category: str
    enabled: bool = True
    module: str | None = None
    command: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class IncidentConfig(BaseModel):
    store_path: str = "incidents"
    similarity_threshold: float = 0.85
    similarity_method: Literal["keyword", "embedding"] = "keyword"


class Paths(BaseModel):
    skills_dir: str = "config/skills"
    incidents_dir: str = "incidents"


class InterventionConfig(BaseModel):
    confidence_threshold: float = 0.75
    escalation_teams: list[str] = Field(
        default_factory=lambda: [
            "platform-oncall", "data-oncall", "security-oncall",
        ],
    )


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    incidents: IncidentConfig = Field(default_factory=IncidentConfig)
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"]
    )
    paths: Paths = Field(default_factory=Paths)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)


import os
import re
from pathlib import Path
import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate(value):
    if isinstance(value, str):
        def replace(m):
            name = m.group(1)
            if name not in os.environ:
                raise KeyError(f"Required env var not set: {name}")
            return os.environ[name]
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_config(path: str | Path) -> AppConfig:
    from dotenv import load_dotenv
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text())
    resolved = _interpolate(raw)
    return AppConfig(**resolved)

# ====== module: orchestrator/incident.py ======

IncidentStatus = Literal[
    "new", "in_progress", "matched", "resolved",
    "escalated", "awaiting_input", "stopped",
]


class Reporter(BaseModel):
    id: str
    team: str


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_rationale: str | None = None


class Findings(BaseModel):
    triage: Any = None
    deep_investigator: Any = None


class Incident(BaseModel):
    id: str
    status: IncidentStatus
    created_at: str
    updated_at: str
    query: str
    environment: str
    reporter: Reporter
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    severity: str | None = None
    category: str | None = None
    matched_prior_inc: str | None = None
    embedding: list[float] | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    findings: Findings = Field(default_factory=Findings)
    resolution: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)


from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


class IncidentStore:
    """JSON-file-backed incident store. One file per INC."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        today = _utc_today()
        prefix = f"INC-{today}-"
        existing = [p.stem for p in self.base_dir.glob(f"{prefix}*.json")]
        max_seq = 0
        for stem in existing:
            try:
                max_seq = max(max_seq, int(stem.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"{prefix}{max_seq + 1:03d}"

    def create(self, *, query: str, environment: str,
               reporter_id: str, reporter_team: str) -> Incident:
        inc_id = self._next_id()
        now = _utc_now_iso()
        inc = Incident(
            id=inc_id,
            status="new",
            created_at=now,
            updated_at=now,
            query=query,
            environment=environment,
            reporter=Reporter(id=reporter_id, team=reporter_team),
        )
        self.save(inc)
        return inc

    def save(self, incident: Incident) -> None:
        incident.updated_at = _utc_now_iso()
        path = self.base_dir / f"{incident.id}.json"
        path.write_text(incident.model_dump_json(indent=2))

    def load(self, incident_id: str) -> Incident:
        path = self.base_dir / f"{incident_id}.json"
        if not path.exists():
            raise FileNotFoundError(incident_id)
        return Incident.model_validate_json(path.read_text())

    def list_all(self) -> list[Incident]:
        return [self.load(p.stem) for p in self.base_dir.glob("INC-*.json")]

    def list_recent(self, limit: int = 20) -> list[Incident]:
        all_inc = self.list_all()
        all_inc.sort(key=lambda i: (i.created_at, i.id), reverse=True)
        return all_inc[:limit]

# ====== module: orchestrator/similarity.py ======

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "the", "of", "in", "on", "to", "and", "or", "is", "was", "with", "for",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1}


class Scorer(Protocol):
    def score(self, a: str, b: str) -> float: ...


class KeywordSimilarity:
    """Jaccard overlap on tokenized text. Cheap, deterministic."""

    def score(self, a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


def find_similar(*, query: str, candidates: list[dict], text_field: str,
                 scorer: Scorer, threshold: float, limit: int) -> list[tuple[dict, float]]:
    scored = [(c, scorer.score(query, c[text_field])) for c in candidates]
    passing = [(c, s) for c, s in scored if s >= threshold]
    passing.sort(key=lambda x: x[1], reverse=True)
    return passing[:limit]

# ====== module: orchestrator/skill.py ======

_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _validate_agent_name(name: str, *, source: str) -> None:
    """Reject directory names that can't safely become agent identifiers.

    Agent names appear in route targets (``next: triage``), LangGraph node
    IDs, and INC tags. Restricting them to ``[a-z][a-z0-9_]{0,63}`` keeps
    them grep-able, shell-safe, and free of case-sensitivity surprises
    across filesystems.
    """
    if not _AGENT_NAME_RE.match(name):
        raise ValueError(
            f"invalid agent name {name!r} (from {source}): must match "
            f"[a-z][a-z0-9_]{{0,63}} — lowercase, start with a letter, "
            f"alphanumerics + underscore only, max 64 chars"
        )


class RouteRule(BaseModel):
    when: str
    next: str


class Skill(BaseModel):
    name: str
    description: str
    model: str | None = None
    temperature: float | None = None
    tools: list[str] = Field(default_factory=list)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        return v.strip()


def _concat_md(md_files: list[Path]) -> str:
    return "\n\n".join(p.read_text().strip() for p in md_files)


def _load_common_prompt(skills_dir: Path) -> str:
    """Read every ``*.md`` under ``<skills_dir>/_common/`` (if present) and
    return them concatenated in alphabetical order.

    Returns the empty string when no ``_common/`` directory exists or it
    contains no markdown files — ``load_all_skills`` then leaves agent
    prompts unchanged.
    """
    common_dir = skills_dir / "_common"
    if not common_dir.is_dir():
        return ""
    return _concat_md(sorted(common_dir.glob("*.md")))


def load_skill(agent_dir: str | Path, *, common: str = "") -> Skill:
    """Load one agent from its directory.

    The directory name is the agent's ``name`` (single source of truth).
    Reads ``config.yaml`` for the rest of the structured metadata and
    concatenates every ``*.md`` file (sorted alphabetically) into
    ``system_prompt``. If ``common`` is non-empty, it is appended after
    the agent's own prompt so shared sections (Confidence, Output) only
    need to be authored once.

    Raises ``ValueError`` if ``config.yaml`` declares its own ``name`` —
    the duplication used to drift silently when a directory was renamed
    without updating the config.
    """
    base = Path(agent_dir)
    config_path = base / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.yaml in skill dir: {base}")
    _validate_agent_name(base.name, source=f"directory {base}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    if "name" in cfg:
        raise ValueError(
            f"config.yaml at {config_path} must not declare 'name' — the "
            f"agent name is taken from the directory ({base.name!r})"
        )
    cfg["name"] = base.name
    md_files = sorted(base.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"no .md prompt files in skill dir: {base}")
    agent_prompt = _concat_md(md_files)
    cfg["system_prompt"] = (
        f"{agent_prompt}\n\n{common}".strip() if common else agent_prompt
    )
    return Skill(**cfg)


def load_all_skills(skills_dir: str | Path) -> dict[str, Skill]:
    base = Path(skills_dir)
    if not base.exists():
        raise FileNotFoundError(f"skills dir not found: {base}")
    common = _load_common_prompt(base)
    skills: dict[str, Skill] = {}
    for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # Reserve the leading underscore for shared content (_common,
        # _drafts, etc.) — never treat those as agents.
        if agent_dir.name.startswith("_"):
            continue
        if not (agent_dir / "config.yaml").exists():
            continue
        skill = load_skill(agent_dir, common=common)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills

# ====== module: orchestrator/llm.py ======

class StubChatModel(BaseChatModel):
    """Deterministic chat model for tests/CI. Returns canned text per role.

    Optionally emits one tool call on first invocation if `tool_call_plan` is set.
    """
    role: str = "default"
    canned_responses: dict[str, str] = Field(default_factory=dict)
    tool_call_plan: list[dict] | None = None
    _called_once: bool = False

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        text = self.canned_responses.get(self.role, f"[stub:{self.role}] no canned response")
        tool_calls: list[dict] = []
        if self.tool_call_plan and not self._called_once:
            for tc in self.tool_call_plan:
                tool_calls.append({"name": tc["name"], "args": tc.get("args", {}), "id": str(uuid4())})
            self._called_once = True
        msg = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """No-op binder: stub emits tool calls only via `tool_call_plan`, not via real binding."""
        return self


def get_llm(cfg: LLMConfig, *, role: str = "default", model: str | None = None,
            temperature: float | None = None,
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None) -> BaseChatModel:
    actual_model = model or cfg.default_model
    actual_temp = temperature if temperature is not None else cfg.default_temperature

    if cfg.provider == "stub":
        return StubChatModel(
            role=role,
            canned_responses=stub_canned or {},
            tool_call_plan=stub_tool_plan,
        )
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama
        if cfg.ollama is None:
            raise ValueError("ollama provider requires llm.ollama config")
        kwargs: dict[str, Any] = {
            "base_url": cfg.ollama.base_url,
            "model": actual_model,
            "temperature": actual_temp,
        }
        api_key = cfg.ollama.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return ChatOllama(**kwargs)
    if cfg.provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        if cfg.azure_openai is None:
            raise ValueError("azure_openai provider requires llm.azure_openai config")
        return AzureChatOpenAI(
            azure_endpoint=cfg.azure_openai.endpoint,
            api_version=cfg.azure_openai.api_version,
            azure_deployment=cfg.azure_openai.deployment,
            api_key=cfg.azure_openai.api_key or os.environ.get("AZURE_OPENAI_KEY"),
            temperature=actual_temp,
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")

# ====== module: orchestrator/mcp_servers/incident.py ======

_SEVERITY_MAP = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low", "informational": "low",
    "low": "low",
}


def normalize_severity(value: str | None) -> str | None:
    """Coerce assorted severity inputs (sev1/p2/critical/etc.) to low/medium/high.

    Unknown inputs pass through untouched so callers can flag them; the
    normalized vocabulary surfaced to the UI and stored on disk is restricted
    to {low, medium, high}.
    """
    if value is None:
        return None
    return _SEVERITY_MAP.get(value.strip().lower(), value)


@dataclass
class IncidentMCPServer:
    """Per-instance container holding a FastMCP server and its scoped state.

    Each Orchestrator constructs its own :class:`IncidentMCPServer`, so two
    orchestrators in the same process (e.g. two test fixtures, two web users
    behind one Streamlit deployment) no longer share a store reference.
    """
    store: IncidentStore | None = None
    similarity_threshold: float = 0.85
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        # Bind the tool implementations to *this* server's state. We pass an
        # explicit ``name=`` because FastMCP defaults to the function's
        # ``__name__`` (which would expose the leading underscore). The names
        # below match the original module-level tool functions, so the MCP
        # tool surface (and the LangChain registry that consumes it) is
        # unchanged.
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(self, *, store: IncidentStore, similarity_threshold: float) -> None:
        self.store = store
        self.similarity_threshold = similarity_threshold

    def _require_store(self) -> IncidentStore:
        if self.store is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.store

    # FastMCP introspects parameter type hints to build tool schemas, so the
    # bound methods below take the same signatures as the previous module-level
    # functions (no `self` from the LLM's POV — `self` is captured by the bound
    # method when we pass it to ``mcp.tool()``).

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        store = self._require_store()
        resolved = [i for i in store.list_all() if i.status == "resolved"]
        candidates = [
            {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
             "summary": i.summary, "resolution": i.resolution, "environment": i.environment}
            for i in resolved if i.environment == environment
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(), threshold=self.similarity_threshold, limit=5,
        )
        return {"matches": [
            {"id": r["id"], "summary": r["summary"], "resolution": r["resolution"], "score": round(s, 3)}
            for r, s in results
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    reporter_id: str = "user-mock",
                                    reporter_team: str = "platform") -> dict:
        """Create a new INC ticket and persist it."""
        inc = self._require_store().create(query=query, environment=environment,
                                           reporter_id=reporter_id,
                                           reporter_team=reporter_team)
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC. Allowed keys: status, severity, category, summary, tags,
        matched_prior_inc, resolution, findings_triage, findings_deep_investigator."""
        store = self._require_store()
        inc = store.load(incident_id)
        if "status" in patch:
            inc.status = patch["status"]
        if "severity" in patch:
            inc.severity = normalize_severity(patch["severity"])
        if "category" in patch:
            inc.category = patch["category"]
        if "summary" in patch:
            inc.summary = patch["summary"]
        if "tags" in patch:
            inc.tags = list(patch["tags"])
        if "matched_prior_inc" in patch:
            inc.matched_prior_inc = patch["matched_prior_inc"]
        if "resolution" in patch:
            inc.resolution = patch["resolution"]
        if "findings_triage" in patch:
            inc.findings.triage = patch["findings_triage"]
        if "findings_deep_investigator" in patch:
            inc.findings.deep_investigator = patch["findings_deep_investigator"]
        store.save(inc)
        return inc.model_dump()


# ---------------------------------------------------------------------------
# Module-level default server (back-compat for the existing MCP loader path).
#
# The MCP loader imports ``mcp`` from this module by name (it doesn't know
# about IncidentMCPServer). To preserve that contract, we expose:
#
#   - ``mcp``   : a FastMCP instance owned by ``_default_server``
#   - ``set_state(...)`` : configure ``_default_server``'s store/threshold
#   - ``lookup_similar_incidents`` / ``create_incident`` / ``update_incident``
#     : thin shims so direct callers in tests still work
#
# Per-Orchestrator scoping is achieved by constructing a fresh
# :class:`IncidentMCPServer` (planned in a follow-up that wires it through
# ``mcp_loader.load_tools``). This commit removes the *module-global dict* —
# state now lives on the instance — without breaking the existing import path.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, store: IncidentStore, similarity_threshold: float) -> None:
    """Configure the default IncidentMCPServer instance.

    Kept for backwards compatibility with callers that import this function
    directly (the orchestrator and several tests). New code should construct an
    :class:`IncidentMCPServer` and call ``configure`` on it.
    """
    _default_server.configure(store=store, similarity_threshold=similarity_threshold)


# Public function aliases so test modules importing these names directly keep
# working. They forward to the bound methods on ``_default_server``.
async def lookup_similar_incidents(query: str, environment: str) -> dict:
    return await _default_server._tool_lookup_similar_incidents(query, environment)


async def create_incident(query: str, environment: str,
                          reporter_id: str = "user-mock",
                          reporter_team: str = "platform") -> dict:
    return await _default_server._tool_create_incident(
        query, environment, reporter_id, reporter_team
    )


async def update_incident(incident_id: str, patch: dict) -> dict:
    return await _default_server._tool_update_incident(incident_id, patch)

# ====== module: orchestrator/mcp_servers/observability.py ======

mcp = FastMCP("observability")


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


@mcp.tool()
async def get_logs(service: str, environment: str, minutes: int = 15) -> dict:
    """Return canned recent log lines for a service in an environment."""
    seed = _seed(service, environment, str(minutes))
    rng = (seed >> 4) % 4
    base = [
        f"{datetime.now(timezone.utc).isoformat()} INFO {service} request_id=abc123 path=/v1/items dur=42ms",
        f"{datetime.now(timezone.utc).isoformat()} WARN {service} slow_query duration=820ms table=orders",
        f"{datetime.now(timezone.utc).isoformat()} ERROR {service} upstream_timeout target=payments duration=5000ms",
        f"{datetime.now(timezone.utc).isoformat()} INFO {service} cache_miss key=user:42",
    ]
    return {"service": service, "environment": environment, "lines": base[rng:] + base[:rng]}


@mcp.tool()
async def get_metrics(service: str, environment: str, minutes: int = 15) -> dict:
    """Return canned metrics snapshot."""
    seed = _seed(service, environment)
    return {
        "service": service,
        "environment": environment,
        "window_minutes": minutes,
        "p50_latency_ms": 50 + (seed % 50),
        "p99_latency_ms": 800 + (seed % 1500),
        "error_rate": round(((seed % 100) / 100) * 0.05, 4),
        "rps": 120 + (seed % 300),
        "cpu_pct": 30 + (seed % 60),
        "mem_pct": 40 + (seed % 50),
    }


@mcp.tool()
async def get_service_health(environment: str) -> dict:
    """Return overall environment health summary."""
    seed = _seed(environment)
    statuses = ["healthy", "degraded", "unhealthy"]
    status = statuses[seed % 3]
    return {
        "environment": environment,
        "status": status,
        "services": {
            "api": "healthy" if status == "healthy" else status,
            "db": "healthy",
            "cache": "healthy",
            "queue": status,
        },
    }


@mcp.tool()
async def check_deployment_history(environment: str, hours: int = 24) -> dict:
    """Return canned recent deployments."""
    now = datetime.now(timezone.utc)
    seed = _seed(environment, str(hours))
    deployments = [
        {"service": "api", "version": f"v1.{(seed % 50) + 100}", "deployed_at":
         (now - timedelta(hours=2)).isoformat(), "deployer": "deploy-bot"},
        {"service": "worker", "version": f"v2.{(seed % 30) + 50}", "deployed_at":
         (now - timedelta(hours=8)).isoformat(), "deployer": "deploy-bot"},
    ]
    return {"environment": environment, "window_hours": hours, "deployments": deployments}

# ====== module: orchestrator/mcp_servers/remediation.py ======

mcp = FastMCP("remediation")


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


@mcp.tool()
async def propose_fix(hypothesis: str, environment: str) -> dict:
    """Generate a remediation proposal for a hypothesis. Returns auto_apply_safe flag."""
    seed = _seed(hypothesis, environment)
    safe = (seed % 3) == 0  # ~33% of mock proposals are auto-safe
    return {
        "proposal_id": f"prop-{seed % 1000:03d}",
        "proposal": f"Restart the affected service in {environment}; investigate {hypothesis}.",
        "auto_apply_safe": safe,
        "estimated_impact": "low" if safe else "medium",
    }


@mcp.tool()
async def apply_fix(proposal_id: str, environment: str) -> dict:
    """Apply a previously proposed fix. Mock returns success/failure deterministically."""
    seed = _seed(proposal_id, environment)
    success = (seed % 4) != 0
    return {
        "proposal_id": proposal_id,
        "environment": environment,
        "status": "applied" if success else "failed",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "details": "Mock remediation completed." if success else "Mock remediation failed.",
    }


@mcp.tool()
async def notify_oncall(incident_id: str, message: str) -> dict:
    """Page the oncall engineer."""
    return {
        "incident_id": incident_id,
        "page_id": f"page-{abs(hash(incident_id)) % 10000:04d}",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }

# ====== module: orchestrator/mcp_servers/user_context.py ======

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

# ====== module: orchestrator/mcp_loader.py ======

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
    return await load_mcp_tools(client.session)


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
    return await load_mcp_tools(client.session)


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
            registry.add(ToolEntry(
                name=t.name, description=t.description or "",
                server=server_cfg.name, category=server_cfg.category, tool=t,
            ))
    return registry

# ====== module: orchestrator/graph.py ======

logger = logging.getLogger(__name__)


_CONFIDENCE_LABELS: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


def _coerce_confidence(raw) -> float | None:
    """Coerce a raw confidence value emitted by an LLM to a clamped float in
    [0.0, 1.0], or None when the value cannot be interpreted.

    Order matters: bool **must** be rejected before float because Python treats
    ``True``/``False`` as instances of ``int`` (and therefore acceptable to
    ``float()``). Strings are matched against the canonical {high, medium, low}
    labels case-insensitively; any other string emits a warning and yields
    None. Floats outside [0.0, 1.0] are clamped (with a warning) rather than
    dropped — clamping is more forgiving when an LLM is on the right track but
    miscalibrated.
    """
    if isinstance(raw, bool):
        logger.warning("confidence value is bool (%r); rejecting", raw)
        return None
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _CONFIDENCE_LABELS:
            mapped = _CONFIDENCE_LABELS[key]
            logger.warning("coerced string confidence %r -> %s", raw, mapped)
            return mapped
        logger.warning("unknown confidence string %r; treating as None", raw)
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning("uncoercible confidence value %r (%s); treating as None",
                       raw, type(raw).__name__)
        return None
    clamped = max(0.0, min(1.0, val))
    if clamped != val:
        logger.warning("clamped out-of-range confidence %s -> %s", val, clamped)
    return clamped


def _coerce_rationale(raw) -> str | None:
    """Coerce a confidence_rationale value to a stripped string, or None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning("confidence_rationale is bool (%r); rejecting", raw)
        return None
    try:
        return str(raw)
    except Exception:  # noqa: BLE001 — defensive; any object should be str-able
        logger.warning("uncoercible confidence_rationale %r; dropping", raw)
        return None

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END







class GraphState(TypedDict, total=False):
    incident: Incident
    next_route: str | None
    last_agent: str | None
    error: str | None


def route_from_skill(skill: Skill, signal: str) -> str:
    if not skill.routes:
        raise ValueError(f"Skill '{skill.name}' has no routes defined")
    for rule in skill.routes:
        if rule.when == signal:
            return rule.next
    for rule in skill.routes:
        if rule.when == "default":
            return rule.next
    return skill.routes[0].next


class AgentRunRecorder:
    """Helper to capture an agent's run + tool calls into the incident."""

    def __init__(self, *, agent: str, incident: Incident):
        self.agent = agent
        self.incident = incident
        self._started_at: str | None = None

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record_tool_call(self, tool: str, args: dict, result) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.incident.tool_calls.append(
            ToolCall(agent=self.agent, tool=tool, args=args, result=result, ts=ts)
        )

    def finish(self, *, summary: str) -> None:
        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.incident.agents_run.append(AgentRun(
            agent=self.agent,
            started_at=self._started_at or ended_at,
            ended_at=ended_at,
            summary=summary,
        ))


_TRANSIENT_MARKERS = (
    "internal server error",
    "status code: -1",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "remoteprotocolerror",
    "incomplete chunked read",
    "connection reset",
)


async def _ainvoke_with_retry(executor, input_, *, max_attempts: int = 3,
                              base_delay: float = 1.5):
    """Wrap a LangGraph agent invocation with retry on transient cloud errors.

    Retries on common Ollama Cloud / streaming hiccups (500, status -1, etc.).
    Non-transient exceptions (4xx, validation, etc.) propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await executor.ainvoke(input_)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = any(m in msg for m in _TRANSIENT_MARKERS)
            if not transient or attempt == max_attempts - 1:
                raise
            last_exc = exc
            await asyncio.sleep(base_delay * (attempt + 1))
    raise last_exc  # pragma: no cover  (unreachable)


def _format_intake_input(incident: Incident) -> str:
    base = (
        f"Incident {incident.id}\n"
        f"Environment: {incident.environment}\n"
        f"Query: {incident.query}\n"
        f"Status: {incident.status}\n"
        f"Findings (triage): {incident.findings.triage}\n"
        f"Findings (deep_investigator): {incident.findings.deep_investigator}\n"
    )
    if incident.user_inputs:
        bullets = "\n".join(f"- {ui}" for ui in incident.user_inputs)
        base += (
            "\nUser-provided context (appended via intervention):\n"
            f"{bullets}\n"
        )
    return base


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Incident], str],
    store: IncidentStore,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route."""
    agent_executor = create_react_agent(llm, tools, prompt=skill.system_prompt)

    async def node(state: GraphState) -> dict:
        incident = state["incident"]
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_intake_input(incident))]},
            )
        except Exception as exc:  # noqa: BLE001
            # Reload to absorb any partial writes from tools that ran before the failure.
            try:
                incident = store.load(inc_id)
            except FileNotFoundError:
                pass
            ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            incident.agents_run.append(AgentRun(
                agent=skill.name, started_at=started_at, ended_at=ended_at,
                summary=f"agent failed: {exc}",
                token_usage=TokenUsage(),
            ))
            store.save(incident)
            return {"incident": incident, "next_route": None,
                    "last_agent": skill.name, "error": str(exc)}

        # Tools (e.g. update_incident) write straight to disk. Reload so the
        # node's own append of agent_run + tool_calls happens against the
        # tool-mutated state — otherwise saving the stale in-memory object
        # clobbers the tools' writes.
        incident = store.load(inc_id)

        # Record tool calls from the agent's message trace. While iterating,
        # also harvest the latest `confidence` / `confidence_rationale` carried
        # in any `update_incident` patch — those keys are stamped on the
        # AgentRun (the MCP tool itself silently ignores extra patch keys).
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        agent_confidence: float | None = None
        agent_rationale: str | None = None
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("args", {}) or {}
                incident.tool_calls.append(ToolCall(
                    agent=skill.name,
                    tool=tc_name,
                    args=tc_args,
                    result=None,
                    ts=ts,
                ))
                if tc_name == "update_incident":
                    patch = tc_args.get("patch") or {}
                    if "confidence" in patch:
                        agent_confidence = _coerce_confidence(patch["confidence"])
                    if "confidence_rationale" in patch:
                        agent_rationale = _coerce_rationale(patch["confidence_rationale"])

        # Pair tool responses with their tool calls.
        for msg in result.get("messages", []):
            if msg.__class__.__name__ == "ToolMessage":
                for entry in reversed(incident.tool_calls):
                    if entry.tool == getattr(msg, "name", None) and entry.result is None:
                        entry.result = getattr(msg, "content", None)
                        break

        # Final summary text from the agent's last AIMessage. Persist it
        # verbatim — concision is enforced via the skill prompts (each one
        # instructs the agent to keep the final reply ≤150 words). Storing
        # the full message preserves the audit trail; mid-fence truncation
        # used to corrupt downstream markdown rendering.
        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage" and msg.content:
                final_text = str(msg.content)
                break

        # Sum token usage across every message that reports it. langchain-ollama
        # populates `usage_metadata` on AIMessages from Ollama's
        # prompt_eval_count / eval_count fields. Stub/test models leave it
        # absent — those simply contribute zero.
        agent_in = agent_out = 0
        for msg in result.get("messages", []):
            um = getattr(msg, "usage_metadata", None) or {}
            agent_in += int(um.get("input_tokens") or 0)
            agent_out += int(um.get("output_tokens") or 0)
        agent_total = agent_in + agent_out

        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        incident.agents_run.append(AgentRun(
            agent=skill.name, started_at=started_at, ended_at=ended_at,
            summary=final_text or f"{skill.name} completed",
            token_usage=TokenUsage(
                input_tokens=agent_in,
                output_tokens=agent_out,
                total_tokens=agent_total,
            ),
            confidence=agent_confidence,
            confidence_rationale=agent_rationale,
        ))
        incident.token_usage.input_tokens += agent_in
        incident.token_usage.output_tokens += agent_out
        incident.token_usage.total_tokens += agent_total

        next_route_signal = decide_route(incident)
        store.save(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"incident": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


# Per-agent route decision functions.
def _decide_intake(inc: Incident) -> str:
    return "default"


def _decide_triage(inc: Incident) -> str:
    return "default"


def _decide_deep_investigator(inc: Incident) -> str:
    return "default"


def _decide_resolution(inc: Incident) -> str:
    return "default"


_DECIDERS: dict[str, Callable[[Incident], str]] = {
    "intake": _decide_intake,
    "triage": _decide_triage,
    "deep_investigator": _decide_deep_investigator,
    "resolution": _decide_resolution,
}


_STUB_CANNED = {
    "intake": "Created INC, no prior matches. Routing to triage.",
    "triage": "Severity medium, category latency. No recent deploys correlate.",
    "deep_investigator": "Hypothesis: upstream payments timeout. Evidence: log line 'upstream_timeout target=payments'.",
    "resolution": "Proposed fix: restart api service. Auto-applied. INC resolved.",
}


def _latest_di_confidence(incident: Incident) -> float | None:
    """Return the most recent deep_investigator AgentRun confidence, or None."""
    for run in reversed(incident.agents_run):
        if run.agent == "deep_investigator":
            return run.confidence
    return None


def make_gate_node(*, cfg: AppConfig, store: IncidentStore):
    """Build the intervention gate node placed between DI and resolution.

    If the latest deep_investigator confidence is below the configured
    threshold (or absent), the gate marks the incident as `awaiting_input`,
    populates `pending_intervention`, and routes to END. Otherwise it routes
    to `resolution`.

    Implemented as a plain async coroutine (not via ``make_agent_node``) so
    it does not invoke an LLM — but it IS a real graph node, so streamed
    events surface ``enter gate`` / ``exit gate``.
    """
    threshold = cfg.intervention.confidence_threshold
    teams = list(cfg.intervention.escalation_teams)

    async def gate(state: GraphState) -> dict:
        incident = state["incident"]
        # Reload from disk in case earlier nodes wrote tool-driven patches.
        try:
            incident = store.load(incident.id)
        except FileNotFoundError:
            pass
        di_conf = _latest_di_confidence(incident)
        if di_conf is None or di_conf < threshold:
            incident.status = "awaiting_input"
            incident.pending_intervention = {
                "reason": "low_confidence",
                "confidence": di_conf,
                "threshold": threshold,
                "options": ["resume_with_input", "escalate", "stop"],
                "escalation_teams": teams,
            }
            store.save(incident)
            return {"incident": incident, "next_route": "__end__",
                    "last_agent": "gate", "error": None}
        # Confidence met threshold — clear any stale intervention payload.
        if incident.pending_intervention is not None:
            incident.pending_intervention = None
            store.save(incident)
        return {"incident": incident, "next_route": "default",
                "last_agent": "gate", "error": None}

    return gate


def _build_agent_nodes(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                       registry: ToolRegistry) -> dict:
    """Materialize agent nodes from skills + registry. Reused by main + resume graphs."""
    nodes: dict = {}
    for agent_name, skill in skills.items():
        llm = get_llm(
            cfg.llm,
            role=agent_name,
            model=skill.model,
            temperature=skill.temperature,
            stub_canned=_STUB_CANNED,
        )
        tools = registry.get(skill.tools)
        decide = _DECIDERS.get(agent_name, lambda inc: "default")
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
        )
    return nodes


def _gate_router(state: GraphState):
    nr = state.get("next_route")
    if nr in (None, "__end__"):
        return END
    # gate's "default" means "advance to resolution".
    return "resolution"


async def build_graph(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                      registry: ToolRegistry):
    """Compile the main LangGraph: intake -> triage -> deep_investigator -> gate -> resolution.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.
    """
    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    # Insert the human-in-the-loop gate between DI and resolution.
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))

    sg.set_entry_point("intake")

    # Standard router for agent nodes: next_route is a target node or __end__.
    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        # deep_investigator's "default" forwards through the gate.
        if state.get("last_agent") == "deep_investigator" and nr == "resolution":
            return "gate"
        return nr

    for agent_name in skills.keys():
        possible_targets = {s.name for s in skills.values()} | {END, "gate"}
        target_map = {name: name for name in possible_targets if name != END}
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)

    # Gate's edges: route to resolution on default, END on __end__.
    sg.add_conditional_edges("gate", _gate_router, {
        "resolution": "resolution", END: END,
    })

    return sg.compile()


async def build_resume_graph(*, cfg: AppConfig, skills: dict,
                             store: IncidentStore, registry: ToolRegistry):
    """Compile a sub-graph that re-runs only deep_investigator -> gate -> resolution.

    Used by ``Orchestrator.resume_investigation`` after the user supplies new
    context: intake/triage already ran, so we resume from DI with the updated
    incident. Same gate semantics — if the new run is still low-confidence,
    we'll pause again.
    """
    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    # Only DI + resolution agents participate; intake/triage are skipped.
    for agent_name in ("deep_investigator", "resolution"):
        if agent_name in nodes:
            sg.add_node(agent_name, nodes[agent_name])
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))
    sg.set_entry_point("deep_investigator")

    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        if state.get("last_agent") == "deep_investigator" and nr == "resolution":
            return "gate"
        return nr

    for agent_name in ("deep_investigator", "resolution"):
        sg.add_conditional_edges(agent_name, _router, {
            "deep_investigator": "deep_investigator",
            "resolution": "resolution",
            "gate": "gate",
            END: END,
        })
    sg.add_conditional_edges("gate", _gate_router, {
        "resolution": "resolution", END: END,
    })
    return sg.compile()

# ====== module: orchestrator/orchestrator.py ======

class Orchestrator:
    """High-level facade. Construct via ``await Orchestrator.create(cfg)``.

    The Orchestrator owns the lifecycle of the FastMCP clients underpinning the
    tool registry. Always call :meth:`aclose` (or use ``async with``) when done.
    """

    def __init__(self, cfg: AppConfig, store: IncidentStore,
                 skills: dict[str, Skill], registry: ToolRegistry, graph,
                 resume_graph, exit_stack: AsyncExitStack):
        self.cfg = cfg
        self.store = store
        self.skills = skills
        self.registry = registry
        self.graph = graph
        self.resume_graph = resume_graph
        self._exit_stack = exit_stack

    @classmethod
    async def create(cls, cfg: AppConfig) -> "Orchestrator":
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            store = IncidentStore(cfg.paths.incidents_dir)
            _set_inc_state(store=store, similarity_threshold=cfg.incidents.similarity_threshold)
            skills = load_all_skills(cfg.paths.skills_dir)
            registry = await load_tools(cfg.mcp, stack)
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry)
            resume_graph = await build_resume_graph(
                cfg=cfg, skills=skills, store=store, registry=registry,
            )
            return cls(cfg, store, skills, registry, graph, resume_graph, stack)
        except BaseException:
            await stack.aclose()
            raise

    async def aclose(self) -> None:
        """Close all owned MCP clients/transports. Idempotent."""
        await self._exit_stack.aclose()

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "model": s.model or self.cfg.llm.default_model,
                "tools": list(s.tools),
                "routes": [r.model_dump() for r in s.routes],
            }
            for s in self.skills.values()
        ]

    def list_tools(self) -> list[dict]:
        bindings: dict[str, list[str]] = {}
        for skill in self.skills.values():
            for tool_name in skill.tools:
                bindings.setdefault(tool_name, []).append(skill.name)
        return [
            {
                "name": e.name,
                "description": e.description,
                "category": e.category,
                "server": e.server,
                "bound_agents": bindings.get(e.name, []),
            }
            for e in self.registry.entries.values()
        ]

    def get_incident(self, incident_id: str) -> dict:
        return self.store.load(incident_id).model_dump()

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        return [i.model_dump() for i in self.store.list_recent(limit)]

    async def start_investigation(self, *, query: str, environment: str,
                                  reporter_id: str = "user-mock",
                                  reporter_team: str = "platform") -> str:
        inc = self.store.create(query=query, environment=environment,
                                reporter_id=reporter_id, reporter_team=reporter_team)
        await self.graph.ainvoke(GraphState(incident=inc, next_route=None,
                                            last_agent=None, error=None))
        return inc.id

    async def stream_investigation(self, *, query: str, environment: str,
                                   reporter_id: str = "user-mock",
                                   reporter_team: str = "platform"
                                   ) -> AsyncIterator[dict]:
        inc = self.store.create(query=query, environment=environment,
                                reporter_id=reporter_id, reporter_team=reporter_team)
        yield {"event": "investigation_started", "incident_id": inc.id,
               "ts": _now()}
        async for ev in self.graph.astream_events(
            GraphState(incident=inc, next_route=None, last_agent=None, error=None),
            version="v2",
        ):
            yield self._to_ui_event(ev, inc.id)
        yield {"event": "investigation_completed", "incident_id": inc.id, "ts": _now()}

    async def resume_investigation(self, incident_id: str,
                                   decision: dict) -> AsyncIterator[dict]:
        """Resume a paused INC. ``decision`` shapes:

        - ``{"action": "resume_with_input", "input": "<text>"}``
        - ``{"action": "escalate", "team": "<team-name>"}``
        - ``{"action": "stop"}``

        Yields a small set of UI events: a ``resume_started`` event up front,
        the underlying graph events for ``resume_with_input``, then a
        ``resume_completed`` event with ``status`` set to the final state.
        """
        action = decision.get("action")
        yield {"event": "resume_started", "incident_id": incident_id,
               "action": action, "ts": _now()}

        inc = self.store.load(incident_id)

        # Guard: only paused INCs are resumable. A resolved/stopped/escalated
        # INC must not be advanced again — that would silently corrupt state
        # (e.g. re-pinging on-call after the incident has already closed).
        if inc.status != "awaiting_input":
            yield {"event": "resume_rejected", "incident_id": incident_id,
                   "reason": f"not_awaiting_input (status={inc.status})",
                   "ts": _now()}
            return

        if action == "stop":
            inc.status = "stopped"
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "stopped", "ts": _now()}
            return

        if action == "escalate":
            team = decision.get("team") or "platform-oncall"
            allowed = list(self.cfg.intervention.escalation_teams)
            if team not in allowed:
                # Reject the request entirely. The INC stays awaiting_input
                # so the user can retry with a valid team. Logging the
                # allowed roster on the event makes it actionable in the UI.
                yield {"event": "resume_rejected", "incident_id": incident_id,
                       "reason": (
                           f"team '{team}' not in allowed escalation_teams "
                           f"({allowed})"
                       ),
                       "ts": _now()}
                return
            message = (
                f"INC {incident_id} escalated by user — team {team}. "
                "Confidence below threshold."
            )
            tool_args = {"incident_id": incident_id, "message": message}
            tool_result = await self._invoke_tool("notify_oncall", tool_args)
            inc = self.store.load(incident_id)
            inc.tool_calls.append(ToolCall(
                agent="orchestrator",
                tool="notify_oncall",
                args=tool_args,
                result=tool_result,
                ts=_now(),
            ))
            inc.status = "escalated"
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "escalated", "team": team, "ts": _now()}
            return

        if action == "resume_with_input":
            user_text = (decision.get("input") or "").strip()
            if not user_text:
                raise ValueError("resume_with_input requires a non-empty 'input'")
            # Snapshot the intervention payload BEFORE we mutate the INC, so
            # we can restore it if the sub-graph blows up. Without this an
            # apply_fix exception leaves the INC stuck at in_progress with a
            # cleared pending_intervention — the user can no longer resolve it
            # via the UI.
            saved_pi = inc.pending_intervention
            inc.user_inputs.append(user_text)
            inc.pending_intervention = None
            inc.status = "in_progress"
            self.store.save(inc)
            inc = self.store.load(incident_id)  # reload as canonical state
            try:
                async for ev in self.resume_graph.astream_events(
                    GraphState(incident=inc, next_route=None, last_agent=None,
                               error=None),
                    version="v2",
                ):
                    yield self._to_ui_event(ev, incident_id)
            except Exception as exc:  # noqa: BLE001 — restore on any failure
                # Reload from disk to absorb any partial writes from tools
                # that ran before the failure, then restore intervention
                # state so the UI can reprompt the user.
                try:
                    inc = self.store.load(incident_id)
                except FileNotFoundError:
                    pass
                inc.pending_intervention = saved_pi
                inc.status = "awaiting_input"
                self.store.save(inc)
                yield {"event": "resume_failed", "incident_id": incident_id,
                       "error": str(exc), "ts": _now()}
                return
            final = self.store.load(incident_id)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": final.status, "ts": _now()}
            return

        raise ValueError(f"Unknown resume action: {action!r}")

    async def _invoke_tool(self, name: str, args: dict):
        """Call an MCP tool by name, going through the LangChain wrapper.

        Used for orchestrator-driven tool calls (e.g. notify_oncall on
        escalate) that aren't initiated by an LLM.
        """
        entry = self.registry.entries.get(name)
        if entry is None:
            raise KeyError(f"tool '{name}' not registered")
        return await entry.tool.ainvoke(args)

    @staticmethod
    def _to_ui_event(raw: dict, incident_id: str) -> dict:
        kind = raw.get("event", "unknown")
        node = raw.get("name") or raw.get("metadata", {}).get("langgraph_node")
        return {
            "event": kind,
            "node": node,
            "incident_id": incident_id,
            "ts": _now(),
            "data": raw.get("data"),
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ====== module: orchestrator/api.py ======

class InvestigateRequest(BaseModel):
    query: str
    environment: str
    reporter_id: str = "user-mock"
    reporter_team: str = "platform"


class InvestigateResponse(BaseModel):
    incident_id: str


async def build_app(cfg: AppConfig) -> FastAPI:
    orch = await Orchestrator.create(cfg)
    app = FastAPI(title="ASR — Agent Orchestrator")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/agents")
    async def agents():
        return orch.list_agents()

    @app.get("/tools")
    async def tools():
        return orch.list_tools()

    @app.get("/incidents")
    async def incidents(limit: int = 20):
        return orch.list_recent_incidents(limit=limit)

    @app.get("/incidents/{incident_id}")
    async def incident(incident_id: str):
        return orch.get_incident(incident_id)

    @app.post("/investigate", response_model=InvestigateResponse)
    async def investigate(req: InvestigateRequest) -> InvestigateResponse:
        inc_id = await orch.start_investigation(
            query=req.query, environment=req.environment,
            reporter_id=req.reporter_id, reporter_team=req.reporter_team,
        )
        return InvestigateResponse(incident_id=inc_id)

    @app.on_event("shutdown")
    async def _shutdown():
        # Release MCP clients/transports owned by the orchestrator on app
        # shutdown. Without this the FastMCP transports leak past app teardown.
        await orch.aclose()

    app.state.orchestrator = orch
    return app

# ====== module: ui/streamlit_app.py ======

CONFIG_PATH = Path("config/config.yaml")


def _load_metadata_dicts(cfg: AppConfig) -> tuple[list[dict], list[dict], list[str]]:
    """Build a transient orchestrator, snapshot agents/tools/envs, then aclose.

    Per-rerun cost is dominated by FastMCP client startup (~100-200ms total for
    in-process servers); acceptable for a UI rerun.
    """
    async def _go():
        orch = await Orchestrator.create(cfg)
        try:
            return orch.list_agents(), orch.list_tools(), list(orch.cfg.environments)
        finally:
            await orch.aclose()
    return asyncio.run(_go())


# Color palette for st.badge — Streamlit accepts: blue/green/orange/red/violet/gray/primary.
_STATUS_COLOR = {
    "new": "gray",
    "in_progress": "blue",
    "matched": "violet",
    "resolved": "green",
    "escalated": "red",
    "awaiting_input": "orange",
    "stopped": "gray",
}

# Human-readable labels — awaiting_input is highlighted as the action-required state.
_STATUS_LABEL = {
    "new": "NEW",
    "in_progress": "IN PROGRESS",
    "matched": "MATCHED",
    "resolved": "RESOLVED",
    "escalated": "ESCALATED",
    "awaiting_input": "⚠ NEEDS INPUT",
    "stopped": "STOPPED",
}

_SEVERITY_COLOR = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}

_CATEGORY_COLOR = {
    "latency": "orange",
    "availability": "red",
    "data": "violet",
    "security": "red",
    "capacity": "blue",
    "performance": "orange",
    "config": "gray",
}


def _badge(label: str, color: str) -> None:
    """Render an inline coloured pill via st.badge.

    Centralised so the small label/colour decisions live in one place and
    the rest of the UI can call ``_status_badge(inc)`` etc. without
    touching the palette dicts directly.
    """
    st.badge(label, color=color)


def _status_badge(status: str | None) -> None:
    if not status:
        return
    _badge(_STATUS_LABEL.get(status, status.upper()),
           _STATUS_COLOR.get(status, "gray"))


def _severity_badge(severity: str | None) -> None:
    if not severity:
        return
    _badge(severity.upper(), _SEVERITY_COLOR.get(severity, "gray"))


def _category_badge(category: str | None) -> None:
    if not category:
        return
    _badge(category, _CATEGORY_COLOR.get(category, "gray"))


def render_sidebar(store: IncidentStore) -> None:
    with st.sidebar:
        st.markdown("### Recent INCs")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            statuses = ["all", "new", "in_progress", "matched", "resolved",
                        "escalated", "awaiting_input", "stopped"]
            status_filter = st.selectbox("Filter", statuses, key="status_filter",
                                         label_visibility="collapsed")
        with col_r:
            if st.button("↻", help="Refresh"):
                st.rerun()

        recent = [i.model_dump() for i in store.list_recent(50)]
        if status_filter != "all":
            recent = [i for i in recent if i["status"] == status_filter]

        if not recent:
            st.caption("No incidents.")
            return
        for inc in recent[:20]:
            with st.container(border=True):
                top_l, top_r = st.columns([3, 2])
                with top_l:
                    st.markdown(
                        f"**`{inc['id']}`** · _{inc['environment']}_"
                    )
                with top_r:
                    _status_badge(inc.get("status"))
                meta_l, meta_r = st.columns(2)
                with meta_l:
                    _severity_badge(inc.get("severity"))
                with meta_r:
                    _category_badge(inc.get("category"))
                toks = (inc.get("token_usage") or {}).get("total_tokens", 0)
                tok_str = (f"{toks/1000:.1f}k tok" if toks >= 1000
                           else f"{toks} tok")
                if st.button(f"View · {tok_str}",
                             key=f"inc_{inc['id']}",
                             use_container_width=True):
                    st.session_state["selected_incident"] = inc["id"]


def _render_kv_block(d: dict) -> None:
    """Render a dict as labeled markdown lines, recursing into nested
    dicts and lists.

    Replaces ``st.json`` everywhere structured agent output bleeds into
    the main detail panel — the JSON braces look out of place between
    bordered cards and prose. The deliberately raw views (the "Args"
    block on a tool call and the bottom-of-page "Raw JSON" expander)
    keep ``st.json`` because they exist precisely to expose the wire
    shape.
    """
    for k, v in d.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        key = k.replace("_", " ").capitalize()
        if isinstance(v, dict):
            st.markdown(f"**{key}:**")
            with st.container(border=True):
                _render_kv_block(v)
        elif isinstance(v, list):
            st.markdown(f"**{key}:**")
            for item in v:
                if isinstance(item, dict):
                    with st.container(border=True):
                        _render_kv_block(item)
                else:
                    st.markdown(f"- {item}")
        elif isinstance(v, bool):
            st.markdown(f"**{key}:** `{str(v).lower()}`")
        else:
            st.markdown(f"**{key}:** {v}")


def _render_value(v) -> None:
    """Render an agent-produced value (str / dict / list / None) safely.

    Strings → ``st.write`` (markdown-aware). Dicts and lists → the labeled
    block renderer above. ``st.json`` is reserved for the explicit raw
    viewers; routing everything through it leaks JSON braces into the
    prose-y main flow.
    """
    if v is None:
        st.caption("_(none)_")
    elif isinstance(v, dict):
        _render_kv_block(v)
    elif isinstance(v, list):
        if not v:
            st.caption("_(empty)_")
            return
        for i, item in enumerate(v):
            if isinstance(item, dict):
                if i > 0:
                    st.markdown("---")
                _render_kv_block(item)
            else:
                st.markdown(f"- {item}")
    else:
        st.write(v)


def _parse_iso(ts: str) -> datetime | None:
    """Parse the project's ISO-like timestamp (`YYYY-MM-DDTHH:MM:SSZ`)."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def _duration_seconds(start: str, end: str) -> int:
    s, e = _parse_iso(start), _parse_iso(end)
    if not s or not e:
        return 0
    return max(0, int((e - s).total_seconds()))


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _fmt_confidence_badge(conf: float | None) -> str:
    """Inline coloured badge for an agent confidence value.

    Green ≥0.75, amber 0.5–0.75, red <0.5, grey when None. Markdown only —
    no HTML — so the badge survives Streamlit's sanitizer.
    """
    if conf is None:
        return "⚪ confidence —"
    if conf >= 0.75:
        glyph = "🟢"
    elif conf >= 0.5:
        glyph = "🟡"
    else:
        glyph = "🔴"
    return f"{glyph} confidence {conf:.2f}"


def _render_hypothesis_list(items: list, label: str) -> None:
    """Render a list of hypothesis-shaped dicts (cause/evidence/next_steps)
    as bordered cards. Strings or scalar entries fall back to bullets.
    """
    for i, h in enumerate(items, 1):
        with st.container(border=True):
            if isinstance(h, dict):
                st.markdown(f"**{label} {i}:** {h.get('cause', '—')}")
                ev = h.get("evidence")
                if ev:
                    st.markdown("**Evidence:**")
                    for e in (ev if isinstance(ev, list) else [ev]):
                        st.markdown(f"- {e}")
                ns = h.get("next_steps") or h.get("next_step") or h.get("probe")
                if ns:
                    st.markdown(f"**Next steps:** {ns}")
                # Anything else in the dict that we haven't surfaced.
                extra = {k: v for k, v in h.items()
                         if k not in {"cause", "evidence", "next_steps",
                                      "next_step", "probe"}}
                if extra:
                    _render_kv_block(extra)
            else:
                st.markdown(f"**{label} {i}:** {h}")


def _render_findings_section(value, label: str) -> None:
    """Findings can be a list of hypothesis dicts, a single dict, or free
    prose. Pick the right renderer; never silently truncate.
    """
    if isinstance(value, list):
        _render_hypothesis_list(value, label="Hypothesis")
    elif isinstance(value, dict):
        _render_kv_block(value)
    elif isinstance(value, str):
        st.write(value)
    else:
        _render_value(value)


def render_incident_detail(store: IncidentStore) -> None:
    inc_id = st.session_state.get("selected_incident")
    if not inc_id:
        return
    with st.expander(f"INC detail: {inc_id}", expanded=True):
        inc = store.load(inc_id).model_dump()

        # --- Top status / severity / category badges ----------------------
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            st.caption("Status")
            _status_badge(inc.get("status"))
        with b2:
            st.caption("Severity")
            _severity_badge(inc.get("severity"))
        with b3:
            st.caption("Category")
            _category_badge(inc.get("category"))

        # Prominent action-required call-out when the gate has paused the
        # graph. Sits above the (separate) intervention prompt block so the
        # user can't miss it on a long page.
        if inc.get("status") == "awaiting_input":
            st.warning(
                "**Human intervention required.** The intervention gate "
                "paused this INC because deep-investigator confidence was "
                "below the configured threshold. Use the controls below to "
                "resume with input, escalate, or stop."
            )

        # --- Numeric metrics ----------------------------------------------
        token_total = (inc.get("token_usage") or {}).get("total_tokens", 0)
        duration_s = _duration_seconds(inc.get("created_at", ""),
                                       inc.get("updated_at", ""))
        m1, m2 = st.columns(2)
        m1.metric("Total tokens", _fmt_tokens(token_total))
        m2.metric("Duration", f"{duration_s}s")

        # --- Header block -------------------------------------------------
        st.markdown(f"**Query:** {inc['query']}")
        st.markdown(f"**Environment:** `{inc['environment']}`")
        if inc.get("tags"):
            st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in inc["tags"]))
        if inc.get("summary"):
            st.markdown(f"**Summary:** {inc['summary']}")
        if inc.get("matched_prior_inc"):
            tags = inc.get("tags") or []
            if "hypothesis:prior_match_supported" in tags:
                stance = "supported by current evidence"
                callout = st.success
            elif "hypothesis:prior_match_rejected" in tags:
                stance = "rejected — fresh evidence diverges from prior cause"
                callout = st.warning
            else:
                stance = "not yet validated"
                callout = st.info
            callout(
                f"**Prior similar incident (hypothesis):** "
                f"`{inc['matched_prior_inc']}` — {stance}.  \n"
                f"_Same symptom can have different root causes "
                f"(code bug vs. network vs. resource overload), so the prior "
                f"cause is one ranked hypothesis for the deep investigator — "
                f"not the answer._"
            )

        # --- Intervention prompt (only when paused on low confidence) -----
        if inc.get("status") == "awaiting_input" and inc.get("pending_intervention"):
            _render_intervention_block(inc, inc_id)

        # --- Agents run ---------------------------------------------------
        agents_run = inc.get("agents_run", [])
        if agents_run:
            st.markdown("### Agents run")
            for ar in agents_run:
                a_dur = _duration_seconds(ar.get("started_at", ""),
                                          ar.get("ended_at", ""))
                a_tok = (ar.get("token_usage") or {}).get("total_tokens", 0)
                conf = ar.get("confidence")
                badge = _fmt_confidence_badge(conf)
                with st.container(border=True):
                    st.markdown(
                        f"**{ar['agent']}** — {a_dur}s — "
                        f"{_fmt_tokens(a_tok)} tokens — {badge}"
                    )
                    rationale = ar.get("confidence_rationale")
                    if rationale:
                        st.caption(f"Why: {rationale}")
                    st.write(ar.get("summary") or "_(no summary)_")

        # --- Findings -----------------------------------------------------
        findings = inc.get("findings") or {}
        f_triage = findings.get("triage")
        f_di = findings.get("deep_investigator")
        if f_triage is not None or f_di is not None:
            st.markdown("### Findings")
        if f_triage is not None:
            with st.container(border=True):
                st.markdown("**Triage**")
                _render_findings_section(f_triage, label="Finding")
        if f_di is not None:
            with st.container(border=True):
                st.markdown("**Deep investigator**")
                _render_findings_section(f_di, label="Hypothesis")

        # --- Resolution ---------------------------------------------------
        if inc.get("resolution") is not None:
            st.markdown("### Resolution")
            with st.container(border=True):
                _render_value(inc["resolution"])

        # --- Tool calls ---------------------------------------------------
        tool_calls = inc.get("tool_calls", [])
        if tool_calls:
            st.markdown("### Tool calls")
            for idx, tc in enumerate(tool_calls):
                with st.expander(f"`{tc['agent']}` → `{tc['tool']}`"):
                    st.markdown("**Args:**")
                    st.json(tc.get("args") or {})
                    st.markdown("**Result:**")
                    _render_value(tc.get("result"))

        with st.expander("Raw JSON"):
            st.json(inc)


def _format_event(ev: dict) -> str | None:
    kind = ev.get("event")
    node = ev.get("node") or ""
    ts = ev.get("ts", "")
    if kind == "investigation_started":
        return f"[{ts}] start  inc={ev.get('incident_id')}"
    if kind == "investigation_completed":
        return f"[{ts}] done   inc={ev.get('incident_id')}"
    if kind == "on_chain_start" and node in {"intake", "triage", "deep_investigator", "resolution"}:
        return f"[{ts}] enter  {node}"
    if kind == "on_chain_end" and node in {"intake", "triage", "deep_investigator", "resolution"}:
        return f"[{ts}] exit   {node}"
    if kind == "on_tool_start":
        return f"[{ts}] tool   {node}"
    if kind == "on_tool_end":
        result = (ev.get("data") or {}).get("output")
        snippet = str(result)[:120] if result is not None else ""
        return f"[{ts}] tool→  {node} {snippet}"
    return None


async def _run_investigation_async(cfg: AppConfig, query: str, environment: str,
                                   log_area, lines: list[str]) -> None:
    """Build a fresh Orchestrator, stream events, aclose. One asyncio.run frame."""
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.stream_investigation(query=query, environment=environment):
            line = _format_event(ev)
            if line:
                lines.append(line)
                log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()


async def _resume_async(cfg: AppConfig, inc_id: str, decision: dict,
                        log_area, lines: list[str]) -> dict:
    """Build a fresh Orchestrator, stream resume events, aclose.

    Returns a small summary dict describing the outcome so the caller can show
    a banner: ``{"rejected": <reason or None>}``. ``rejected`` is set when the
    orchestrator emits a ``resume_rejected`` event (e.g. INC no longer
    awaiting_input, invalid escalation team).
    """
    outcome: dict = {"rejected": None}
    orch = await Orchestrator.create(cfg)
    try:
        async for ev in orch.resume_investigation(inc_id, decision):
            kind = ev.get("event")
            ts = ev.get("ts", "")
            if kind == "resume_started":
                lines.append(f"[{ts}] resume {ev.get('action')}")
            elif kind == "resume_completed":
                lines.append(f"[{ts}] done   status={ev.get('status')}")
            elif kind == "resume_rejected":
                lines.append(f"[{ts}] rejected {ev.get('reason')}")
                outcome["rejected"] = ev.get("reason")
            elif kind == "resume_failed":
                lines.append(f"[{ts}] failed {ev.get('error')}")
                outcome["rejected"] = ev.get("error")
            else:
                line = _format_event(ev)
                if line:
                    lines.append(line)
            log_area.code("\n".join(lines), language="text")
    finally:
        await orch.aclose()
    return outcome


def _render_intervention_block(inc: dict, inc_id: str) -> None:
    """Render the intervention prompt above the agents_run section.

    Shows the confidence vs. threshold and a single form with an action
    selector that swaps inputs (text box / team dropdown / nothing) and a
    submit button. On submit, calls `_resume_async` and then reruns.
    """
    cfg = load_config(CONFIG_PATH)
    pi = inc.get("pending_intervention") or {}
    conf = pi.get("confidence")
    threshold = pi.get("threshold", 0.75)
    teams = pi.get("escalation_teams") or list(cfg.intervention.escalation_teams)

    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
    with st.container(border=True):
        st.markdown(
            f"#### 🟠 Intervention required — confidence {conf_str} "
            f"< threshold {threshold:.2f}"
        )
        st.caption(
            "The deep investigator's confidence is below the configured "
            "threshold. Choose how to proceed."
        )
        action = st.selectbox(
            "Action", ["resume_with_input", "escalate", "stop"],
            key=f"intervention_action_{inc_id}",
        )
        decision: dict = {"action": action}
        if action == "resume_with_input":
            decision["input"] = st.text_area(
                "Add context for the investigator", height=120,
                placeholder="Anything the agent should know — recent changes, "
                            "logs you've already checked, suspected services…",
                key=f"intervention_input_{inc_id}",
            )
        elif action == "escalate":
            decision["team"] = st.selectbox(
                "Escalate to team", teams, key=f"intervention_team_{inc_id}",
            )

        submit = st.button("Submit", type="primary",
                           key=f"intervention_submit_{inc_id}")
        if submit:
            if action == "resume_with_input" and not (decision.get("input") or "").strip():
                st.warning("Add some context before resuming.")
                return
            log_area = st.empty()
            lines: list[str] = []
            outcome = asyncio.run(_resume_async(cfg, inc_id, decision, log_area, lines))
            if outcome.get("rejected"):
                # Don't auto-rerun — let the user read the warning before the
                # form goes away. Common causes: INC already closed, invalid
                # escalation team, or a sub-graph exception that restored the
                # INC to awaiting_input.
                st.warning(f"Resume rejected: {outcome['rejected']}")
                return
            st.success(f"Resume complete (action: {action}).")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="ASR — Agent Orchestrator", layout="wide")
    cfg = load_config(CONFIG_PATH)
    store = IncidentStore(cfg.paths.incidents_dir)

    # One-shot snapshot of agent/tool metadata + environments. ~100-200ms per
    # rerun; acceptable, and keeps async resources strictly scoped.
    agents, tools, environments = _load_metadata_dicts(cfg)

    render_sidebar(store)

    tab_investigate, tab_registry = st.tabs(["Investigate", "Agents & Tools"])

    with tab_investigate:
        st.header("Start an investigation")
        with st.form("investigate_form"):
            query = st.text_area("What's happening?", height=100, key="form_query")
            environment = st.selectbox("Impacted environment", environments,
                                       key="form_env")
            submitted = st.form_submit_button("Start investigation", type="primary")

        if submitted and query.strip():
            timeline_box = st.container()
            timeline_box.markdown("### Live timeline")
            log_area = timeline_box.empty()
            lines: list[str] = []

            asyncio.run(_run_investigation_async(cfg, query, environment, log_area, lines))

            # Surface the resulting INC for one-click drill-in
            recent = [i.model_dump() for i in store.list_recent(1)]
            if recent:
                st.session_state["selected_incident"] = recent[0]["id"]
                st.success(f"Investigation complete — {recent[0]['id']} ({recent[0]['status']})")
                st.rerun()

    with tab_registry:
        st.header("Agents & Tools registry")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Agents")
            for a in agents:
                with st.container(border=True):
                    st.markdown(f"**{a['name']}** — `{a['model']}`")
                    st.caption(a["description"])
                    st.markdown("Tools: " + ", ".join(f"`{t}`" for t in a["tools"]))
                    if a["routes"]:
                        st.caption("Routes: " + ", ".join(
                            f"`{r['when']}→{r['next']}`" for r in a["routes"]))

        with col_b:
            st.subheader("Tools by category")
            by_cat: dict[str, list[dict]] = {}
            for t in tools:
                by_cat.setdefault(t["category"], []).append(t)
            for cat in sorted(by_cat):
                st.markdown(f"**{cat}**")
                for t in by_cat[cat]:
                    bound = ", ".join(f"`{a}`" for a in t["bound_agents"]) or "_(unbound)_"
                    st.markdown(f"- `{t['name']}` — {t['description'][:80]}  \n  bound to: {bound}")

    render_incident_detail(store)


if __name__ == "__main__":
    main()
