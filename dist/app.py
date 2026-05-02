from __future__ import annotations
# ----- imports for config.py -----
"""Config schemas for the orchestrator."""

import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
import yaml


# ----- imports for incident.py -----
"""Incident domain model."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field

# ----- imports for similarity.py -----
"""Similarity scoring for incident matching."""

from typing import Protocol

# ----- imports for skill.py -----
"""Skill loader.

Each agent lives in its own subdirectory under ``config/skills/``::

    config/skills/
      _common/                # OPTIONAL: prompt fragments shared by all agents
        confidence.md         # appended to every agent's system_prompt, in
        output.md             # alphabetical order, joined with blank lines
      intake/
        config.yaml           # description, tools, routes, model
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

from pydantic import BaseModel, Field, field_validator


# ----- imports for llm.py -----
"""LLM provider abstraction with stub/ollama/azure_openai backends.

Models are resolved by name from ``LLMConfig``. Each named entry binds a
provider (kind + connection) to a model id and optional temperature/deployment.
``get_llm(cfg, "smart")`` looks up ``cfg.models["smart"]`` and uses its
referenced ``cfg.providers[<name>]`` to build a langchain ``BaseChatModel``.
"""

from typing import Any
from uuid import uuid4
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr



# ----- imports for storage/types.py -----
"""Custom SQLAlchemy column types that bridge SQLite and Postgres.

- ``VectorColumn(dim)`` — ``pgvector.sqlalchemy.Vector(dim)`` on Postgres,
  ``LargeBinary`` (numpy float32 bytes) on SQLite. Python value is always
  ``list[float] | None`` regardless of dialect.
- ``JSONColumn`` — thin alias over SQLAlchemy ``JSON``; auto-routes to
  ``JSONB`` on Postgres, ``TEXT`` on SQLite. Centralized so we can adjust
  serialization (e.g. datetime support) in one place later.
"""

import numpy as np
from sqlalchemy import JSON, LargeBinary
from sqlalchemy.types import TypeDecorator


# ----- imports for storage/models.py -----
"""SQLAlchemy declarative model for the ``incidents`` table.

Hybrid schema: scalar/queryable fields as columns, nested Pydantic
structures as JSON columns (JSONB on Postgres, TEXT on SQLite), and a
native vector column for embeddings.
"""

from datetime import datetime
from sqlalchemy import DateTime, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column



# ----- imports for storage/engine.py -----
"""SQLAlchemy engine factory + sqlite-vec extension loader.

Behaviour
---------
- ``sqlite://``     → engine with ``NullPool``, ``check_same_thread=False``,
                      and a ``connect`` event hook that loads sqlite-vec into
                      every new dbapi connection.
- ``postgresql://`` → engine with the configured pool size, plus a
                      one-time ``CREATE EXTENSION IF NOT EXISTS vector``.
"""

import ctypes
import ctypes.util
from sqlalchemy import event, text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool


# Python 3.14 on many distros is compiled without SQLITE_ENABLE_LOAD_EXTENSION,
# so conn.enable_load_extension / conn.load_extension don't exist as Python
# methods.  We call the underlying C functions directly via ctypes instead.
# ----- imports for storage/embeddings.py -----
"""LangChain ``Embeddings`` facade + deterministic stub for tests.

Construction is config-driven via :func:`build_embedder`. Provider kind
dispatches to ``OllamaEmbeddings`` / ``AzureOpenAIEmbeddings`` / a local
stub. Stubs are deterministic so tests can assert similarity ordering
without external services.
"""

import hashlib



# ----- imports for storage/repository.py -----
"""SQLAlchemy-backed Incident store.

Public methods mirror the previous JSON ``IncidentStore`` 1:1 so call
sites in the MCP server and orchestrator change minimally. The repository
also owns the embedder; ``find_similar`` (Task G) does the dialect dispatch.
"""

from typing import Optional

from sqlalchemy import and_, desc, func, literal, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session



# ----- imports for mcp_servers/incident.py -----
"""FastMCP server: incident_management tools, backed by IncidentRepository.

State scoping
-------------
Each Orchestrator constructs its own :class:`IncidentMCPServer`, so two
orchestrators in the same process do not share repository state. The
module-level ``mcp`` and ``set_state`` symbols are kept as a back-compat
surface for the MCP loader (``getattr(mod, "mcp")``) and for tests that
import these names directly.
"""

from dataclasses import dataclass, field
from fastmcp import FastMCP



# ----- imports for mcp_servers/observability.py -----
"""FastMCP server: observability mock tools."""

from datetime import datetime, timezone, timedelta

# ----- imports for mcp_servers/remediation.py -----
"""FastMCP server: remediation mock tools."""


# ----- imports for mcp_servers/user_context.py -----
"""FastMCP server: user_context mock tool."""


# ----- imports for mcp_loader.py -----
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

import importlib
from contextlib import AsyncExitStack
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools



# ----- imports for graph.py -----
"""LangGraph state, routing helpers, and node runner."""

import asyncio
import logging
from typing import TypedDict, Callable, Awaitable

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END






# ----- imports for orchestrator.py -----
"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""

from typing import AsyncIterator










# ----- imports for api.py -----
"""FastAPI app — health, listings, and incident endpoints.

``build_app(cfg)`` is sync and constructs the FastAPI instance. The
orchestrator (which holds long-lived FastMCP transports) is created during
the app's startup lifespan and stored on ``app.state.orchestrator``; the
shutdown hook closes it cleanly. Routes read the orchestrator via
``app.state``.

The module-level ``get_app()`` is a no-arg factory suitable for
``uvicorn --factory``: it reads ``ASR_CONFIG`` (default
``config/config.yaml``) and returns a fresh app.
"""

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel






# ====== module: orchestrator/config.py ======

ProviderKind = Literal["ollama", "azure_openai", "stub"]


class ProviderConfig(BaseModel):
    """Connection settings for one upstream LLM provider.

    Multiple named ``ModelConfig`` entries can reference the same provider
    so that, e.g., two Ollama models share a single base_url + api_key.
    """
    kind: ProviderKind
    base_url: str | None = None       # ollama
    api_key: str | None = None        # ollama, azure_openai
    endpoint: str | None = None       # azure_openai
    api_version: str | None = None    # azure_openai


class ModelConfig(BaseModel):
    """Named chat model entry. ``provider`` references a key in ``LLMConfig.providers``."""
    provider: str
    model: str = ""           # raw upstream model id (ignored for stub kind)
    temperature: float = 0.0
    deployment: str | None = None  # azure_openai


class EmbeddingConfig(BaseModel):
    """Single embedding model. ``provider`` references a key in ``LLMConfig.providers``."""
    provider: str
    model: str
    deployment: str | None = None  # azure_openai
    dim: int = 1024


class LLMConfig(BaseModel):
    """Named-model registry. Skills reference chat models by name; the orchestrator
    resolves name → model entry → provider entry at LLM build time.

    ``default`` is used when a skill's ``model`` field is ``None``.
    ``embedding`` is the single embedding model (for similarity / retrieval).
    """
    default: str = "stub_default"
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {"stub": ProviderConfig(kind="stub")}
    )
    models: dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "stub_default": ModelConfig(provider="stub", model="stub-1"),
        }
    )
    embedding: EmbeddingConfig | None = None

    @model_validator(mode="after")
    def _validate_refs(self) -> "LLMConfig":
        if self.default not in self.models:
            raise ValueError(
                f"llm.default={self.default!r} not found in llm.models "
                f"(known: {sorted(self.models)})"
            )
        for name, m in self.models.items():
            if m.provider not in self.providers:
                raise ValueError(
                    f"llm.models[{name!r}].provider={m.provider!r} not found "
                    f"in llm.providers (known: {sorted(self.providers)})"
                )
        if self.embedding and self.embedding.provider not in self.providers:
            raise ValueError(
                f"llm.embedding.provider={self.embedding.provider!r} not found "
                f"in llm.providers (known: {sorted(self.providers)})"
            )
        return self

    @classmethod
    def stub(cls) -> "LLMConfig":
        """Convenience factory for tests/CI — single stub model."""
        return cls()


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


class StorageConfig(BaseModel):
    """Database backend. SQLite (with sqlite-vec) for dev, Postgres (with pgvector) for prod."""
    url: str = "sqlite:///incidents.db"
    pool_size: int = 5      # postgres only; sqlite uses NullPool
    echo: bool = False


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


class OrchestratorConfig(BaseModel):
    entry_agent: str = "intake"
    # Signals an agent may emit (via ``update_incident.patch.signal``) that
    # the router will accept and look up against the skill's ``routes`` table.
    # Anything outside this set falls through to ``when: default``. Override
    # in YAML to extend the vocabulary; the default keeps current behaviour.
    signals: list[str] = Field(
        default_factory=lambda: ["success", "failed", "needs_input"],
    )
    # Mapping from raw severity inputs to canonical severity labels.
    # Override in YAML to adapt to domain-specific taxonomies.
    # Default reproduces the original hardcoded _SEVERITY_MAP in incident.py.
    severity_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
            "critical": "high", "urgent": "high", "high": "high",
            "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
            "sev4": "low", "p4": "low", "info": "low", "informational": "low",
            "low": "low",
        }
    )


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    incidents: IncidentConfig = Field(default_factory=IncidentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"]
    )
    paths: Paths = Field(default_factory=Paths)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
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

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


IncidentStatus = Literal[
    "new", "in_progress", "matched", "resolved",
    "escalated", "awaiting_input", "stopped", "deleted",
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
    signal: str | None = None


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
    # Findings is an open mapping keyed by agent name (or any agent-declared
    # output key). Old saves with {"triage": ..., "deep_investigator": ...}
    # load transparently because Pydantic accepts those as dict entries.
    findings: dict[str, Any] = Field(default_factory=dict)
    resolution: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
    deleted_at: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_UTC_TS_FMT)


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
        if not _INC_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected INC-YYYYMMDD-NNN"
            )
        incident.updated_at = _utc_now_iso()
        path = self.base_dir / f"{incident.id}.json"
        path.write_text(incident.model_dump_json(indent=2))

    def load(self, incident_id: str) -> Incident:
        if not _INC_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected INC-YYYYMMDD-NNN"
            )
        path = self.base_dir / f"{incident_id}.json"
        if not path.exists():
            raise FileNotFoundError(incident_id)
        return Incident.model_validate_json(path.read_text())

    def list_all(self) -> list[Incident]:
        return [self.load(p.stem) for p in self.base_dir.glob("INC-*.json")]

    def list_recent(self, limit: int = 20,
                    include_deleted: bool = False) -> list[Incident]:
        all_inc = self.list_all()
        if not include_deleted:
            all_inc = [i for i in all_inc if i.status != "deleted"]
        all_inc.sort(key=lambda i: (i.created_at, i.id), reverse=True)
        return all_inc[:limit]

    def delete(self, incident_id: str) -> Incident:
        """Soft-delete: mark status='deleted' + set deleted_at timestamp.

        Idempotent — re-deleting a deleted INC returns it unchanged. The
        JSON file is preserved for audit; ``list_recent`` hides it by
        default.
        """
        inc = self.load(incident_id)
        if inc.status == "deleted":
            return inc
        inc.status = "deleted"
        inc.deleted_at = _utc_now_iso()
        inc.pending_intervention = None
        self.save(inc)
        return inc

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
    gate: str | None = None


class Skill(BaseModel):
    name: str
    description: str
    model: str | None = None
    tools: dict[str, list[str]] = Field(default_factory=dict)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str
    stub_response: str | None = None
    """Per-skill canned response used by ``StubChatModel`` when
    ``provider.kind == "stub"``.  Takes precedence over any entry in
    ``_DEFAULT_STUB_CANNED`` for the same agent name."""

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for server, names in v.items():
            if not names:
                raise ValueError(
                    f"empty tool list for server {server!r}; "
                    f"remove the key or use ['*']"
                )
            if "*" in names and len(names) != 1:
                raise ValueError(
                    f"'*' must be the sole entry for server "
                    f"{server!r}; got {names!r}"
                )
        return v

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


def _build_ollama_chat(provider: ProviderConfig, model_id: str,
                       temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    kwargs: dict[str, Any] = {
        "base_url": provider.base_url or "https://ollama.com",
        "model": model_id,
        "temperature": temperature,
    }
    api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
    if api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return ChatOllama(**kwargs)


def _build_azure_chat(provider: ProviderConfig, model: ModelConfig) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI
    if provider.endpoint is None:
        raise ValueError("azure_openai provider requires 'endpoint'")
    if model.deployment is None:
        raise ValueError(
            f"azure_openai model {model.model!r} requires 'deployment'"
        )
    _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
    return AzureChatOpenAI(
        azure_endpoint=provider.endpoint,
        api_version=provider.api_version or "2024-08-01-preview",
        azure_deployment=model.deployment,
        api_key=SecretStr(_ak) if _ak else None,
        temperature=model.temperature,
    )


def get_llm(cfg: LLMConfig, model_name: str | None = None, *,
            role: str = "default",
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None) -> BaseChatModel:
    """Build a chat model by named entry from ``cfg.models``.

    ``model_name`` defaults to ``cfg.default``. Validation that the name
    exists is enforced by ``LLMConfig`` itself (model_validator), so a
    missing name here means caller passed a typo — raise loudly.
    """
    name = model_name or cfg.default
    model = cfg.models.get(name)
    if model is None:
        raise KeyError(
            f"llm model {name!r} not found in llm.models "
            f"(known: {sorted(cfg.models)})"
        )
    provider = cfg.providers[model.provider]  # validated at config load

    if provider.kind == "stub":
        return StubChatModel(
            role=role,
            canned_responses=stub_canned or {},
            tool_call_plan=stub_tool_plan,
        )
    if provider.kind == "ollama":
        return _build_ollama_chat(provider, model.model, model.temperature)
    if provider.kind == "azure_openai":
        return _build_azure_chat(provider, model)
    raise ValueError(f"Unknown provider kind: {provider.kind!r}")


def get_embedding(cfg: LLMConfig) -> Embeddings:
    """Build the configured embedding model. Raises if ``cfg.embedding`` is None."""
    if cfg.embedding is None:
        raise ValueError("llm.embedding is not configured")
    provider = cfg.providers[cfg.embedding.provider]
    if provider.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        kwargs: dict[str, Any] = {
            "base_url": provider.base_url or "https://ollama.com",
            "model": cfg.embedding.model,
        }
        api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return OllamaEmbeddings(**kwargs)
    if provider.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        if provider.endpoint is None:
            raise ValueError("azure_openai provider requires 'endpoint'")
        deployment = cfg.embedding.deployment or cfg.embedding.model
        _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
        return AzureOpenAIEmbeddings(
            azure_endpoint=provider.endpoint,
            api_version=provider.api_version or "2024-08-01-preview",
            azure_deployment=deployment,
            api_key=SecretStr(_ak) if _ak else None,
        )
    raise ValueError(
        f"Embedding not supported for provider kind {provider.kind!r}"
    )

# ====== module: orchestrator/storage/types.py ======

JSONColumn = JSON  # SQLAlchemy auto-dialects: JSONB on pg, TEXT on sqlite.


class VectorColumn(TypeDecorator):
    """Vector column backed by pgvector on Postgres, BLOB on SQLite.

    Python value: ``list[float] | None``.
    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.dim,):
            raise ValueError(
                f"vector dim {arr.shape[0]} != column dim {self.dim}"
            )
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return np.frombuffer(value, dtype=np.float32).tolist()

# ====== module: orchestrator/storage/models.py ======

EMBEDDING_DIM = 1024  # bge-m3; if you change embed model, re-embed corpus.


class Base(DeclarativeBase):
    pass


class IncidentRow(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(String, nullable=False)
    reporter_id: Mapped[str] = mapped_column(String, nullable=False)
    reporter_team: Mapped[str] = mapped_column(String, nullable=False)

    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    matched_prior_inc: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)

    tags: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    agents_run: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)
    findings: Mapped[dict] = mapped_column(JSONColumn, nullable=False, default=dict)
    pending_intervention: Mapped[dict | None] = mapped_column(JSONColumn, nullable=True)
    user_inputs: Mapped[list] = mapped_column(JSONColumn, nullable=False, default=list)

    embedding: Mapped[list[float] | None] = mapped_column(
        VectorColumn(EMBEDDING_DIM), nullable=True
    )

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_incidents_status_env_active", "status", "environment",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
        Index("ix_incidents_created_at_active", "created_at",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
    )

# ====== module: orchestrator/storage/engine.py ======

_libsqlite = ctypes.CDLL(ctypes.util.find_library("sqlite3"))
_libsqlite.sqlite3_enable_load_extension.restype = ctypes.c_int
_libsqlite.sqlite3_enable_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_int]
_libsqlite.sqlite3_load_extension.restype = ctypes.c_int
_libsqlite.sqlite3_load_extension.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
]
# The CPython sqlite3.Connection C struct begins with PyObject header
# (ob_refcnt + ob_type = 2 pointers), followed immediately by sqlite3 *db.
_DB_PTR_OFFSET = 2 * ctypes.sizeof(ctypes.c_void_p)


def _ctypes_load_vec(dbapi_conn) -> None:  # type: ignore[misc]
    """Load sqlite-vec into *dbapi_conn* using the C-level SQLite API.

    Required because CPython may be built without SQLITE_ENABLE_LOAD_EXTENSION,
    which removes the Python-level enable_load_extension / load_extension methods
    but leaves the underlying C functions available in libsqlite3.
    """
    import sqlite_vec
    db_ptr = ctypes.c_void_p.from_address(id(dbapi_conn) + _DB_PTR_OFFSET).value
    _libsqlite.sqlite3_enable_load_extension(db_ptr, 1)
    errmsg = ctypes.c_char_p()
    path = sqlite_vec.loadable_path().encode()
    rc = _libsqlite.sqlite3_load_extension(db_ptr, path, None, ctypes.byref(errmsg))
    _libsqlite.sqlite3_enable_load_extension(db_ptr, 0)
    if rc != 0:
        raise RuntimeError(f"sqlite3_load_extension failed: {errmsg.value!r}")


def _attach_sqlite_vec(engine: Engine) -> None:
    """Register the sqlite-vec loader on every new SQLite dbapi connection."""
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):  # type: ignore[misc]
        _ctypes_load_vec(dbapi_conn)


def _ensure_pgvector(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def build_engine(cfg: StorageConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )
        _attach_sqlite_vec(engine)
        return engine
    engine = create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)
    _ensure_pgvector(engine)
    return engine

# ====== module: orchestrator/storage/embeddings.py ======

class _StubEmbeddings(Embeddings):
    """Deterministic dummy embedder.

    Same text → same vector; different texts → different vectors. Useful
    for CI and unit tests without a network or model server.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(
    cfg: EmbeddingConfig | None,
    providers: dict[str, ProviderConfig],
) -> Embeddings | None:
    """Build a LangChain ``Embeddings`` from config; ``None`` if not configured."""
    if cfg is None:
        return None
    p = providers[cfg.provider]
    if p.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.model,
            base_url=p.base_url or "http://localhost:11434",
        )
    if p.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=cfg.deployment,
            model=cfg.model,
            azure_endpoint=p.endpoint,
            api_version=p.api_version,
            api_key=p.api_key,
        )
    if p.kind == "stub":
        return _StubEmbeddings(dim=cfg.dim)
    raise ValueError(f"unknown provider kind: {p.kind!r}")

# ====== module: orchestrator/storage/repository.py ======

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now().strftime("%Y%m%d")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """DB datetime -> Incident model ISO string (UTC, 'Z' suffix)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Incident ISO string -> DB datetime (UTC-aware)."""
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class IncidentRepository:
    """SQLAlchemy-backed Incident store. Drop-in for the old ``IncidentStore``.

    Threading note: methods open short-lived sessions; safe for the
    orchestrator's coarse-grained concurrency model.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        embedder: Optional[Embeddings] = None,
        similarity_threshold: float = 0.85,
        severity_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.severity_aliases = severity_aliases or {}

    # ---------- ID minting ----------
    def _next_id(self, session: Session) -> str:
        prefix = f"INC-{_today_str()}-"
        like = f"{prefix}%"
        rows = session.execute(
            select(IncidentRow.id).where(IncidentRow.id.like(like))
        ).scalars().all()
        max_seq = 0
        for r in rows:
            try:
                max_seq = max(max_seq, int(r.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"{prefix}{max_seq + 1:03d}"

    # ---------- public API ----------
    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> Incident:
        with Session(self.engine) as session:
            now = _now()
            inc_id = self._next_id(session)
            row = IncidentRow(
                id=inc_id,
                status="new",
                created_at=now,
                updated_at=now,
                query=query,
                environment=environment,
                reporter_id=reporter_id,
                reporter_team=reporter_team,
                summary="",
                tags=[],
                agents_run=[],
                tool_calls=[],
                findings={},
                user_inputs=[],
                embedding=self._maybe_embed(query),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    def load(self, incident_id: str) -> Incident:
        if not _INC_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected INC-YYYYMMDD-NNN"
            )
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def save(self, incident: Incident) -> None:
        if not _INC_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected INC-YYYYMMDD-NNN"
            )
        incident.updated_at = _iso(_now())
        with Session(self.engine) as session:
            existing = session.get(IncidentRow, incident.id)
            new_embedding = self._compute_save_embedding(existing, incident)
            data = self._incident_to_row_dict(incident, embedding=new_embedding)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()

    def delete(self, incident_id: str) -> Incident:
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            if row.status != "deleted":
                row.status = "deleted"
                row.deleted_at = _now()
                row.pending_intervention = None
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    def list_all(self, *, include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_recent(self, limit: int = 20, *,
                    include_deleted: bool = False) -> list[Incident]:
        with Session(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            stmt = stmt.order_by(
                desc(IncidentRow.created_at), desc(IncidentRow.id)
            ).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    # ---------- similarity search ----------
    def find_similar(
        self, *, query: str, environment: str,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
    ) -> list[tuple[Incident, float]]:
        """Return up to ``limit`` similar resolved incidents for the same env.

        Embedding path uses native vector ops (pgvector ``cosine_distance`` /
        sqlite-vec ``vec_distance_cosine``). Keyword path falls back to
        the existing ``KeywordSimilarity`` scorer to preserve behaviour
        when no embedder is configured.
        """
        if self.embedder is None:
            return self._keyword_similar(
                query=query, environment=environment,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        return self._vector_similar(
            query=query, environment=environment,
            status_filter=status_filter,
            threshold=threshold, limit=limit,
        )

    def _vector_similar(self, *, query, environment, status_filter, threshold, limit):
        import numpy as np
        vec = self.embedder.embed_query(query)
        threshold = self.similarity_threshold if threshold is None else threshold
        with Session(self.engine) as session:
            if self.engine.dialect.name == "postgresql":
                score = (literal(1.0) - IncidentRow.embedding.cosine_distance(vec)).label("score")
            else:
                blob = np.asarray(vec, dtype=np.float32).tobytes()
                score = (literal(1.0) - func.vec_distance_cosine(IncidentRow.embedding, blob)).label("score")
            stmt = (
                select(IncidentRow, score)
                .where(and_(
                    IncidentRow.deleted_at.is_(None),
                    IncidentRow.status == status_filter,
                    IncidentRow.environment == environment,
                    IncidentRow.embedding.is_not(None),
                ))
                .order_by(desc("score"))
                .limit(limit)
            )
            rows = session.execute(stmt).all()
        out: list[tuple[Incident, float]] = []
        for row, s in rows:
            s = float(s)
            if s < threshold:
                continue
            out.append((self._row_to_incident(row), s))
        return out

    def _keyword_similar(self, *, query, environment, status_filter, threshold, limit):

        candidates_inc = [
            i for i in self.list_all()
            if i.environment == environment
            and i.status == status_filter
            and i.deleted_at is None
        ]
        candidates = [
            {"id": i.id, "text": f"{i.query} {i.summary} {' '.join(i.tags)}",
             "incident": i}
            for i in candidates_inc
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(c["incident"], float(s)) for c, s in results]

    # ---------- mapping helpers ----------
    def _row_to_incident(self, row: IncidentRow) -> Incident:
        agents_run = [AgentRun.model_validate(a) for a in (row.agents_run or [])]
        tool_calls = [ToolCall.model_validate(t) for t in (row.tool_calls or [])]
        token_usage = TokenUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
        )
        return Incident(
            id=row.id,
            status=row.status,
            created_at=_iso(row.created_at),
            updated_at=_iso(row.updated_at),
            deleted_at=_iso(row.deleted_at) if row.deleted_at else None,
            query=row.query,
            environment=row.environment,
            reporter=Reporter(id=row.reporter_id, team=row.reporter_team),
            summary=row.summary or "",
            tags=list(row.tags or []),
            severity=row.severity,
            category=row.category,
            matched_prior_inc=row.matched_prior_inc,
            embedding=row.embedding,
            agents_run=agents_run,
            tool_calls=tool_calls,
            findings=dict(row.findings or {}),
            resolution=row.resolution,
            token_usage=token_usage,
            pending_intervention=row.pending_intervention,
            user_inputs=list(row.user_inputs or []),
        )

    def _incident_to_row_dict(
        self, inc: Incident, *, embedding: Optional[list[float]],
    ) -> dict:
        return {
            "id": inc.id,
            "status": inc.status,
            "created_at": _parse_iso(inc.created_at),
            "updated_at": _parse_iso(inc.updated_at),
            "deleted_at": _parse_iso(inc.deleted_at) if inc.deleted_at else None,
            "query": inc.query,
            "environment": inc.environment,
            "reporter_id": inc.reporter.id,
            "reporter_team": inc.reporter.team,
            "summary": inc.summary or "",
            "severity": inc.severity,
            "category": inc.category,
            "matched_prior_inc": inc.matched_prior_inc,
            "resolution": inc.resolution,
            "tags": list(inc.tags),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "embedding": embedding,
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
        }

    # ---------- embedding lifecycle ----------
    def _maybe_embed(self, text: str) -> Optional[list[float]]:
        if self.embedder is None or not text:
            return None
        return self.embedder.embed_query(text)

    def _compute_save_embedding(
        self, existing: Optional[IncidentRow], inc: Incident,
    ) -> Optional[list[float]]:
        """Re-embed only when the source text materially changed."""
        if self.embedder is None:
            return existing.embedding if existing is not None else None
        text = _embed_source(inc)
        if existing is not None:
            prior = _embed_source_from_row(existing)
            if prior == text and existing.embedding is not None:
                return existing.embedding
        return self.embedder.embed_query(text) if text else None


def _embed_source(inc: Incident) -> str:
    return (inc.query or "").strip()


def _embed_source_from_row(row: IncidentRow) -> str:
    return (row.query or "").strip()

# ====== module: orchestrator/mcp_servers/incident.py ======

_DEFAULT_SEVERITY_ALIASES: dict[str, str] = {
    "sev1": "high", "sev2": "high", "p1": "high", "p2": "high",
    "critical": "high", "urgent": "high", "high": "high",
    "sev3": "medium", "p3": "medium", "moderate": "medium", "medium": "medium",
    "sev4": "low", "p4": "low", "info": "low", "informational": "low",
    "low": "low",
}


def normalize_severity(
    value: str | None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if aliases is None:
        return lowered
    return aliases.get(lowered, value)


@dataclass
class IncidentMCPServer:
    """FastMCP server bound to a single :class:`IncidentRepository`."""
    repository: IncidentRepository | None = None
    severity_aliases: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SEVERITY_ALIASES)
    )
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("incident_management")
        self.mcp.tool(name="lookup_similar_incidents")(self._tool_lookup_similar_incidents)
        self.mcp.tool(name="create_incident")(self._tool_create_incident)
        self.mcp.tool(name="update_incident")(self._tool_update_incident)

    def configure(
        self, *,
        repository: IncidentRepository,
        severity_aliases: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        if severity_aliases is not None:
            self.severity_aliases = severity_aliases

    def _require_repo(self) -> IncidentRepository:
        if self.repository is None:
            raise RuntimeError(
                "incident_management server not initialized — "
                "call configure() (or the module-level set_state) first"
            )
        return self.repository

    async def _tool_lookup_similar_incidents(self, query: str, environment: str) -> dict:
        """Search past resolved INCs for similar issues. Returns top 5 by similarity score."""
        repo = self._require_repo()
        hits = repo.find_similar(query=query, environment=environment, limit=5)
        return {"matches": [
            {"id": i.id, "summary": i.summary, "resolution": i.resolution,
             "score": round(s, 3)}
            for i, s in hits
        ]}

    async def _tool_create_incident(self, query: str, environment: str,
                                    reporter_id: str = "user-mock",
                                    reporter_team: str = "platform") -> dict:
        """Create a new INC ticket and persist it."""
        inc = self._require_repo().create(query=query, environment=environment,
                                          reporter_id=reporter_id,
                                          reporter_team=reporter_team)
        return inc.model_dump()

    async def _tool_update_incident(self, incident_id: str, patch: dict) -> dict:
        """Apply a flat patch to an INC.

        Allowed keys:
          - status, severity, category, summary, tags, matched_prior_inc, resolution
          - findings_<agent_name> — writes ``inc.findings[<agent_name>] = value``.
        """
        repo = self._require_repo()
        inc = repo.load(incident_id)
        if "status" in patch:
            inc.status = patch["status"]
        if "severity" in patch:
            inc.severity = normalize_severity(patch["severity"], self.severity_aliases)
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
        for key, value in patch.items():
            if key.startswith("findings_"):
                inc.findings[key[len("findings_"):]] = value
        repo.save(inc)
        return inc.model_dump()


# ---------------------------------------------------------------------------
# Module-level default server (back-compat for the MCP loader path).
# The MCP loader imports ``mcp`` from this module by name; this keeps that
# contract working unchanged.
# ---------------------------------------------------------------------------

_default_server = IncidentMCPServer()
mcp = _default_server.mcp


def set_state(*, repository: IncidentRepository,
              severity_aliases: dict[str, str] | None = None) -> None:
    """Configure the default IncidentMCPServer instance."""
    _default_server.configure(
        repository=repository,
        severity_aliases=severity_aliases,
    )


# Direct-call shims kept for tests that import these names.
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


# Fallback signal vocabulary used by tests and any caller that doesn't
# thread ``cfg.orchestrator.signals`` through. Production code passes the
# configured set down via ``make_agent_node``.
_DEFAULT_SIGNALS: frozenset[str] = frozenset({"success", "failed", "needs_input"})


def _coerce_signal(raw, valid_signals: frozenset[str] | None = None) -> str | None:
    """Coerce a raw signal value emitted by an LLM to a canonical lowercase
    string, or None when the value cannot be interpreted.

    The accepted vocabulary comes from ``cfg.orchestrator.signals`` (passed
    in as ``valid_signals``); when the caller omits it, the historical
    ``{success, failed, needs_input}`` default is used. Any value outside
    the set emits a warning and yields None — the route lookup then falls
    back to ``when: default``. ``bool`` is rejected explicitly because
    Python treats it as ``int`` and string-coerces to ``"True"``/``"False"``.
    """
    allowed = valid_signals if valid_signals is not None else _DEFAULT_SIGNALS
    if isinstance(raw, bool):
        logger.warning("signal value is bool (%r); rejecting", raw)
        return None
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning("non-string signal %r (%s); rejecting",
                       raw, type(raw).__name__)
        return None
    key = raw.strip().lower()
    if key in allowed:
        return key
    logger.warning("unknown signal %r; treating as None (will fall through to default)", raw)
    return None


class GraphState(TypedDict, total=False):
    incident: Incident
    next_route: str | None
    last_agent: str | None
    gated_target: str | None  # set by gate node; the downstream target if gate passes
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
        self._started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

    def record_tool_call(self, tool: str, args: dict, result) -> None:
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
        self.incident.tool_calls.append(
            ToolCall(agent=self.agent, tool=tool, args=args, result=result, ts=ts)
        )

    def finish(self, *, summary: str) -> None:
        ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
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
    raise last_exc or RuntimeError("retry exhausted with no attempts")  # pragma: no cover


def _format_agent_input(incident: Incident) -> str:
    """Build the human-message preamble each agent receives.

    Findings are now a free-form mapping (one entry per upstream agent),
    so emit one ``Findings (<agent>): <value>`` line per recorded finding
    instead of pinning the prompt to two specific agent identities.
    """
    base = (
        f"Incident {incident.id}\n"
        f"Environment: {incident.environment}\n"
        f"Query: {incident.query}\n"
        f"Status: {incident.status}\n"
    )
    for agent_key, finding in incident.findings.items():
        base += f"Findings ({agent_key}): {finding}\n"
    if incident.user_inputs:
        bullets = "\n".join(f"- {ui}" for ui in incident.user_inputs)
        base += (
            "\nUser-provided context (appended via intervention):\n"
            f"{bullets}\n"
        )
    return base


def _merge_patch_metadata(
    patch: dict,
    confidence: float | None,
    rationale: str | None,
    signal: str | None,
    valid_signals: frozenset[str] | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Update the (confidence, rationale, signal) trio with whatever the
    patch carries; preserve the prior value when the patch is silent on a
    field. Centralises the per-key conditional that used to nest 3 deep
    inside ``_harvest_tool_calls_and_patches``.
    """
    new_conf = _coerce_confidence(patch["confidence"]) if "confidence" in patch else confidence
    new_rationale = (
        _coerce_rationale(patch["confidence_rationale"])
        if "confidence_rationale" in patch else rationale
    )
    new_signal = (
        _coerce_signal(patch["signal"], valid_signals)
        if "signal" in patch else signal
    )
    return new_conf, new_rationale, new_signal


def _harvest_tool_calls_and_patches(
    messages: list,
    skill_name: str,
    incident: Incident,
    ts: str,
    valid_signals: frozenset[str] | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Iterate agent messages, record ToolCall entries on the incident, and
    harvest any confidence / confidence_rationale / signal from update_incident
    patches.

    Returns ``(agent_confidence, agent_rationale, agent_signal)``.
    """
    agent_confidence: float | None = None
    agent_rationale: str | None = None
    agent_signal: str | None = None
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            tc_name = tc.get("name", "unknown")
            tc_args = tc.get("args", {}) or {}
            # Tool names are now namespaced as ``<server>:<original>``;
            # match on the un-prefixed suffix so the bare and prefixed
            # forms both harvest confidence/signal patches.
            tc_original = tc_name.rsplit(":", 1)[-1]
            incident.tool_calls.append(ToolCall(
                agent=skill_name,
                tool=tc_name,
                args=tc_args,
                result=None,
                ts=ts,
            ))
            if tc_original == "update_incident":
                patch = tc_args.get("patch") or {}
                agent_confidence, agent_rationale, agent_signal = _merge_patch_metadata(
                    patch, agent_confidence, agent_rationale, agent_signal,
                    valid_signals,
                )
    return agent_confidence, agent_rationale, agent_signal


def _pair_tool_responses(messages: list, incident: Incident) -> None:
    """Match ToolMessage responses back to their corresponding ToolCall entries."""
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage":
            for entry in reversed(incident.tool_calls):
                if entry.tool == getattr(msg, "name", None) and entry.result is None:
                    entry.result = getattr(msg, "content", None)
                    break


def _extract_final_text(messages: list) -> str:
    """Return the text content of the last non-empty AIMessage, or empty string."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            return str(msg.content)
    return ""


def _sum_token_usage(messages: list) -> TokenUsage:
    """Sum input/output token counts across all messages that report usage_metadata."""
    agent_in = agent_out = 0
    for msg in messages:
        um = getattr(msg, "usage_metadata", None) or {}
        agent_in += int(um.get("input_tokens") or 0)
        agent_out += int(um.get("output_tokens") or 0)
    return TokenUsage(
        input_tokens=agent_in,
        output_tokens=agent_out,
        total_tokens=agent_in + agent_out,
    )


def _handle_agent_failure(
    *,
    skill_name: str,
    started_at: str,
    exc: Exception,
    inc_id: str,
    store: "IncidentStore",
    fallback: "Incident",
) -> dict:
    """Reload incident (absorbing partial tool writes), stamp a failure AgentRun,
    persist, and return the error state dict for the LangGraph node.

    ``fallback`` is the in-memory incident from the caller; we use it only
    when the on-disk state has gone missing (FileNotFoundError on reload).
    """
    try:
        incident = store.load(inc_id)
    except FileNotFoundError:
        incident = fallback
    ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
    incident.agents_run.append(AgentRun(
        agent=skill_name, started_at=started_at, ended_at=ended_at,
        summary=f"agent failed: {exc}",
        token_usage=TokenUsage(),
    ))
    store.save(incident)
    return {"incident": incident, "next_route": None,
            "last_agent": skill_name, "error": str(exc)}


def _record_success_run(
    *,
    incident: "Incident",
    skill_name: str,
    started_at: str,
    final_text: str,
    usage: "TokenUsage",
    confidence: float | None,
    rationale: str | None,
    signal: str | None,
    store: "IncidentStore",
) -> None:
    """Append the success-path AgentRun, update the incident's running token
    totals, and persist. Mutates ``incident`` in place."""
    ended_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)
    incident.agents_run.append(AgentRun(
        agent=skill_name, started_at=started_at, ended_at=ended_at,
        summary=final_text or f"{skill_name} completed",
        token_usage=usage,
        confidence=confidence,
        confidence_rationale=rationale,
        signal=signal,
    ))
    incident.token_usage.input_tokens += usage.input_tokens
    incident.token_usage.output_tokens += usage.output_tokens
    incident.token_usage.total_tokens += usage.total_tokens
    store.save(incident)


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Incident], str],
    store: IncidentStore,
    valid_signals: frozenset[str] | None = None,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route.

    ``valid_signals`` is the orchestrator-wide accepted signal vocabulary
    (``cfg.orchestrator.signals``). When omitted, the legacy
    ``{success, failed, needs_input}`` default is used so older callers and
    tests keep working.
    """
    agent_executor = create_react_agent(llm, tools, prompt=skill.system_prompt)

    async def node(state: GraphState) -> dict:
        incident = state["incident"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies incident
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except Exception as exc:  # noqa: BLE001
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        # Tools (e.g. update_incident) write straight to disk. Reload so the
        # node's own append of agent_run + tool_calls happens against the
        # tool-mutated state — otherwise saving the stale in-memory object
        # clobbers the tools' writes.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Record tool calls and harvest confidence/signal from update_incident patches.
        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
        )

        # Pair tool responses with their tool calls.
        _pair_tool_responses(messages, incident)

        # Final summary text and token usage.
        final_text = _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=agent_confidence, rationale=agent_rationale, signal=agent_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)
        return {"incident": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Incident) -> str:
    """Return the latest agent's emitted signal, or "default" if absent.

    Agents emit one of {success, failed, needs_input} via the ``signal``
    key of their final ``update_incident`` patch (see ``_coerce_signal``).
    The node harvests it onto ``AgentRun.signal``; this decider then reads
    the *most recent* run (which is the one that just finished, since the
    node has already appended it). If no signal is present we return
    "default" so ``route_from_skill`` picks the fallback rule.
    """
    if not inc.agents_run:
        return "default"
    return inc.agents_run[-1].signal or "default"


_DEFAULT_STUB_CANNED: dict[str, str] = {
    # Back-compat defaults for the four canonical agents.  Any YAML-defined
    # agent can override or extend this via ``skill.stub_response``; new agents
    # without an entry here fall through to StubChatModel's generic placeholder.
    "intake": "Created INC, no prior matches. Routing to triage.",
    "triage": "Severity medium, category latency. No recent deploys correlate.",
    "deep_investigator": "Hypothesis: upstream payments timeout. Evidence: log line 'upstream_timeout target=payments'.",
    "resolution": "Proposed fix: restart api service. Auto-applied. INC resolved.",
}


def _latest_run_for(incident: Incident, agent_name: str | None):
    """Return the most recent ``AgentRun`` for ``agent_name``, or None.

    ``agent_name`` is whichever agent ran immediately before the gate,
    derived at runtime from ``state["last_agent"]`` — so the gate is
    config-driven and works for any YAML-defined upstream, not just
    ``deep_investigator``.
    """
    if not agent_name:
        return None
    for run in reversed(incident.agents_run):
        if run.agent == agent_name:
            return run
    return None


def make_gate_node(*, cfg: AppConfig, store: IncidentStore):
    """Build the intervention gate node placed before a gated downstream.

    The gate evaluates the confidence of whichever agent ran immediately
    before it (``state["last_agent"]``). If that confidence is below the
    configured threshold (or absent), the gate marks the incident
    ``awaiting_input``, populates ``pending_intervention``, and routes
    to END. Otherwise it routes to the gated target.

    Implemented as a plain async coroutine (not via ``make_agent_node``)
    so it does not invoke an LLM — but it IS a real graph node, so
    streamed events surface ``enter gate`` / ``exit gate``.

    The gate is fully agent-agnostic: any YAML route carrying
    ``gate: confidence`` causes the gate to evaluate the upstream agent
    named on that route, regardless of agent identity.
    """
    threshold = cfg.intervention.confidence_threshold
    teams = list(cfg.intervention.escalation_teams)

    async def gate(state: GraphState) -> dict:
        incident = state["incident"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies incident
        upstream = state.get("last_agent")
        # Capture the intended downstream target before we overwrite next_route.
        # The upstream agent set next_route to the gated target; we stash it in
        # gated_target so _make_gate_to can route correctly for multi-target graphs.
        intended_target = state.get("next_route")
        # Reload from disk in case earlier nodes wrote tool-driven patches.
        try:
            incident = store.load(incident.id)
        except FileNotFoundError:
            pass
        upstream_run = _latest_run_for(incident, upstream)
        upstream_conf = upstream_run.confidence if upstream_run else None
        if upstream_conf is None or upstream_conf < threshold:
            incident.status = "awaiting_input"
            # Surface the upstream agent's own summary + rationale so the
            # human reviewer can decide what input to give without scrolling
            # through every step of the agents-run log.
            incident.pending_intervention = {
                "reason": "low_confidence",
                "confidence": upstream_conf,
                "threshold": threshold,
                "upstream_agent": upstream,
                "summary": upstream_run.summary if upstream_run else "",
                "rationale": upstream_run.confidence_rationale if upstream_run else "",
                "options": ["resume_with_input", "escalate", "stop"],
                "escalation_teams": teams,
            }
            store.save(incident)
            return {"incident": incident, "next_route": "__end__",
                    "gated_target": intended_target, "last_agent": "gate", "error": None}
        # Confidence met threshold — clear any stale intervention payload.
        if incident.pending_intervention is not None:
            incident.pending_intervention = None
            store.save(incident)
        return {"incident": incident, "next_route": "default",
                "gated_target": intended_target, "last_agent": "gate", "error": None}

    return gate


def _build_agent_nodes(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                       registry: ToolRegistry) -> dict:
    """Materialize agent nodes from skills + registry. Reused by main + resume graphs."""
    valid_signals = frozenset(cfg.orchestrator.signals)
    nodes: dict = {}
    for agent_name, skill in skills.items():
        if skill.stub_response is not None:
            stub_canned: dict[str, str] | None = {skill.name: skill.stub_response}
        elif agent_name in _DEFAULT_STUB_CANNED:
            stub_canned = {agent_name: _DEFAULT_STUB_CANNED[agent_name]}
        else:
            stub_canned = None
        llm = get_llm(
            cfg.llm,
            skill.model,
            role=agent_name,
            stub_canned=stub_canned,
        )
        tools = registry.resolve(skill.tools, cfg.mcp)
        decide = _decide_from_signal
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
            valid_signals=valid_signals,
        )
    return nodes


def _make_router(gated_edges: dict[tuple[str, str], str]):
    """Build a state router that intercepts gated edges into the gate node.

    Used by both ``build_graph`` and ``build_resume_graph`` — they share
    the same routing semantics, so this factory eliminates duplication.
    """
    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        la = state.get("last_agent")
        if (la, nr) in gated_edges:
            return "gate"
        return nr
    return _router


def _make_gate_to(gate_targets: set[str]):
    """Build the gate's outbound router.

    On a low-confidence fail the gate sets ``next_route="__end__"`` and
    we terminate. On a pass the gate sets ``next_route="default"`` and also
    stamps ``gated_target`` with the intended downstream node (captured from
    the incoming ``next_route`` before the gate ran). We read ``gated_target``
    here so this router handles any number of gated targets without needing
    a hardcoded closure over a single target name.
    """
    def _gate_to(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        gt = state.get("gated_target")
        if gt in gate_targets:
            return gt
        return END
    return _gate_to


def _collect_gated_edges(skills: dict) -> dict[tuple[str, str], str]:
    """Return ``{(from_agent, to_node): gate_type}`` for every route rule
    whose ``gate`` is set. Today only ``gate: confidence`` is recognised."""
    edges: dict[tuple[str, str], str] = {}
    for agent_name, skill in skills.items():
        for rule in skill.routes:
            if rule.gate:
                edges[(agent_name, rule.next)] = rule.gate
    return edges


async def build_graph(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                      registry: ToolRegistry):
    """Compile the main LangGraph from configured skills and routes.

    The entry agent is read from ``cfg.orchestrator.entry_agent``. Gate
    insertions are derived from each skill's route rules: a rule with
    ``gate: confidence`` causes the router to redirect ``(this_agent, next)``
    through the ``gate`` node.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.
    """
    entry = cfg.orchestrator.entry_agent
    if entry not in skills:
        raise ValueError(
            f"orchestrator.entry_agent={entry!r} is not a known skill "
            f"(known: {sorted(skills.keys())})"
        )
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))

    sg.set_entry_point(entry)

    _router = _make_router(gated_edges)

    for agent_name, _skill in skills.items():
        possible_targets = {s.name for s in skills.values()} | {END, "gate"}
        # Exclude targets that are intercepted via a gated edge for this agent:
        # the router redirects (agent_name, gated_target) -> "gate", so the
        # gated_target must NOT appear in this agent's target_map. Leaving it
        # in would cause LangGraph to register a visible direct edge in the
        # compiled graph, defeating the structural assertion in the test (and
        # misleading graph visualisations).
        gated_targets_for_agent = {to for (frm, to) in gated_edges.keys() if frm == agent_name}
        target_map = {
            name: name
            for name in possible_targets
            if name != END and name not in gated_targets_for_agent
        }
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel

    # Build the gate's outbound target map from all distinct gated downstream
    # nodes. _make_gate_to reads next_route from state (set by the upstream
    # agent) so it routes to the correct target regardless of how many gated
    # edges exist — the multi-target restriction is no longer needed.
    gate_targets = {to for (_from, to) in gated_edges.keys()}
    if gate_targets:
        _gate_to = _make_gate_to(gate_targets)
        gate_target_map: dict = {t: t for t in gate_targets}
        gate_target_map[END] = END
        sg.add_conditional_edges("gate", _gate_to, gate_target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel
    else:
        sg.add_edge("gate", END)

    return sg.compile()


async def build_resume_graph(*, cfg: AppConfig, skills: dict,
                             store: IncidentStore, registry: ToolRegistry):
    """Compile a sub-graph that re-runs from the upstream end of whichever
    gated edge the INC was awaiting input from.

    Supports any number of gated edges. The correct upstream agent is
    determined at runtime from ``incident.pending_intervention["upstream_agent"]``
    (stamped by the gate node). A lightweight dispatcher node is used as the
    fixed entry point; it reads the upstream agent from state and sets
    ``next_route`` so the shared router forwards execution to it.

    Used by ``Orchestrator.resume_investigation`` after the user supplies
    new context. Same gate semantics — if the new run is still low-confidence
    the gate will pause again.
    """
    gated_edges = _collect_gated_edges(skills)
    if not gated_edges:
        raise ValueError(
            "build_resume_graph requires at least one route with gate set; "
            "no gated edges were found in the configured skills — add "
            "'gate: confidence' to the relevant agent's route in "
            "config/skills/<agent>/config.yaml"
        )

    upstream_agents = {frm for (frm, _to) in gated_edges.keys()}
    gate_targets = {to for (_frm, to) in gated_edges.keys()}

    async def _resume_dispatcher(state: GraphState) -> dict:
        """Route to whichever upstream agent the INC was awaiting input from.

        Reads ``pending_intervention["upstream_agent"]`` from the incident
        so the resume graph works for any gated topology without being
        rebuilt per call.
        """
        incident = state["incident"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        pi = getattr(incident, "pending_intervention", None)
        upstream = (pi or {}).get("upstream_agent") if pi else None
        if upstream not in upstream_agents:
            # Fallback: pick the lexically first upstream to avoid a hard crash;
            # callers should only resume incidents that are awaiting_input.
            upstream = next(iter(sorted(upstream_agents)))
            logger.warning(
                "resume_dispatcher: upstream_agent %r not in gated upstreams %r; "
                "falling back to %r",
                upstream, sorted(upstream_agents), upstream,
            )
        return {"next_route": upstream, "last_agent": None, "error": None}

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    sg.add_node("__resume_dispatcher__", _resume_dispatcher)
    for agent_name in upstream_agents | gate_targets:
        if agent_name in nodes:
            sg.add_node(agent_name, nodes[agent_name])
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))
    sg.set_entry_point("__resume_dispatcher__")

    _router = _make_router(gated_edges)

    all_resume_nodes = upstream_agents | gate_targets
    shared_target_map: dict = {name: name for name in all_resume_nodes | {"gate"}}
    shared_target_map[END] = END

    # Dispatcher routes to an upstream agent via next_route.
    sg.add_conditional_edges("__resume_dispatcher__", _router, shared_target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel

    for agent_name in all_resume_nodes:
        sg.add_conditional_edges(agent_name, _router, shared_target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel

    _gate_to = _make_gate_to(gate_targets)
    gate_target_map: dict = {t: t for t in gate_targets}
    gate_target_map[END] = END
    sg.add_conditional_edges("gate", _gate_to, gate_target_map)  # pyright: ignore[reportArgumentType] — langgraph typing limitation with END sentinel
    return sg.compile()

# ====== module: orchestrator/orchestrator.py ======

_INCIDENT_MCP_MODULE = "orchestrator.mcp_servers.incident"


def _storage_url(cfg: AppConfig) -> str:
    """Derive the SQLite URL for the current config.

    When ``cfg.storage.url`` is still the default sentinel (``sqlite:///incidents.db``),
    use ``cfg.paths.incidents_dir`` so that per-test ``tmp_path`` isolation is
    respected. Production deployments that set an explicit ``storage.url``
    (e.g. a Postgres DSN or a non-default SQLite path) are left untouched.
    """
    default_url = StorageConfig().url
    if cfg.storage.url != default_url:
        return cfg.storage.url
    return f"sqlite:///{Path(cfg.paths.incidents_dir) / 'incidents.db'}"


class Orchestrator:
    """High-level facade. Construct via ``await Orchestrator.create(cfg)``.

    The Orchestrator owns the lifecycle of the FastMCP clients underpinning the
    tool registry. Always call :meth:`aclose` (or use ``async with``) when done.
    """

    def __init__(self, cfg: AppConfig, store: IncidentRepository,
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
            engine = build_engine(StorageConfig(url=_storage_url(cfg),
                                                pool_size=cfg.storage.pool_size,
                                                echo=cfg.storage.echo))
            Base.metadata.create_all(engine)
            embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
            store = IncidentRepository(
                engine=engine,
                embedder=embedder,
                similarity_threshold=cfg.incidents.similarity_threshold,
                severity_aliases=cfg.orchestrator.severity_aliases,
            )
            # Configure incident_management state via importlib so we hit the
            # *same* module instance the MCP loader will import. In the
            # single-file dist bundle a direct ``set_state`` call would
            # configure a bundled-local ``_default_server`` while the loader
            # imports ``orchestrator.mcp_servers.incident`` from src and uses
            # a *different* singleton — leaving FastMCP tools unconfigured.
            for srv in cfg.mcp.servers:
                if (srv.transport == "in_process" and srv.enabled
                        and srv.module == _INCIDENT_MCP_MODULE):
                    importlib.import_module(_INCIDENT_MCP_MODULE).set_state(
                        repository=store,
                        severity_aliases=cfg.orchestrator.severity_aliases,
                    )
                    break
            skills = load_all_skills(cfg.paths.skills_dir)
            for s in skills.values():
                if s.model is not None and s.model not in cfg.llm.models:
                    raise ValueError(
                        f"skill {s.name!r} references llm model {s.model!r} "
                        f"which is not defined in llm.models "
                        f"(known: {sorted(cfg.llm.models)})"
                    )
            registry = await load_tools(cfg.mcp, stack)
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry)
            # Build the resume graph only when at least one skill declares
            # a gated route. Without gates an INC can never enter the
            # ``awaiting_input`` state, so the resume path is dead code —
            # and ``build_resume_graph`` raises by design when gated_edges
            # is empty. This unblocks intake-only YAML configurations.
            has_gates = any(
                r.gate for s in skills.values() for r in s.routes
            )
            resume_graph = (
                await build_resume_graph(
                    cfg=cfg, skills=skills, store=store, registry=registry,
                ) if has_gates else None
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
                # The named model entry the agent will use (resolved against
                # cfg.llm.default when the skill leaves model unset).
                "model": s.model or self.cfg.llm.default,
                # Expose the flat list of prefixed tool names the LLM sees.
                # resolve() returns list[BaseTool], so .name is on the tool directly.
                "tools": [
                    t.name
                    for t in self.registry.resolve(s.tools, self.cfg.mcp)
                ],
                "routes": [r.model_dump() for r in s.routes],
            }
            for s in self.skills.values()
        ]

    def list_tools(self) -> list[dict]:
        # Build reverse map: prefixed tool name -> list of skill names that bind it.
        # resolve() returns list[BaseTool]; tool.name is the prefixed form.
        bindings: dict[str, list[str]] = {}
        for skill in self.skills.values():
            for t in self.registry.resolve(skill.tools, self.cfg.mcp):
                bindings.setdefault(t.name, []).append(skill.name)
        # Map server name -> transport so callers can tell local from remote.
        transport_by_server = {s.name: s.transport for s in self.cfg.mcp.servers}
        return [
            {
                "name": e.tool.name,          # prefixed: "<server>:<original>"
                "original_name": e.name,      # original tool name as exposed by server
                "description": e.description,
                "category": e.category,
                "server": e.server,
                "transport": transport_by_server.get(e.server, "unknown"),
                "bound_agents": bindings.get(e.tool.name, []),
            }
            for e in self.registry.entries.values()
        ]

    def get_incident(self, incident_id: str) -> dict:
        return self.store.load(incident_id).model_dump()

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        return [i.model_dump() for i in self.store.list_recent(limit)]

    def delete_incident(self, incident_id: str) -> dict:
        return self.store.delete(incident_id).model_dump()

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
            async for ev in self._resume_with_input(incident_id, inc, decision):
                yield ev
            return

        raise ValueError(f"Unknown resume action: {action!r}")

    async def _resume_with_input(self, incident_id: str, inc, decision: dict):
        """Handle the resume_with_input action: append user text, re-run sub-graph,
        restore state on failure. Yields UI events."""
        user_text = (decision.get("input") or "").strip()
        if not user_text:
            raise ValueError("resume_with_input requires a non-empty 'input'")
        # The resume sub-graph only exists when the YAML declares at least
        # one gated route. An intake-only configuration has nothing to
        # resume into — bail with a rejection event rather than crashing.
        if self.resume_graph is None:
            yield {"event": "resume_rejected", "incident_id": incident_id,
                   "reason": "resume_with_input not available: no gated route configured",
                   "ts": _now()}
            return
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

    async def _invoke_tool(self, name: str, args: dict):
        """Call an MCP tool by original name, going through the LangChain wrapper.

        Searches the registry for any entry whose original ``name`` matches.
        Used for orchestrator-driven tool calls (e.g. notify_oncall on
        escalate) that aren't initiated by an LLM.
        """
        entry = next(
            (e for e in self.registry.entries.values() if e.name == name),
            None,
        )
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


class ResumeRequest(BaseModel):
    decision: Literal["resume_with_input", "escalate", "stop"]
    user_input: str | None = None


def _make_lifespan(cfg: AppConfig):
    """Build the lifespan context manager for an app constructed with ``cfg``.

    The orchestrator owns FastMCP transports tied to an asyncio event loop;
    creating it inside the lifespan ensures it lives on uvicorn's loop and
    is closed cleanly on shutdown.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        orch = await Orchestrator.create(cfg)
        app.state.orchestrator = orch
        app.state.environments = list(cfg.environments)
        try:
            yield
        finally:
            await orch.aclose()
    return _lifespan


def build_app(cfg: AppConfig) -> FastAPI:
    """Construct the FastAPI app. Synchronous.

    The ``orchestrator`` is created during the app's startup lifespan and
    is reachable as ``app.state.orchestrator`` from any route handler.
    """
    fastapi_app = FastAPI(
        title="ASR — Agent Orchestrator",
        lifespan=_make_lifespan(cfg),
    )

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    @fastapi_app.get("/agents")
    async def agents():
        return fastapi_app.state.orchestrator.list_agents()

    @fastapi_app.get("/tools")
    async def tools():
        return fastapi_app.state.orchestrator.list_tools()

    @fastapi_app.get("/incidents")
    async def incidents(limit: int = 20):
        return fastapi_app.state.orchestrator.list_recent_incidents(limit=limit)

    @fastapi_app.get("/incidents/{incident_id}")
    async def incident(incident_id: str):
        return fastapi_app.state.orchestrator.get_incident(incident_id)

    @fastapi_app.delete("/incidents/{incident_id}")
    async def delete_incident(incident_id: str):
        return fastapi_app.state.orchestrator.delete_incident(incident_id)

    @fastapi_app.post("/investigate")
    async def investigate(req: InvestigateRequest) -> InvestigateResponse:
        inc_id = await fastapi_app.state.orchestrator.start_investigation(
            query=req.query, environment=req.environment,
            reporter_id=req.reporter_id, reporter_team=req.reporter_team,
        )
        return InvestigateResponse(incident_id=inc_id)

    @fastapi_app.get("/environments")
    async def environments():
        return fastapi_app.state.environments

    @fastapi_app.post("/investigate/stream")
    async def investigate_stream(req: InvestigateRequest) -> StreamingResponse:
        orch = fastapi_app.state.orchestrator

        async def _events():
            async for ev in orch.stream_investigation(
                query=req.query, environment=req.environment,
                reporter_id=req.reporter_id, reporter_team=req.reporter_team,
            ):
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    @fastapi_app.post("/incidents/{incident_id}/resume")
    async def resume_incident(incident_id: str, req: ResumeRequest) -> StreamingResponse:
        orch = fastapi_app.state.orchestrator
        decision: dict = {"action": req.decision}
        if req.user_input is not None:
            decision["input"] = req.user_input

        async def _events():
            try:
                async for ev in orch.resume_investigation(incident_id, decision):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'event': 'error', 'error': str(exc)}, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    return fastapi_app


def get_app() -> FastAPI:
    """No-arg factory for ``uvicorn --factory``.

    Reads config from the ``ASR_CONFIG`` env var (default
    ``config/config.yaml``) and returns a fresh FastAPI app. The
    orchestrator is created lazily during the app's startup lifespan,
    not eagerly here, so this factory is safe to call from inside
    uvicorn's running event loop.

    Launch::

        python -m uvicorn --app-dir dist app:get_app --factory --port 37776
    """
    cfg_path = Path(os.environ.get("ASR_CONFIG", "config/config.yaml"))
    return build_app(load_config(cfg_path))
