from __future__ import annotations
# ----- imports for runtime/config.py -----
"""Config schemas for the orchestrator."""

import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
import yaml


# ----- imports for runtime/state.py -----
"""Generic session model — the framework's unit of work.

A ``Session`` is the in-progress (or archived) record of one agent run.
Applications extend this via subclassing::

    class IncidentState(Session):
        environment: str
        reporter: Reporter
        ...

``Session`` deliberately contains *no* domain-specific fields. Adding one
here is a framework regression — domain fields belong in the example
app's ``state.py``.
"""



from pydantic import BaseModel, Field

# ----- imports for runtime/state_resolver.py -----
"""Resolve ``RuntimeConfig.state_class`` (a dotted path) to a class object.

The orchestrator calls :func:`resolve_state_class` once at construction
time and threads the resulting class through the storage layer. Doing the
import here (rather than relying on type-var introspection at runtime)
sidesteps PEP 484 generic erasure: ``Orchestrator[IncidentState]`` is
compiled away by the time we need a callable class.

Errors on:

- A dotted path that does not parse (no ``.`` separator).
- A module that fails to import.
- A module that imports but lacks the named attribute.
- An attribute that is not a subclass of :class:`runtime.state.Session`.
"""


import importlib
from typing import Type



# ----- imports for runtime/similarity.py -----
"""Similarity scoring for incident matching."""

from typing import Protocol

# ----- imports for runtime/skill.py -----
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

P6 — Agent kinds
----------------

Each ``Skill`` declares a ``kind`` discriminator; the loader validates
per-kind field shape so misconfigured skills fail loudly at startup
instead of at runtime. Three kinds are supported:

* ``responsive`` — the today-default LLM agent that responds inside a
  session graph (existing behaviour, preserved by default).
* ``supervisor`` — a no-LLM router that dispatches work to subordinate
  agents via LangGraph ``Send()``. No ``AgentRun`` row.
* ``monitor`` — a long-running observer that runs out-of-band on a
  schedule, evaluates an emit condition, and fires a Phase-5 trigger.
"""

import ast
from typing import Any, Callable, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ----- imports for runtime/llm.py -----
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



# ----- imports for runtime/storage/models.py -----
"""SQLAlchemy declarative model for the ``incidents`` table.

Hybrid schema: scalar/queryable fields as columns, nested Pydantic
structures as JSON columns (JSONB on Postgres, TEXT on SQLite).
Vector similarity lives in a separate LangChain VectorStore (landed in M3).
"""

from datetime import datetime
from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ----- imports for runtime/storage/engine.py -----
"""SQLAlchemy engine factory.

Sync engine for SQLite (dev) or Postgres (prod). No vector-extension
loading — vectors live in a separate LangChain VectorStore (see
:mod:`orchestrator.storage.vector`, landed in M3).

When the metadata store and the LangGraph ``AsyncSqliteSaver``
checkpointer share a SQLite file, two writers contend on the same DB.
SQLite's default ``BEGIN DEFERRED`` transaction acquires SHARED on the
first read and only escalates to RESERVED on the first write — and the
escalation is **non-retryable** when the connection has already read
inside the same transaction (busy_timeout does *not* apply). The losing
writer raises ``database is locked`` immediately. The fix is to start
write transactions with ``BEGIN IMMEDIATE`` so the RESERVED lock is
acquired up front, before any reads, and busy_timeout's wait-and-retry
loop can correctly serialize the two writers. This is the
SQLAlchemy-recommended pattern for concurrent SQLite writers; see
https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl
"""

from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import NullPool


# Generous timeout: tests can run under load with multiple async writers
# interleaving on the same DB file. 30s leaves headroom for the slowest
# checkpointer commit while still failing fast on a true deadlock.
# ----- imports for runtime/storage/embeddings.py -----
"""LangChain ``Embeddings`` facade + deterministic stub for tests.

Construction is config-driven via :func:`build_embedder`. Provider kind
dispatches to ``OllamaEmbeddings`` / ``AzureOpenAIEmbeddings`` / a local
stub. Stubs are deterministic so tests can assert similarity ordering
without external services.
"""

import hashlib
import numpy as np



# ----- imports for runtime/storage/vector.py -----
"""LangChain ``VectorStore`` factory.

Backends
--------
- ``faiss``    -> ``langchain_community.vectorstores.FAISS`` (file-backed, dev).
- ``pgvector`` -> ``langchain_postgres.PGVector`` (DB-backed, prod).
- ``none``     -> ``None``; caller falls back to keyword similarity.

FAISS persistence: callers invoke :meth:`vector_store.save_local` after
each mutation. The factory loads from disk if a saved index exists at
the configured ``path``; otherwise it constructs an empty index by
seeding with a placeholder doc and immediately deleting it (LangChain's
FAISS constructor doesn't accept an empty docstore).
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore



# ----- imports for runtime/storage/history_store.py -----
"""Read-only similarity search over closed sessions.

``HistoryStore`` is the read-side companion of ``SessionStore``: it
operates on the same engine + vector store but never writes. The vector
path is preferred when both ``vector_store`` and ``embedder`` are
configured; otherwise it falls back to keyword similarity.

Like ``SessionStore``, ``HistoryStore`` is parametrised on ``StateT`` so
that find_similar / load surface the configured app state class rather
than the framework default.

``find_similar`` accepts an arbitrary ``filter_kwargs`` mapping — keys
must correspond to ``IncidentRow`` columns. This decouples the
framework from incident-specific filter dimensions: apps with a
``severity``-only schema, or a multi-tenant ``tenant_id`` schema, or
anything else, build their filter on the fly.
"""

from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


# Mirrors the bound on ``SessionStore.StateT`` — kept permissive at
# ``BaseModel`` so framework code does not need to import the
# example-app subclass. The resolver in :mod:`runtime.state_resolver`
# enforces a ``runtime.state.Session`` subclass at config time.
# ----- imports for runtime/storage/session_store.py -----
"""Active session lifecycle store.

``SessionStore`` owns the write path for the row schema:
create, load, save, delete, list_all, list_recent. It also owns the
vector write-through (``_persist_vector``, ``_add_vector``,
``_refresh_vector``) and the row<->model converters shared with
``HistoryStore``.

The class is parametrised as ``Generic[StateT]`` and routes row
hydration through ``self._state_cls(...)`` so apps can plug in their
own ``Session`` subclass via ``RuntimeConfig.state_class``. The row
schema remains incident-shaped, but unused fields are dropped via
Pydantic's default ``extra='ignore'`` when a narrower ``state_cls`` is
supplied.
"""

import json
from datetime import datetime, timezone
from typing import Generic, Optional, Type, TypeVar

from sqlalchemy import desc, select
from sqlalchemy.orm import Session as SqlSession



# The legacy ``INC-YYYYMMDD-NNN`` pattern stays here for back-compat
# validation against on-disk rows minted before the ``Session.id_format``
# hook existed. New rows are validated by ``_SESSION_ID_RE`` which
# accepts any ``PREFIX-YYYYMMDD-NNN`` shape the app's ``id_format`` may
# emit (e.g. ``CR-...`` for code-review).
# ----- imports for runtime/mcp_servers/observability.py -----
"""FastMCP server: observability mock tools."""

from datetime import datetime, timezone, timedelta
from fastmcp import FastMCP

# ----- imports for runtime/mcp_servers/remediation.py -----
"""FastMCP server: remediation mock tools."""


# ----- imports for runtime/mcp_servers/user_context.py -----
"""FastMCP server: user_context mock tool."""


# ----- imports for runtime/mcp_loader.py -----
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

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools



# ----- imports for runtime/graph.py -----
"""LangGraph state, routing helpers, and node runner."""

import asyncio
import logging
from typing import TypedDict, Callable, Awaitable

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END








# ----- imports for runtime/checkpointer_postgres.py -----
"""Postgres checkpointer wrapper.

Loaded only when ``cfg.storage.metadata.url`` resolves to a Postgres
URL. Uses a *separate* :class:`psycopg_pool.AsyncConnectionPool` (not
SQLAlchemy's pool) so the LangGraph checkpoint saver doesn't contend
with the metadata-store writes on the same connection.

The pool's lifecycle is bound to the orchestrator via the returned
async cleanup callable; the orchestrator awaits it from ``aclose``.
"""


from typing import Awaitable, Callable, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver


# ----- imports for runtime/checkpointer.py -----
"""LangGraph checkpointer factory.

Reuses ``cfg.storage.metadata.url`` so the LangGraph durable-state
checkpointer and the application metadata store live in the same
database (one URL to configure, easier ops). Per-backend a *separate*
connection pool is created so the two paths never deadlock:

- SQLite: dedicated ``aiosqlite.Connection`` with ``PRAGMA journal_mode=WAL``
  so the SQLAlchemy session pool and the checkpoint saver can both write
  to the same on-disk file without blocking each other.
- Postgres: a separate ``psycopg_pool.AsyncConnectionPool`` rather than
  reusing SQLAlchemy's pool, so checkpointer writes don't contend with
  metadata writes on the same connection.

The factory is async because the orchestrator drives the graph through
async ``ainvoke`` / ``astream_events`` — and LangGraph's async Pregel
loop calls ``aget_tuple`` on the saver, which in turn requires an
asyncio-friendly DB driver (aiosqlite for SQLite,
``psycopg_pool.AsyncConnectionPool`` for Postgres).

Returns ``(saver, cleanup)`` where ``cleanup`` is an *async* callable
that closes the dedicated connection / pool. The caller owns lifecycle
and must await ``cleanup()`` on shutdown — typically via the
orchestrator's ``aclose()``.
"""


from urllib.parse import urlparse




# ----- imports for runtime/triggers/base.py -----
"""ABC and DTOs shared by every trigger transport.

A ``TriggerTransport`` owns the inbound side of one transport flavour
(api / webhook / schedule / plugin). The lifecycle is exactly two
async methods so the FastAPI lifespan in ``runtime/api.py`` can sequence
startup / shutdown deterministically.

A ``TriggerInfo`` is the provenance record attached to every session
started via a trigger. It rides along through ``Orchestrator.start_session``
purely for traceability; the orchestrator does not branch on it.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# ----- imports for runtime/triggers/config.py -----
"""Pydantic discriminated union for the ``triggers:`` block in app config.

A ``TriggerConfig`` declares ONE inbound dispatch path. The ``transport``
literal selects the concrete shape:

    - ``api``      — built-in HTTP route (back-compat with /investigate)
    - ``webhook``  — third-party POST /triggers/{name}; bearer auth
    - ``schedule`` — APScheduler in-process cron job
    - ``plugin``   — entry-point or explicitly-registered custom transport

Validation is fail-fast: bad dotted paths, missing auth env vars, and
malformed cron strings raise at config load time, never at request time.
"""


from typing import Annotated, Literal, Union


# Dotted-path regex used by ``payload_schema`` and ``transform`` fields.
# Accepts ``a.b.c`` or ``a.b:c`` (the colon form is tolerated for parity
# with entry-point syntax; ``runtime.triggers.resolve`` normalises both).
# ----- imports for runtime/triggers/resolve.py -----
"""Resolve dotted paths to live Python objects at registry init time.

Used to bind ``payload_schema`` (a Pydantic ``BaseModel`` subclass) and
``transform`` (a callable) declared in YAML. Resolution happens once,
during ``TriggerRegistry.create`` — never per-request — so a typo
fails at startup, not at first webhook delivery.
"""


from typing import Any, Callable, Type



# ----- imports for runtime/triggers/idempotency.py -----
"""Idempotency-Key dedup store: in-memory LRU + SQLite write-through.

Same DB as session metadata (``storage.metadata.url``); one connection
pool, one filesystem path, one backup story. SQLite WAL mode (already
enabled by ``runtime.storage.engine.build_engine``) handles concurrent
reads from the LRU and the orchestrator.

Cold-restart survival: the LRU is rebuilt on demand; ``get`` falls
through to SQLite when the LRU misses, so a fresh process still
returns the cached ``session_id`` for an unexpired ``Idempotency-Key``.

Schema (registered against ``runtime.storage.models.Base`` so
``Base.metadata.create_all(engine)`` picks it up — no Alembic change
required, matching the existing P3 pattern):

    trigger_idempotency_keys
        trigger_name TEXT NOT NULL
        key          TEXT NOT NULL
        session_id   TEXT NOT NULL
        created_at   TIMESTAMP NOT NULL
        expires_at   TIMESTAMP NOT NULL
        PRIMARY KEY (trigger_name, key)
"""


import threading
from collections import OrderedDict

from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.orm import Mapped, Session as SqlaSession, mapped_column


# ----- imports for runtime/triggers/auth.py -----
"""Bearer auth dependency for webhook trigger routes.

A small FastAPI dependency that compares the inbound ``Authorization``
header against the env-var named in ``WebhookTriggerConfig.auth_token_env``.
Constant-time comparison via ``hmac.compare_digest``; missing/bad/wrong
header all answer ``401``.

Tokens are read once at app startup (when the dependency is built) so
rotating a secret requires a process restart — same model as every other
config-derived secret in the runtime.
"""


import hmac
from typing import Callable

from fastapi import Header, HTTPException, status


# ----- imports for runtime/triggers/transports/api.py -----
"""Built-in ``api`` transport.

The api transport is a no-op lifecycle wrapper — the actual HTTP route
(``POST /investigate`` and ``POST /sessions``) is mounted directly on
the FastAPI app for back-compat. Existing in the registry purely so
operators can list ``api`` triggers in YAML for symmetry, and so the
provenance ``TriggerInfo.transport == "api"`` is available for sessions
created via the legacy route.
"""





# ----- imports for runtime/triggers/transports/webhook.py -----
"""Webhook transport — ``POST /triggers/{name}``.

Mounted by ``runtime/api.py`` during the FastAPI lifespan. Each
``WebhookTriggerConfig`` becomes one route under the same ``/triggers``
prefix; per-route bearer auth is wired via ``runtime.triggers.auth``.

Per-request flow:

1. Bearer dep validates ``Authorization: Bearer <token>`` (when configured).
2. Body parsed against the resolved ``payload_schema`` (Pydantic).
   Validation failure -> ``422``.
3. Optional ``Idempotency-Key`` header is forwarded to
   :meth:`TriggerRegistry.dispatch`. A cache hit returns the existing
   session id; misses run the full transform + ``start_session`` path.
4. Transform errors (any exception from the ``transform`` callable)
   surface as ``422 Unprocessable Entity`` with the exception message;
   per the plan we do not auto-retry and do not cache the failure.
5. Success: ``202 Accepted`` with ``{"session_id": "..."}``.
"""


from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import ValidationError




# ----- imports for runtime/triggers/transports/schedule.py -----
"""APScheduler-backed ``schedule`` transport.

Single ``AsyncIOScheduler`` per process, started during FastAPI lifespan
and stopped on shutdown. Each ``ScheduleTriggerConfig`` becomes one cron
job that calls ``registry.dispatch(name, payload)`` on fire.

Cron flavour: standard 5-field via ``CronTrigger.from_crontab``. The
6-field APScheduler-native form is rejected by ``ScheduleTriggerConfig``
itself; this transport never sees it.

Drift / accuracy: APScheduler in-process is good for ±1 minute under
normal load. Tighter SLOs need an external scheduler (Celery beat,
k8s CronJob) — not supported here.
"""



from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger



# ----- imports for runtime/triggers/transports/plugin.py -----
"""Documentation hook for plugin trigger transports.

Plugin transports are concrete subclasses of
:class:`runtime.triggers.base.TriggerTransport` registered via either
the ``runtime.triggers`` setuptools entry-point group or the
``plugin_transports={kind: cls}`` kwarg to ``TriggerRegistry.create``.

This module is intentionally lightweight — the contract lives entirely
in ``base.py``. It exists as a single discoverable entry-point for
operators reading the source tree.

Example minimal plugin transport::



    class SQSTransport(TriggerTransport):
        def __init__(self, config: PluginTriggerConfig) -> None:
            self._cfg = config
            self._task = None

        async def start(self, registry):
            self._task = asyncio.create_task(self._poll(registry))

        async def stop(self):
            if self._task:
                self._task.cancel()

        async def _poll(self, registry):
            ...  # poll SQS, call registry.dispatch(self._cfg.name, payload)

Register via ``pyproject.toml``::

    [project.entry-points."runtime.triggers"]
    sqs = "myapp.triggers:SQSTransport"

Or explicitly::

    TriggerRegistry.create(
        configs, start_session_fn=...,
        plugin_transports={"sqs": SQSTransport},
    )
"""



# ----- imports for runtime/triggers/registry.py -----
"""TriggerRegistry — owns transport instances and dispatch.

The registry is the single sink every transport calls when it wants to
fire a session. It:

- Resolves dotted paths (``payload_schema`` / ``transform``) at init time.
- Holds an :class:`runtime.triggers.idempotency.IdempotencyStore`.
- Exposes a single async ``dispatch(name, payload, *, idempotency_key=None)``
  entrypoint that performs:

      transform(payload) -> kwargs
      orchestrator.start_session(**kwargs, trigger=info)

The registry also owns each transport's lifecycle: ``start_all`` and
``stop_all`` mirror FastAPI's lifespan handshake.

Plugin transports are merged from two sources:

1. Setuptools entry-points in group ``runtime.triggers``: ``key = kind``,
   ``value = importable subclass of TriggerTransport``.
2. Explicit registration via ``plugin_transports={...}`` on
   :meth:`TriggerRegistry.create`. Explicit wins on key collision.
"""


import importlib.metadata
from typing import Any, Awaitable, Callable, Type






# ----- imports for runtime/dedup.py -----
"""Two-stage dedup pipeline (P7).

Stage 1 — embedding similarity over closed sessions in the same
environment via :meth:`HistoryStore.find_similar`.
Stage 2 — LLM confirmation on the top-K candidates with Pydantic-typed
structured output {is_duplicate, confidence, rationale}.

The pipeline is **framework-level** and never imports the
incident-management state class (R4 in the Phase-7 plan). Apps inject
domain-specific text via a ``text_extractor: Callable[[Session], str]``
callable.

Outcome semantics (locked):
  * Stage 2 short-circuits on the first confirmed match (R3 cost cap).
  * Stage 1 ordering already prioritises by similarity; stage 2 honours
    that order and does not re-rank.
  * Pipeline is non-mutating — the orchestrator owns mutation. This
    keeps unit tests simple and supports a future dry-run mode.

The configuration surface lives in :class:`DedupConfig` and is exposed
to the runtime via a generic provider hook
(``RuntimeConfig.dedup_config_path``). Framework default is *off*;
each example app's YAML opts in.
"""


import enum
from typing import Any, Callable, Generic, Literal, TYPE_CHECKING, TypeVar


# ----- imports for runtime/intake.py -----
"""Framework default intake runner.

Every app's first step is a ``kind: supervisor`` skill named
``intake``. The skill's ``runner`` field defaults to
``runtime.intake:default_intake_runner`` when not overridden. The
default runner does two generic things, both opt-in:

1. **Prior-similar retrieval** — if ``app_cfg.intake_context.history_store``
   is wired, call ``HistoryStore.find_similar`` with the new session's
   ``to_agent_input()`` text and stash the top-K matches as a small
   list of dicts on ``state.findings['prior_similar']``. Downstream
   agents read this as a hypothesis surface, not a verdict.
2. **Dedup short-circuit** — if ``app_cfg.intake_context.dedup_pipeline``
   is wired and reports a duplicate, the runner stamps
   ``parent_session_id`` and ``status='duplicate'`` on the session and
   returns ``next_route='__end__'`` to skip the rest of the graph.

When neither is wired the runner returns ``None`` and the supervisor
falls through to its dispatch table.

Apps that need additional preparation (memory hydration, plan
loading, etc.) compose the framework default with their own runner
via :func:`compose_runners`.
"""


from typing import Any, Callable


# ----- imports for runtime/memory/session_state.py -----
"""ASR memory-layer slots that ride on ``IncidentState.memory``.

Each layer in the ASR.md §3 7-layer model that the MVP slice exercises
(L2 / L5 / L7) gets a small pydantic model here so an investigation
can attach the context it fetched from that layer to the session
state. The whole bundle round-trips losslessly through the framework's
``extra_fields`` mechanism — no row schema changes are needed.

Read-only by construction: agents *consume* these slots; mutation via
MCP tools is not exposed.
"""


from pydantic import BaseModel, ConfigDict, Field


# ----- imports for runtime/memory/knowledge_graph.py -----
"""L2 Knowledge Graph — filesystem backend.

Read-only thin class over two JSON files on disk:

- ``components.json`` — list of ``{id, name, owner, criticality, environment}``.
- ``edges.json``      — list of ``{from, to, kind}`` where
  ``kind in {"calls", "deploys", "reads", "writes"}``.

The store accepts a ``root: Path`` (the layer directory, conventionally
``incidents/kg/``) for testability. If that directory is missing or
empty, the store falls back to the seed bundle at
``examples/incident_management/asr/seeds/kg/`` so a freshly cloned
checkout has a working graph for tests and demos. Air-gapped friendly:
no network, no Neo4j.

Mutation is out of scope for this batch (post-MVP); the surface here
is ``get_component`` / ``find_by_name`` / ``neighbors`` / ``subgraph``.
``subgraph`` returns the assembled :class:`L2KGContext` ready to drop
onto ``IncidentState.memory.l2_kg``.
"""


from typing import Iterable


# ----- imports for runtime/memory/release_context.py -----
"""L5 Release Context — filesystem backend.

Read-only thin class over a single JSON file:

- ``recent.json`` — list of release records ``{id, service, sha,
  deployed_at, author, summary}`` sorted descending by ``deployed_at``.

Accepts ``root: Path`` (the layer directory, conventionally
``incidents/releases/``) for testability. Falls back to the seed
bundle at ``examples/incident_management/asr/seeds/releases/`` when
the configured directory is missing or empty. No Postgres/pgvector
dependency in this batch.

Surface:

- ``recent_for_service(service, *, hours=24)``
- ``suspect_at(*, services, at, window_minutes=60)``
- ``context(services, incident_at)`` -> :class:`L5ReleaseContext` ready
  to attach to ``IncidentState.memory.l5_release``.
"""


from datetime import datetime, timedelta, timezone


# ----- imports for runtime/memory/playbook_store.py -----
"""L7 Playbook Store — filesystem backend.

Read-only thin class over a directory of YAML playbooks. Each file
follows the schema:

.. code-block:: yaml

    id: pb-payments-latency
    title: "Payments service latency spike"
    match_signals:
      service: payments
      metric: p99_latency
      threshold_breach: true
    hypothesis_steps:
      - "Check recent payments deploys (L5)"
      - "Check downstream dependencies (L2)"
    remediation:
      - tool: restart_service
        args: { service: payments }
    required_approval: true

Accepts ``root: Path`` (the layer directory, conventionally
``incidents/playbooks/``) for testability. Falls back to the seed
bundle at ``examples/incident_management/asr/seeds/playbooks/`` when
the configured directory is missing or empty. No FAISS / pgvector
dependency in this batch — semantic match comes in 9d-vector later.

Surface: ``get`` / ``list_all`` / ``match``. ``match`` produces a list
of :class:`L7PlaybookSuggestion` ranked by signal-overlap score, ready
to drop onto ``IncidentState.memory.l7_playbooks``.
"""





# ----- imports for runtime/memory/hypothesis.py -----
"""ASR triage hypothesis-refinement loop helpers.

The triage agent runs an iterative pattern: generate a hypothesis →
gather evidence (L1 current findings, L3-equivalent past similar
incidents, L5 recent releases) → score → refine OR accept. The loop
is bounded so a stuck hypothesis doesn't spin forever.

This module ships the *deterministic* primitives that gate the loop:

* :func:`score_hypothesis` — token-overlap heuristic. Pure, no LLM.
  Returns a normalised score in ``[0.0, 1.0]`` plus a one-sentence
  rationale. Tests can assert exact behaviour.

* :func:`should_refine` — boolean decision based on the current score
  and the iteration counter. Refines while score < 0.7 AND
  iterations < 3.

The agent's LLM-driven generation step (the *hypothesis* itself) lives
in the system prompt at ``skills/triage/system.md``; only the scoring
and continue/stop predicates are deterministic Python so the loop's
boundary conditions are exercised in unit tests without spinning the
LLM.

Design note: tokenisation mirrors :mod:`runtime.similarity` — same
regex, same stopword list — so a hypothesis containing service names
and timing words ranks consistent with the dedup pipeline's notion of
"similar".
"""


from typing import TypedDict


# Loop bounds.
# ----- imports for runtime/memory/resolution.py -----
"""Resolution agent helpers — playbook → tool-call translation.

The resolution agent matches the L7 PlaybookStore for the session's
signals and produces a list of suggested tool calls. The framework's
risk-rated gateway (``runtime.tools.gateway``) decides whether each
call runs auto / notify-soft / require-approval based on its policy
and the prod-environment override.

This module is a thin, deterministic translator. The agent's prompt
(``skills/resolution/system.md``) describes the reasoning; the helpers
here are pure functions exercised by unit tests.

Surface:

* :class:`ToolCallSpec` — typed dict for a single suggested tool call.
* :func:`playbook_to_tool_calls` — given a playbook dict (the YAML
  shape :class:`PlaybookStore` already validates), return the list of
  :class:`ToolCallSpec` entries the agent should issue.
* :func:`top_playbook` — given the full
  ``IncidentState.memory.l7_playbooks`` suggestion list, return the
  highest-scoring playbook id (None if empty).
"""


from typing import Any, TypedDict



# ----- imports for runtime/orchestrator.py -----
"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""

import warnings
from typing import AsyncIterator, Generic, Type, TypeVar





# ----- imports for runtime/api.py -----
"""FastAPI app — health, listings, incident, and multi-session endpoints.

``build_app(cfg)`` is sync and constructs the FastAPI instance. The long-lived
:class:`runtime.service.OrchestratorService` is created during the app's
startup lifespan and stored on ``app.state.service``; its underlying
:class:`runtime.orchestrator.Orchestrator` is exposed as
``app.state.orchestrator`` so legacy routes keep working without
double-building the FastMCP transports / SQLite engines.

The shutdown hook calls ``service.shutdown()`` which cancels in-flight
session tasks, closes MCP clients, joins the background loop thread, and
resets the process-singleton.

``POST /sessions``, ``GET /sessions``, ``DELETE /sessions/{id}`` delegate
to ``OrchestratorService``. The legacy ``POST /investigate`` is preserved
as a deprecated alias and delegates to the same long-lived service so
old clients keep working.

The module-level ``get_app()`` is a no-arg factory suitable for
``uvicorn --factory``: it reads ``ASR_CONFIG`` (default
``config/config.yaml``) and returns a fresh app.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse



# ----- imports for runtime/api_dedup.py -----
"""Dedup retraction HTTP routes.

Exposes ``register_dedup_routes(app, *, store_provider)`` — a side-car
router so we don't need to inline these routes in ``runtime.api``. The
caller wires it into the main FastAPI app at lifespan startup; tests
construct a tiny FastAPI app and register only this router so they
don't pull in the full lifespan stack.

Endpoint:
  ``POST /sessions/{session_id}/un-duplicate``
    Body: ``{"retracted_by": str | None, "note": str | None}``
    On success: 200 with the updated session.
    409 when ``status != "duplicate"``.
    404 when the session id is unknown.

The endpoint never re-runs the agent graph — operators trigger that
explicitly. The audit row is inserted in the same transaction as the
status flip via :meth:`SessionStore.un_duplicate`.
"""



from fastapi import FastAPI, HTTPException




# ====== module: runtime/config.py ======

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


class MetadataConfig(BaseModel):
    """Relational store for incident metadata. SQLite (dev) or Postgres (prod)."""
    url: str = "sqlite:///incidents/incidents.db"
    pool_size: int = 5      # postgres only; sqlite uses NullPool
    echo: bool = False


VectorBackend = Literal["faiss", "pgvector", "none"]
DistanceStrategy = Literal["cosine", "euclidean", "inner_product"]


class VectorConfig(BaseModel):
    """Vector store backing. FAISS (dev) or PGVector (prod) or none (keyword-only)."""
    backend: VectorBackend = "faiss"
    path: str = "incidents/faiss"
    collection_name: str = "incidents"
    distance_strategy: DistanceStrategy = "cosine"


class StorageConfig(BaseModel):
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)


class Paths(BaseModel):
    skills_dir: str | None = None
    incidents_dir: str = "incidents"


class OrchestratorConfig(BaseModel):
    entry_agent: str = "intake"
    # Signals an agent may emit (via ``update_incident.patch.signal``) that
    # the router will accept and look up against the skill's ``routes`` table.
    # Anything outside this set falls through to ``when: default``. Override
    # in YAML to extend the vocabulary; the default keeps current behaviour.
    signals: list[str] = Field(
        default_factory=lambda: ["success", "failed", "needs_input"],
    )


RiskLevel = Literal["low", "medium", "high"]


class ProdOverrides(BaseModel):
    """Per-environment HITL tightening rules for the gateway.

    When the live ``Session.environment`` is in ``prod_environments`` AND
    the tool name matches one of the globs in ``resolution_trigger_tools``,
    the gateway forces ``require-approval`` regardless of the tool's
    risk-tier lookup. The override can only TIGHTEN, never relax — it runs
    BEFORE the risk-tier dispatch in ``effective_action``.

    Globs use ``fnmatch`` semantics (``*`` matches any run of characters,
    ``?`` matches one). E.g. ``"remediation:*"`` matches all tools whose
    name starts with ``remediation:``.
    """

    prod_environments: list[str] = Field(default_factory=lambda: ["production"])
    resolution_trigger_tools: list[str] = Field(default_factory=list)


class GatewayConfig(BaseModel):
    """Risk-rated tool gateway configuration.

    ``policy`` maps a tool name to a declared risk level. The level drives
    the hybrid HITL action:

      * ``low``    -> auto-execute, no operator action;
      * ``medium`` -> notify-soft (no graph pause);
      * ``high``   -> require-approval (LangGraph ``interrupt()`` pauses).

    Tools absent from the policy default to ``low`` (auto). Apps that need
    stricter prod behaviour configure ``prod_overrides``.

    ``notify_channel`` is an opaque routing hint passed to the notify
    sink (Slack handle, log channel, webhook id …); the gateway itself
    does not interpret it.
    """

    policy: dict[str, RiskLevel] = Field(default_factory=dict)
    notify_channel: str | None = None
    prod_overrides: ProdOverrides | None = None
    # Pending-approval timeout (seconds). When a high-risk tool call
    # enters ``interrupt()`` and the operator never returns, the
    # session sits in ``awaiting_input`` indefinitely and counts against
    # ``OrchestratorService.max_concurrent_sessions`` — eventually
    # leaking the slot. The :class:`runtime.tools.approval_watchdog`
    # asyncio task scans active sessions every 60s and resumes any
    # ``pending_approval`` ToolCall whose ``ts`` is older than this
    # value with ``decision="timeout"``. Default: 1 hour.
    approval_timeout_seconds: int = 3600


class RuntimeConfig(BaseModel):
    """Framework-runtime knobs that apps can override.

    ``state_class`` is a dotted import path to a ``runtime.state.Session``
    subclass. ``None`` (or omitted) means "use the framework default
    (``runtime.state.Session``)". Apps that ship a custom domain state set
    this to e.g. ``"examples.incident_management.state.IncidentState"`` so
    that the orchestrator and storage layer hydrate rows into the right
    class.

    ``framework_app_config_path`` is a dotted reference of the form
    ``module.path:callable`` resolving to a no-arg callable that returns
    a :class:`FrameworkAppConfig` instance. Used by
    ``Orchestrator.create`` so the runtime never has to import an
    app-specific config module. ``None`` (default) falls back to a bare
    ``FrameworkAppConfig()``.
    """

    state_class: str | None = None
    framework_app_config_path: str | None = None
    # Optional dotted reference of the form ``module.path:callable``
    # resolving to a no-arg callable returning a :class:`DedupConfig`
    # (or ``None`` if dedup is not configured). Apps that want the
    # two-stage dedup pipeline expose this on their YAML so the
    # runtime never has to import an app-specific config module to
    # discover dedup settings.
    dedup_config_path: str | None = None
    # Optional dotted reference for the app-specific list of
    # environments rendered on the ``GET /environments`` endpoint.
    # Apps that don't expose environments leave this unset; the
    # endpoint then returns an empty list.
    environments_provider_path: str | None = None
    # Hard cap on concurrent in-flight sessions a single
    # ``OrchestratorService`` will run. ``start_session`` raises
    # ``SessionCapExceeded`` once the registry holds this many entries
    # — fail fast, do not queue. Tune per deployment; the default is
    # generous enough for an interactive desk while keeping a single
    # process from saturating MCP transports.
    max_concurrent_sessions: int = 8
    # Optional risk-rated tool gateway. When ``None``, the gateway is
    # bypassed entirely and tools execute as before.
    gateway: GatewayConfig | None = None


# ---------------------------------------------------------------------------
# FrameworkAppConfig — generic cross-cutting application knobs the
# framework reads at runtime. Apps compose this inside their own
# ``AppConfig`` (``IncidentAppConfig``, ``CodeReviewAppConfig``) and
# expose a no-arg provider via ``RuntimeConfig.framework_app_config_path``.
# Keeps app-specific config modules out of ``runtime/`` imports.
# ---------------------------------------------------------------------------


_DEFAULT_DEDUP_SYSTEM_PROMPT = (
    "You are deduplicating sessions in an agent-orchestration framework. "
    "Decide whether the new session is a duplicate of the prior session. "
    "Return strict JSON: {\"is_duplicate\": bool, \"confidence\": float, "
    "\"rationale\": string}."
)


class UIBadge(BaseModel):
    """One badge entry — label + Streamlit color."""
    label: str
    color: str  # streamlit-allowed: red|orange|yellow|blue|green|violet|gray|primary

    model_config = {"frozen": True, "extra": "forbid"}


class UIDetailField(BaseModel):
    """A configured detail-pane field. ``key`` is a dotted path resolved
    against ``Session.extra_fields`` (or the session dict itself)."""
    key: str
    label: str
    section: str = "summary"  # "summary" | "metrics" | "meta"

    model_config = {"frozen": True, "extra": "forbid"}


class UIConfig(BaseModel):
    """App-driven UI rendering knobs. Keeps the generic Streamlit shell
    in ``runtime/ui.py`` agnostic of any specific domain — colors, labels,
    and tag prefixes come from YAML.

    ``badges`` is a 2-level dict: ``{field_name: {value: UIBadge}}``.
    Example: ``{"status": {"open": {"label": "OPEN", "color": "red"}}}``.

    ``detail_fields`` lists fields the detail pane renders, in order.
    Each entry may target a section (``summary``/``metrics``/``meta``).

    ``tags`` is an opaque key->tag-string map the UI consults for
    cross-skill signals (e.g. ``prior_match_supported`` -> the literal
    tag a skill emits).
    """
    badges: dict[str, dict[str, UIBadge]] = Field(default_factory=dict)
    detail_fields: list[UIDetailField] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class FrameworkAppConfig(BaseModel):
    """Generic application-supplied knobs the framework reads at runtime.

    Apps compose this inside their own AppConfig and surface it via
    a no-arg provider callable referenced by
    ``RuntimeConfig.framework_app_config_path``. The framework never
    imports app-specific config modules; it only reads these fields.
    """

    confidence_threshold: float = 0.75
    similarity_threshold: float = 0.2
    escalation_teams: list[str] = Field(default_factory=list)
    severity_aliases: dict[str, str] = Field(default_factory=dict)
    dedup_system_prompt: str = _DEFAULT_DEDUP_SYSTEM_PROMPT
    # Intake runner knobs: forwarded into IntakeContext at graph-build time.
    intake_top_k: int = 3
    intake_similarity_threshold: float = 0.7
    # UI rendering knobs surfaced to the generic runtime UI. Mirrors
    # AppConfig.ui — the FrameworkAppConfig provider can either copy
    # AppConfig.ui or supply its own. Defaults to empty so apps that
    # don't render with the generic UI pay nothing.
    ui: UIConfig = Field(default_factory=UIConfig)


def resolve_framework_app_config(
    dotted: str | None,
) -> FrameworkAppConfig:
    """Resolve a ``module:callable`` provider into a ``FrameworkAppConfig``.

    Returns a bare ``FrameworkAppConfig()`` when ``dotted`` is ``None``.
    Raises ``ValueError`` for malformed paths and ``ImportError`` /
    ``AttributeError`` propagate from the underlying resolution so that
    misconfiguration fails loud at boot.

    The provider must be a no-arg callable returning a
    ``FrameworkAppConfig``; anything else raises ``TypeError``.
    """
    if dotted is None:
        return FrameworkAppConfig()
    if ":" not in dotted:
        raise ValueError(
            f"framework_app_config_path={dotted!r} must be in "
            "'module.path:callable' form"
        )
    module_name, _, attr = dotted.partition(":")
    import importlib
    mod = importlib.import_module(module_name)
    provider = getattr(mod, attr)
    cfg = provider()
    if not isinstance(cfg, FrameworkAppConfig):
        raise TypeError(
            f"provider {dotted!r} returned {type(cfg).__name__}; "
            "expected FrameworkAppConfig"
        )
    return cfg


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: Paths = Field(default_factory=Paths)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    # Cross-cutting framework knobs (confidence threshold, escalation
    # roster, severity aliases, dedup prompt, intake tuning) read by
    # the runtime directly off the loaded ``AppConfig`` — no
    # app-specific provider callable required. Apps configure these
    # under the ``framework:`` block of their YAML; tests build them
    # in code via ``FrameworkAppConfig(...)``. Defaults are framework-
    # neutral so unconfigured apps still validate cleanly.
    framework: FrameworkAppConfig = Field(default_factory=FrameworkAppConfig)
    # Two-stage dedup pipeline shape. Typed as ``Any`` because
    # ``DedupConfig`` lives in ``runtime.dedup`` and importing it here
    # would introduce a circular import (``runtime.dedup`` ->
    # ``runtime.config``). The ``_coerce_dedup`` validator below
    # promotes a raw dict (the YAML shape) to a real ``DedupConfig``;
    # callers reading ``cfg.dedup`` get the typed object.
    dedup: Any | None = None
    # App-specific environments roster surfaced on the UI's
    # ``GET /environments`` endpoint and the env selector. Empty list
    # means "this app doesn't expose environments".
    environments: list[str] = Field(default_factory=list)
    # Declarative trigger registry. Each entry is one transport-flavoured
    # ``TriggerConfig`` (api/webhook/schedule/plugin). Typed as
    # ``list[Any]`` because Pydantic v2's discriminated-union binding
    # pulls in the trigger module at import time, which would introduce
    # a circular import. The ``_coerce_triggers`` validator below
    # promotes raw dicts to the proper TriggerConfig variants.
    triggers: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coerce_dedup(self) -> "AppConfig":
        # Lazy import to avoid the circular dep with ``runtime.dedup``
        # (which imports things that re-import ``runtime.config``).

        if self.dedup is None:
            return self
        if isinstance(self.dedup, DedupConfig):
            return self
        if isinstance(self.dedup, dict):
            self.__dict__["dedup"] = DedupConfig(**self.dedup)
            return self
        raise ValueError(
            f"app.dedup must be a DedupConfig or dict; got "
            f"{type(self.dedup).__name__}"
        )

    @model_validator(mode="after")
    def _coerce_triggers(self) -> "AppConfig":
        # Lazy import inside the validator to avoid a circular import:
        # ``runtime.triggers.config`` is free to ``import AppConfig`` for
        # typing without a forward-declaration dance.

        variants = {
            "api": APITriggerConfig,
            "webhook": WebhookTriggerConfig,
            "schedule": ScheduleTriggerConfig,
            "plugin": PluginTriggerConfig,
        }
        coerced: list[Any] = []
        for raw in self.triggers:
            if isinstance(
                raw,
                (APITriggerConfig, WebhookTriggerConfig,
                 ScheduleTriggerConfig, PluginTriggerConfig),
            ):
                coerced.append(raw)
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"trigger entries must be dicts; got {type(raw).__name__}"
                )
            t = raw.get("transport", "api")
            cls = variants.get(t)
            if cls is None:
                raise ValueError(
                    f"unknown trigger transport {t!r}; "
                    f"expected one of {sorted(variants)}"
                )
            coerced.append(cls(**raw))
        # Pydantic v2 stores fields in ``__dict__``; assigning here is
        # the documented way to mutate after validation.
        self.__dict__["triggers"] = coerced
        return self


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

# ====== module: runtime/state.py ======

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


# Per-call audit metadata for the risk-rated tool gateway.
ToolRisk = Literal["low", "medium", "high"]
ToolStatus = Literal[
    "executed",                 # auto / legacy default
    "executed_with_notify",     # medium-risk: ran + soft-notify
    "pending_approval",         # high-risk: graph paused on interrupt()
    "approved",                 # high-risk: operator approved, then ran
    "rejected",                 # high-risk: operator rejected, did not run
    "timeout",                  # high-risk: approval window expired
]


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | list | int | float | bool | None
    ts: str
    # Audit fields for the risk-rated gateway. All optional and
    # default-permissive so legacy rows in the JSON column hydrate with
    # ``status="executed"`` and the rest of the fields ``None`` —
    # preserving back-compat with older sessions.
    risk: ToolRisk | None = None
    status: ToolStatus = "executed"
    approver: str | None = None
    approved_at: str | None = None
    approval_rationale: str | None = None


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


class Session(BaseModel):
    """Framework base session. Lifecycle + telemetry fields only.

    Applications subclass this and add domain fields. The framework only
    reads/writes the fields declared here.
    """

    id: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    agents_run: list[AgentRun] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    findings: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pending_intervention: dict | None = None
    user_inputs: list[str] = Field(default_factory=list)
    # Dedup linkage. NULL by default; set when this session is
    # confirmed as a duplicate of a prior closed session. The value is
    # the prior session's id; the link is non-destructive — both
    # sessions remain queryable. See ``runtime.dedup``.
    parent_session_id: str | None = None
    # Stage-2 LLM rationale for the dedup decision. Stored on the
    # session row so the UI can render "why was this marked duplicate?"
    # without needing a separate join.
    dedup_rationale: str | None = None
    # Bag for app-specific session data the framework doesn't touch.
    # Apps that previously subclassed Session to add typed fields now
    # store them here. The storage layer round-trips this via the
    # matching ``IncidentRow.extra_fields`` JSON column.
    extra_fields: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # App-overridable agent-input formatter hook.
    # ------------------------------------------------------------------
    def to_agent_input(self) -> str:
        """Return the human-message preamble each agent receives.

        Apps subclass ``Session`` and override this to surface the
        domain shape (``Incident X / Environment Y / Query Z`` for the
        incident-management app, ``PR title / repo / diff stats`` for
        code review, etc.). The framework default keeps the prompt
        framework-agnostic — id + status only — so that any app that
        has not overridden the hook still gets a usable preamble.

        Findings, prior agent output, and operator-supplied user input
        are appended at the end so the surface stays useful even when
        the subclass keeps the default prefix.
        """
        base = (
            f"Session {self.id}\n"
            f"Status: {self.status}\n"
        )
        for agent_key, finding in self.findings.items():
            base += f"Findings ({agent_key}): {finding}\n"
        if self.user_inputs:
            bullets = "\n".join(f"- {ui}" for ui in self.user_inputs)
            base += (
                "\nUser-provided context (appended via intervention):\n"
                f"{bullets}\n"
            )
        return base

    # ------------------------------------------------------------------
    # App-overridable session id minting hook.
    # ------------------------------------------------------------------
    @classmethod
    def id_format(cls, *, seq: int) -> str:
        """Return the canonical session id for the given sequence number.

        Apps override this on their ``Session`` subclass to produce an
        id format that suits their domain (e.g. ``PR-{repo}-{number}``
        for code review). The framework default keeps the legacy
        ``INC-YYYYMMDD-NNN`` shape so existing on-disk rows and any app
        that has not opted in continue to round-trip cleanly.

        ``seq`` is the per-day monotonic sequence supplied by
        ``SessionStore._next_id``; it lets the default format produce
        the expected zero-padded suffix without each subclass
        re-implementing the SQL scan.
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"INC-{today}-{seq:03d}"

# ====== module: runtime/state_resolver.py ======

def resolve_state_class(dotted_path: str | None) -> Type[Session]:
    """Resolve ``dotted_path`` to a concrete ``Session`` subclass.

    ``None`` and ``""`` are treated as "use the framework default
    (``runtime.state.Session``)". Any other input must be a fully
    qualified dotted import path (``pkg.module.ClassName``) pointing at a
    class that ``issubclass(Session)``.

    Raises:
        ValueError: if ``dotted_path`` is not a dotted path.
        ImportError: if the target module cannot be imported.
        AttributeError: if the module does not define the attribute.
        TypeError: if the resolved attribute is not a Session subclass.
    """
    if not dotted_path:
        return Session

    if "." not in dotted_path:
        raise ValueError(
            f"state_class must be a dotted path 'pkg.module.ClassName'; "
            f"got {dotted_path!r}"
        )

    module_path, _, attr_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"cannot import state_class module {module_path!r} "
            f"(from {dotted_path!r}): {exc}"
        ) from exc

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"module {module_path!r} has no attribute {attr_name!r} "
            f"(state_class={dotted_path!r})"
        )

    cls = getattr(module, attr_name)
    if not isinstance(cls, type) or not issubclass(cls, Session):
        raise TypeError(
            f"state_class {dotted_path!r} must be a Session subclass; "
            f"got {cls!r}"
        )
    return cls

# ====== module: runtime/similarity.py ======

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

# ====== module: runtime/skill.py ======

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


class DispatchRule(BaseModel):
    """One condition/target pair used by ``kind: supervisor`` skills with
    ``dispatch_strategy: rule``.

    ``when`` is a safe-eval expression (see :func:`_validate_safe_expr`)
    evaluated against the live session payload at dispatch time. The
    first matching rule wins; ``target`` names a subordinate agent.
    """

    when: str
    target: str


SkillKind = Literal["responsive", "supervisor", "monitor"]


# Cron-expression sanity check (5-field form: minute hour dom month dow).
# Each field accepts: '*', or a comma-separated list of ints / int ranges
# (a-b) / step ('* /n' or 'a-b/n'). This is intentionally a small subset
# of POSIX cron — broad enough for monitor schedules, narrow enough to
# parse with a regex (no external ``croniter`` dep, which is unavailable
# in the air-gapped target env per ``rules/build.md``).
_CRON_FIELD_RE = re.compile(
    r"^(\*|\*/\d+|\d+(-\d+)?(/\d+)?(,\d+(-\d+)?(/\d+)?)*)$"
)


def _validate_cron(expr: str) -> None:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(
            f"schedule {expr!r} is not a 5-field cron expression "
            f"(got {len(parts)} fields)"
        )
    for i, field in enumerate(parts):
        if not _CRON_FIELD_RE.match(field):
            raise ValueError(
                f"schedule {expr!r}: field #{i+1} {field!r} is not a "
                f"valid cron expression component"
            )


# Safe-eval AST whitelist for monitor ``emit_signal_when`` and
# supervisor ``DispatchRule.when``. We intentionally implement this with
# the stdlib ``ast`` module rather than depend on ``simpleeval`` —
# ``simpleeval`` is not available in the air-gapped target env. The
# whitelist is the smallest set that lets operators write conditions
# like ``observation['error_rate'] > 0.05 and status == 'open'`` without
# enabling arbitrary code execution. See plan R7.
_SAFE_AST_NODES: tuple[type, ...] = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Subscript, ast.Index, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.IfExp,
)


def _validate_safe_expr(expr: str, *, source: str) -> None:
    """Reject non-whitelisted AST nodes in user-supplied expressions.

    Raises ``ValueError`` if the expression is not parseable or contains
    nodes outside :data:`_SAFE_AST_NODES`. Callable invocations,
    attribute access, comprehensions, lambdas, walrus, and the like are
    explicitly rejected.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{source}: cannot parse expression {expr!r}: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise ValueError(
                f"{source}: expression {expr!r} uses disallowed "
                f"construct {type(node).__name__!r} (safe-eval only "
                f"permits constants, names, comparisons, boolean ops, "
                f"arithmetic, subscripts, and literals)"
            )


def _resolve_dotted_callable(path: str, *, source: str) -> Callable[..., Any]:
    """Resolve a ``module.path:attr`` (or ``module.path.attr``) string to a callable.

    Used by ``kind: supervisor`` skills' ``runner`` field so app-level
    extension hooks can be wired in via YAML and validated at
    skill-load time. Raises ``ValueError`` on any failure mode
    (malformed path, missing module, missing attribute, non-callable
    target). The error names ``source`` so the YAML author sees which
    skill is broken.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{source}: dotted path must be a non-empty string")
    text = path.strip()
    if ":" in text:
        mod_part, _, attr_part = text.partition(":")
    elif "." in text:
        mod_part, _, attr_part = text.rpartition(".")
    else:
        raise ValueError(
            f"{source}: dotted path {path!r} must include an attribute "
            f"(use 'pkg.module:func' or 'pkg.module.func')"
        )
    if not mod_part or not attr_part:
        raise ValueError(
            f"{source}: dotted path {path!r} is missing the module or "
            f"attribute component"
        )
    try:
        module = importlib.import_module(mod_part)
    except ImportError as exc:
        raise ValueError(
            f"{source}: cannot import module {mod_part!r} from path "
            f"{path!r}: {exc}"
        ) from exc
    try:
        target = getattr(module, attr_part)
    except AttributeError as exc:
        raise ValueError(
            f"{source}: module {mod_part!r} has no attribute "
            f"{attr_part!r} (path={path!r})"
        ) from exc
    if not callable(target):
        raise ValueError(
            f"{source}: target {path!r} resolved to a non-callable "
            f"({type(target).__name__})"
        )
    return target


class Skill(BaseModel):
    """Single skill definition with a ``kind`` discriminator.

    The ``kind`` field selects the agent's execution model. Per-kind
    fields are declared at the model level for ergonomic YAML
    authoring; a ``model_validator`` rejects any combination that
    doesn't match the declared kind.

    Default kind is ``responsive`` so existing YAML (and historic
    Skill(...) construction in tests) keeps working without an explicit
    ``kind:`` field.
    """

    name: str
    description: str
    kind: SkillKind = "responsive"

    # ----- responsive (today's default behaviour) -----
    model: str | None = None
    tools: dict[str, list[str]] = Field(default_factory=dict)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str = ""
    stub_response: str | None = None
    """Per-skill canned response used by ``StubChatModel`` when
    ``provider.kind == "stub"``.  Takes precedence over any entry in
    ``_DEFAULT_STUB_CANNED`` for the same agent name."""

    # ----- supervisor (no-LLM router) -----
    subordinates: list[str] = Field(default_factory=list)
    dispatch_strategy: Literal["llm", "rule"] = "llm"
    dispatch_prompt: str | None = None
    dispatch_rules: list[DispatchRule] = Field(default_factory=list)
    max_dispatch_depth: int = 3
    # Optional dotted-path extension hook for app-specific supervisor
    # logic (e.g. memory-layer hydration, single-active-investigation
    # gates). The runner is invoked BEFORE the dispatch table and may
    # either mutate state or short-circuit to ``__end__``. Resolved at
    # skill-load time so misconfigured YAML fails fast.
    runner: str | None = None

    # ----- monitor (out-of-band, scheduled) -----
    schedule: str | None = None             # cron expression
    observe: list[str] = Field(default_factory=list)  # tool names
    emit_signal_when: str | None = None     # safe-eval expression
    trigger_target: str | None = None       # trigger registry name
    tick_timeout_seconds: float = 30.0      # per-tick timeout

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

    @field_validator("max_dispatch_depth")
    @classmethod
    def _validate_max_depth(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError(
                f"max_dispatch_depth must be between 1 and 10 (got {v})"
            )
        return v

    @model_validator(mode="after")
    def _validate_kind_shape(self) -> "Skill":
        """Per-kind field-shape validation.

        Each kind has a strict allow-list of fields. Anything from
        another kind raises ValueError naming the offending field and
        the kind. Required fields (e.g. monitor.schedule) are also
        enforced here.
        """
        kind = self.kind
        if kind == "responsive":
            self._validate_responsive()
        elif kind == "supervisor":
            self._validate_supervisor()
        elif kind == "monitor":
            self._validate_monitor()
        return self

    # ----- per-kind shape validators (called from _validate_kind_shape) -----

    def _validate_responsive(self) -> None:
        forbidden = {
            "subordinates": bool(self.subordinates),
            "dispatch_prompt": self.dispatch_prompt is not None,
            "dispatch_rules": bool(self.dispatch_rules),
            "schedule": self.schedule is not None,
            "observe": bool(self.observe),
            "emit_signal_when": self.emit_signal_when is not None,
            "trigger_target": self.trigger_target is not None,
            "runner": self.runner is not None,
        }
        # dispatch_strategy is allowed to keep its default; only flag it if
        # the user explicitly set it to non-default.
        if self.dispatch_strategy != "llm":
            forbidden["dispatch_strategy"] = True
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=responsive) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        # ``system_prompt`` is sourced from the agent's *.md files at load
        # time (see ``load_skill``); the model itself permits an empty
        # string for tests and ad-hoc constructors that don't go through
        # the loader. The loader enforces .md presence for responsive
        # skills.

    def _validate_supervisor(self) -> None:
        forbidden = {
            "system_prompt": bool(self.system_prompt),
            "tools": bool(self.tools),
            "routes": bool(self.routes),
            "stub_response": self.stub_response is not None,
            "schedule": self.schedule is not None,
            "observe": bool(self.observe),
            "emit_signal_when": self.emit_signal_when is not None,
            "trigger_target": self.trigger_target is not None,
        }
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=supervisor) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        if not self.subordinates:
            raise ValueError(
                f"skill {self.name!r} (kind=supervisor) requires a non-empty "
                f"subordinates list"
            )
        if self.runner is None:
            # Default every supervisor to the framework intake runner
            # (similarity retrieval + dedup gate). Apps override by
            # setting ``runner:`` in YAML.
            self.runner = "runtime.intake:default_intake_runner"
        # Resolve at skill-load time so a typo in YAML surfaces here,
        # not in the middle of a session. The resolver itself raises
        # ``ValueError`` with a helpful message — bubble that up.
        _resolve_dotted_callable(
            self.runner,
            source=f"skill {self.name!r} runner",
        )
        if self.dispatch_strategy == "llm" and not self.dispatch_prompt:
            raise ValueError(
                f"skill {self.name!r} (kind=supervisor, strategy=llm) requires "
                f"dispatch_prompt"
            )
        if self.dispatch_strategy == "rule":
            if not self.dispatch_rules:
                raise ValueError(
                    f"skill {self.name!r} (kind=supervisor, strategy=rule) "
                    f"requires dispatch_rules"
                )
            for i, rule in enumerate(self.dispatch_rules):
                _validate_safe_expr(
                    rule.when,
                    source=f"skill {self.name!r} dispatch_rules[{i}].when",
                )
                if rule.target not in self.subordinates:
                    raise ValueError(
                        f"skill {self.name!r}: dispatch_rules[{i}].target="
                        f"{rule.target!r} not found in subordinates "
                        f"({sorted(self.subordinates)})"
                    )

    def _validate_monitor(self) -> None:
        forbidden = {
            "system_prompt": bool(self.system_prompt),
            "routes": bool(self.routes),
            "stub_response": self.stub_response is not None,
            "subordinates": bool(self.subordinates),
            "dispatch_prompt": self.dispatch_prompt is not None,
            "dispatch_rules": bool(self.dispatch_rules),
            "runner": self.runner is not None,
        }
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=monitor) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        if not self.schedule:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires a schedule "
                f"(5-field cron expression)"
            )
        _validate_cron(self.schedule)
        if not self.observe:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires a non-empty "
                f"observe list"
            )
        if not self.emit_signal_when:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires emit_signal_when"
            )
        _validate_safe_expr(
            self.emit_signal_when,
            source=f"skill {self.name!r} emit_signal_when",
        )
        if not self.trigger_target:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires trigger_target"
            )
        if self.tick_timeout_seconds <= 0:
            raise ValueError(
                f"skill {self.name!r}: tick_timeout_seconds must be positive "
                f"(got {self.tick_timeout_seconds})"
            )


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
    # P6: only ``responsive`` skills require a system_prompt assembled from
    # the directory's .md files. ``supervisor`` and ``monitor`` skills are
    # configured purely via YAML, so the .md requirement is relaxed for
    # those kinds. The default kind is still ``responsive`` so existing
    # YAML keeps the historical "config.yaml + system.md" requirement.
    kind = cfg.get("kind", "responsive")
    md_files = sorted(base.glob("*.md"))
    if kind == "responsive":
        if not md_files:
            raise FileNotFoundError(f"no .md prompt files in skill dir: {base}")
        agent_prompt = _concat_md(md_files)
        cfg["system_prompt"] = (
            f"{agent_prompt}\n\n{common}".strip() if common else agent_prompt
        )
    else:
        # Non-responsive kinds may still ship descriptive .md alongside
        # config.yaml, but it's optional and never used as a system prompt.
        cfg.setdefault("system_prompt", "")
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

# ====== module: runtime/llm.py ======

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

# ====== module: runtime/storage/models.py ======

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

    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    agents_run: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_calls: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    findings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pending_intervention: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    user_inputs: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Dedup linkage. NULL by default; set when this session is
    # confirmed as a duplicate of a prior closed session. Indexed so
    # ``list_children(parent)`` is fast.
    parent_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Stage-2 LLM rationale persisted on the row so the UI can surface
    # "why was this flagged?" without a separate decisions table.
    # Mirror of ``Session.dedup_rationale``.
    dedup_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Bag for any state-class field that does not have a typed column
    # above. ``SessionStore`` walks ``state_cls.model_fields`` and
    # routes unknown fields here on save; ``_row_to_session`` merges
    # them back into the model on load. Additive: legacy rows written
    # before this column existed have ``NULL`` and round-trip cleanly.
    extra_fields: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_incidents_status_env_active", "status", "environment",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
        Index("ix_incidents_created_at_active", "created_at",
              postgresql_where=text("deleted_at IS NULL"),
              sqlite_where=text("deleted_at IS NULL")),
        Index("ix_incidents_parent_session_id", "parent_session_id"),
    )


# Append-only audit log of dedup retractions. No FK to ``incidents``
# so retraction history survives session deletion. Indexed on
# ``session_id`` for the parent detail pane lookup.
class DedupRetractionRow(Base):
    __tablename__ = "dedup_retractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    retracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    retracted_by: Mapped[str | None] = mapped_column(String, nullable=True)
    original_match_id: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_dedup_retractions_session_id", "session_id"),
    )


SessionRow = IncidentRow  # generic alias

# ====== module: runtime/storage/engine.py ======

_SQLITE_BUSY_TIMEOUT_MS = 30_000


def build_engine(cfg: MetadataConfig) -> Engine:
    if cfg.url.startswith("sqlite"):
        engine = create_engine(
            cfg.url,
            poolclass=NullPool,
            echo=cfg.echo,
            connect_args={"check_same_thread": False, "isolation_level": None},
        )
        _install_sqlite_concurrency_pragmas(engine)
        return engine
    return create_engine(cfg.url, pool_size=cfg.pool_size, echo=cfg.echo)


def _install_sqlite_concurrency_pragmas(engine: Engine) -> None:
    """Configure every new SQLite connection so it plays nicely with a
    concurrent LangGraph checkpointer on the same DB file.

    Three things happen here:

    1. ``connect_args["isolation_level"]=None`` (set in ``build_engine``)
       puts the underlying ``sqlite3.Connection`` in autocommit mode so
       Python's stdlib doesn't sneak a ``BEGIN`` in front of our SQL.
       SQLAlchemy then drives transactions explicitly via the ``begin``
       event hook below — this is the documented escape hatch for
       custom transaction modes.
    2. ``PRAGMA journal_mode=WAL`` + ``PRAGMA synchronous=NORMAL`` +
       ``PRAGMA busy_timeout`` are issued on every new connection so
       readers and writers don't block each other and writers wait
       briefly on contention rather than failing immediately.
    3. ``BEGIN IMMEDIATE`` on the ``begin`` hook acquires the RESERVED
       lock at transaction start (before any read), so the
       ``busy_timeout`` retry loop can wait out a concurrent writer
       cleanly. Without this, two writers each in a ``BEGIN DEFERRED``
       transaction race to escalate and the loser hits ``database is
       locked`` instantly.
    """
    @sa_event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _conn_record):  # noqa: ANN001 — sqlalchemy event sig
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
        finally:
            cur.close()

    @sa_event.listens_for(engine, "begin")
    def _on_begin(conn):  # noqa: ANN001 — sqlalchemy event sig
        # Replace SQLAlchemy's default ``BEGIN`` (deferred) with
        # ``BEGIN IMMEDIATE``. Required so concurrent writers serialize
        # cleanly via busy_timeout instead of failing on lock-escalation.
        conn.exec_driver_sql("BEGIN IMMEDIATE")

# ====== module: runtime/storage/embeddings.py ======

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
        v = rng.standard_normal(self.dim).astype(np.float32)
        # Normalize to unit length so cosine similarity = dot product in [−1, 1].
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

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

# ====== module: runtime/storage/vector.py ======

_PLACEHOLDER_ID = "__seed__"


def _faiss_distance_strategy(name: str):
    from langchain_community.vectorstores.utils import DistanceStrategy
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
    }[name]


def _pgvector_distance_strategy(name: str):
    from langchain_postgres.vectorstores import DistanceStrategy
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN,
        "inner_product": DistanceStrategy.INNER_PRODUCT,
    }[name]


def _build_faiss(cfg: VectorConfig, embedder: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    folder = Path(cfg.path)
    index_file = folder / f"{cfg.collection_name}.faiss"
    if index_file.exists():
        return FAISS.load_local(
            folder_path=str(folder),
            index_name=cfg.collection_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
            distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
        )
    folder.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(
        [Document(page_content=_PLACEHOLDER_ID, metadata={"id": _PLACEHOLDER_ID})],
        embedding=embedder,
        ids=[_PLACEHOLDER_ID],
        distance_strategy=_faiss_distance_strategy(cfg.distance_strategy),
    )
    vs.delete(ids=[_PLACEHOLDER_ID])
    return vs


def _build_pgvector(cfg: VectorConfig, embedder: Embeddings,
                    engine) -> VectorStore:
    from langchain_postgres import PGVector
    return PGVector(
        embeddings=embedder,
        collection_name=cfg.collection_name,
        connection=engine,
        distance_strategy=_pgvector_distance_strategy(cfg.distance_strategy),
        use_jsonb=True,
    )


def build_vector_store(
    cfg: VectorConfig,
    embedder: Optional[Embeddings],
    metadata_engine=None,
) -> Optional[VectorStore]:
    if cfg.backend == "none" or embedder is None:
        return None
    if cfg.backend == "faiss":
        return _build_faiss(cfg, embedder)
    if cfg.backend == "pgvector":
        if metadata_engine is None:
            raise ValueError(
                "pgvector backend requires metadata_engine (the SQLAlchemy "
                "engine, used as the connection)"
            )
        return _build_pgvector(cfg, embedder, metadata_engine)
    raise ValueError(f"unknown vector backend: {cfg.backend!r}")


def distance_to_similarity(distance: float, strategy: str) -> float:
    """Normalize a backend-native distance to a similarity in roughly [0, 1].

    - cosine: ``1 - distance`` (LangChain's cosine-distance in [0, 2]).
    - inner_product: returns ``distance`` unchanged (already a similarity).
    - euclidean: ``1 / (1 + distance)`` -- monotonic, compressed to (0, 1].
    """
    if strategy == "cosine":
        return 1.0 - distance
    if strategy == "inner_product":
        return distance
    if strategy == "euclidean":
        return 1.0 / (1.0 + distance)
    raise ValueError(f"unknown distance strategy: {strategy!r}")

# ====== module: runtime/storage/history_store.py ======

StateT = TypeVar("StateT", bound=BaseModel)

# Allowed ``filter_kwargs`` keys = IncidentRow column names.
# Computed at module load so we can produce a precise error for typos.
_ALLOWED_FILTER_COLUMNS: frozenset[str] = frozenset(
    c.name for c in IncidentRow.__table__.columns
)


class HistoryStore(Generic[StateT]):
    """Read-only similarity search over the same row store, parametrised on ``StateT``.

    Never mutates. Reuses ``SessionStore``'s row->state converter via a
    private internal instance to avoid duplicating mapping logic; that
    converter inherits the same ``state_cls`` so hydration stays consistent.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        state_cls: Optional[Type[StateT]] = None,
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_threshold: float = 0.85,
        distance_strategy: str = "cosine",
    ) -> None:
        # Imported lazily so a bare ``HistoryStore`` import has no
        # side-effect of pulling SessionStore's heavier dep tree.


        self.engine = engine
        self._state_cls = state_cls
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.distance_strategy = distance_strategy
        # Private converter helper. We never call its mutating methods.
        # ``state_cls=None`` lets SessionStore pick its own default
        # (``runtime.state.Session``).
        ss_kwargs: dict[str, Any] = {
            "engine": engine, "embedder": None, "vector_store": None,
            "distance_strategy": distance_strategy,
        }
        if state_cls is not None:
            ss_kwargs["state_cls"] = state_cls
        self._converter: SessionStore[StateT] = SessionStore(**ss_kwargs)

    def _row_to_incident(self, row: IncidentRow) -> StateT:
        return self._converter._row_to_incident(row)

    def _load(self, incident_id: str) -> StateT:
        with Session(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def _list_filtered(self, *, filter_kwargs: Mapping[str, Any]) -> list[StateT]:
        """List non-deleted rows matching the given column filters.

        Pure SQL prefilter — used by both vector and keyword paths.
        """
        with Session(self.engine) as session:
            stmt = select(IncidentRow).where(IncidentRow.deleted_at.is_(None))
            for col, val in filter_kwargs.items():
                stmt = stmt.where(getattr(IncidentRow, col) == val)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    @staticmethod
    def _validate_filter_kwargs(filter_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
        """Reject filter keys that aren't IncidentRow columns.

        Returns a plain ``dict`` (defensive copy). ``None`` is treated as
        an empty filter for ergonomic callers.
        """
        if filter_kwargs is None:
            return {}
        bad = [k for k in filter_kwargs if k not in _ALLOWED_FILTER_COLUMNS]
        if bad:
            raise ValueError(
                f"unsupported filter_kwargs key(s): {bad}. "
                f"Allowed columns: {sorted(_ALLOWED_FILTER_COLUMNS)}"
            )
        return dict(filter_kwargs)

    def find_similar(
        self, *, query: str,
        filter_kwargs: Mapping[str, Any] | None = None,
        status_filter: str = "resolved",
        threshold: Optional[float] = None,
        limit: int = 5,
        # Back-compat: accept ``environment=`` so existing callers keep
        # compiling. New code should pass ``filter_kwargs={"environment": ...}``.
        environment: Optional[str] = None,
    ) -> list[tuple[StateT, float]]:
        """Return up to ``limit`` similar sessions matching the given filters.

        ``filter_kwargs`` is a mapping of ``IncidentRow`` column -> value
        (e.g. ``{"environment": "production"}``); each entry becomes an
        equality predicate in the SQL prefilter. ``status_filter`` is
        also applied (defaulting to ``"resolved"``). Vector path uses
        the configured VectorStore when both ``vector_store`` and
        ``embedder`` are present; otherwise keyword similarity.
        """
        filter_dict = self._validate_filter_kwargs(filter_kwargs)
        if environment is not None and "environment" not in filter_dict:
            # Convenience for legacy callers — translate to the new dict.
            filter_dict["environment"] = environment

        if self.vector_store is None or self.embedder is None:
            return self._keyword_similar(
                query=query, filter_kwargs=filter_dict,
                status_filter=status_filter,
                threshold=threshold, limit=limit,
            )
        threshold = self.similarity_threshold if threshold is None else threshold

        vec = self.embedder.embed_query(query)
        raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit * 4)
        out: list[tuple[StateT, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(float(distance), self.distance_strategy)
            if score < threshold:
                continue
            inc_id = doc.metadata.get("id")
            if inc_id is None:
                continue
            try:
                inc = self._load(inc_id)
            except (FileNotFoundError, ValueError):
                continue
            if getattr(inc, "status", None) != status_filter:
                continue
            if getattr(inc, "deleted_at", None) is not None:
                continue
            # Generic column-equality check via getattr — works for any
            # field declared on the configured state subclass.
            if not all(getattr(inc, k, None) == v for k, v in filter_dict.items()):
                continue
            out.append((inc, score))
            if len(out) >= limit:
                break
        return out

    def _keyword_similar(self, *, query, filter_kwargs, status_filter, threshold, limit):

        # SQL prefilter narrows the row set; status is filtered in
        # Python because it lives on the row but apps occasionally
        # override it via custom states.
        all_filtered = self._list_filtered(filter_kwargs=filter_kwargs)
        candidates_inc = [
            i for i in all_filtered
            if getattr(i, "status", None) == status_filter
            and getattr(i, "deleted_at", None) is None
        ]
        def _ef(i, key, default=""):
            """Read a field from typed attribute first, then extra_fields."""
            val = getattr(i, key, None)
            if val:
                return val
            return (getattr(i, "extra_fields", None) or {}).get(key, default)

        candidates = [
            {
                "id": i.id,
                "text": " ".join(filter(None, [
                    _ef(i, "query", "") or "",
                    _ef(i, "summary", "") or "",
                    " ".join(_ef(i, "tags", []) or []),
                ])),
                "incident": i,
            }
            for i in candidates_inc
        ]
        results = find_similar(
            query=query, candidates=candidates, text_field="text",
            scorer=KeywordSimilarity(),
            threshold=self.similarity_threshold if threshold is None else threshold,
            limit=limit,
        )
        return [(c["incident"], float(s)) for c, s in results]

# ====== module: runtime/storage/session_store.py ======

_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*-\d{8}-\d{3}$")

# StateT is bound to ``BaseModel`` so callers can pass either bare
# ``Session`` or any pydantic subclass. The resolver in
# :mod:`runtime.state_resolver` enforces a ``runtime.state.Session``
# subclass at config time; the looser bound here keeps the storage
# layer usable by ad-hoc tests that build a ``BaseModel`` directly.
StateT = TypeVar("StateT", bound=BaseModel)


def _embed_source(inc: BaseModel) -> str:
    """Produce the text that represents a session in the vector store.

    Reads ``query`` from a typed field on a Session subclass first; for
    bare ``runtime.state.Session`` instances, falls back to
    ``extra_fields["query"]``. Returns "" only when neither carries a
    value — those rows aren't vectorised.
    """
    typed = (getattr(inc, "query", "") or "").strip()
    if typed:
        return typed
    extras = getattr(inc, "extra_fields", None) or {}
    return str(extras.get("query") or "").strip()


def _embed_source_from_row(row: "IncidentRow") -> str:
    """Same as ``_embed_source`` but from a raw ORM row (used in save)."""
    return (row.query or "").strip()


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


def _deserialize_resolution(raw: Optional[str]):
    """Attempt JSON parse of stored resolution; return raw string on failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class SessionStore(Generic[StateT]):
    """Active session/incident lifecycle store, parametrised on ``StateT``.

    Owns CRUD on the row schema plus the vector write-through. Read-only
    similarity search lives in ``HistoryStore``.

    Threading note: methods open short-lived sessions; safe for the
    orchestrator's coarse-grained concurrency model.

    The ``state_cls`` ctor argument controls row->model hydration. Default
    is :class:`runtime.state.Session` (the framework base). Apps inject
    their own ``Session`` subclass (e.g. ``IncidentState``) via
    ``RuntimeConfig.state_class`` to surface domain-specific fields.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        state_cls: Type[StateT] = Session,  # type: ignore[assignment]
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        vector_path: Optional[str] = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
    ) -> None:
        self.engine = engine
        self._state_cls = state_cls
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_path = vector_path
        self.vector_index_name = vector_index_name
        self.distance_strategy = distance_strategy

    # ---------- ID minting ----------
    def _next_id(self, session: SqlSession) -> str:
        """Mint a new session id via ``state_cls.id_format(seq=...)``.

        The per-app id format lives on the ``Session`` subclass so each
        app picks its own prefix (``INC-`` for incidents, ``CR-`` for
        code-review, anything else custom apps want). The store still
        owns the monotonic sequence — it scans for prior rows whose id
        starts with the same ``PREFIX-YYYYMMDD-`` stem and returns
        ``max(seq) + 1``.
        """
        # Probe today's prefix by asking the state class to format seq=1
        # and stripping the ``-001`` suffix. Apps that override
        # ``id_format`` to return a non-``PREFIX-YYYYMMDD-NNN`` shape
        # (e.g. opaque ULIDs) fall through to the simple count path
        # below.
        sample = self._state_cls.id_format(seq=1)
        m = _SESSION_ID_RE.match(sample)
        if m is None:
            # Custom format — count all rows as the sequence base. Apps
            # that want collision-free ids should mint ULIDs in
            # ``id_format`` and ignore ``seq``.
            count = session.execute(
                select(IncidentRow.id)
            ).scalars().all()
            return self._state_cls.id_format(seq=len(count) + 1)

        # Extract the ``PREFIX-YYYYMMDD-`` stem (everything up to and
        # including the second hyphen).
        stem = sample.rsplit("-", 1)[0] + "-"
        like = f"{stem}%"
        rows = session.execute(
            select(IncidentRow.id).where(IncidentRow.id.like(like))
        ).scalars().all()
        max_seq = 0
        for r in rows:
            try:
                max_seq = max(max_seq, int(r.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return self._state_cls.id_format(seq=max_seq + 1)

    # ---------- public API ----------
    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> StateT:
        with SqlSession(self.engine) as session:
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
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            inc = self._row_to_incident(row)
        self._add_vector(inc)
        return inc

    def load(self, incident_id: str) -> StateT:
        if not _SESSION_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected PREFIX-YYYYMMDD-NNN"
            )
        with SqlSession(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def save(self, incident: StateT) -> None:
        if not _SESSION_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected PREFIX-YYYYMMDD-NNN"
            )
        incident.updated_at = _iso(_now())
        with SqlSession(self.engine) as session:
            existing = session.get(IncidentRow, incident.id)
            prior_text = _embed_source_from_row(existing) if existing is not None else ""
            data = self._incident_to_row_dict(incident)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()
        self._refresh_vector(incident, prior_text=prior_text)

    def delete(self, incident_id: str) -> StateT:
        with SqlSession(self.engine) as session:
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

    def list_all(self, *, include_deleted: bool = False) -> list[StateT]:
        with SqlSession(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_recent(self, limit: int = 20, *,
                    include_deleted: bool = False,
                    include_duplicates: bool = False) -> list[StateT]:
        """Most-recent sessions first.

        ``include_duplicates`` defaults to ``False`` so the main UI list
        stays clean of the long tail of dedup'd sessions. The UI opts in
        via ``include_duplicates=True`` to render the collapsed-
        duplicates row.
        """
        with SqlSession(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            if not include_duplicates:
                stmt = stmt.where(IncidentRow.status != "duplicate")
            stmt = stmt.order_by(
                desc(IncidentRow.created_at), desc(IncidentRow.id)
            ).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_children(self, parent_session_id: str) -> list[StateT]:
        """Return all sessions whose ``parent_session_id`` equals the given id.

        Powers the parent-session detail pane "children" section.
        Soft-deleted children are excluded; ordering is oldest-first so
        the UI shows them in the order they were flagged.
        """
        with SqlSession(self.engine) as session:
            stmt = (
                select(IncidentRow)
                .where(IncidentRow.parent_session_id == parent_session_id)
                .where(IncidentRow.deleted_at.is_(None))
                .order_by(IncidentRow.created_at, IncidentRow.id)
            )
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def un_duplicate(self, session_id: str, *,
                     retracted_by: str | None = None,
                     note: str | None = None) -> StateT:
        """Retract a duplicate flag.

        Behaviour (one transaction):
          * Loads the row; raises ``FileNotFoundError`` if missing.
          * Raises ``ValueError`` if ``status != "duplicate"`` so the
            HTTP layer can return ``409 Conflict``.
          * Captures ``parent_session_id`` as ``original_match_id``.
          * Sets ``status="new"`` and clears ``parent_session_id`` /
            ``dedup_rationale``.
          * Inserts a row in ``dedup_retractions`` for the audit trail.

        Idempotent at the boundary: a second call on the same id raises
        ``ValueError`` (the row is no longer a duplicate). The retraction
        does **not** auto-rerun the agent graph — operators trigger that
        explicitly.
        """
        # Imported here so the storage layer doesn't take a hard
        # import-time dependency on the audit table for callers that
        # never invoke retraction.

        with SqlSession(self.engine) as session:
            row = session.get(IncidentRow, session_id)
            if row is None:
                raise FileNotFoundError(session_id)
            if row.status != "duplicate":
                raise ValueError(
                    f"session {session_id!r} is not a duplicate "
                    f"(status={row.status!r})"
                )
            original_match_id = row.parent_session_id or ""
            row.status = "new"
            row.parent_session_id = None
            row.dedup_rationale = None
            row.updated_at = _now()
            session.add(DedupRetractionRow(
                session_id=session_id,
                retracted_at=_now(),
                retracted_by=retracted_by,
                original_match_id=original_match_id,
                note=note,
            ))
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    # ---------- vector helpers ----------
    def _persist_vector(self) -> None:
        """If FAISS-backed (has save_local) and a path is configured, persist to disk."""
        if self.vector_store is None:
            return
        if not hasattr(self.vector_store, "save_local"):
            return
        if not self.vector_path:
            return
        from pathlib import Path
        folder = Path(self.vector_path)
        folder.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(
            folder_path=str(folder),
            index_name=self.vector_index_name,
        )

    def _add_vector(self, inc: BaseModel) -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    def _refresh_vector(self, inc: BaseModel, *, prior_text: str) -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        if prior_text == text:
            return
        self.vector_store.delete(ids=[inc.id])
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    # ---------- mapping helpers ----------
    #
    # Round-trip is driven by ``state_cls.model_fields`` so any
    # ``Session`` subclass — incident-shaped, code-review-shaped, or
    # whatever a future app brings — round-trips losslessly. The
    # ``IncidentRow`` schema keeps its incident-shaped typed columns
    # for back-compat indexing (``query``, ``environment``,
    # ``severity``, ...); fields the row schema doesn't have a column
    # for land in the ``extra_fields`` JSON column on save and merge
    # back into the model on load.

    # Fields handled out-of-band by the row<->model converters; do not
    # send these to ``extra_fields`` on save.
    _STATE_TOP_LEVEL_FIELDS = frozenset({
        "id", "status", "created_at", "updated_at", "deleted_at",
        "agents_run", "tool_calls", "findings", "token_usage",
        "pending_intervention", "user_inputs",
        "parent_session_id", "dedup_rationale",
        # ``extra_fields`` is the bag itself — round-tripped via the
        # JSON column directly, never nested inside the bag.
        "extra_fields",
    })

    # Incident-shaped typed columns the row carries for back-compat
    # indexing. Apps whose state declares any of these get full
    # round-trip identity through the typed column; apps without them
    # leave the column at its DB default (empty string / NULL).
    _ROW_TYPED_DOMAIN_COLUMNS = frozenset({
        "query", "environment", "summary", "tags", "severity",
        "category", "matched_prior_inc", "resolution",
    })

    def _row_to_incident(self, row: IncidentRow) -> StateT:
        """Hydrate ``row`` into ``self._state_cls``.

        Fields are pulled from typed columns when the state class
        declares them; everything else is merged in from the
        ``extra_fields`` JSON bag. ``reporter`` is reconstituted from
        the typed ``reporter_id`` / ``reporter_team`` columns *only* when
        the state class has a ``reporter`` field — otherwise it's
        omitted so apps without a reporter concept (code-review) don't
        receive an unexpected attribute.
        """
        model_fields = self._state_cls.model_fields

        agents_run = [AgentRun.model_validate(a) for a in (row.agents_run or [])]
        tool_calls = [ToolCall.model_validate(t) for t in (row.tool_calls or [])]
        token_usage = TokenUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
        )

        kwargs: dict[str, object] = {
            "id": row.id,
            "status": row.status,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            "deleted_at": _iso(row.deleted_at) if row.deleted_at else None,
            "agents_run": agents_run,
            "tool_calls": tool_calls,
            "findings": dict(row.findings or {}),
            "token_usage": token_usage,
            "pending_intervention": row.pending_intervention,
            "user_inputs": list(row.user_inputs or []),
            "parent_session_id": row.parent_session_id,
            "dedup_rationale": row.dedup_rationale,
        }

        # Incident-shaped typed columns: include only fields the state
        # class actually declares.
        if "query" in model_fields:
            kwargs["query"] = row.query
        if "environment" in model_fields:
            kwargs["environment"] = row.environment
        if "reporter" in model_fields:
            kwargs["reporter"] = {"id": row.reporter_id, "team": row.reporter_team}
        if "summary" in model_fields:
            kwargs["summary"] = row.summary or ""
        if "tags" in model_fields:
            kwargs["tags"] = list(row.tags or [])
        if "severity" in model_fields:
            kwargs["severity"] = row.severity
        if "category" in model_fields:
            kwargs["category"] = row.category
        if "matched_prior_inc" in model_fields:
            kwargs["matched_prior_inc"] = row.matched_prior_inc
        if "resolution" in model_fields:
            kwargs["resolution"] = _deserialize_resolution(row.resolution)

        # Merge in any non-typed-column fields from ``extra_fields``.
        # Pydantic's ``extra='ignore'`` will drop any keys the state
        # class doesn't declare (e.g. legacy fields written by an older
        # binary), keeping the round-trip robust.
        extra = dict(row.extra_fields or {})

        # Route typed-column values into extra_fields when the state
        # class does NOT declare a typed Python field AND the column
        # actually has data. This lets a bare ``runtime.state.Session``
        # surface domain values (severity, environment, query…) via
        # ``state.extra_fields`` without the app needing a Session
        # subclass.
        typed_to_extra: dict[str, object] = {}
        if "query" not in model_fields and row.query:
            typed_to_extra["query"] = row.query
        if "environment" not in model_fields and row.environment:
            typed_to_extra["environment"] = row.environment
        if "reporter" not in model_fields and (row.reporter_id or row.reporter_team):
            typed_to_extra["reporter"] = {
                "id": row.reporter_id or "", "team": row.reporter_team or "",
            }
        if "summary" not in model_fields and row.summary:
            typed_to_extra["summary"] = row.summary
        if "severity" not in model_fields and row.severity:
            typed_to_extra["severity"] = row.severity
        if "category" not in model_fields and row.category:
            typed_to_extra["category"] = row.category
        if "matched_prior_inc" not in model_fields and row.matched_prior_inc:
            typed_to_extra["matched_prior_inc"] = row.matched_prior_inc
        if "tags" not in model_fields and row.tags:
            typed_to_extra["tags"] = list(row.tags)
        if "resolution" not in model_fields and row.resolution:
            typed_to_extra["resolution"] = _deserialize_resolution(row.resolution)

        # Fan extra_fields keys out as top-level kwargs for subclasses
        # that declare typed fields for them. Pydantic's ``extra=ignore``
        # silently drops keys the subclass doesn't declare — we re-stash
        # those into the bare ``extra_fields`` kwarg below so the data
        # survives.
        for k, v in extra.items():
            if k in self._STATE_TOP_LEVEL_FIELDS:
                continue  # handled above
            kwargs[k] = v

        # If the state class itself has an ``extra_fields`` field
        # (Session and any subclass that opts in), pass the row's
        # extra_fields content + typed-column-derived values through as
        # a single dict. Subclass-specific typed fields are handled by
        # the fan-out above; bare Session collects everything here.
        if "extra_fields" in model_fields:
            merged_extras: dict[str, object] = {}
            # 1. Typed-column values (when the subclass doesn't declare them)
            merged_extras.update(typed_to_extra)
            # 2. Row's extra_fields JSON column (subclass-specific
            # fields go to top-level kwargs above; whatever the subclass
            # doesn't declare lives only here)
            for k, v in extra.items():
                if k in self._STATE_TOP_LEVEL_FIELDS:
                    continue
                if k in model_fields:
                    # Subclass declared this as a typed field — it's
                    # already routed via top-level kwargs.
                    continue
                merged_extras[k] = v
            kwargs["extra_fields"] = merged_extras

        return self._state_cls(**kwargs)

    def _incident_to_row_dict(self, inc: StateT) -> dict:
        """Serialize a state instance into a row-shaped dict.

        Fields with a typed column on ``IncidentRow`` are written there;
        everything else (any field declared by the state class but not
        present on the row schema) lands in ``extra_fields`` JSON.
        """
        model_fields = type(inc).model_fields
        # Apps may pass either a Session subclass with the full
        # incident-shaped fields (round-trip identity) or a bare
        # Session whose app data lives in ``extra_fields``. Helper
        # ``_field`` reads from a typed attribute first, then falls back
        # to extra_fields[key] — so both subclass and bare-Session paths
        # round-trip cleanly through the typed columns.
        bare_extra = getattr(inc, "extra_fields", {}) or {}

        def _field(name: str, default=None):
            if name in model_fields:
                return getattr(inc, name, default)
            return bare_extra.get(name, default)

        reporter = _field("reporter", None)
        if isinstance(reporter, dict):
            reporter_id = reporter.get("id")
            reporter_team = reporter.get("team")
        else:
            reporter_id = getattr(reporter, "id", None) if reporter is not None else None
            reporter_team = getattr(reporter, "team", None) if reporter is not None else None
        resolution = _field("resolution", None)

        # Build ``extra_fields``: every state-class field that is *not*
        # a top-level Session field and *not* one of the incident-shaped
        # typed columns ends up here as JSON-safe dict.
        extra: dict[str, object] = {}
        for fname in model_fields:
            if fname in self._STATE_TOP_LEVEL_FIELDS:
                continue
            if fname in self._ROW_TYPED_DOMAIN_COLUMNS:
                continue
            if fname == "reporter":
                # Already projected onto reporter_id / reporter_team
                # typed columns above; do not also persist to extra.
                continue
            value = getattr(inc, fname, None)
            # Pydantic v2: prefer model_dump for nested BaseModels and
            # collections-of-BaseModels so the JSON column gets a
            # JSON-safe representation.
            if isinstance(value, BaseModel):
                extra[fname] = value.model_dump(mode="json")
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                extra[fname] = [v.model_dump(mode="json") for v in value]
            elif isinstance(value, dict):
                # Convert any embedded BaseModel values to JSON-safe form
                # via model_dump where appropriate; otherwise pass through.
                extra[fname] = {
                    k: (v.model_dump(mode="json") if isinstance(v, BaseModel) else v)
                    for k, v in value.items()
                }
            else:
                extra[fname] = value

        return {
            "id": inc.id,
            "status": inc.status,
            "created_at": _parse_iso(inc.created_at),
            "updated_at": _parse_iso(inc.updated_at),
            "deleted_at": _parse_iso(inc.deleted_at) if inc.deleted_at else None,
            "query": _field("query", "") or "",
            "environment": _field("environment", "") or "",
            "reporter_id": reporter_id or "",
            "reporter_team": reporter_team or "",
            "summary": _field("summary", "") or "",
            "severity": _field("severity", None),
            "category": _field("category", None),
            "matched_prior_inc": _field("matched_prior_inc", None),
            "resolution": (
                resolution if resolution is None or isinstance(resolution, str)
                else json.dumps(resolution)
            ),
            "tags": list(_field("tags", []) or []),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
            # Dedup linkage + rationale columns. ``getattr`` so bare
            # ``Session`` instances (without the dedup fields) round-
            # trip with NULL.
            "parent_session_id": getattr(inc, "parent_session_id", None),
            "dedup_rationale": getattr(inc, "dedup_rationale", None),
            # Everything not covered by a typed column. Subclass fields
            # come from the loop above; bare-Session callers stash app
            # data in ``state.extra_fields`` directly. Merge both, with
            # subclass fields taking precedence (parity with load path).
            "extra_fields": ({**bare_extra, **extra}) or None,
        }

# ====== module: runtime/mcp_servers/observability.py ======

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

# ====== module: runtime/mcp_servers/remediation.py ======

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
async def notify_oncall(incident_id: str, message: str,
                       team: str = "") -> dict:
    """Page the oncall engineer for the named team.

    ``team`` should be one of the framework's configured
    ``escalation_teams``. The result echoes ``team`` so callers and the
    UI can record which roster was paged.
    """
    return {
        "incident_id": incident_id,
        "team": team,
        "page_id": f"page-{abs(hash(incident_id + team)) % 10000:04d}",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }

# ====== module: runtime/mcp_servers/user_context.py ======

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

# ====== module: runtime/mcp_loader.py ======

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

# ====== module: runtime/graph.py ======

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
    session: Session
    next_route: str | None
    last_agent: str | None
    gated_target: str | None  # set by gate node; the downstream target if gate passes
    # Depth counter for supervisor recursion. The supervisor node bumps
    # it on entry and aborts at ``skill.max_dispatch_depth``. Carrying
    # it on graph state — rather than stashing it on the session —
    # keeps audit fields off the session and the depth check cheap
    # (no store reload).
    dispatch_depth: int | None
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

    def __init__(self, *, agent: str, incident: Session):
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


def _format_agent_input(incident: Session) -> str:
    """Build the human-message preamble each agent receives.

    Delegates to ``Session.to_agent_input`` so each app subclass owns the
    domain-shape of its prompt. The framework default surfaces only the
    session id + status; ``IncidentState`` and ``CodeReviewState``
    override with their respective shapes.
    """
    return incident.to_agent_input()


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
    incident: Session,
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


def _pair_tool_responses(messages: list, incident: Session) -> None:
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
    store: "SessionStore",
    fallback: "Session",
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
    return {"session": incident, "next_route": None,
            "last_agent": skill_name, "error": str(exc)}


def _record_success_run(
    *,
    incident: "Session",
    skill_name: str,
    started_at: str,
    final_text: str,
    usage: "TokenUsage",
    confidence: float | None,
    rationale: str | None,
    signal: str | None,
    store: "SessionStore",
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
    decide_route: Callable[[Session], str],
    store: SessionStore,
    valid_signals: frozenset[str] | None = None,
    gateway_cfg: GatewayConfig | None = None,
) -> Callable[[GraphState], Awaitable[dict]]:
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route.

    ``valid_signals`` is the orchestrator-wide accepted signal vocabulary
    (``cfg.orchestrator.signals``). When omitted, the legacy
    ``{success, failed, needs_input}`` default is used so older callers and
    tests keep working.

    ``gateway_cfg`` is the optional risk-rated tool gateway config.
    When supplied, every ``BaseTool`` in ``tools`` is wrapped via
    :func:`runtime.tools.gateway.wrap_tool` *inside the node body* so
    the closure captures the live ``Session`` per agent invocation.
    When ``None``, tools are passed through untouched.
    """

    async def node(state: GraphState) -> dict:
        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Wrap tools per-invocation so each wrap closes over the live
        # ``Session`` for this run. When the gateway is unconfigured,
        # the original tools pass through untouched and
        # ``create_react_agent`` sees the same surface as before.
        if gateway_cfg is not None:
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name)
                for t in tools
            ]
        else:
            run_tools = tools
        agent_executor = create_react_agent(
            llm, run_tools, prompt=skill.system_prompt,
        )

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
        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Session) -> str:
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


def _latest_run_for(incident: Session, agent_name: str | None):
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


def make_gate_node(
    *,
    cfg: AppConfig,
    store: SessionStore,
    threshold: float | None = None,
    teams: list[str] | None = None,
):
    """Build the intervention gate node placed before a gated downstream.

    The gate evaluates the confidence of whichever agent ran immediately
    before it (``state["last_agent"]``). If that confidence is below the
    configured threshold (or absent), the gate persists a
    ``pending_intervention`` payload on the Session row **and** raises
    ``langgraph.types.interrupt(payload)``. The checkpointer captures
    the suspended state; ``Orchestrator.resume_session`` resumes the
    graph via ``Command(resume=user_input)``. On resume the node body
    re-executes from the start, ``interrupt()`` returns the resume
    value (instead of raising), the user input is appended to
    ``session.user_inputs``, ``pending_intervention`` is cleared, and
    execution falls through to the gated downstream.

    The dual-write order is intentional: persist the row **before**
    calling ``interrupt()`` so the Streamlit UI (which polls
    ``Session.pending_intervention``) sees the pending state even on a
    cold restart. Reversing the order would leave the UI blind to a
    pending intervention captured only inside the LangGraph checkpoint.

    Implemented as a plain async coroutine (not via ``make_agent_node``)
    so it does not invoke an LLM — but it IS a real graph node, so
    streamed events surface ``enter gate`` / ``exit gate``.

    The gate is fully agent-agnostic: any YAML route carrying
    ``gate: confidence`` causes the gate to evaluate the upstream agent
    named on that route, regardless of agent identity.

    ``threshold`` and ``teams`` are the cross-cutting confidence cutoff
    and escalation roster the gate exposes to the operator on
    intervention. They are supplied by ``build_graph`` from the
    orchestrator's resolved :class:`FrameworkAppConfig` so the gate is
    completely free of any app-specific config import. When ``None``
    (legacy callers / unit tests that build the gate directly), the
    framework defaults are used (0.75 / empty roster).
    """
    if threshold is None:
        threshold = FrameworkAppConfig().confidence_threshold
    if teams is None:
        teams = []
    teams = list(teams)

    async def gate(state: GraphState) -> dict:
        # Imported lazily so unit tests that import graph.py without a
        # checkpointer don't pay the import cost. ``interrupt`` raises
        # ``GraphInterrupt`` on first execution and returns the resume
        # value on subsequent executions of the same node.
        from langgraph.types import interrupt

        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
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
            payload = {
                "reason": "low_confidence",
                "confidence": upstream_conf,
                "threshold": threshold,
                "upstream_agent": upstream,
                "summary": upstream_run.summary if upstream_run else "",
                "rationale": upstream_run.confidence_rationale if upstream_run else "",
                "options": ["resume_with_input", "escalate", "stop"],
                "escalation_teams": teams,
                "intended_target": intended_target,
            }
            incident.pending_intervention = payload
            # CRITICAL ORDERING: persist the Session row BEFORE calling
            # ``interrupt()``. ``interrupt()`` raises ``GraphInterrupt`` on
            # first execution; if we reversed the order the UI (which
            # polls Session.pending_intervention) would never see the
            # pending state. See plan R4 / "Streamlit hand-off".
            store.save(incident)
            # First execution: this raises GraphInterrupt and the
            # checkpointer captures the paused state.
            # Resume: this returns the value supplied via
            # ``Command(resume=...)``.
            decision = interrupt(payload)
            # Post-resume continuation. ``decision`` is whatever the
            # caller passed to ``Command(resume=...)`` — typically the
            # operator's free-text note (str) or a dict with an
            # ``input`` key. Empty/None decisions are treated as a
            # no-op (the gate still clears and falls through to the
            # gated target — the orchestrator wraps non-text actions
            # like ``stop``/``escalate`` outside the graph).
            user_text: str | None = None
            if isinstance(decision, str):
                stripped = decision.strip()
                if stripped:
                    user_text = stripped
            elif isinstance(decision, dict):
                raw = decision.get("input")
                if isinstance(raw, str) and raw.strip():
                    user_text = raw.strip()
            if user_text is not None:
                incident.user_inputs.append(user_text)
            incident.pending_intervention = None
            incident.status = "in_progress"
            store.save(incident)
            return {"session": incident, "next_route": "default",
                    "gated_target": intended_target, "last_agent": "gate", "error": None}
        # Confidence met threshold — clear any stale intervention payload.
        if incident.pending_intervention is not None:
            incident.pending_intervention = None
            store.save(incident)
        return {"session": incident, "next_route": "default",
                "gated_target": intended_target, "last_agent": "gate", "error": None}

    return gate


def _build_agent_nodes(*, cfg: AppConfig, skills: dict, store: SessionStore,
                       registry: ToolRegistry) -> dict:
    """Materialize agent nodes from skills + registry. Reused by main + resume graphs.

    Dispatches on ``skill.kind``:

    * ``responsive`` — builds a ReAct LLM node via :func:`make_agent_node`
      (today's path).
    * ``supervisor`` — builds a no-LLM router node via
      :func:`runtime.agents.supervisor.make_supervisor_node`. Supervisor
      skills with ``dispatch_strategy=llm`` get a small LLM bound; rule
      strategy uses ``None``.
    * ``monitor``   — **skipped**: monitor skills run out-of-band under
      :class:`runtime.agents.monitor.MonitorRunner`, not inside the
      session graph.
    """
    # Local import — agents package depends on this module's helpers.


    valid_signals = frozenset(cfg.orchestrator.signals)
    gateway_cfg = getattr(cfg.runtime, "gateway", None)
    nodes: dict = {}
    for agent_name, skill in skills.items():
        kind = getattr(skill, "kind", "responsive")
        if kind == "monitor":
            # Monitors are not graph nodes; skip silently.
            continue
        if kind == "supervisor":
            llm = None
            if skill.dispatch_strategy == "llm":
                llm = get_llm(cfg.llm, skill.model, role=agent_name)
            nodes[agent_name] = make_supervisor_node(skill=skill, llm=llm)
            continue
        # Default / "responsive" path.
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
            gateway_cfg=gateway_cfg,
        )
    return nodes


def _make_router(gated_edges: dict[tuple[str, str], str]):
    """Build a state router that intercepts gated edges into the gate node.

    Used by ``build_graph`` to route an agent's outbound edges through
    the gate when the corresponding ``RouteRule`` carries ``gate=...``.
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


async def build_graph(*, cfg: AppConfig, skills: dict, store: SessionStore,
                      registry: ToolRegistry,
                      checkpointer=None,
                      framework_cfg: FrameworkAppConfig | None = None):
    """Compile the main LangGraph from configured skills and routes.

    The entry agent is read from ``cfg.orchestrator.entry_agent``. Gate
    insertions are derived from each skill's route rules: a rule with
    ``gate: confidence`` causes the router to redirect ``(this_agent, next)``
    through the ``gate`` node.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.

    ``checkpointer`` is an optional :class:`BaseCheckpointSaver` that
    LangGraph uses for durable per-thread state. When ``None``, the
    graph compiles without one (back-compat for the few callers that
    build a graph outside the orchestrator, e.g. unit tests).

    ``framework_cfg`` carries the cross-cutting confidence/escalation
    knobs the gate node reads. When ``None`` (legacy / unit-test
    callers that compile a graph without an orchestrator), the
    runtime falls back to ``resolve_framework_app_config(None)`` which
    yields a bare ``FrameworkAppConfig()``.
    """
    entry = cfg.orchestrator.entry_agent
    if entry not in skills:
        raise ValueError(
            f"orchestrator.entry_agent={entry!r} is not a known skill "
            f"(known: {sorted(skills.keys())})"
        )
    if framework_cfg is None:
        # Prefer the YAML-driven ``AppConfig.framework`` field; fall
        # back to the legacy provider-callable path for backward
        # compatibility with deployments that still wire it.
        if getattr(cfg.runtime, "framework_app_config_path", None) is not None:
            framework_cfg = resolve_framework_app_config(
                cfg.runtime.framework_app_config_path,
            )
        else:
            framework_cfg = getattr(cfg, "framework", None) or resolve_framework_app_config(None)
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    sg.add_node("gate", make_gate_node(
        cfg=cfg, store=store,
        threshold=framework_cfg.confidence_threshold,
        teams=list(framework_cfg.escalation_teams),
    ))

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

    return sg.compile(checkpointer=checkpointer) if checkpointer is not None else sg.compile()

# ====== module: runtime/checkpointer_postgres.py ======

async def make_postgres_checkpointer(
    url: str,
) -> Tuple[BaseCheckpointSaver, Callable[[], Awaitable[None]]]:
    """Build a Postgres checkpointer + async cleanup callable.

    The orchestrator runs in async mode, so we use the async variant
    (:class:`AsyncPostgresSaver`) backed by an
    :class:`AsyncConnectionPool` rather than the sync ``PostgresSaver``
    -- LangGraph's async Pregel loop calls ``aget_tuple`` which the
    sync saver does not support.

    The pool is configured with ``autocommit=True`` because LangGraph
    issues each checkpoint write as a single statement and the
    enclosing transaction would otherwise hold the row lock until
    explicit commit.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    # Translate SQLAlchemy URL -> libpq connection string. SQLAlchemy
    # accepts ``postgresql+psycopg://...`` while psycopg's
    # AsyncConnectionPool wants the bare ``postgresql://...``. Strip
    # any dialect suffix on the scheme so both URL flavours work.
    if "+" in url.split("://", 1)[0]:
        _, rest = url.split("://", 1)
        url = f"postgresql://{rest}"

    pool = AsyncConnectionPool(
        conninfo=url,
        max_size=4,
        kwargs={"autocommit": True},
        # ``open=False`` defers the actual TCP connect to ``open()``;
        # we open immediately below so callers see real connection
        # errors at construction time, not on first request.
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type] — pool is the right shape per docs
    # Idempotent: creates the checkpoint tables if they don't yet exist.
    await saver.setup()
    return saver, pool.close

# ====== module: runtime/checkpointer.py ======

def _sqlite_path_from_url(url: str) -> str:
    """Extract the on-disk path from a sqlite SQLAlchemy URL.

    Accepts the SQLAlchemy ``sqlite:///<path>`` form (three slashes ->
    relative or absolute path begins after the third slash). The four-
    slash variant ``sqlite:////abs/path`` is also tolerated for callers
    who explicitly want an absolute path; sqlite3 itself accepts both.
    """
    parsed = urlparse(url)
    path = parsed.path
    if path.startswith("//"):
        # sqlite:////abs/path -> urlparse path "//abs/path" -> "/abs/path"
        return path[1:]
    # sqlite:///<rel> -> urlparse path "/<rel>". SQLAlchemy treats this
    # as a path *relative* to CWD; strip the leading slash so the helper
    # agrees (otherwise mkdir tries the filesystem root). Tests pass
    # absolute paths via tmp_path which composes to the four-slash form
    # above, so this branch only matters for the relative case.
    if path.startswith("/"):
        return path[1:]
    return path


async def make_checkpointer(
    cfg: AppConfig | MetadataConfig,
) -> Tuple[BaseCheckpointSaver, Callable[[], Awaitable[None]]]:
    """Build a checkpointer for the configured metadata DB.

    Accepts either a full :class:`AppConfig` (``cfg.storage.metadata`` is
    read) or a :class:`MetadataConfig` directly. The orchestrator uses
    the direct form because it post-processes the raw URL (resolving
    the default ``incidents/incidents.db`` sentinel against
    ``cfg.paths.incidents_dir``) and needs to pass *that* resolved URL
    through so per-test ``tmp_path`` isolation lands on the same DB
    file as the SQLAlchemy engine.

    Branches on the URL scheme:

    - ``sqlite:`` -> :class:`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
    - ``postgresql:`` / ``postgres:`` -> Postgres path

    Returns ``(saver, cleanup)``. ``cleanup`` is an async callable that
    closes the dedicated connection / pool; the caller is expected to
    await it at orchestrator shutdown.
    """
    if isinstance(cfg, MetadataConfig):
        url = cfg.url
    else:
        url = cfg.storage.metadata.url

    if url.startswith("sqlite:"):
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = _sqlite_path_from_url(url)
        # Ensure the parent directory exists. The orchestrator may build
        # the checkpointer before any storage write that would otherwise
        # have created it (e.g. on first start in a fresh deploy dir).
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Dedicated aiosqlite connection — separate from the SQLAlchemy
        # engine's connection pool. WAL mode lets readers and writers
        # proceed concurrently on the same file.
        #
        # ``isolation_level=None`` puts the underlying ``sqlite3`` driver
        # in autocommit mode so Python's stdlib doesn't sneak a
        # ``BEGIN DEFERRED`` in front of our INSERTs. AsyncSqliteSaver's
        # ``aput`` then runs as an implicit single-statement write
        # followed by ``commit()``; SQLite handles the transaction
        # internally as ``BEGIN IMMEDIATE`` for any DML, so the saver
        # plays nicely with the SQLAlchemy engine's explicit
        # ``BEGIN IMMEDIATE`` (see ``runtime.storage.engine``).
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        # Set WAL + relaxed durability up front so the orchestrator's
        # other path (the SQLAlchemy engine) sees them; AsyncSqliteSaver
        # itself also enables WAL during setup() but doing it here makes
        # the pragma observable to verification tests immediately.
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        # SQLAlchemy and the saver share the same DB file via separate
        # connections. Without a generous busy_timeout, a contended
        # writer races straight into ``database is locked``. 30s matches
        # the engine-side setting (see ``runtime.storage.engine``) so
        # both writers wait the same amount before failing.
        await conn.execute("PRAGMA busy_timeout=30000")
        saver = AsyncSqliteSaver(conn)
        # Create checkpoint tables on first use. Idempotent.
        await saver.setup()
        return saver, conn.close

    if url.startswith("postgresql:") or url.startswith("postgres:"):
        # Imported lazily so SQLite-only deploys don't need psycopg_pool
        # installed.


        return await make_postgres_checkpointer(url)

    raise ValueError(
        f"unsupported checkpointer URL scheme {url!r} — "
        "expected sqlite or postgresql"
    )

# ====== module: runtime/triggers/base.py ======

if TYPE_CHECKING:
    pass



@dataclass(frozen=True)
class TriggerInfo:
    """Provenance attached to every session started via a trigger.

    Stamped onto ``Orchestrator.start_session(trigger=...)`` by every
    transport. The framework does not branch on the contents; the field
    exists so dashboards / audit logs can answer "where did this session
    come from?" without re-deriving from disjoint sources.
    """

    name: str          # the trigger name from config (``triggers[].name``)
    transport: str     # ``api`` / ``webhook`` / ``schedule`` / plugin kind
    target_app: str    # ``triggers[].target_app``
    received_at: datetime


class TriggerTransport(ABC):
    """Lifecycle interface for a transport (api / webhook / schedule / plugin).

    The registry calls ``start(registry)`` on lifespan-enter and ``stop()``
    on lifespan-exit. Transports own their own state (router, scheduler,
    background tasks) and must be safe to construct *before* ``start`` —
    the FastAPI app collects routers from webhook transports during
    ``build_app`` and mounts them once during the lifespan handshake.
    """

    @abstractmethod
    async def start(self, registry: "TriggerRegistry") -> None:
        """Begin accepting inbound traffic. Must be idempotent."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop accepting traffic and release resources. Must be idempotent."""

# ====== module: runtime/triggers/config.py ======

_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)+(:[A-Za-z_][\w]*)?$")

# 5-field cron, optionally allowing ``*/N`` step / ``A,B,C`` list / ``A-B``
# range tokens. Validation here is intentionally loose — APScheduler's
# ``CronTrigger.from_crontab`` is the source of truth and will reject
# semantically bad strings at scheduler-start time.
_CRON_5FIELD_RE = re.compile(
    r"^\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*$"
)


class _BaseTriggerConfig(BaseModel):
    """Shared fields for every trigger transport variant."""

    name: str = Field(..., min_length=1, max_length=128)
    target_app: str = Field(..., min_length=1)
    target_agent: str | None = None
    transform: str | None = None  # dotted path; required for webhook/schedule

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        # Webhook URLs use the name as a path segment — restrict to a safe
        # alphabet so we never have to URL-encode.
        if not re.match(r"^[A-Za-z0-9_\-]+$", v):
            raise ValueError(
                f"trigger name must match [A-Za-z0-9_-]+, got {v!r}"
            )
        return v

    @field_validator("transform")
    @classmethod
    def _validate_transform(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(f"transform must be dotted path, got {v!r}")
        return v


class APITriggerConfig(_BaseTriggerConfig):
    """Built-in HTTP route — preserves ``POST /investigate`` semantics.

    The api transport is implicitly registered for back-compat; explicitly
    listing it in ``triggers:`` is supported for symmetry but not required.
    """

    transport: Literal["api"] = "api"


class WebhookTriggerConfig(_BaseTriggerConfig):
    """Webhook trigger — third-party ``POST /triggers/{name}``.

    ``payload_schema`` is a dotted path to a Pydantic ``BaseModel``; the
    request JSON is validated against it, then ``transform(payload)`` is
    invoked to produce the keyword args for ``Orchestrator.start_session``.

    ``auth_token_env`` is the name of an environment variable holding the
    bearer token; the gateway never reads raw secrets from YAML.
    """

    transport: Literal["webhook"] = "webhook"
    payload_schema: str
    auth: Literal["bearer", "none"] = "bearer"
    auth_token_env: str | None = None
    idempotency_ttl_hours: int = Field(24, ge=1, le=24 * 30)

    @field_validator("payload_schema")
    @classmethod
    def _validate_payload_schema(cls, v: str) -> str:
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(f"payload_schema must be dotted path, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_bearer(self) -> "WebhookTriggerConfig":
        if self.auth == "bearer" and not self.auth_token_env:
            raise ValueError("auth: bearer requires auth_token_env")
        if self.transform is None:
            raise ValueError("webhook trigger requires transform: <dotted.path>")
        return self


class ScheduleTriggerConfig(_BaseTriggerConfig):
    """In-process APScheduler cron job.

    ``schedule`` is a 5-field standard cron string interpreted via
    ``CronTrigger.from_crontab``. APScheduler's native 6-field form is
    rejected here — the registry owns the cron flavour.

    ``payload`` is a static dict passed to ``transform`` on each fire;
    runtime cron jobs have no inbound payload to validate.
    """

    transport: Literal["schedule"] = "schedule"
    schedule: str
    timezone: str = "UTC"
    payload: dict = Field(default_factory=dict)

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, v: str) -> str:
        if not _CRON_5FIELD_RE.match(v):
            raise ValueError(
                f"schedule must be a 5-field cron string, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _check_transform(self) -> "ScheduleTriggerConfig":
        if self.transform is None:
            raise ValueError("schedule trigger requires transform: <dotted.path>")
        return self


class PluginTriggerConfig(_BaseTriggerConfig):
    """Plugin-defined transport, addressed by ``kind``.

    Resolution: the registry merges entry-points (group ``runtime.triggers``)
    with explicit ``plugin_transports`` passed to ``TriggerRegistry.create``;
    explicit entries win. Bad ``kind`` raises at registry init.
    """

    transport: Literal["plugin"] = "plugin"
    kind: str = Field(..., min_length=1)
    options: dict = Field(default_factory=dict)


# Discriminated union — Pydantic uses the ``transport`` literal to pick
# the right shape during validation. Field default makes the runtime
# triggers list ``list[TriggerConfig]`` resolve cleanly.
TriggerConfig = Annotated[
    Union[
        APITriggerConfig,
        WebhookTriggerConfig,
        ScheduleTriggerConfig,
        PluginTriggerConfig,
    ],
    Field(discriminator="transport"),
]

# ====== module: runtime/triggers/resolve.py ======

def _resolve_dotted(path: str) -> Any:
    """Import a module and return the named attribute.

    Accepts both ``a.b.c`` (last segment is the attribute) and
    ``a.b:c`` (colon-delimited per entry-point convention).
    """
    if ":" in path:
        module_path, attr = path.split(":", 1)
    else:
        module_path, _, attr = path.rpartition(".")
        if not module_path:
            raise ImportError(f"dotted path missing module: {path!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"cannot import module for {path!r}: {e}") from e
    if not hasattr(module, attr):
        raise ImportError(
            f"module {module_path!r} has no attribute {attr!r} "
            f"(resolving {path!r})"
        )
    return getattr(module, attr)


def resolve_payload_schema(path: str) -> Type[BaseModel]:
    """Resolve a dotted path to a Pydantic ``BaseModel`` subclass.

    Raises ``TypeError`` if the resolved object isn't a ``BaseModel``
    subclass — keeps the failure on the operator console at startup.
    """
    obj = _resolve_dotted(path)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(
            f"payload_schema {path!r} did not resolve to a Pydantic "
            f"BaseModel subclass; got {obj!r}"
        )
    return obj


def resolve_transform(path: str) -> Callable[..., dict]:
    """Resolve a dotted path to a callable.

    The callable is expected to return a ``dict`` of keyword arguments
    suitable for ``Orchestrator.start_session(**kwargs)``. The framework
    does not enforce a stricter signature — apps own the contract with
    their own transform.
    """
    obj = _resolve_dotted(path)
    if not callable(obj):
        raise TypeError(
            f"transform {path!r} did not resolve to a callable; got {obj!r}"
        )
    return obj

# ====== module: runtime/triggers/idempotency.py ======

_LRU_MAX_PER_TRIGGER = 1024


class IdempotencyRow(Base):
    """SQLite-backed dedup record. One row per (trigger_name, key)."""

    __tablename__ = "trigger_idempotency_keys"

    trigger_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IdempotencyStore:
    """Thread-safe LRU with SQLite write-through.

    The LRU bounds memory; SQLite handles cold-restart survival and
    cross-process sharing (e.g. multi-worker uvicorn). Mutations are
    serialised via a single threading lock — the critical section is
    a few dict ops plus one SQL round-trip, which dwarfs lock overhead.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # Ensure the table exists even if the orchestrator hasn't run
        # ``Base.metadata.create_all`` yet (early lifespan path).
        Base.metadata.create_all(engine, tables=[IdempotencyRow.__table__])
        self._lru: dict[str, OrderedDict[str, str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls, db_url: str) -> "IdempotencyStore":
        """Build a store backed by a fresh SQLAlchemy engine.

        Convenience for tests + standalone tooling. Production paths
        should reuse the orchestrator's engine via the constructor so a
        single SQLite file is opened once.
        """
        engine = create_engine(db_url, poolclass=NullPool, future=True)
        return cls(engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, trigger_name: str, key: str) -> str | None:
        """Return the cached ``session_id`` for the key, or ``None``.

        LRU first, SQLite second; a SQLite hit refills the LRU so
        subsequent reads stay in memory.
        """
        with self._lock:
            cache = self._lru.get(trigger_name)
            if cache is not None and key in cache:
                # Bump recency
                cache.move_to_end(key)
                return cache[key]
        # SQLite fall-through (outside the threading lock — sqlite3 has
        # its own locking, and this path is rare).
        with SqlaSession(self._engine) as s:
            row = s.execute(
                select(IdempotencyRow).where(
                    IdempotencyRow.trigger_name == trigger_name,
                    IdempotencyRow.key == key,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                # SQLite drops tz on round-trip; assume UTC (we always
                # write UTC).
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= _utc_now():
                # Stale row — opportunistic delete.
                s.execute(
                    delete(IdempotencyRow).where(
                        IdempotencyRow.trigger_name == trigger_name,
                        IdempotencyRow.key == key,
                    )
                )
                s.commit()
                return None
            session_id = row.session_id
        # Refill LRU.
        with self._lock:
            cache = self._lru.setdefault(trigger_name, OrderedDict())
            cache[key] = session_id
            cache.move_to_end(key)
            self._evict_if_needed(cache)
        return session_id

    def put(
        self,
        trigger_name: str,
        key: str,
        session_id: str,
        *,
        ttl_hours: int = 24,
    ) -> None:
        """Cache a fresh (key -> session_id) binding.

        Writes through to SQLite with ``expires_at = now + ttl``. The
        LRU receives the same record. Calling ``put`` for an existing
        key overwrites both layers.
        """
        now = _utc_now()
        expires_at = now + timedelta(hours=ttl_hours)
        with SqlaSession(self._engine) as s:
            existing = s.get(IdempotencyRow, (trigger_name, key))
            if existing is None:
                s.add(IdempotencyRow(
                    trigger_name=trigger_name,
                    key=key,
                    session_id=session_id,
                    created_at=now,
                    expires_at=expires_at,
                ))
            else:
                existing.session_id = session_id
                existing.created_at = now
                existing.expires_at = expires_at
            s.commit()
        with self._lock:
            cache = self._lru.setdefault(trigger_name, OrderedDict())
            cache[key] = session_id
            cache.move_to_end(key)
            self._evict_if_needed(cache)
        # Opportunistic purge of expired rows so a long-running process
        # doesn't accumulate dead records. Cheap (range-bounded delete).
        self.purge_expired()

    def purge_expired(self) -> int:
        """Delete all rows whose ``expires_at`` is in the past. Returns
        the number of rows removed."""
        with SqlaSession(self._engine) as s:
            result = s.execute(
                delete(IdempotencyRow).where(
                    IdempotencyRow.expires_at <= _utc_now()
                )
            )
            s.commit()
            return result.rowcount or 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _evict_if_needed(cache: "OrderedDict[str, str]") -> None:
        while len(cache) > _LRU_MAX_PER_TRIGGER:
            cache.popitem(last=False)

# ====== module: runtime/triggers/auth.py ======

def make_bearer_dep(token_env: str) -> Callable:
    """Return a FastAPI dependency that asserts the inbound bearer token
    matches ``$token_env``.

    Snapshots the env var at construction time and raises ``RuntimeError``
    if it isn't set — callers can't accidentally start a webhook without
    a configured secret.
    """
    expected = os.environ.get(token_env)
    if not expected:
        raise RuntimeError(
            f"env var {token_env!r} for webhook auth is not set"
        )

    async def _bearer_dep(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer",
            )
        token = authorization[len("Bearer "):].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer",
            )

    return _bearer_dep

# ====== module: runtime/triggers/transports/api.py ======

class APITransport(TriggerTransport):
    """No-op transport that holds the ``api`` configs.

    Future: when the legacy ``POST /investigate`` route is removed, this
    class can mount its own router. Today it exists as a marker.
    """

    def __init__(self, configs: list[APITriggerConfig]) -> None:
        self._configs = list(configs)

    @property
    def configs(self) -> list[APITriggerConfig]:
        return list(self._configs)

    async def start(self, registry) -> None:  # noqa: D401, ARG002
        return None

    async def stop(self) -> None:
        return None

# ====== module: runtime/triggers/transports/webhook.py ======

if TYPE_CHECKING:
    pass



_log = logging.getLogger(__name__)


class WebhookTransport(TriggerTransport):
    """FastAPI router exposing one ``POST /triggers/{name}`` per webhook."""

    def __init__(
        self,
        configs: list[WebhookTriggerConfig],
        specs: "dict[str, TriggerSpec]",
        idempotency: "IdempotencyStore | None",
    ) -> None:
        self._configs = {c.name: c for c in configs}
        self._specs = specs
        self._idempotency = idempotency
        self.router = APIRouter()
        self._registry: "TriggerRegistry | None" = None
        self._mounted = False

    async def start(self, registry: "TriggerRegistry") -> None:
        if self._mounted:
            return
        self._registry = registry
        for name, cfg in self._configs.items():
            self.router.add_api_route(
                f"/triggers/{name}",
                self._make_handler(name),
                methods=["POST"],
                dependencies=self._auth_deps(cfg),
                status_code=status.HTTP_202_ACCEPTED,
                name=f"trigger:{name}",
            )
        self._mounted = True

    async def stop(self) -> None:
        # The router lives on the FastAPI app for the process lifetime;
        # there's nothing to tear down (FastAPI cleans up its own state
        # when the app object is GC'd). Mark unmounted so a subsequent
        # ``start`` is a no-op only if we were already started.
        self._registry = None
        self._mounted = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _auth_deps(cfg: WebhookTriggerConfig) -> list[Any]:
        if cfg.auth == "none":
            return []
        # cfg.auth == "bearer": the model validator already enforced
        # auth_token_env presence at config load.
        assert cfg.auth_token_env is not None
        return [Depends(make_bearer_dep(cfg.auth_token_env))]

    def _make_handler(self, name: str) -> Callable:
        spec = self._specs[name]
        schema = spec.payload_schema
        assert schema is not None  # webhook configs always carry a schema

        async def handler(
            request: Request,
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        ) -> dict:
            registry = self._registry
            if registry is None:
                raise HTTPException(
                    status_code=503,
                    detail="trigger registry not started",
                )
            # Parse body. We read raw JSON then run through the schema
            # so we get a uniform 422 surface (FastAPI body-binding 422
            # has a different shape).
            try:
                raw = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid json body: {exc}",
                ) from exc
            try:
                payload = schema.model_validate(raw)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            # Dispatch. Translate transform/start_session errors to 422
            # (per R3: log + 422, no retry, no idempotency cache).
            try:
                session_id = await registry.dispatch(
                    name, payload, idempotency_key=idempotency_key
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (ValueError, TypeError, ValidationError) as exc:
                _log.warning(
                    "trigger %r transform/dispatch failed: %s", name, exc
                )
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"session_id": session_id}

        return handler

# ====== module: runtime/triggers/transports/schedule.py ======

if TYPE_CHECKING:
    pass


_log = logging.getLogger(__name__)


class ScheduleTransport(TriggerTransport):
    """In-process APScheduler driving cron-firing triggers."""

    def __init__(self, configs: list[ScheduleTriggerConfig]) -> None:
        self._configs = list(configs)
        self._scheduler: AsyncIOScheduler | None = None
        self._registry: "TriggerRegistry | None" = None

    @property
    def scheduler(self) -> AsyncIOScheduler | None:
        return self._scheduler

    async def start(self, registry: "TriggerRegistry") -> None:
        if self._scheduler is not None:
            return
        self._registry = registry
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        for cfg in self._configs:
            cron = CronTrigger.from_crontab(cfg.schedule, timezone=cfg.timezone)
            self._scheduler.add_job(
                self._fire,
                trigger=cron,
                kwargs={"name": cfg.name, "payload": dict(cfg.payload)},
                id=f"trigger:{cfg.name}",
                replace_existing=True,
            )
        self._scheduler.start()

    async def stop(self) -> None:
        if self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as exc:  # noqa: BLE001
            _log.warning("apscheduler shutdown raised: %s", exc)
        self._scheduler = None
        self._registry = None

    async def _fire(self, *, name: str, payload: dict) -> None:
        """APScheduler job target. Logs and swallows exceptions so a bad
        cron job doesn't poison the scheduler thread."""
        registry = self._registry
        if registry is None:
            _log.warning(
                "schedule trigger %r fired with no registry attached", name
            )
            return
        try:
            await registry.dispatch(name, payload)
        except Exception as exc:  # noqa: BLE001
            _log.exception(
                "schedule trigger %r dispatch failed: %s", name, exc
            )

# ====== module: runtime/triggers/transports/plugin.py ======

__all__ = ["TriggerTransport"]

# ====== module: runtime/triggers/registry.py ======

_log = logging.getLogger(__name__)

# Type aliases for clarity — ``StartSessionFn`` is the closure the
# registry calls; the FastAPI lifespan binds it to the orchestrator
# service so we don't take a hard import dependency on the Orchestrator
# class here (avoids a circular import).
StartSessionFn = Callable[..., Awaitable[str]]


class TriggerSpec:
    """Resolved (live) form of one ``TriggerConfig``.

    Built at registry init: dotted paths bound to live callables /
    classes; the original config is retained for transports to read.
    """

    __slots__ = (
        "config",
        "payload_schema",
        "transform",
    )

    def __init__(
        self,
        config: TriggerConfig,
        payload_schema: Type[BaseModel] | None,
        transform: Callable[..., dict] | None,
    ) -> None:
        self.config = config
        self.payload_schema = payload_schema
        self.transform = transform

    @property
    def name(self) -> str:
        return self.config.name


class TriggerRegistry:
    """Owns trigger lifecycle + dispatch.

    Construct via :meth:`create` so dotted-path resolution and transport
    instantiation happen together. Direct ``__init__`` use is reserved
    for unit tests that pre-build their own spec list.
    """

    def __init__(
        self,
        specs: dict[str, TriggerSpec],
        transports: list[TriggerTransport],
        start_session_fn: StartSessionFn,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        self._specs: dict[str, TriggerSpec] = specs
        self._transports: list[TriggerTransport] = transports
        self._start_session_fn = start_session_fn
        self._idempotency = idempotency
        self._started = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        configs: list[TriggerConfig],
        *,
        start_session_fn: StartSessionFn,
        idempotency: IdempotencyStore | None = None,
        plugin_transports: dict[str, Type[TriggerTransport]] | None = None,
    ) -> "TriggerRegistry":
        """Resolve dotted paths + instantiate transports.

        Raises ``ImportError`` / ``TypeError`` at startup for any bad
        dotted path — fail-fast, never at request time.
        """
        # 1. Resolve specs (dotted paths -> live objects).
        specs: dict[str, TriggerSpec] = {}
        for cfg in configs:
            schema: Type[BaseModel] | None = None
            transform_fn: Callable[..., dict] | None = None
            if isinstance(cfg, WebhookTriggerConfig):
                schema = resolve_payload_schema(cfg.payload_schema)
            if cfg.transform is not None:
                transform_fn = resolve_transform(cfg.transform)
            specs[cfg.name] = TriggerSpec(
                config=cfg, payload_schema=schema, transform=transform_fn
            )

        # 2. Resolve plugin kinds (entry-points + explicit, explicit wins).
        plugin_kinds: dict[str, Type[TriggerTransport]] = (
            cls._load_entry_point_transports()
        )
        if plugin_transports:
            plugin_kinds.update(plugin_transports)

        # 3. Bucket configs by transport flavour.
        api_cfgs: list[APITriggerConfig] = []
        webhook_cfgs: list[WebhookTriggerConfig] = []
        schedule_cfgs: list[ScheduleTriggerConfig] = []
        plugin_cfgs: list[PluginTriggerConfig] = []
        for cfg in configs:
            if isinstance(cfg, APITriggerConfig):
                api_cfgs.append(cfg)
            elif isinstance(cfg, WebhookTriggerConfig):
                webhook_cfgs.append(cfg)
            elif isinstance(cfg, ScheduleTriggerConfig):
                schedule_cfgs.append(cfg)
            elif isinstance(cfg, PluginTriggerConfig):
                plugin_cfgs.append(cfg)

        # 4. Instantiate transports. Lazy import to break import cycles.




        transports: list[TriggerTransport] = []
        if api_cfgs:
            transports.append(APITransport(api_cfgs))
        if webhook_cfgs:
            transports.append(WebhookTransport(webhook_cfgs, specs, idempotency))
        if schedule_cfgs:
            transports.append(ScheduleTransport(schedule_cfgs))
        for pcfg in plugin_cfgs:
            kind_cls = plugin_kinds.get(pcfg.kind)
            if kind_cls is None:
                raise ImportError(
                    f"plugin trigger {pcfg.name!r} requested kind={pcfg.kind!r} "
                    f"but no transport with that kind is registered "
                    f"(known: {sorted(plugin_kinds)})"
                )
            transports.append(kind_cls(pcfg))

        return cls(specs, transports, start_session_fn, idempotency)

    @staticmethod
    def _load_entry_point_transports() -> dict[str, Type[TriggerTransport]]:
        """Discover plugin transports via the ``runtime.triggers`` group.

        Defensive: a missing or malformed entry-point is logged and
        skipped rather than failing registry init. Apps that need strict
        binding pass ``plugin_transports`` explicitly.
        """
        out: dict[str, Type[TriggerTransport]] = {}
        try:
            eps = importlib.metadata.entry_points(group="runtime.triggers")
        except Exception:  # noqa: BLE001
            return out
        for ep in eps:
            try:
                obj = ep.load()
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "trigger entry-point %r failed to load: %s", ep.name, exc
                )
                continue
            if not (isinstance(obj, type) and issubclass(obj, TriggerTransport)):
                _log.warning(
                    "trigger entry-point %r did not resolve to a "
                    "TriggerTransport subclass; got %r",
                    ep.name,
                    obj,
                )
                continue
            out[ep.name] = obj
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def transports(self) -> list[TriggerTransport]:
        return list(self._transports)

    @property
    def specs(self) -> dict[str, TriggerSpec]:
        return dict(self._specs)

    @property
    def idempotency(self) -> IdempotencyStore | None:
        return self._idempotency

    async def start_all(self) -> None:
        """Start every transport. Idempotent."""
        if self._started:
            return
        for t in self._transports:
            await t.start(self)
        self._started = True

    async def stop_all(self) -> None:
        """Stop every transport. Idempotent."""
        if not self._started:
            return
        for t in self._transports:
            try:
                await t.stop()
            except Exception as exc:  # noqa: BLE001
                # Best-effort: one misbehaving transport mustn't block
                # the rest from cleaning up.
                _log.warning("trigger transport %r stop() failed: %s", t, exc)
        self._started = False

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        payload: Any,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Run ``transform(payload)`` and call ``start_session_fn``.

        Returns the session id. If ``idempotency_key`` is provided, the
        cached session id is returned on hit; on miss the call proceeds
        and the (key, session_id) mapping is recorded for the trigger's
        configured TTL.

        Raises ``KeyError`` for an unknown trigger name. Surfaces any
        ``ValueError`` / ``ValidationError`` from ``transform`` to the
        caller — transports translate to HTTP status codes (typically
        ``422 Unprocessable Entity``).
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unknown trigger: {name!r}")

        # Idempotency hit: return cached session id without invoking
        # transform / orchestrator. Per R3 in the plan, transform errors
        # are NOT cached — only successful dispatches.
        if idempotency_key and self._idempotency is not None:
            cached = self._idempotency.get(name, idempotency_key)
            if cached is not None:
                return cached

        # Resolve trigger payload -> start_session kwargs.
        if spec.transform is not None:
            kwargs = spec.transform(payload)
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"transform for trigger {name!r} returned "
                    f"{type(kwargs).__name__}, expected dict"
                )
        else:
            # api transport: payload is already the kwargs dict.
            kwargs = dict(payload) if payload else {}

        info = TriggerInfo(
            name=name,
            transport=spec.config.transport,
            target_app=spec.config.target_app,
            received_at=datetime.now(timezone.utc),
        )

        session_id = await self._start_session_fn(trigger=info, **kwargs)

        # Record successful dispatch for idempotency.
        if idempotency_key and self._idempotency is not None:
            ttl = (
                spec.config.idempotency_ttl_hours
                if isinstance(spec.config, WebhookTriggerConfig)
                else 24
            )
            self._idempotency.put(name, idempotency_key, session_id, ttl_hours=ttl)

        return session_id

# ====== module: runtime/dedup.py ======

if TYPE_CHECKING:  # pragma: no cover — import-only types
    from langchain_core.language_models.chat_models import BaseChatModel



logger = logging.getLogger(__name__)

# Framework-level state type. Permissive at ``BaseModel`` so the dedup
# layer never depends on app-level subclasses (R4 enforcement).
StateT = TypeVar("StateT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DedupScope(BaseModel):
    """Filter knobs that narrow the Stage 1 candidate pool."""

    same_environment: bool = True
    only_closed: bool = True


class DedupConfig(BaseModel):
    """Configuration for the two-stage dedup pipeline.

    All numeric thresholds are inclusive at the lower bound (``>=``),
    so a candidate hitting exactly ``stage1_threshold`` is considered.

    Defaults are tuned for the incident-management example. Apps that
    want different policies override via YAML.
    """

    enabled: bool = False
    stage1_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    stage1_top_k: int = Field(default=5, ge=1, le=20)
    stage2_top_k: int = Field(default=3, ge=1, le=20)
    stage2_model: str = "cheap"
    stage2_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Optional override of the Stage-2 system prompt. When ``None`` the
    # pipeline falls back to ``framework_cfg.dedup_system_prompt`` so
    # apps can either tune the prompt per app (FrameworkAppConfig) or
    # override it inline on the DedupConfig.
    system_prompt: str | None = None
    # Reserved for future modes; only ``post_intake`` is implemented.
    run_at: Literal["post_intake"] = "post_intake"
    scope: DedupScope = Field(default_factory=DedupScope)

    @model_validator(mode="after")
    def _validate_top_k(self) -> "DedupConfig":
        if self.stage2_top_k > self.stage1_top_k:
            raise ValueError(
                f"dedup.stage2_top_k ({self.stage2_top_k}) must be "
                f"<= dedup.stage1_top_k ({self.stage1_top_k})"
            )
        return self

    def assert_model_exists(self, llm_cfg: "LLMConfig") -> None:
        """Fail fast if ``stage2_model`` is missing from the LLM registry.

        Called at orchestrator boot when dedup is enabled. Raising here
        is preferred over discovering the typo on the first incident.
        """
        if self.stage2_model not in llm_cfg.models:
            raise ValueError(
                f"dedup.stage2_model={self.stage2_model!r} not found in "
                f"llm.models (known: {sorted(llm_cfg.models)})"
            )


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


class DedupDecision(BaseModel):
    """Pydantic schema for Stage 2 LLM structured output."""

    is_duplicate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


class DedupResult(BaseModel):
    """Outcome of one ``DedupPipeline.run`` call.

    ``matched=True`` iff Stage 2 confirmed a duplicate. The remaining
    fields are always populated when matched and may be partially
    populated for diagnostics when Stage 2 ran but declined.

    ``parse_failures`` counts the number of Stage 2 LLM responses that
    failed to parse into a :class:`DedupDecision` during this run. Any
    non-zero value is an operator signal that the Stage 2 model is
    drifting off-schema and dedup may be silently false-negative.
    """

    matched: bool = False
    parent_session_id: str | None = None
    candidate_id: str | None = None
    decision: DedupDecision | None = None
    stage1_score: float | None = None
    parse_failures: int = 0


# Internal tagged outcome for Stage 2 parse — distinguishes a legitimate
# "model said not-a-duplicate" from "model returned garbage we couldn't
# parse" so the pipeline can count parse failures separately.
class _Stage2Outcome(enum.Enum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    PARSE_FAILED = "parse_failed"


# ---------------------------------------------------------------------------
# Stage 2 prompt
# ---------------------------------------------------------------------------


# Legacy default — kept for back-compat with callers that referenced
# this module-level constant directly. New code should read the prompt
# off ``DedupConfig.system_prompt`` (when set) or the
# ``FrameworkAppConfig.dedup_system_prompt`` the pipeline holds.
_STAGE2_SYSTEM = (
    "You are deduplicating incident reports for an SRE platform. "
    "Two reports are duplicates only if they describe the same root cause "
    "AND the same service/environment AND overlap in time-of-occurrence. "
    "Surface-level keyword overlap is NOT enough. "
    "Respond with a single JSON object matching this schema: "
    '{"is_duplicate": bool, "confidence": float in [0,1], "rationale": '
    '"1-2 sentences"}.'
)


def _build_stage2_user_prompt(*, prior_text: str, new_text: str,
                              prior_id: str, new_id: str) -> str:
    """Assemble the user-side prompt for one Stage 2 comparison."""
    return (
        f"[INCIDENT A — existing, id={prior_id}]\n"
        f"{prior_text}\n\n"
        f"[INCIDENT B — new, id={new_id}]\n"
        f"{new_text}\n\n"
        "Decide: is B a duplicate of A?"
    )


def _parse_decision_tagged(
    raw: str, *, model_name: str = "<unknown>",
) -> tuple[DedupDecision | None, Exception | None]:
    """Parse the LLM's text into a ``DedupDecision`` with a failure tag.

    Returns ``(decision, None)`` on success, ``(None, exc)`` on any
    parse / validation failure ("treat as not-duplicate, do not retry —
    budget protection"). Empty input is also a parse failure so the
    pipeline can surface model-stopped-responding as a signal.

    Emits a structured ``warning`` log on any failure with fields a
    log aggregator can pick up via the LogRecord ``extra`` namespace:
    ``event``, ``error_type``, ``error_msg``, ``model``,
    ``raw_output_excerpt``.
    """
    text = (raw or "").strip()
    if not text:
        exc = ValueError("empty Stage 2 LLM output")
        _log_parse_failure(exc, model_name=model_name, raw=raw or "")
        return None, exc
    # Tolerate ```json ... ``` fences from chatty models.
    if text.startswith("```"):
        # Strip the first fence line and a trailing fence if present.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        _log_parse_failure(exc, model_name=model_name, raw=raw)
        return None, exc
    try:
        return DedupDecision.model_validate(payload), None
    except Exception as exc:  # noqa: BLE001 — pydantic ValidationError + fallback
        _log_parse_failure(exc, model_name=model_name, raw=raw)
        return None, exc


def _log_parse_failure(exc: Exception, *, model_name: str, raw: str) -> None:
    """Emit a structured warning for a Stage 2 parse failure.

    Fields land on the LogRecord via ``extra`` so structured log
    aggregators (Loki, Datadog, etc.) can index them, while the
    human-readable message stays useful for grep.
    """
    excerpt = (raw or "")[:200]
    logger.warning(
        "dedup stage 2 parse failure: %s (%s)",
        exc, type(exc).__name__,
        extra={
            "event": "dedup_parse_failure",
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:200],
            "model": model_name,
            "raw_output_excerpt": excerpt,
        },
    )


def _parse_decision(raw: str) -> DedupDecision | None:
    """Backward-compatible wrapper around :func:`_parse_decision_tagged`.

    Existing callers / tests that only care about the decision keep
    working; the pipeline uses the tagged variant directly so it can
    distinguish parse failures from legitimate non-matches.
    """
    decision, _err = _parse_decision_tagged(raw)
    return decision


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DedupPipeline(Generic[StateT]):
    """Stage 1 (embedding) + Stage 2 (LLM) dedup orchestrator.

    Construction is cheap; ``run`` is the per-session entry point. The
    pipeline is stateless across runs.

    ``text_extractor`` returns the comparison text for a given session
    (the framework can't know which fields the app considers
    semantically meaningful).

    ``model_factory`` is a no-arg callable that returns a fresh
    ``BaseChatModel`` configured against ``config.stage2_model``. It is
    a callable (not a model instance) so the orchestrator can build the
    LLM lazily and so unit tests can inject a stub without importing
    LangChain.
    """

    def __init__(
        self,
        *,
        config: DedupConfig,
        text_extractor: Callable[[Any], str],
        model_factory: Callable[[], "BaseChatModel"],
        framework_cfg: "FrameworkAppConfig | None" = None,
    ) -> None:
        self.config = config
        self._text_extractor = text_extractor
        self._model_factory = model_factory
        # ``framework_cfg`` carries the cross-cutting prompt the
        # framework uses when ``DedupConfig.system_prompt`` is unset.
        # Imported lazily to avoid a circular import (``runtime.dedup``
        # is imported from ``runtime.config`` test paths).
        if framework_cfg is None:


            framework_cfg = _FAC()
        self._framework_cfg = framework_cfg

    async def run(
        self,
        *,
        session: StateT,
        history_store: "HistoryStore",
    ) -> DedupResult:
        """Run the pipeline for ``session``.

        Returns ``DedupResult(matched=False)`` when the pipeline is
        disabled, when the session text is empty, when Stage 1 finds no
        candidates above ``stage1_threshold``, or when Stage 2 declines
        every candidate (or errors out parsing structured output).
        """
        if not self.config.enabled:
            return DedupResult(matched=False)

        new_text = (self._text_extractor(session) or "").strip()
        if not new_text:
            return DedupResult(matched=False)

        candidates = self._stage1(session=session, new_text=new_text,
                                  history_store=history_store)
        if not candidates:
            return DedupResult(matched=False)

        return await self._stage2(session=session, new_text=new_text,
                                  candidates=candidates)

    # ----- Stage 1 -----

    def _stage1(
        self,
        *,
        session: StateT,
        new_text: str,
        history_store: "HistoryStore",
    ) -> list[tuple[Any, float]]:
        """Embedding similarity prefilter.

        Filters by ``scope.same_environment`` and ``scope.only_closed``,
        drops the current session id, applies the inclusive
        ``stage1_threshold``, and caps to ``stage1_top_k``.
        """
        filter_kwargs: dict[str, Any] = {}
        if self.config.scope.same_environment:
            env = getattr(session, "environment", None)
            if env:
                filter_kwargs["environment"] = env
        # ``status_filter`` is the resolved session bucket — only_closed
        # maps to "resolved" in the incident-management vocabulary.
        # Apps that disable only_closed get all statuses other than
        # in-flight via the empty filter (HistoryStore default behaviour
        # already screens deleted rows).
        status_filter = "resolved" if self.config.scope.only_closed else "*"
        try:
            raw = history_store.find_similar(
                query=new_text,
                filter_kwargs=filter_kwargs or None,
                status_filter=status_filter,
                threshold=self.config.stage1_threshold,
                limit=self.config.stage1_top_k,
            )
        except Exception as exc:  # noqa: BLE001 — never let dedup crash intake
            logger.warning("dedup stage 1: history_store failure: %s", exc)
            return []

        own_id = getattr(session, "id", None)
        out: list[tuple[Any, float]] = []
        for inc, score in raw:
            if getattr(inc, "id", None) == own_id:
                continue
            if score < self.config.stage1_threshold:
                # ``find_similar`` already screens by threshold but apps
                # may pass a custom HistoryStore — defensive double-check.
                continue
            out.append((inc, float(score)))
        return out[: self.config.stage1_top_k]

    # ----- Stage 2 -----

    async def _stage2(
        self,
        *,
        session: StateT,
        new_text: str,
        candidates: list[tuple[Any, float]],
    ) -> DedupResult:
        """LLM confirmation pass — short-circuits on the first confirm."""
        from langchain_core.messages import HumanMessage, SystemMessage

        capped = candidates[: self.config.stage2_top_k]
        # Build the model lazily so the factory error surfaces only when
        # we actually need an LLM (i.e. Stage 1 found candidates).
        try:
            llm = self._model_factory()
        except Exception as exc:  # noqa: BLE001
            logger.error("dedup stage 2: model factory failed: %s", exc)
            return DedupResult(matched=False)

        new_id = getattr(session, "id", "<new>")
        parse_failures = 0
        # Resolve the Stage-2 system prompt: per-config override wins,
        # otherwise the framework default. Apps that want incident-shaped
        # phrasing tune ``framework_cfg.dedup_system_prompt`` (the
        # incident-management example does); apps that want a one-off
        # override set it on ``DedupConfig.system_prompt``.
        system_prompt = (
            self.config.system_prompt
            or self._framework_cfg.dedup_system_prompt
        )
        for prior, stage1_score in capped:
            prior_id = getattr(prior, "id", "<unknown>")
            prior_text = (self._text_extractor(prior) or "").strip()
            user_prompt = _build_stage2_user_prompt(
                prior_text=prior_text,
                new_text=new_text,
                prior_id=str(prior_id),
                new_id=str(new_id),
            )
            try:
                msg = await llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "dedup stage 2: LLM call failed for prior=%s: %s",
                    prior_id, exc,
                )
                continue
            raw = getattr(msg, "content", "") or ""
            decision, parse_err = _parse_decision_tagged(
                raw, model_name=self.config.stage2_model,
            )
            if decision is None:
                # Parse / validation failure — count it so operators can
                # detect schema drift in dashboards / alerts. The legit
                # "model said not-duplicate" branch goes through the
                # ``decision is not None`` arm below and does NOT bump
                # the counter.
                if parse_err is not None:
                    parse_failures += 1
                continue
            if (decision.is_duplicate
                    and decision.confidence >= self.config.stage2_min_confidence):
                return DedupResult(
                    matched=True,
                    parent_session_id=str(prior_id),
                    candidate_id=str(prior_id),
                    decision=decision,
                    stage1_score=stage1_score,
                    parse_failures=parse_failures,
                )
        return DedupResult(matched=False, parse_failures=parse_failures)

# ====== module: runtime/intake.py ======

_log = logging.getLogger("runtime.intake")


@dataclass
class IntakeContext:
    """Optional store handles passed through ``app_cfg.intake_context``.

    The graph builder attaches one of these to the ``app_cfg`` argument
    before invoking the runner so that stateless module-level runners
    can still reach the live stores.
    """

    history_store: Any = None       # Optional[HistoryStore[StateT]]
    dedup_pipeline: Any = None      # Optional[DedupPipeline[StateT]]
    top_k: int = 3
    similarity_threshold: float = 0.7


def _project_prior(session: Session) -> dict[str, Any]:
    """Compact representation suitable for stashing on findings."""
    return {"id": session.id, "status": session.status}


def default_intake_runner(
    state: Any,
    *,
    app_cfg: Any | None = None,
) -> dict[str, Any] | None:
    """Generic similarity retrieval + dedup short-circuit.

    Returns ``None`` when nothing changed, a dict patch otherwise.
    The patch is merged by ``runtime.agents.supervisor`` and may
    include ``next_route='__end__'`` to short-circuit the graph.

    Called synchronously from the supervisor node. When a running event
    loop is detected (e.g. inside a LangGraph async node), the dedup
    step is skipped with a warning rather than raising ``RuntimeError``.
    """
    import asyncio

    session: Session | None = (
        state.get("session") if hasattr(state, "get") else None
    )
    if session is None:
        return None
    ctx: IntakeContext | None = getattr(app_cfg, "intake_context", None)
    if ctx is None:
        return None

    patch: dict[str, Any] = {}
    text = (session.to_agent_input() or "").strip()

    if ctx.history_store is not None and text:
        hits = ctx.history_store.find_similar(
            query=text,
            filter_kwargs=None,
            limit=ctx.top_k,
            threshold=ctx.similarity_threshold,
        )
        # hits is list[tuple[Session, float]]
        session.findings["prior_similar"] = [_project_prior(h) for h, _ in hits]
        patch["session"] = session

    if ctx.dedup_pipeline is not None:
        try:
            result = asyncio.run(
                ctx.dedup_pipeline.run(
                    session=session,
                    history_store=ctx.history_store,
                )
            )
        except RuntimeError:
            # Already inside a running event loop (e.g. LangGraph async
            # node). Fall through without dedup — the caller must use
            # the async API directly in that context.
            _log.warning(
                "default_intake_runner: asyncio.run() called from a running "
                "event loop; dedup short-circuit skipped for session %s",
                session.id,
            )
            result = None

        if result is not None and getattr(result, "matched", False):
            session.parent_session_id = result.parent_session_id
            session.status = "duplicate"
            rationale = None
            if result.decision is not None:
                rationale = getattr(result.decision, "rationale", None)
            if rationale:
                session.dedup_rationale = rationale
            patch["session"] = session
            patch["next_route"] = "__end__"

    return patch or None


def compose_runners(
    *runners: Callable[..., dict[str, Any] | None],
) -> Callable[..., dict[str, Any] | None]:
    """Chain multiple runners; first one to return ``next_route`` wins.

    Each runner is called in order with the same ``(state, *, app_cfg)``
    signature. Non-route patches are merged left-to-right (later
    runners can overwrite earlier keys, except ``next_route`` which is
    sticky once set).
    """

    def _composed(state: Any, *, app_cfg: Any | None = None) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        for r in runners:
            out = r(state, app_cfg=app_cfg)
            if not out:
                continue
            merged.update({k: v for k, v in out.items() if k != "next_route"})
            if "next_route" in out and "next_route" not in merged:
                merged["next_route"] = out["next_route"]
                # Short-circuit: subsequent runners do not run.
                return merged
        return merged or None

    return _composed


def hydrate_from_memory(
    state: Any,
    *,
    kg_store: Any = None,
    playbook_store: Any = None,
    release_store: Any = None,
    hydrator: Callable[..., Any] | None = None,
    gate: Callable[..., str | None] | None = None,
) -> dict[str, Any] | None:
    """Generic memory-hydration runner shell.

    Apps that wire L2 / L5 / L7 stores via :mod:`runtime.memory` plug
    them in here. The framework supplies the runner-shape contract
    (``state.session`` access, ``next_route='__end__'`` short-circuit,
    duplicate-metadata stamping) so per-app supervisors collapse to:



        def app_hydration(state, *, app_cfg=None):
            return hydrate_from_memory(
                state,
                kg_store=...,
                playbook_store=...,
                release_store=...,
                hydrator=app_specific_hydrate_callable,
                gate=app_specific_gate_callable,  # optional
            )

        default_supervisor_runner = compose_runners(
            default_intake_runner, app_hydration,
        )

    ``hydrator`` signature: ``(session, *, kg_store, playbook_store,
    release_store) -> Any`` where the returned object is stamped on
    ``session.memory`` if that attribute is settable. If ``hydrator``
    is ``None`` the function is a no-op (returns ``None``).

    ``gate`` signature: ``(session, *, kg_store) -> str | None`` —
    return a parent session id to mark the new session as a duplicate
    (caller stamps ``status='duplicate'`` and emits
    ``next_route='__end__'``). When ``None`` is returned (or no gate
    supplied) the session proceeds normally.

    Returns ``None`` when no hydration occurred, otherwise a runner
    patch suitable for merging by ``runtime.agents.supervisor``.
    """
    session = state.get("session") if hasattr(state, "get") else None
    if session is None:
        return None
    if hydrator is None and gate is None:
        return None

    patch: dict[str, Any] = {}

    if hydrator is not None:
        try:
            memory = hydrator(
                session,
                kg_store=kg_store,
                playbook_store=playbook_store,
                release_store=release_store,
            )
        except Exception:  # noqa: BLE001 — defensive, keep graph alive
            _log.exception(
                "hydrate_from_memory: hydrator raised; routing through",
            )
            memory = None

        if memory is not None and hasattr(session, "memory"):
            try:
                session.memory = memory
            except Exception:  # noqa: BLE001 — frozen / read-only field
                _log.warning(
                    "hydrate_from_memory: cannot set session.memory; "
                    "downstream agents will not see hydrated context",
                )
        patch["session"] = session

    if gate is not None:
        try:
            parent = gate(session, kg_store=kg_store)
        except Exception:  # noqa: BLE001 — defensive
            _log.exception(
                "hydrate_from_memory: gate raised; routing through",
            )
            parent = None

        if parent is not None:
            try:
                session.status = "duplicate"
                if hasattr(session, "parent_session_id"):
                    session.parent_session_id = parent
            except Exception:  # noqa: BLE001
                _log.warning(
                    "hydrate_from_memory: cannot stamp duplicate metadata "
                    "on session %s", getattr(session, "id", "?"),
                )
            patch["session"] = session
            patch["next_route"] = "__end__"

    return patch or None

# ====== module: runtime/memory/session_state.py ======

class L2KGContext(BaseModel):
    """L2 Knowledge Graph subgraph snapshot.

    Mirrors ASR.md §3 L2 / §6: a small projection over the affected
    components plus their immediate upstream / downstream neighbours.
    ``raw`` carries the full assembled subgraph so downstream agents
    can render or re-traverse without a second store call.
    """

    model_config = ConfigDict(extra="forbid")

    components: list[str] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)


class L5ReleaseContext(BaseModel):
    """L5 Release Context window relevant to an investigation.

    Each entry in ``recent_releases`` is a release record dict with at
    least ``service``, ``sha``, ``deployed_at``, ``author``.
    ``suspect_releases`` is the subset of release ids correlated to
    the session's anchor time within the configured window.
    """

    model_config = ConfigDict(extra="forbid")

    recent_releases: list[dict] = Field(default_factory=list)
    suspect_releases: list[str] = Field(default_factory=list)


class L7PlaybookSuggestion(BaseModel):
    """A single L7 playbook the matcher proposes for this investigation."""

    model_config = ConfigDict(extra="forbid")

    playbook_id: str
    score: float = Field(ge=0.0, le=1.0)
    matched_signals: list[str] = Field(default_factory=list)


class MemoryLayerState(BaseModel):
    """Container for the memory-layer slots attached to ``IncidentState``.

    The whole object is optional / empty by default so legacy sessions
    written before this field existed round-trip cleanly: the field
    hydrates to a default ``MemoryLayerState`` even when
    ``extra_fields`` is missing the key entirely.
    """

    model_config = ConfigDict(extra="forbid")

    l2_kg: L2KGContext | None = None
    l5_release: L5ReleaseContext | None = None
    l7_playbooks: list[L7PlaybookSuggestion] = Field(default_factory=list)

# ====== module: runtime/memory/knowledge_graph.py ======

_VALID_EDGE_KINDS: frozenset[str] = frozenset(
    {"calls", "deploys", "reads", "writes"}
)

_SEED_ROOT = Path(__file__).parent / "seeds" / "kg"


class KnowledgeGraphStore:
    """Filesystem-backed L2 Knowledge Graph reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._components: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        comp_path = self._root / "components.json"
        edges_path = self._root / "edges.json"

        # Fall back to the bundled seed when the configured layer dir
        # is missing or empty. Keeps tests and fresh checkouts working
        # without forcing the operator to seed ``incidents/kg/``.
        if not comp_path.exists() or not edges_path.exists():
            comp_path = _SEED_ROOT / "components.json"
            edges_path = _SEED_ROOT / "edges.json"

        components_raw = json.loads(comp_path.read_text())
        edges_raw = json.loads(edges_path.read_text())

        for c in components_raw:
            cid = c.get("id")
            if not cid:
                continue
            self._components[cid] = dict(c)

        for e in edges_raw:
            kind = e.get("kind")
            if kind not in _VALID_EDGE_KINDS:
                # Skip silently rather than raise — air-gapped operators
                # frequently hand-edit these files. A logged warning is
                # the right move once observability lands; for now,
                # ``list_edges()`` exposes the loaded set for tests.
                continue
            if "from" not in e or "to" not in e:
                continue
            self._edges.append({
                "from": e["from"],
                "to": e["to"],
                "kind": kind,
            })

    # ----- introspection (mostly for tests) -----------------------------

    def list_components(self) -> list[dict]:
        return list(self._components.values())

    def list_edges(self) -> list[dict]:
        return list(self._edges)

    # ----- public read API ----------------------------------------------

    def get_component(self, comp_id: str) -> dict | None:
        return self._components.get(comp_id)

    def find_by_name(self, name: str) -> list[dict]:
        """Case-insensitive substring match on the ``name`` field."""
        if not name:
            return []
        needle = name.lower()
        return [
            dict(c)
            for c in self._components.values()
            if needle in (c.get("name") or "").lower()
        ]

    def neighbors(
        self,
        comp_id: str,
        *,
        kinds: set[str] | None = None,
        hops: int = 1,
    ) -> set[str]:
        """Return the set of component ids reachable from ``comp_id``.

        - ``kinds`` filters edges to a subset of the valid edge kinds.
          ``None`` means "any kind".
        - ``hops`` is the BFS depth (>= 1). The starting node itself is
          *not* included in the returned set.
        """
        if hops < 1:
            return set()
        if comp_id not in self._components:
            return set()

        kind_filter = (
            None
            if kinds is None
            else (set(kinds) & _VALID_EDGE_KINDS)
        )

        visited: set[str] = {comp_id}
        frontier: set[str] = {comp_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for edge in self._edges:
                    if kind_filter is not None and edge["kind"] not in kind_filter:
                        continue
                    # Treat edges as undirected for neighbour expansion;
                    # ``subgraph`` records direction explicitly via the
                    # upstream / downstream split.
                    if edge["from"] == node and edge["to"] not in visited:
                        next_frontier.add(edge["to"])
                    elif edge["to"] == node and edge["from"] not in visited:
                        next_frontier.add(edge["from"])
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(comp_id)
        return visited

    def subgraph(
        self,
        comp_ids: Iterable[str],
        hops: int = 1,
    ) -> L2KGContext:
        """Assemble an :class:`L2KGContext` for the given component set.

        - ``components`` — the input set (filtered to ones we know about).
        - ``upstream``   — distinct ids on the ``from`` side of any edge
          whose ``to`` is in the input set.
        - ``downstream`` — distinct ids on the ``to`` side of any edge
          whose ``from`` is in the input set.
        - ``raw``        — full subgraph snapshot
          (``{"nodes": [...], "edges": [...]}``) including ``hops``
          worth of neighbour expansion, useful for UI rendering.
        """
        seeds = {c for c in comp_ids if c in self._components}

        upstream: set[str] = set()
        downstream: set[str] = set()
        for edge in self._edges:
            if edge["to"] in seeds and edge["from"] not in seeds:
                upstream.add(edge["from"])
            if edge["from"] in seeds and edge["to"] not in seeds:
                downstream.add(edge["to"])

        # Expand to ``hops`` neighbourhood for the raw snapshot.
        expanded: set[str] = set(seeds)
        for s in seeds:
            expanded |= self.neighbors(s, hops=hops)

        raw_nodes = [
            self._components[i] for i in expanded if i in self._components
        ]
        raw_edges = [
            edge
            for edge in self._edges
            if edge["from"] in expanded and edge["to"] in expanded
        ]

        return L2KGContext(
            components=sorted(seeds),
            upstream=sorted(upstream),
            downstream=sorted(downstream),
            raw={"nodes": raw_nodes, "edges": raw_edges},
        )

# ====== module: runtime/memory/release_context.py ======

_SEED_ROOT = Path(__file__).parent / "seeds" / "releases"


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp tolerating the ``Z`` suffix."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ReleaseContextStore:
    """Filesystem-backed L5 Release Context reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._releases: list[dict] = []
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        path = self._root / "recent.json"
        if not path.exists():
            path = _SEED_ROOT / "recent.json"

        records = json.loads(path.read_text())
        cleaned: list[dict] = []
        for r in records:
            if not r.get("id") or not r.get("service") or not r.get("deployed_at"):
                continue
            try:
                _parse_iso(r["deployed_at"])
            except ValueError:
                continue
            cleaned.append(dict(r))

        cleaned.sort(
            key=lambda r: _parse_iso(r["deployed_at"]),
            reverse=True,
        )
        self._releases = cleaned

    # ----- introspection (mostly for tests) -----------------------------

    def list_all(self) -> list[dict]:
        return [dict(r) for r in self._releases]

    # ----- public read API ----------------------------------------------

    def recent_for_service(
        self,
        service: str,
        *,
        hours: int = 24,
    ) -> list[dict]:
        """Releases for ``service`` deployed within the last ``hours``.

        ``hours`` is measured against ``datetime.now(UTC)``; for
        deterministic correlation work prefer :meth:`context` /
        :meth:`suspect_at` which take an explicit ``at``/``incident_at``
        anchor.
        """
        if hours <= 0:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out = [
            dict(r)
            for r in self._releases
            if r["service"] == service and _parse_iso(r["deployed_at"]) >= cutoff
        ]
        # Already in descending order from ``_load``.
        return out

    def suspect_at(
        self,
        *,
        services: list[str],
        at: datetime,
        window_minutes: int = 60,
    ) -> list[str]:
        """Release ids for ``services`` deployed within ``window_minutes``
        of ``at``.

        The window is symmetric around ``at`` so a release shipped right
        before *or* right after that anchor time is surfaced — useful
        for both "deploy caused it" and "deploy is the rollback" cases.
        Returns release ids sorted by ``deployed_at`` descending.
        """
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if window_minutes <= 0:
            return []

        wanted = set(services)
        delta = timedelta(minutes=window_minutes)
        lo = at - delta
        hi = at + delta

        suspects: list[tuple[datetime, str]] = []
        for r in self._releases:
            if r["service"] not in wanted:
                continue
            deployed_at = _parse_iso(r["deployed_at"])
            if lo <= deployed_at <= hi:
                suspects.append((deployed_at, r["id"]))

        suspects.sort(key=lambda t: t[0], reverse=True)
        return [rid for _, rid in suspects]

    def context(
        self,
        services: list[str],
        incident_at: datetime,
    ) -> L5ReleaseContext:
        """Assemble an :class:`L5ReleaseContext` for the investigation.

        - ``recent_releases`` — all releases for the given services in
          the last 24h relative to ``incident_at`` (descending).
        - ``suspect_releases`` — release ids inside a 60-minute window
          around ``incident_at``.
        """
        if incident_at.tzinfo is None:
            incident_at = incident_at.replace(tzinfo=timezone.utc)

        wanted = set(services)
        cutoff = incident_at - timedelta(hours=24)
        recent = [
            dict(r)
            for r in self._releases
            if r["service"] in wanted and cutoff <= _parse_iso(r["deployed_at"]) <= incident_at
        ]
        suspects = self.suspect_at(
            services=services, at=incident_at, window_minutes=60
        )
        return L5ReleaseContext(
            recent_releases=recent,
            suspect_releases=suspects,
        )

# ====== module: runtime/memory/playbook_store.py ======

_SEED_ROOT = Path(__file__).parent / "seeds" / "playbooks"


def _normalise(value: Any) -> str:
    """Lowercase a scalar for case-insensitive equality."""
    if isinstance(value, bool):
        # ``str(True) == "True"`` would never match user-supplied
        # ``"true"``; standardise on the JSON/YAML lowercase form.
        return "true" if value else "false"
    return str(value).strip().lower()


class PlaybookStore:
    """Filesystem-backed L7 Playbook reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._playbooks: dict[str, dict] = {}
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        roots: list[Path] = [self._root]
        # Fall back to the bundled seed when the configured layer dir
        # has no playbooks yet.
        if not self._has_yaml(self._root):
            roots = [_SEED_ROOT]

        for r in roots:
            if not r.exists() or not r.is_dir():
                continue
            for path in sorted(r.iterdir()):
                if path.suffix.lower() not in {".yaml", ".yml"}:
                    continue
                try:
                    data = yaml.safe_load(path.read_text())
                except yaml.YAMLError:
                    continue
                if not isinstance(data, dict):
                    continue
                pid = data.get("id")
                if not pid or not isinstance(pid, str):
                    continue
                self._playbooks[pid] = data

    @staticmethod
    def _has_yaml(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        return any(
            p.suffix.lower() in {".yaml", ".yml"} for p in root.iterdir()
        )

    # ----- public read API ----------------------------------------------

    def get(self, playbook_id: str) -> dict | None:
        pb = self._playbooks.get(playbook_id)
        return None if pb is None else dict(pb)

    def list_all(self) -> list[dict]:
        return [dict(p) for p in self._playbooks.values()]

    def match(self, signals: dict) -> list[L7PlaybookSuggestion]:
        """Score every playbook against ``signals`` (case-insensitive eq).

        Score = ``matched / total`` where ``total`` is the number of
        keys declared on the playbook's ``match_signals`` block. A
        playbook with no declared signals scores 0 and is dropped from
        the result. Suggestions are returned in descending score, then
        ascending ``playbook_id`` for deterministic ties.
        """
        if not signals:
            return []

        norm_signals = {
            str(k).strip().lower(): _normalise(v)
            for k, v in signals.items()
        }

        out: list[L7PlaybookSuggestion] = []
        for pid, pb in self._playbooks.items():
            declared = pb.get("match_signals") or {}
            if not isinstance(declared, dict) or not declared:
                continue

            total = len(declared)
            matched_keys: list[str] = []
            for key, expected in declared.items():
                k = str(key).strip().lower()
                if k in norm_signals and norm_signals[k] == _normalise(expected):
                    matched_keys.append(f"{key}={expected}")

            if not matched_keys:
                continue

            out.append(
                L7PlaybookSuggestion(
                    playbook_id=pid,
                    score=len(matched_keys) / total,
                    matched_signals=sorted(matched_keys),
                )
            )

        out.sort(key=lambda s: (-s.score, s.playbook_id))
        return out

# ====== module: runtime/memory/hypothesis.py ======

MAX_ITERATIONS: int = 3
ACCEPT_THRESHOLD: float = 0.7


# Mirror runtime.similarity's tokenisation so the score's notion of
# "overlap" matches the dedup / lookup_similar_incidents path.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "to", "and", "or",
    "is", "was", "with", "for", "be", "are", "as", "at",
    "by", "from", "has", "had", "have", "it", "that", "this",
    "we", "i", "you", "they", "but", "not", "no", "if",
})


class HypothesisScore(TypedDict):
    """Result returned by :func:`score_hypothesis`."""

    score: float
    rationale: str
    matched_terms: list[str]


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOP and len(t) > 1
    }


def score_hypothesis(
    hypothesis: str,
    evidence: list[str],
) -> HypothesisScore:
    """Score how well ``evidence`` supports ``hypothesis``.

    Token-overlap heuristic. The score is the fraction of hypothesis
    tokens that appear in *any* evidence string, capped at 1.0.

    - Empty hypothesis -> ``0.0`` (defensive; the LLM should never
      produce an empty hypothesis but the loop must not crash).
    - Empty evidence list -> ``0.0`` (no support).
    - Score is always in ``[0.0, 1.0]`` inclusive.

    Returns a :class:`HypothesisScore` with the score, a short
    machine-generated rationale, and the matched tokens (handy for
    rendering in the UI's hypothesis trail).
    """
    h_tokens = _tokens(hypothesis)
    if not h_tokens:
        return HypothesisScore(
            score=0.0,
            rationale="Empty hypothesis — no tokens to score against evidence.",
            matched_terms=[],
        )
    if not evidence:
        return HypothesisScore(
            score=0.0,
            rationale=f"No evidence supplied to support {len(h_tokens)} hypothesis terms.",
            matched_terms=[],
        )

    e_tokens: set[str] = set()
    for snippet in evidence:
        e_tokens |= _tokens(snippet)

    matched = h_tokens & e_tokens
    score = len(matched) / len(h_tokens)
    # Round to 3 dp to keep the audit trail readable; score is still a
    # float for callers that want a tighter comparison.
    score = round(score, 3)

    rationale = (
        f"Matched {len(matched)}/{len(h_tokens)} hypothesis terms "
        f"in {len(evidence)} evidence snippets."
    )
    return HypothesisScore(
        score=score,
        rationale=rationale,
        matched_terms=sorted(matched),
    )


def should_refine(score: float, iterations: int) -> bool:
    """Loop-control predicate: True when the agent should refine again.

    Refines while:

    * the current score is below :data:`ACCEPT_THRESHOLD`, AND
    * the iteration count is strictly less than :data:`MAX_ITERATIONS`.

    The iteration counter is the number of *completed* rounds — so
    ``should_refine(score=0.5, iterations=0)`` returns ``True`` (we've
    done 0 rounds, want at least 1), ``should_refine(score=0.5,
    iterations=3)`` returns ``False`` (cap hit).

    Defensive on bad inputs: negative iterations are clamped to 0;
    out-of-range scores are clamped to ``[0.0, 1.0]``.
    """
    if score is None:
        return iterations < MAX_ITERATIONS
    s = max(0.0, min(1.0, float(score)))
    n = max(0, int(iterations))
    return s < ACCEPT_THRESHOLD and n < MAX_ITERATIONS


__all__ = [
    "ACCEPT_THRESHOLD",
    "HypothesisScore",
    "MAX_ITERATIONS",
    "score_hypothesis",
    "should_refine",
]

# ====== module: runtime/memory/resolution.py ======

class ToolCallSpec(TypedDict):
    """A single suggested tool call sourced from a playbook step."""

    tool: str
    args: dict[str, Any]
    requires_approval: bool


def playbook_to_tool_calls(playbook: dict) -> list[ToolCallSpec]:
    """Translate a playbook dict's ``remediation`` block into tool calls.

    Each ``remediation`` entry is expected to have:

    - ``tool`` (str, required) — the tool name as known to the gateway.
    - ``args`` (dict, optional) — keyword args; defaults to ``{}``.

    The playbook's top-level ``required_approval`` flag (see
    ``examples/incident_management/asr/seeds/playbooks/*.yaml``) is
    propagated to every emitted spec — it represents the playbook
    author's stated risk posture. The gateway's risk policy still has
    final say at execution time; this flag is purely advisory metadata
    for the agent's prompt and the UI.

    Returns an empty list when ``playbook`` is None / lacks
    ``remediation`` / has malformed entries — the caller can branch on
    "no suggestion" without a try/except.
    """
    if not playbook or not isinstance(playbook, dict):
        return []
    remediation = playbook.get("remediation") or []
    if not isinstance(remediation, list):
        return []

    requires_approval = bool(playbook.get("required_approval"))
    out: list[ToolCallSpec] = []
    for entry in remediation:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if not tool or not isinstance(tool, str):
            continue
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCallSpec(
            tool=tool,
            args=dict(args),
            requires_approval=requires_approval,
        ))
    return out


def top_playbook(
    suggestions: list[L7PlaybookSuggestion],
) -> str | None:
    """Return the playbook_id of the highest-scoring suggestion, or None.

    ``suggestions`` is the list ``PlaybookStore.match`` already sorts
    by descending score; we still pick by ``max`` so callers that pass
    a re-ordered list (e.g. tests) still get a deterministic answer.
    """
    if not suggestions:
        return None
    best = max(suggestions, key=lambda s: (s.score, -hash(s.playbook_id)))
    return best.playbook_id


__all__ = [
    "ToolCallSpec",
    "playbook_to_tool_calls",
    "top_playbook",
]

# ====== module: runtime/orchestrator.py ======

if TYPE_CHECKING:
    # Avoid a runtime circular import — ``runtime.triggers.base`` only
    # defines a dataclass, and the type appears in a method annotation.
    pass






from langgraph.types import Command











def _default_text_extractor(session) -> str:
    """Default text extraction for the incident-management example.

    Concatenates the operator-supplied ``query``, the intake-summary
    (when present), and any tags. Keeps the framework agnostic of the
    domain class — apps with a different shape can install their own
    extractor by subclassing the orchestrator (or by setting one of the
    expected attributes on their state class).
    """
    parts = []
    q = getattr(session, "query", None)
    if q:
        parts.append(str(q))
    s = getattr(session, "summary", None)
    if s:
        parts.append(str(s))
    tags = getattr(session, "tags", None)
    if tags:
        parts.append(" ".join(str(t) for t in tags))
    return " ".join(parts).strip()

_INCIDENT_MCP_MODULE = "examples.incident_management.mcp_server"


def _resolve_dedup_config(dotted: str | None) -> "DedupConfig | None":
    """Resolve ``module.path:callable`` into a :class:`DedupConfig` (or None).

    Returns ``None`` when the path is unset; raises ``ValueError`` for
    malformed paths (mirrors :func:`resolve_framework_app_config`).
    The provider must be a no-arg callable returning ``DedupConfig | None``.
    """
    if dotted is None:
        return None
    if ":" not in dotted:
        raise ValueError(
            f"dedup_config_path={dotted!r} must be in 'module.path:callable' form"
        )
    module_name, _, attr = dotted.partition(":")
    mod = importlib.import_module(module_name)
    provider = getattr(mod, attr)
    cfg = provider()
    if cfg is None:
        return None
    if not isinstance(cfg, DedupConfig):
        raise TypeError(
            f"dedup provider {dotted!r} returned {type(cfg).__name__}; "
            "expected DedupConfig | None"
        )
    return cfg

# StateT mirrors the bound used in ``runtime.storage.session_store``;
# kept permissive at ``BaseModel`` so the storage layer is usable by
# ad-hoc tests that build a plain ``BaseModel`` directly. Real apps
# inject a ``runtime.state.Session`` subclass via
# ``RuntimeConfig.state_class`` and the resolver enforces that bound.
StateT = TypeVar("StateT", bound=BaseModel)


def _coerce_state_overrides(
    state_overrides: dict | None,
    environment: str | None,
) -> dict | None:
    """Resolve the generic ``state_overrides`` kwarg from the public
    ``start_session`` surface, coercing the deprecated ``environment``
    kwarg when present.

    Rules mirror :func:`_coerce_submitter`:
      - If neither is supplied, return ``None``.
      - If only ``state_overrides`` is supplied, return it unchanged.
      - If only ``environment`` is supplied, emit a single
        ``DeprecationWarning`` and return ``{"environment": environment}``.
      - If both are supplied, raise ``TypeError`` — silent precedence
        would mask caller bugs.
    """
    if state_overrides is not None and environment is not None:
        raise TypeError(
            "start_session() received both state_overrides and "
            "environment; pass state_overrides only "
            "(environment is deprecated)"
        )
    if environment is not None:
        warnings.warn(
            "environment is a deprecated kwarg on start_session(); pass "
            "state_overrides={'environment': ...} instead. The legacy kwarg "
            "will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return {"environment": environment}
    return state_overrides


def _coerce_submitter(
    submitter: dict | None,
    reporter_id: str | None,
    reporter_team: str | None,
) -> dict | None:
    """Resolve the generic ``submitter`` kwarg from the public start_session
    surface, coercing the deprecated ``reporter_id``/``reporter_team`` pair
    when present.

    Rules:
      - If neither ``submitter`` nor the legacy pair is supplied, return ``None``.
      - If only ``submitter`` is supplied, return it unchanged.
      - If only the legacy kwargs are supplied, emit a single
        ``DeprecationWarning`` per call and return
        ``{"id": reporter_id, "team": reporter_team}`` (defaulting missing
        halves to the historical ``"user-mock"``/``"platform"`` so existing
        callers behave identically).
      - If both ``submitter`` and either legacy kwarg are supplied, raise
        ``TypeError`` — there's no sensible merge and silent precedence
        would mask a caller bug.
    """
    legacy_supplied = reporter_id is not None or reporter_team is not None
    if submitter is not None and legacy_supplied:
        raise TypeError(
            "start_session() received both submitter and "
            "reporter_id/reporter_team; pass submitter only "
            "(reporter_id/reporter_team are deprecated)"
        )
    if legacy_supplied:
        warnings.warn(
            "reporter_id and reporter_team are deprecated kwargs on "
            "start_session(); pass submitter={'id': ..., 'team': ...} "
            "instead. The legacy kwargs will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return {
            "id": reporter_id if reporter_id is not None else "user-mock",
            "team": reporter_team if reporter_team is not None else "platform",
        }
    return submitter


def _metadata_url(cfg: AppConfig) -> str:
    """Derive the metadata DB URL for the current config.

    When ``cfg.storage.metadata.url`` is still the default sentinel, use
    ``cfg.paths.incidents_dir`` so that per-test ``tmp_path`` isolation is
    respected. Production deployments that set an explicit URL are left
    untouched.
    """
    default_url = MetadataConfig().url
    if cfg.storage.metadata.url != default_url:
        return cfg.storage.metadata.url
    return f"sqlite:///{Path(cfg.paths.incidents_dir) / 'incidents.db'}"


class Orchestrator(Generic[StateT]):
    """High-level facade. Construct via ``await Orchestrator.create(cfg)``.

    The Orchestrator owns the lifecycle of the FastMCP clients underpinning the
    tool registry. Always call :meth:`aclose` (or use ``async with``) when done.

    ``Generic[StateT]`` carries the resolved app state class through to the
    store. The actual class is resolved at construction time via
    :func:`runtime.state_resolver.resolve_state_class` (PEP 484 generics are
    erased at runtime, so we can't introspect ``StateT`` directly).
    """

    def __init__(self, cfg: AppConfig, store: SessionStore,
                 skills: dict[str, Skill], registry: ToolRegistry, graph,
                 exit_stack: AsyncExitStack,
                 framework_cfg: FrameworkAppConfig | None = None,
                 state_cls: Type[StateT] = Session,  # type: ignore[assignment]
                 history: HistoryStore | None = None,
                 checkpointer=None,
                 checkpointer_close=None,
                 dedup_pipeline: "DedupPipeline | None" = None):
        self.cfg = cfg
        self.store = store
        # Optional two-stage dedup pipeline. Built in ``create`` only
        # when the resolved DedupConfig has ``enabled=True``; otherwise
        # ``None`` so the lifecycle hook is a no-op.
        self.dedup_pipeline = dedup_pipeline
        # The active ``SessionStore`` (CRUD) plus a separate read-only
        # ``HistoryStore`` (similarity search) share the same engine and
        # vector store; ``history`` is optional for callers that don't
        # need similarity lookups.
        self.history = history
        self.skills = skills
        self.registry = registry
        # A single compiled graph drives both fresh runs and resume-
        # from-interrupt. Resumes go through ``ainvoke`` /
        # ``astream_events`` with ``Command(resume=...)`` against the
        # same ``thread_id`` — the checkpointer rehydrates the paused
        # state.
        self.graph = graph
        self._exit_stack = exit_stack
        self.state_cls = state_cls
        # Durable LangGraph checkpointer keyed off the same metadata URL
        # as the relational store. ``checkpointer`` is the saver; the
        # ``checkpointer_close`` callable is invoked from ``aclose`` so
        # the underlying connection / pool is released on orchestrator
        # shutdown.
        self.checkpointer = checkpointer
        self._checkpointer_close = checkpointer_close
        # Cross-cutting domain-flavored knobs (confidence threshold,
        # escalation roster, severity aliases, dedup prompt) now live
        # on a generic FrameworkAppConfig the runtime can consume
        # without importing app-specific config modules.
        self.framework_cfg = framework_cfg or FrameworkAppConfig()

    @classmethod
    async def create(cls, cfg: AppConfig) -> "Orchestrator":
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            # Cross-cutting framework knobs read directly off
            # ``AppConfig.framework`` — the YAML carries them under the
            # ``framework:`` block, no app-specific provider callable.
            # Falls back to the dotted-path provider for backward
            # compatibility with deployments that still wire it (the
            # provider, when set, wins over the YAML default since
            # historical configs relied on it).
            if cfg.runtime.framework_app_config_path is not None:
                framework_cfg = resolve_framework_app_config(
                    cfg.runtime.framework_app_config_path,
                )
            else:
                framework_cfg = cfg.framework
            # Resolve the app state class once. ``None`` (default) keeps the
            # framework-default ``Session`` shape; apps point this at e.g.
            # ``examples.incident_management.state.IncidentState`` via YAML.
            resolved_state_cls: Type[BaseModel] = resolve_state_class(
                cfg.runtime.state_class
            )
            # SQLite concurrency PRAGMAs (WAL, busy_timeout,
            # synchronous=NORMAL) and ``BEGIN IMMEDIATE`` are installed
            # inside ``build_engine`` so any caller (orchestrator, tests,
            # ad-hoc scripts) gets a saver-friendly engine without
            # duplicating the connect-event hook.
            engine = build_engine(MetadataConfig(
                url=_metadata_url(cfg),
                pool_size=cfg.storage.metadata.pool_size,
                echo=cfg.storage.metadata.echo,
            ))
            Base.metadata.create_all(engine)
            embedder = build_embedder(cfg.llm.embedding, cfg.llm.providers)
            vector_store = build_vector_store(cfg.storage.vector, embedder, engine)
            # Build SessionStore (CRUD) and HistoryStore (similarity)
            # directly. ``state_class`` is resolved via the runtime
            # resolver; when an app doesn't set it the bare ``Session``
            # is used.
            repo_state_cls: Type[BaseModel] = resolved_state_cls
            store = SessionStore(
                engine=engine,
                state_cls=repo_state_cls,
                embedder=embedder,
                vector_store=vector_store,
                vector_path=(cfg.storage.vector.path
                             if cfg.storage.vector.backend == "faiss" else None),
                vector_index_name=cfg.storage.vector.collection_name,
                distance_strategy=cfg.storage.vector.distance_strategy,
            )
            history = HistoryStore(
                engine=engine,
                state_cls=repo_state_cls,
                embedder=embedder,
                vector_store=vector_store,
                similarity_threshold=framework_cfg.similarity_threshold,
                distance_strategy=cfg.storage.vector.distance_strategy,
            )
            # Attach intake_context onto framework_cfg so supervisor nodes can
            # reach the live stores via app_cfg.intake_context. FrameworkAppConfig
            # is a Pydantic model; use object.__setattr__ to set a runtime
            # attribute without triggering Pydantic's frozen-model guard.
            object.__setattr__(
                framework_cfg,
                "intake_context",
                IntakeContext(
                    history_store=history,
                    dedup_pipeline=None,  # dedup_pipeline built below; patched after
                    top_k=framework_cfg.intake_top_k,
                    similarity_threshold=framework_cfg.intake_similarity_threshold,
                ),
            )
            # Configure incident_management state via importlib so we hit the
            # *same* module instance the MCP loader will import. In the
            # single-file dist bundle a direct ``set_state`` call would
            # configure a bundled-local ``_default_server`` while the loader
            # imports ``examples.incident_management.mcp_server`` and uses
            # a *different* singleton — leaving FastMCP tools unconfigured.
            for srv in cfg.mcp.servers:
                if (srv.transport == "in_process" and srv.enabled
                        and srv.module == _INCIDENT_MCP_MODULE):
                    importlib.import_module(_INCIDENT_MCP_MODULE).set_state(
                        store=store,
                        history=history,
                        severity_aliases=framework_cfg.severity_aliases,
                    )
                    break
            if cfg.paths.skills_dir is None:
                raise RuntimeError(
                    "paths.skills_dir is not configured; apps must set it "
                    "via config.yaml or env"
                )
            skills = load_all_skills(cfg.paths.skills_dir)
            for s in skills.values():
                if s.model is not None and s.model not in cfg.llm.models:
                    raise ValueError(
                        f"skill {s.name!r} references llm model {s.model!r} "
                        f"which is not defined in llm.models "
                        f"(known: {sorted(cfg.llm.models)})"
                    )
            registry = await load_tools(cfg.mcp, stack)
            # Build the durable checkpointer once and pass it into the
            # compiled graph. Stays attached to the orchestrator so
            # aclose() can release the underlying connection / pool.
            # Pass the *resolved* metadata config (URL rewritten via
            # ``_metadata_url`` so per-test ``tmp_path`` isolation lands
            # on the same DB file the SQLAlchemy engine just opened).
            checkpointer, checkpointer_close = await make_checkpointer(
                MetadataConfig(
                    url=_metadata_url(cfg),
                    pool_size=cfg.storage.metadata.pool_size,
                    echo=cfg.storage.metadata.echo,
                )
            )
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry,
                                      checkpointer=checkpointer,
                                      framework_cfg=framework_cfg)
            # Build the dedup pipeline iff the app has opted in AND the
            # configured stage 2 model resolves in the LLM registry.
            # When the registry doesn't include the configured model
            # (e.g. CI uses ``LLMConfig.stub()``), dedup is silently
            # treated as off so unrelated tests don't need to know
            # about it. Production deployments with a real registry hit
            # the strict validation path.
            #
            # DedupConfig is now a first-class field on ``AppConfig``
            # (read from the YAML's top-level ``dedup:`` block). The
            # legacy provider-callable path is honoured when set so
            # existing deployments don't break, but the YAML wins for
            # bare apps.
            dedup_pipeline: DedupPipeline | None = None
            if cfg.runtime.dedup_config_path is not None:
                dedup_cfg: DedupConfig | None = _resolve_dedup_config(
                    cfg.runtime.dedup_config_path,
                )
            else:
                dedup_cfg = cfg.dedup
            if dedup_cfg is not None and dedup_cfg.enabled:
                if dedup_cfg.stage2_model in cfg.llm.models:
                    _llm_cfg_capture = cfg.llm
                    _model_name = dedup_cfg.stage2_model

                    def _factory():
                        return get_llm(
                            _llm_cfg_capture, _model_name, role="dedup",
                        )

                    dedup_pipeline = DedupPipeline(
                        config=dedup_cfg,
                        text_extractor=_default_text_extractor,
                        model_factory=_factory,
                        framework_cfg=framework_cfg,
                    )
            # Backfill dedup_pipeline into the IntakeContext now that it is built.
            # The IntakeContext was constructed with dedup_pipeline=None above
            # because the pipeline is built after graph construction.
            if dedup_pipeline is not None:
                framework_cfg.intake_context.dedup_pipeline = dedup_pipeline
            # No bespoke resume graph — resume runs through the main
            # graph via ``Command(resume=...)`` against the same
            # thread_id, with the checkpointer rehydrating paused state.
            return cls(cfg, store, skills, registry, graph,
                       stack, framework_cfg=framework_cfg,
                       state_cls=repo_state_cls,
                       history=history,
                       checkpointer=checkpointer,
                       checkpointer_close=checkpointer_close,
                       dedup_pipeline=dedup_pipeline)
        except BaseException:
            # Best-effort: close the checkpointer connection if it was
            # built before we hit the failure, so we don't leak FDs.
            try:
                await checkpointer_close()  # pyright: ignore[reportPossiblyUnboundVariable]
            except Exception:  # noqa: BLE001
                pass
            await stack.aclose()
            raise

    async def aclose(self) -> None:
        """Close all owned MCP clients/transports + checkpointer. Idempotent."""
        # Drop the checkpointer first so its pool drains before the
        # AsyncExitStack tears down anything that might have observed it.
        if self._checkpointer_close is not None:
            try:
                await self._checkpointer_close()
            except Exception:  # noqa: BLE001
                pass
            self._checkpointer_close = None
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

    def _thread_config(self, incident_id: str) -> dict:
        """Build the LangGraph ``config`` dict for a per-session thread.

        With a checkpointer attached, every ``ainvoke`` / ``astream_events``
        call must carry a ``configurable.thread_id`` so LangGraph can scope
        the durable state. Using the incident id keeps each INC's graph
        state isolated and lets the checkpointer act as a resume index.
        """
        return {"configurable": {"thread_id": incident_id}}

    def get_session(self, incident_id: str) -> dict:
        """Load a session by id and return its serialized form."""
        return self.store.load(incident_id).model_dump()

    def get_incident(self, incident_id: str) -> dict:
        """Deprecated alias for ``get_session``."""
        return self.get_session(incident_id)

    def list_recent_sessions(self, limit: int = 20) -> list[dict]:
        """List the most recent sessions, newest first."""
        return [i.model_dump() for i in self.store.list_recent(limit)]

    def list_recent_incidents(self, limit: int = 20) -> list[dict]:
        """Deprecated alias for ``list_recent_sessions``."""
        return self.list_recent_sessions(limit)

    def delete_session(self, incident_id: str) -> dict:
        """Soft-delete a session and return its final serialized form."""
        return self.store.delete(incident_id).model_dump()

    def delete_incident(self, incident_id: str) -> dict:
        """Deprecated alias for ``delete_session``."""
        return self.delete_session(incident_id)

    async def _run_dedup_check(self, inc) -> bool:
        """Run the ``dedup_check`` lifecycle hook.

        Returns ``True`` iff the session was confirmed as a duplicate
        and marked accordingly — callers should skip the agent graph in
        that case. Idempotent: a second invocation on a session that
        already has ``status="duplicate"`` is a no-op (returns ``True``).

        Failure modes (history error, LLM error, malformed structured
        output) all degrade to "not a duplicate" so dedup never crashes
        intake. Uses ``getattr`` for defensive access so unit-test stubs
        that build the orchestrator via ``__new__`` (bypassing
        ``__init__``) still get a working passthrough.
        """
        pipeline = getattr(self, "dedup_pipeline", None)
        history = getattr(self, "history", None)
        if pipeline is None:
            return False
        if getattr(inc, "status", None) == "duplicate":
            return True
        if history is None:
            return False
        try:
            result: DedupResult = await pipeline.run(
                session=inc, history_store=history,
            )
        except Exception:  # noqa: BLE001 — dedup must never crash intake
            return False
        if not result.matched:
            return False
        # Mark + persist. The session row is non-destructively linked to
        # the matched parent; the agent graph is skipped.
        inc.status = "duplicate"
        inc.parent_session_id = result.parent_session_id
        if result.decision is not None:
            inc.dedup_rationale = result.decision.rationale
        self.store.save(inc)
        return True

    async def start_session(self, *, query: str,
                            state_overrides: dict | None = None,
                            environment: str | None = None,
                            submitter: dict | None = None,
                            reporter_id: str | None = None,
                            reporter_team: str | None = None,
                            trigger: "TriggerInfo | None" = None) -> str:
        """Start a new agent session and run the entry agent.

        ``state_overrides`` is a free-form dict of domain-specific
        fields the app stamps onto the new session row. The framework
        only projects ``environment`` onto the storage column (the row
        schema's last domain leak); other keys flow through the
        app-specific MCP tools (or, in future, ``state_cls(...)``
        kwargs once the row schema is fully generic).

        ``submitter`` is a free-form dict the app interprets. For
        incident-management it is ``{"id": "...", "team": "..."}``; for
        other apps it can carry app-specific keys (e.g. code-review's
        ``{"id": "<github-username>", "pr_url": "..."}``). The framework
        only projects ``id``/``team`` onto the row's reporter columns;
        apps unpack the rest via their own MCP tools.

        Deprecated kwargs (coerced into ``state_overrides`` / ``submitter``
        and warned about):
          * ``environment`` -> ``state_overrides={"environment": ...}``
          * ``reporter_id`` / ``reporter_team`` -> ``submitter={"id": ...,
            "team": ...}``

        Passing both a generic kwarg and its legacy partner raises
        ``TypeError``.

        ``trigger`` is the optional provenance record from
        :mod:`runtime.triggers`. When supplied, ``name``/``transport``/
        ``target_app`` are written to ``inc.findings['trigger']`` for
        post-hoc audit; the orchestrator does not branch on its
        contents.

        If the dedup pipeline is configured and stage 2 confirms a
        duplicate of a prior closed session, the new session is marked
        ``status="duplicate"`` with ``parent_session_id`` set and the
        agent graph is skipped entirely.
        """
        state_overrides = _coerce_state_overrides(state_overrides, environment)
        submitter = _coerce_submitter(submitter, reporter_id, reporter_team)
        sub_id = (submitter or {}).get("id", "user-mock")
        sub_team = (submitter or {}).get("team", "platform")
        env = (state_overrides or {}).get("environment", "")
        inc = self.store.create(query=query, environment=env,
                                reporter_id=sub_id, reporter_team=sub_team)
        if trigger is not None:
            inc.findings["trigger"] = {
                "name": trigger.name,
                "transport": trigger.transport,
                "target_app": trigger.target_app,
                "received_at": trigger.received_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self.store.save(inc)
        # dedup_check before the graph fires.
        if await self._run_dedup_check(inc):
            return inc.id
        await self.graph.ainvoke(
            GraphState(session=inc, next_route=None,
                       last_agent=None, error=None),
            config=self._thread_config(inc.id),
        )
        return inc.id

    async def start_investigation(self, *, query: str, environment: str,
                                  reporter_id: str = "user-mock",
                                  reporter_team: str = "platform") -> str:
        """Deprecated alias for ``start_session``.

        Coerces the legacy positional surface into the generic
        ``submitter`` + ``state_overrides`` kwargs so the runtime
        deprecation paths never fire from the alias.
        """
        return await self.start_session(
            query=query,
            state_overrides={"environment": environment},
            submitter={"id": reporter_id, "team": reporter_team},
        )

    async def stream_session(self, *, query: str, environment: str,
                             reporter_id: str = "user-mock",
                             reporter_team: str = "platform"
                             ) -> AsyncIterator[dict]:
        """Start a new session and stream UI events as it runs.

        Internally builds a ``submitter`` dict so the row's reporter
        columns are populated through the same coercion path
        ``start_session`` uses.
        """
        sub = {"id": reporter_id, "team": reporter_team}
        inc = self.store.create(
            query=query,
            environment=environment,
            reporter_id=sub["id"],
            reporter_team=sub["team"],
        )
        yield {"event": "investigation_started", "incident_id": inc.id,
               "ts": _event_ts()}
        # dedup_check before the graph fires. Surface a one-shot
        # ``dedup_matched`` event so the UI can render the "marked
        # duplicate" banner without polling.
        if await self._run_dedup_check(inc):
            yield {"event": "dedup_matched", "incident_id": inc.id,
                   "parent_session_id": inc.parent_session_id,
                   "ts": _event_ts()}
            yield {"event": "investigation_completed", "incident_id": inc.id,
                   "ts": _event_ts()}
            return
        async for ev in self.graph.astream_events(
            GraphState(session=inc, next_route=None, last_agent=None, error=None),
            version="v2",
            config=self._thread_config(inc.id),
        ):
            yield self._to_ui_event(ev, inc.id)
        yield {"event": "investigation_completed", "incident_id": inc.id, "ts": _event_ts()}

    async def stream_investigation(self, *, query: str, environment: str,
                                   reporter_id: str = "user-mock",
                                   reporter_team: str = "platform"
                                   ) -> AsyncIterator[dict]:
        """Deprecated alias for ``stream_session``.

        Forwards the legacy positional surface into ``stream_session``;
        the underlying flow already coerces the reporter pair into
        a submitter dict internally so no runtime deprecation fires.
        """
        async for event in self.stream_session(
            query=query, environment=environment,
            reporter_id=reporter_id, reporter_team=reporter_team,
        ):
            yield event

    async def resume_session(self, incident_id: str,
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
               "action": action, "ts": _event_ts()}

        inc = self.store.load(incident_id)

        # Guard: only paused INCs are resumable. A resolved/stopped/escalated
        # INC must not be advanced again — that would silently corrupt state
        # (e.g. re-pinging on-call after the incident has already closed).
        if inc.status != "awaiting_input":
            yield {"event": "resume_rejected", "incident_id": incident_id,
                   "reason": f"not_awaiting_input (status={inc.status})",
                   "ts": _event_ts()}
            return

        if action == "stop":
            inc.status = "stopped"
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "stopped", "ts": _event_ts()}
            return

        if action == "escalate":
            team = decision.get("team") or "platform-oncall"
            allowed = list(self.framework_cfg.escalation_teams)
            if team not in allowed:
                # Reject the request entirely. The INC stays awaiting_input
                # so the user can retry with a valid team. Logging the
                # allowed roster on the event makes it actionable in the UI.
                yield {"event": "resume_rejected", "incident_id": incident_id,
                       "reason": (
                           f"team '{team}' not in allowed escalation_teams "
                           f"({allowed})"
                       ),
                       "ts": _event_ts()}
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
                ts=_event_ts(),
            ))
            inc.status = "escalated"
            inc.extra_fields["escalated_to"] = team
            inc.pending_intervention = None
            self.store.save(inc)
            yield {"event": "resume_completed", "incident_id": incident_id,
                   "status": "escalated", "team": team, "ts": _event_ts()}
            return

        if action == "resume_with_input":
            async for ev in self._resume_with_input(incident_id, inc, decision):
                yield ev
            return

        raise ValueError(f"Unknown resume action: {action!r}")

    async def resume_investigation(self, incident_id: str,
                                   decision: dict) -> AsyncIterator[dict]:
        """Deprecated alias for ``resume_session``."""
        async for event in self.resume_session(incident_id, decision):
            yield event

    async def _resume_with_input(self, incident_id: str, inc, decision: dict):
        """Handle the resume_with_input action.

        Drives the *same* compiled graph with ``Command(resume=user_text)``
        against the paused thread_id. The checkpointer rehydrates the
        suspended gate node; the gate body appends the input to
        ``session.user_inputs``, clears ``pending_intervention``, and
        falls through to the gated downstream target. On failure the
        intervention payload is restored so the UI can reprompt.
        """
        user_text = (decision.get("input") or "").strip()
        if not user_text:
            raise ValueError("resume_with_input requires a non-empty 'input'")
        # Snapshot the intervention payload BEFORE we hand off to the
        # graph. The gate's post-resume continuation clears it on
        # the way out; if the downstream graph blows up we restore from
        # this snapshot so the UI can reprompt the user.
        saved_pi = inc.pending_intervention
        try:
            async for ev in self.graph.astream_events(
                Command(resume=user_text),
                version="v2",
                config=self._thread_config(incident_id),
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
                   "error": str(exc), "ts": _event_ts()}
            return
        final = self.store.load(incident_id)
        yield {"event": "resume_completed", "incident_id": incident_id,
               "status": final.status, "ts": _event_ts()}

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
            "ts": _event_ts(),
            "data": raw.get("data"),
        }


def _event_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ====== module: runtime/api.py ======

def _resolve_environments(dotted: str | None) -> list[str]:
    """Resolve ``RuntimeConfig.environments_provider_path`` to a list.

    Returns an empty list when ``dotted`` is unset (apps that don't
    expose an environments roster). Provider callables must return a
    sequence of strings; anything else raises ``TypeError``.
    """
    if dotted is None:
        return []
    if ":" not in dotted:
        raise ValueError(
            f"environments_provider_path={dotted!r} must be in "
            "'module.path:callable' form"
        )
    import importlib
    module_name, _, attr = dotted.partition(":")
    mod = importlib.import_module(module_name)
    provider = getattr(mod, attr)
    envs = provider()
    if not isinstance(envs, (list, tuple)):
        raise TypeError(
            f"environments provider {dotted!r} returned "
            f"{type(envs).__name__}; expected list[str]"
        )
    return [str(e) for e in envs]


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


# ---------------------------------------------------------------------------
# Multi-session schemas
# ---------------------------------------------------------------------------


class SessionStartBody(BaseModel):
    query: str
    environment: str
    # Generic submitter dict — the framework projects ``id``/``team``
    # onto the row's reporter columns; apps interpret the rest. The
    # legacy ``reporter_id`` / ``reporter_team`` fields were removed
    # from this body because the deprecation path on the runtime
    # would emit a warning at every request — production-log noise.
    submitter: dict | None = None


class SessionStartResponse(BaseModel):
    session_id: str


class SessionStatus(BaseModel):
    session_id: str
    status: str
    started_at: str
    current_agent: str | None = None


# ---------------------------------------------------------------------------
# HITL approval schemas (risk-rated tool gateway)
# ---------------------------------------------------------------------------


class ApprovalDecisionBody(BaseModel):
    """Request body for ``POST /sessions/{sid}/approvals/{tool_call_id}``.

    The wrap_tool closure interprets ``decision`` as either ``approve``
    (run the tool, audit) or ``reject`` (skip the tool, audit rejection).
    ``approver`` is the operator id; ``rationale`` is optional free text.
    """

    decision: Literal["approve", "reject"]
    approver: str
    rationale: str | None = None


class PendingApproval(BaseModel):
    """Snapshot of one pending tool approval read from session.tool_calls."""

    tool_call_id: str
    agent: str
    tool: str
    args: dict
    ts: str


def _make_lifespan(cfg: AppConfig):
    """Build the lifespan context manager for an app constructed with ``cfg``.

    Constructs the :class:`runtime.service.OrchestratorService` singleton,
    starts its background loop, eagerly builds the underlying
    :class:`runtime.orchestrator.Orchestrator` (so legacy routes that
    expect ``app.state.orchestrator`` keep working), and builds the
    :class:`runtime.triggers.TriggerRegistry` from ``cfg.triggers``. The
    webhook router is mounted on the FastAPI app here; APScheduler is
    started by the schedule transport's ``start``.

    On shutdown, the registry's ``stop_all`` runs first (drains
    APScheduler), then ``service.shutdown()`` tears the orchestrator down.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Lazy import: ``runtime.service`` transitively pulls a lot of
        # heavyweight modules (FastMCP, SQLAlchemy). Importing at function
        # scope keeps ``import runtime.api`` cheap for tests/tools that
        # only need ``build_app``.




        svc = OrchestratorService.get_or_create(cfg)
        svc.start()
        # Eagerly build the shared Orchestrator so legacy routes can read
        # it via ``app.state.orchestrator`` without racing on the
        # lazy-build path. ``_ensure_orchestrator`` is on the loop thread,
        # so we hop through the sync bridge.
        orch = svc.submit_and_wait(svc._ensure_orchestrator(), timeout=30.0)
        app.state.service = svc
        app.state.orchestrator = orch
        # Environments roster is app-specific (incident-management has
        # production/staging/dev/local; code-review doesn't expose one).
        # Read it from the YAML's top-level ``environments:`` block;
        # fall back to the legacy ``environments_provider_path`` callable
        # for deployments that still wire it.
        if cfg.environments:
            app.state.environments = list(cfg.environments)
        else:
            app.state.environments = _resolve_environments(
                getattr(cfg.runtime, "environments_provider_path", None),
            )

        # ------------------------------------------------------------
        # Build & start the trigger registry
        # ------------------------------------------------------------
        plugin_transports = getattr(app.state, "plugin_transports", None)

        async def _start_session_fn(**kwargs):
            # The registry's dispatch sink. Bridges through the
            # OrchestratorService so we share the one MCP pool / one
            # orchestrator / one DB engine that the rest of the app uses.
            return svc.submit_and_wait(
                _trigger_dispatch(svc, kwargs), timeout=60.0
            )

        async def _trigger_dispatch(service, kwargs):
            # ``svc.start_session`` is sync (returns the session id); the
            # registry awaits us. Trampoline through the loop's
            # default executor.
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: service.start_session(**kwargs)
            )

        idempotency: IdempotencyStore | None = None
        if cfg.triggers:
            try:
                idempotency = IdempotencyStore(orch.store.engine)
            except AttributeError:
                # Older test stubs don't expose ``store.engine``; the
                # registry tolerates ``idempotency=None`` (no caching).
                idempotency = None
        registry = TriggerRegistry.create(
            list(cfg.triggers),
            start_session_fn=_start_session_fn,
            idempotency=idempotency,
            plugin_transports=plugin_transports,
        )
        app.state.trigger_registry = registry
        await registry.start_all()
        # Mount any webhook routers onto the FastAPI app so the routes
        # become live.
        for t in registry.transports:
            if isinstance(t, WebhookTransport):
                app.include_router(t.router)
        try:
            yield
        finally:
            try:
                await registry.stop_all()
            except Exception:  # noqa: BLE001
                pass
            # ``shutdown()`` cancels in-flight session tasks, closes the
            # underlying Orchestrator + MCP pool, joins the loop thread,
            # and resets the process-singleton.
            svc.shutdown()
    return _lifespan


def build_app(cfg: AppConfig) -> FastAPI:
    """Construct the FastAPI app. Synchronous.

    The :class:`OrchestratorService` and its underlying
    :class:`Orchestrator` are created during the app's startup lifespan
    and are reachable as ``app.state.service`` / ``app.state.orchestrator``
    from any route handler.
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
    async def investigate(req: InvestigateRequest, request: Request) -> InvestigateResponse:
        """Legacy alias for ``POST /sessions`` — kept for back-compat.

        .. deprecated::
            Prefer ``POST /sessions``. This route now delegates to
            :meth:`OrchestratorService.start_session` so old clients keep
            working with the long-lived service backing.
        """
        svc = request.app.state.service
        # Coerce the legacy HTTP body into the generic runtime kwargs
        # BEFORE delegating, so the runtime's deprecation path never
        # fires on a hot HTTP route. Production logs stay quiet.
        try:
            sid = svc.start_session(
                query=req.query,
                state_overrides={"environment": req.environment},
                submitter={
                    "id": req.reporter_id,
                    "team": req.reporter_team,
                },
            )
        except Exception as e:  # noqa: BLE001
            # ``SessionCapExceeded`` is matched by class name to avoid a
            # hard import dependency at module-load time.
            if e.__class__.__name__ == "SessionCapExceeded":
                raise HTTPException(status_code=429, detail=str(e)) from e
            raise
        return InvestigateResponse(incident_id=sid)

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

    # ------------------------------------------------------------------
    # Multi-session endpoints
    # ------------------------------------------------------------------

    @fastapi_app.post(
        "/sessions",
        response_model=SessionStartResponse,
        status_code=201,
    )
    async def start_session_endpoint(
        body: SessionStartBody, request: Request
    ) -> SessionStartResponse:
        """Start a new long-running session. Returns ``201 {session_id}``.

        Returns ``429`` if the configured concurrent-session cap is hit
        (raised by ``OrchestratorService.start_session``). The exception
        class is matched by name so this handler does not depend on a
        hard import.
        """
        svc = request.app.state.service
        try:
            sid = svc.start_session(
                query=body.query,
                state_overrides={"environment": body.environment},
                submitter=body.submitter,
            )
        except Exception as e:  # noqa: BLE001
            if e.__class__.__name__ == "SessionCapExceeded":
                raise HTTPException(status_code=429, detail=str(e)) from e
            raise
        return SessionStartResponse(session_id=sid)

    @fastapi_app.get("/sessions", response_model=list[SessionStatus])
    async def list_sessions_endpoint(request: Request) -> list[SessionStatus]:
        """Snapshot of in-flight sessions (running / awaiting_input / error)."""
        svc = request.app.state.service
        return [SessionStatus(**row) for row in svc.list_active_sessions()]

    # ------------------------------------------------------------------
    # HITL approval endpoints (risk-rated tool gateway)
    # ------------------------------------------------------------------

    @fastapi_app.get(
        "/sessions/{session_id}/approvals",
        response_model=list[PendingApproval],
    )
    async def list_pending_approvals(
        session_id: str, request: Request
    ) -> list[PendingApproval]:
        """Return the list of pending tool approvals for a session.

        Filters ``session.tool_calls`` to entries with
        ``status="pending_approval"``. Returns an empty list when the
        session has no pending approvals; ``404`` when the session id
        is unknown.
        """
        svc = request.app.state.service
        orch = request.app.state.orchestrator
        try:
            inc = orch.store.load(session_id)
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            # ``ValueError`` covers the SessionStore id-format guard
            # (``Invalid incident id ...``) which we treat as a 404
            # at the API boundary — the client passed an id that
            # cannot exist, semantically equivalent to "not found".
            raise HTTPException(
                status_code=404, detail="session not found"
            ) from e
        # Defensive: ``svc`` is unused here today — the read goes through
        # the orchestrator's store. We keep the reference so a future
        # observability hook (per-call metrics) lives next to the read.
        _ = svc
        out: list[PendingApproval] = []
        for idx, tc in enumerate(inc.tool_calls):
            if tc.status == "pending_approval":
                # tool_call_id is the index in the audit list; stable
                # within the lifetime of the session because tool calls
                # are append-only.
                out.append(PendingApproval(
                    tool_call_id=str(idx),
                    agent=tc.agent,
                    tool=tc.tool,
                    args=tc.args,
                    ts=tc.ts,
                ))
        return out

    @fastapi_app.post(
        "/sessions/{session_id}/approvals/{tool_call_id}",
        status_code=200,
    )
    async def submit_approval_decision(
        session_id: str,
        tool_call_id: str,
        body: ApprovalDecisionBody,
        request: Request,
    ) -> dict:
        """Resolve a pending tool approval by resuming the paused graph.

        Resumes via ``Command(resume={decision, approver, rationale})``
        against the session's thread_id. The wrap_tool closure reads the
        resume value and either runs the tool (``approve``) or short-
        circuits with ``status="rejected"`` (``reject``).
        """
        svc = request.app.state.service
        orch = request.app.state.orchestrator
        try:
            orch.store.load(session_id)
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            raise HTTPException(
                status_code=404, detail="session not found"
            ) from e

        decision_payload = {
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

        async def _resume() -> None:
            from langgraph.types import Command

            await orch.graph.ainvoke(
                Command(resume=decision_payload),
                config=orch._thread_config(session_id),
            )

        # Submit the resume onto the long-lived service loop so we
        # don't fight the lifespan thread for the same FastMCP/SQLite
        # transports. We use the async bridge (``submit_async``) rather
        # than ``submit_and_wait`` because this handler may run on the
        # very loop the service is hosting (FastAPI under
        # ``httpx.AsyncClient + ASGITransport``, or any single-loop
        # deployment): blocking that loop while waiting for work
        # scheduled onto it would deadlock.
        await svc.submit_async(_resume())
        return {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "decision": body.decision,
            "approver": body.approver,
            "rationale": body.rationale,
        }

    @fastapi_app.delete("/sessions/{session_id}", status_code=204)
    async def stop_session_endpoint(
        session_id: str, request: Request
    ) -> Response:
        """Cancel an in-flight session and evict its registry entry.

        Returns ``501 Not Implemented`` when the service does not expose
        ``stop_session`` rather than crashing.
        """
        svc = request.app.state.service
        if not hasattr(svc, "stop_session"):
            raise HTTPException(
                status_code=501,
                detail="stop_session not available",
            )
        try:
            svc.stop_session(session_id)
        except Exception as e:  # noqa: BLE001
            # Translate a "session not found" condition into 404 when
            # the underlying error class is recognisable. Otherwise
            # re-raise.
            name = e.__class__.__name__
            if name in {"KeyError", "SessionNotFound"}:
                raise HTTPException(status_code=404, detail=str(e)) from e
            raise
        return Response(status_code=204)

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

# ====== module: runtime/api_dedup.py ======

class UnDuplicateRequest(BaseModel):
    """Request body for the retraction endpoint.

    Both fields are optional. ``retracted_by`` is self-claimed (the
    framework does not authenticate the operator id); ``note`` is free
    text persisted on the audit row.
    """

    retracted_by: str | None = None
    note: str | None = Field(default=None, max_length=2000)


class UnDuplicateResponse(BaseModel):
    """Successful retraction payload."""

    session_id: str
    status: str
    parent_session_id: str | None
    original_match_id: str
    retracted_by: str | None = None
    note: str | None = None


def register_dedup_routes(
    app: FastAPI,
    *,
    store_provider: Callable[[], Any],
) -> None:
    """Register the un-duplicate route on ``app``.

    ``store_provider`` is a no-arg callable that returns the live
    ``SessionStore``. We accept a callable (rather than the store
    directly) so apps can defer construction until first request — the
    route handler itself never caches the store.
    """

    @app.post(
        "/sessions/{session_id}/un-duplicate",
        response_model=UnDuplicateResponse,
        status_code=200,
        tags=["dedup"],
    )
    async def un_duplicate(
        session_id: str,
        body: UnDuplicateRequest | None = None,
    ) -> UnDuplicateResponse:
        store = store_provider()
        # Pre-flight: capture the parent id BEFORE the flip so we can
        # echo it on the response. The store does the same capture
        # internally for the audit row.
        try:
            current = store.load(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            # ``load`` validates the id format; map malformed -> 404.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if current.status != "duplicate":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not a duplicate",
                    "status": current.status,
                },
            )
        original_match_id = current.parent_session_id or ""
        payload = body or UnDuplicateRequest()
        try:
            updated = store.un_duplicate(
                session_id,
                retracted_by=payload.retracted_by,
                note=payload.note,
            )
        except FileNotFoundError as exc:
            # Race: deleted between load and un_duplicate.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            # Race: status flipped between load and un_duplicate.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return UnDuplicateResponse(
            session_id=updated.id,
            status=updated.status,
            parent_session_id=updated.parent_session_id,
            original_match_id=original_match_id,
            retracted_by=payload.retracted_by,
            note=payload.note,
        )
