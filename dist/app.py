from __future__ import annotations
# ----- imports for runtime/errors.py -----
"""Typed runtime errors. Phase 13 lands the LLM-call surface; future
hardening (HARD-04 silent-failure sweep, HARD-03 pyright flip,
real-LLM follow-ups) extends here.

Importable as ``from runtime.errors import LLMTimeoutError, LLMConfigError``.
"""



# ----- imports for runtime/terminal_tools.py -----
"""Generic terminal-tool registry types.

Apps register their terminal-tool rules and status vocabulary via
``OrchestratorConfig.terminal_tools`` / ``OrchestratorConfig.statuses``;
the framework reads these models without knowing app-specific tool
or status names. Cf. .planning/phases/06-generic-terminal-tool-registry/
06-CONTEXT.md (D-06-01, D-06-02, D-06-05).
"""


from typing import Literal

from pydantic import BaseModel, Field


# ----- imports for runtime/config.py -----
"""Config schemas for the orchestrator."""

import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml




# Session-id prefix grammar. The framework mints session ids of the form
# ``{PREFIX}-YYYYMMDD-NNN`` (see ``runtime.state.Session.id_format``);
# the prefix is the only piece an app picks. Allow alphanumerics + hyphens,
# bound the length so the id stays scannable in logs and DB indexes, and
# refuse the empty string so the resulting id never starts with a stray ``-``.
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

Phase 13 (HARD-01 / HARD-05): every chat + embedding HTTP call is bounded
by an effective ``request_timeout`` resolved as
``provider.request_timeout if not None else default_llm_request_timeout``
(default 120.0s on ``OrchestratorConfig``). The native langchain timeout
knob is wired AND an ``asyncio.wait_for`` wrapper raises
``LLMTimeoutError(provider, model, elapsed_ms)`` on hang -- defence in
depth against partial-byte stalls where the httpx layer doesn't fire.
The hardcoded public-Ollama fallback is removed; ollama providers
must declare ``base_url`` (validated at config-load via
``LLMConfigError``).
"""

import asyncio
import time
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
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, text
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
from pydantic import SecretStr



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

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlaSession



# Mirrors the bound on ``SessionStore.StateT`` — tightened from
# ``BaseModel`` to ``runtime.state.Session`` in Phase 19 (HARD-03) so
# pyright sees the typed fields (``id``, ``status``, ``deleted_at`` …)
# this store reads. The resolver in :mod:`runtime.state_resolver`
# already enforces a ``Session`` subclass at config time, and every
# in-tree caller passes either bare ``Session`` or a ``Session``
# subclass.
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

from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session as SqlSession



# The legacy ``INC-YYYYMMDD-NNN`` pattern stays here for back-compat
# validation against on-disk rows minted before the ``Session.id_format``
# hook existed. New rows are validated by ``_SESSION_ID_RE`` which
# accepts any ``PREFIX-YYYYMMDD-NNN`` shape the app's ``id_format`` may
# emit (e.g. ``CR-...`` for code-review).
# ----- imports for runtime/storage/event_log.py -----
"""Append-only session event log.

Events drive the status finalizer's inference (e.g. a registered
``<terminal_tool>`` event appearing in the log -> session reached
the corresponding terminal status). They are never mutated or
deleted.
"""


from dataclasses import dataclass
from typing import Any, Iterator, Literal, get_args

from sqlalchemy.orm import Session


# M2 (per-step telemetry): stable kind vocabulary for the event log.
# Adding a new kind without updating callers is intentional — but
# emitting a kind outside this Literal is a typo and raises at
# record() time so the typo doesn't silently pollute the log.
# ----- imports for runtime/storage/migrations.py -----
"""Idempotent migrations for the JSON-shaped row payloads.

Fills the per-call audit fields on :class:`runtime.state.ToolCall` for
legacy rows. The risk-rated tool gateway uses five optional audit fields:

  * ``risk``          — ``"low" | "medium" | "high" | None``
  * ``status``        — ``ToolStatus`` literal (default ``"executed"``)
  * ``approver``      — operator id, set when status in {approved, rejected}
  * ``approved_at``   — ISO-8601 timestamp of the decision
  * ``approval_rationale`` — free-text justification

Older rows in the ``incidents.tool_calls`` JSON column lack these
fields. Pydantic hydrates the missing keys with their defaults at read
time so reading is already back-compat — but the on-disk JSON still
shows the legacy shape until something rewrites the row.

This migration walks every session, normalises the JSON-shaped
``tool_calls`` list to the current audit schema, and saves the row back
when (and only when) at least one entry changed. Idempotent — running
twice is safe (the second pass is a no-op because every row already
has the fields).

The function operates on the row's JSON list directly (not via the
``ToolCall`` Pydantic model) so we don't accidentally widen the
migration's contract — for example, dropping unknown extra keys via
Pydantic's ``extra='ignore'`` would silently delete forward-compat
fields in a downgrade scenario. JSON-walk is conservative: only fill
what's missing; leave everything else alone.
"""


from typing import Any, Iterable

from sqlalchemy import inspect, text


# Columns added after the initial schema. Each entry is
# ``(column_name, sql_type, default_clause_or_None)``. SQLite ``ADD
# COLUMN`` cannot add a non-nullable column without a constant default,
# so every entry here is nullable — Pydantic hydrates the missing keys
# at read time. Append-only: never reorder, never delete. Removing a
# column needs a separate destructive migration with explicit sign-off.
# ----- imports for runtime/storage/lesson_store.py -----
"""M5: vector-indexed corpus of past resolved sessions ("lessons").

``LessonStore`` mirrors :class:`HistoryStore`'s public surface — ``add``
persists a row + vector embedding, ``find_similar`` runs k-NN over the
corpus and returns the top hits above a threshold.

The relational rows live in ``session_lessons`` (see
:class:`SessionLessonRow`); the embeddings live in whatever LangChain
``VectorStore`` the caller wires (FAISS dir or pgvector collection,
typically ``<vector.path>/lessons`` or collection ``lessons``).

Both writes are best-effort serialised: the relational row is persisted
FIRST so a vector-store failure leaves a recoverable on-disk record
the M7 refresher can re-embed.
"""


import logging



# ----- imports for runtime/learning/extractor.py -----
"""M5: lesson extractor — distills a terminal session's event log +
final session row into a :class:`SessionLessonRow` suitable for the
:class:`LessonStore` corpus.

Pure data-flow: walks ``event_log.iter_for(session.id)`` for tool calls,
reads ``session.agents_run`` for the final confidence + summary, and
composes a canonical ``embedding_text`` string the vector backend
embeds for retrieval. The same input session + event log always
produces the same ``embedding_text`` (modulo the ``created_at``
timestamp and uuid id) so M7's idempotency check can compare
``embedding_text`` to decide whether a re-extract is needed.
"""


from typing import Any, Optional





# ----- imports for runtime/learning/scheduler.py -----
"""M7: nightly batch refresher for the lesson corpus.

Runs an APScheduler ``AsyncIOScheduler`` that fires on
:attr:`FrameworkAppConfig.lesson_refresh_cron` (default ``0 3 * * *`` —
03:00 UTC daily). On each tick it walks the recently-terminated
sessions inside the configured window, dispatches
:class:`LessonExtractor.extract` for any that don't already have a
current-version lesson row, and persists the result via the existing
:class:`LessonStore`.

Idempotency contract: rerunning :meth:`run_once` after a previous
successful pass produces zero new rows (the source_session_id +
``provenance.extractor_version`` pair is unique-by-content). When the
extractor version bumps in a future release, the refresher writes a
fresh row — older lessons stay queryable (append-only corpus).

Tests drive the refresher synchronously via :meth:`run_once`; the
cron loop only exists to fire ``run_once`` on a schedule.
"""


from datetime import datetime, timedelta, timezone






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



# ----- imports for runtime/service.py -----
"""Long-lived orchestrator service.

Owns a background asyncio event loop and a shared FastMCP client pool.
All session execution will run as asyncio tasks on this loop. Sync callers
(Streamlit, FastAPI request handlers, CLI) submit coroutines via
``submit(coro) -> concurrent.futures.Future``.

Lifecycle::

    svc = OrchestratorService.get_or_create(cfg)
    svc.start()    # spins up background thread + loop
    fut = svc.submit(some_coro)
    result = fut.result(timeout=30)
    svc.shutdown() # cancels in-flight tasks, closes MCP clients, joins thread

Capabilities:
  - Skeleton + singleton + start/shutdown lifecycle.
  - ``submit()`` / ``submit_and_wait()`` thread-safe bridge.
  - Shared ``MCPClientPool`` with per-server ``asyncio.Lock``.
  - ``start_session()`` schedules a per-session asyncio task on the
    service's loop and returns the session id immediately (the agent run
    continues in the background). Active tasks are tracked in an
    in-memory registry that evicts on completion / cancellation.
  - ``list_active_sessions()`` returns a thread-safe snapshot of
    the in-flight registry; the snapshot coroutine runs on the loop so
    readers from any thread see a point-in-time consistent view.
  - ``stop_session(sid)`` cancels the in-flight task, waits up
    to 5 s for graceful exit, and persists ``status="stopped"`` on the
    row (clearing ``pending_intervention``). Idempotent — a no-op for
    unknown ids or already-completed sessions.
  - Hard cap on concurrent sessions. ``start_session`` raises
    ``SessionCapExceeded`` once ``len(self._registry) >=
    self.max_concurrent_sessions``. Fail fast; queueing is not supported.

The singleton is process-scoped and reset on ``shutdown()`` so that test
suites can build, tear down, and rebuild the service without leaking
state across cases.
"""


import concurrent.futures
import threading
from typing import Any, Awaitable, Coroutine, TypeVar, cast



# ----- imports for runtime/agents/turn_output.py -----
"""Phase 10 (FOC-03) — AgentTurnOutput envelope + reconciliation helpers.

The envelope is the structural contract every responsive agent invocation
must satisfy: content + confidence in [0,1] + confidence_rationale + optional
signal. The framework wires it as ``response_format=AgentTurnOutput`` into
``langchain.agents.create_agent`` (see Phase 15 / LLM-COMPAT-01); the
agent loop terminates on the same turn the LLM emits the envelope-shaped
tool call, populating ``result["structured_response"]``, which the
framework reads and persists onto the ``AgentRun`` row.

D-10-02 — pydantic envelope wrapped via ``response_format``.
D-10-03 — when a typed-terminal-tool was called this turn, the framework
reconciles its ``confidence`` arg against the envelope's. Tolerance 0.05
inclusive; tool-arg wins on mismatch with an INFO log.

This is a leaf module: no imports from ``runtime.graph`` or
``runtime.orchestrator``. Both of those depend on it; the dependency
graph is acyclic.
"""



from pydantic import BaseModel, ConfigDict, Field

# ----- imports for runtime/tools/gateway.py -----
"""Risk-rated tool gateway: pure resolver + ``BaseTool`` HITL wrapper.

The gateway sits between the ReAct agent and each tool the orchestrator
configures. It enforces the *hybrid* HITL policy resolved by
``effective_action``:

  ``auto``    -> call the underlying tool directly (no plumbing)
  ``notify``  -> call the tool, then persist a soft-notify audit entry
  ``approve`` -> raise ``langgraph.types.interrupt(...)`` BEFORE calling
                 the tool; on resume re-invoke

The resolver is a plain function with no I/O so it can be unit-tested
exhaustively without spinning up Pydantic Sessions, MCP servers, or a
LangGraph runtime. The wrapper is a closure factory deliberately built
inside ``make_agent_node`` so the closure captures the live ``Session``
per agent invocation (mitigation R2 in the Phase-4 plan).
"""


from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Literal




# ``GateDecision`` is imported lazily inside ``_evaluate_gate`` (function
# body) to avoid a runtime cycle (policy.py imports gateway types). The
# type-only import below lets pyright resolve the string-literal return
# annotation on ``_evaluate_gate`` without forming a real cycle.
# ----- imports for runtime/tools/arg_injection.py -----
"""Session-derived tool-arg injection (Phase 9 / FOC-01 / FOC-02).

Two responsibilities, one module:

1. :func:`strip_injected_params` — clones a ``BaseTool``'s args_schema with
   one or more parameters removed. The LLM only sees the stripped sig and
   therefore cannot hallucinate values for those params (D-09-01). The
   original tool is left untouched so direct downstream callers (tests,
   scripts, in-process MCP fixtures) keep working.

2. :func:`inject_injected_args` — at tool-invocation time, re-adds the
   real values resolved from the live :class:`runtime.state.Session` via
   the configured dotted paths. When the LLM still supplied a value for
   an injected arg, the framework's session-derived value wins and an
   INFO log captures the override (D-09-03).

The framework stays generic — apps declare which args to inject and from
where via :attr:`runtime.config.OrchestratorConfig.injected_args` (D-09-02).
"""



from pydantic import BaseModel, create_model



# Module-private logger. Tests assert against logger name
# ``"runtime.orchestrator"`` so the override-log line shows up alongside
# the rest of the orchestrator-side observability without requiring a
# separate caplog target.
# ----- imports for runtime/tools/approval_watchdog.py -----
"""Pending-approval timeout watchdog.

A high-risk tool call enters ``langgraph.types.interrupt()`` and the
session sits in ``awaiting_input`` indefinitely. Without a watchdog
the slot leaks against ``OrchestratorService.max_concurrent_sessions``
forever — the cap eventually starves out new traffic.

The :class:`ApprovalWatchdog` is an asyncio task that runs on the
service's background loop. Every ``poll_interval_seconds`` it:

  1. Snapshots the in-flight session registry.
  2. For each session whose row has ``status="awaiting_input"``,
     scans ``tool_calls`` for entries with ``status="pending_approval"``
     whose ``ts`` is older than ``approval_timeout_seconds``.
  3. Resumes each such session via ``Command(resume={"decision":
     "timeout", "approver": "system", "rationale": "approval window
     expired"})``. The wrapped tool's resume path updates the audit
     row to ``status="timeout"``.

Failures during polling (DB hiccup, malformed row) are logged and
swallowed so a single bad session cannot kill the watchdog.
"""


from typing import TYPE_CHECKING, Any


# ----- imports for runtime/policy.py -----
"""Pure HITL gating policy (Phase 11 / FOC-04).

The :func:`should_gate` function is the SOLE place the framework decides
whether a tool call requires human-in-the-loop approval. It composes
three orthogonal inputs:

  1. ``effective_action(tool_call.tool, env=session.environment,
     gateway_cfg=cfg.gateway)`` -- preserves the v1.0 PVC-08
     prefixed-form lookup invariant.
  2. ``session.environment`` -- gated when in
     ``cfg.gate_policy.gated_environments``.
  3. ``confidence`` -- gated when below
     ``cfg.gate_policy.confidence_threshold``.

Pure: same inputs always yield identical :class:`GateDecision`; no I/O,
no skill-prompt input, no mutation.

Precedence (descending):

  1. ``effective_action`` returns a value in
     ``cfg.gate_policy.gated_risk_actions``
     -> ``GateDecision(gate=True, reason="high_risk_tool")``
  2. ``session.environment`` in ``cfg.gate_policy.gated_environments``
     AND ``effective_action != "auto"``
     -> ``GateDecision(gate=True, reason="gated_env")``
  3. ``confidence`` is not None AND
     ``confidence < cfg.gate_policy.confidence_threshold``
     AND ``effective_action != "auto"``
     -> ``GateDecision(gate=True, reason="low_confidence")``
  4. otherwise -> ``GateDecision(gate=False, reason="auto")``

The literal ``"blocked"`` is reserved on :class:`GateDecision.reason`
for future hard-stop semantics; Phase 11 itself never returns it from a
production code path.
"""



from pydantic import BaseModel, ConfigDict


# Phase 12 (FOC-05) imports for should_retry policy (defined below).
import asyncio as _asyncio
import pydantic as _pydantic


# Phase 11 (FOC-04): forward-reference imports for the should_gate
# signature only; kept inside ``TYPE_CHECKING`` so the bundle's
# intra-import stripper does not remove a load-bearing import. The
# ``pass`` keeps the block syntactically valid after stripping.
# ----- imports for runtime/agents/responsive.py -----
"""Responsive agent kind — the today-default LLM agent.

A responsive skill is a LangGraph node that:

1. Builds a ReAct executor over the skill's ``tools`` and ``model``.
2. Invokes the executor with the live ``Session`` payload as a human
   message preamble.
3. Records ``ToolCall`` and ``AgentRun`` rows on the session, harvests
   the agent's confidence / signal / rationale, and decides the next
   route from ``skill.routes``.

This module owns only the node-factory entrypoint
(``make_agent_node``); the implementation reuses helpers in
:mod:`runtime.graph` so existing call sites and the gate node continue
to work unchanged. Supervisor and monitor factories live alongside it
under :mod:`runtime.agents` rather than piling more kinds into
``graph.py``.
"""


from typing import TYPE_CHECKING, Callable

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langgraph.errors import GraphInterrupt







# ----- imports for runtime/agents/supervisor.py -----
"""Supervisor agent kind — no-LLM router.

A supervisor skill is a LangGraph node that:

1. Reads the live ``Session`` plus the current dispatch depth.
2. Picks one or more subordinate agents per ``dispatch_strategy``:
   ``rule`` (deterministic, evaluated via the same safe-eval AST that
   gates monitor expressions) or ``llm`` (one short LLM call against
   ``dispatch_prompt``).
3. Emits a structured ``supervisor_dispatch`` log entry (no
   ``AgentRun`` row — supervisors are bookkeeping, not token-burning
   agents).
4. Returns ``next_route`` set to the chosen subordinate (or to
   ``__end__`` when the depth limit is hit).

The recursion depth is tracked in :class:`runtime.graph.GraphState`'s
``dispatch_depth`` field; if a supervisor would exceed
``skill.max_dispatch_depth`` the node aborts with a clean error
instead of recursing forever.

This is **not** a fan-out implementation; we always pick a single
target. Multi-target ``Send()`` is intentionally not supported.
"""


from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage



# ----- imports for runtime/agents/monitor.py -----
"""Monitor agent kind — out-of-band scheduled observer.

A monitor skill runs **outside** any session graph. The orchestrator
owns one :class:`MonitorRunner` (a singleton) which schedules registered
monitor skills on a small bounded
:class:`concurrent.futures.ThreadPoolExecutor`.
Each tick:

1. Calls every tool name in ``observe`` via the supplied callable
   (``observe_fn``); aggregates results into one dict keyed by tool.
2. Evaluates ``emit_signal_when`` against the observation using the
   stdlib safe-eval evaluator (R7).
3. If true, looks up ``trigger_target`` in the supplied trigger
   registry / fire callback and fires it with the observation as the
   payload.

APScheduler is intentionally *not* a dependency: the air-gapped target
env doesn't ship it (see ``rules/build.md``). We get away with a tiny
single-threaded scheduler thread because monitor schedules are coarse
(minute-resolution cron) and tool calls are dispatched into the
executor; the scheduler thread itself never blocks on tool I/O.
"""


from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


# ----- imports for runtime/graph.py -----
"""LangGraph state, routing helpers, and node runner."""

from typing import TYPE_CHECKING, Any, TypedDict, Callable, Awaitable

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


from typing import Any, Callable, Type, cast



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


from collections import OrderedDict
from datetime import datetime, timezone, timedelta

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

from pydantic import BaseModel, Field, model_validator

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



# ----- imports for runtime/locks.py -----
"""Per-session asyncio locks.

Status mutations on the same session must serialise. The registry hands
out one ``asyncio.Lock`` per session id; callers acquire it for the
duration of any read-modify-write block on that session's row.

The ``acquire`` context manager is **task-reentrant**: a coroutine that
already holds the lock for a given session id can re-enter it without
deadlocking. This matters when nested helpers (e.g. retry → finalize)
both want to take the lock — without re-entry, the inner ``acquire``
would wait forever for the outer to release.

Locks live in-process. Multi-process deployments must layer SQLite
``BEGIN IMMEDIATE`` (already configured) or move to row-level locking.
"""


from contextlib import asynccontextmanager
from typing import AsyncIterator


# ----- imports for runtime/skill_validator.py -----
"""Load-time validation of skill YAML against the live MCP registry.

Catches:
  * tools.local entries that reference a non-existent (server, tool)
    pair (typically typos that would silently make the tool invisible).
  * routes that omit ``when: default`` (would cause graph hangs at
    __end__ when no signal matches).
"""



# ----- imports for runtime/storage/checkpoint_gc.py -----
"""Garbage-collect orphaned LangGraph checkpoints.

When ``Orchestrator.retry_session`` rebinds a session to a new
``thread_id`` (e.g. ``INC-1:retry-1``), the original ``INC-1`` thread's
checkpoint becomes orphaned — no code path will ever resume it. Over
time these accumulate. ``gc_orphaned_checkpoints`` removes any
checkpoint whose ``thread_id`` does not reference an active session
(or a known retry suffix).

This is intentionally conservative: only checkpoints whose thread_id
prefix matches no live session row at all are removed.
"""


from sqlalchemy import text
from sqlalchemy.exc import OperationalError


# ----- imports for runtime/orchestrator.py -----
"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""

import warnings
from typing import Any, AsyncIterator, Generic, Type, TypeVar





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

from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


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




# ====== module: runtime/errors.py ======

class LLMTimeoutError(TimeoutError):
    """Raised when an LLM provider HTTP call exceeds request_timeout.

    Subclasses ``TimeoutError`` so ``runtime.policy._TRANSIENT_TYPES``
    auto-classifies it as transient via ``isinstance`` -- no policy.py
    edit needed (D-13-04).

    The ``__str__`` includes the substring ``"timed out"`` so existing
    string-matchers in ``runtime.graph._TRANSIENT_MARKERS`` and
    ``runtime.orchestrator._reconstruct_last_error`` also catch it
    without modification.
    """

    def __init__(self, provider: str, model: str, elapsed_ms: int) -> None:
        self.provider = provider
        self.model = model
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"LLM request timed out after {elapsed_ms}ms "
            f"(provider={provider}, model={model})"
        )


class LLMConfigError(ValueError):
    """Raised at config-load when a provider is missing a required field.

    Subclasses ``ValueError`` so pydantic ``@model_validator(mode='after')``
    propagates it cleanly into ``ValidationError`` (D-13-05).
    """

    def __init__(self, provider: str, missing_field: str) -> None:
        self.provider = provider
        self.missing_field = missing_field
        super().__init__(
            f"{provider} provider requires {missing_field!r}"
        )


__all__ = ["LLMTimeoutError", "LLMConfigError"]

# ====== module: runtime/terminal_tools.py ======

class TerminalToolRule(BaseModel):
    """Maps a terminal tool name to the session status it produces.

    ``tool_name`` matches both bare (``set_recommendation``) and prefixed
    (``<server>:set_recommendation``) MCP tool-call names — the framework
    does the suffix check.

    ``status`` must reference a name declared in the same
    ``OrchestratorConfig.statuses`` map; ``OrchestratorConfig``'s
    cross-field validator enforces this at config-load.

    ``extract_fields`` declares per-rule extra-metadata pulls. Each
    key is the destination field name on the session
    (``Session.extra_fields[<key>]``); each value is an ordered list
    of ``args.X`` / ``result.X`` lookup hints. The framework picks
    the first non-falsy match. Empty dict (default) means "no extra
    metadata to capture". Generalises the v1.0
    ``_extract_team(tc, team_keys)`` path; the same lookup syntax is
    preserved (D-06-02).

    ``match_args`` is an optional argument-value discriminator. When
    non-empty, the rule matches a tool call only if EVERY ``(key,
    value)`` pair in ``match_args`` matches ``tool_call.args[key]``
    exactly. Lets one tool name route to multiple statuses based on
    a discriminator argument (e.g. ``set_recommendation`` with
    ``recommendation=approve`` vs ``recommendation=request_changes``).
    Empty default = no arg dispatch; preserves the v1.0 single-rule
    shape (DECOUPLE-07 / D-08-03).
    """

    model_config = {"extra": "forbid"}

    tool_name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    extract_fields: dict[str, list[str]] = Field(default_factory=dict)
    match_args: dict[str, str] = Field(default_factory=dict)


StatusKind = Literal[
    "success",       # e.g. set_recommendation(approve) -> approved
    "failure",       # e.g. set_recommendation(request_changes) -> changes_requested
    "escalation",    # app-defined escalation terminal (e.g. <terminal_tool>)
    "needs_review",  # finalize fired with no rule match
    "pending",       # session in flight
]


class StatusDef(BaseModel):
    """Pydantic record of one app status.

    Framework reads ``terminal`` to decide finalize-vs-pending and
    ``kind`` to dispatch the needs_review fallback path / let UIs
    group statuses without owning their own taxonomy. ``color`` and
    other presentation fields stay in ``UIConfig.badges`` (D-06-05
    rejected alternative — presentation leak).
    """

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1)
    terminal: bool
    kind: StatusKind

# ====== module: runtime/config.py ======

_SESSION_ID_PREFIX_RE = re.compile(r"^[A-Za-z0-9-]{1,16}$")


ProviderKind = Literal["ollama", "azure_openai", "openai_compat", "stub"]


class ProviderConfig(BaseModel):
    """Connection settings for one upstream LLM provider.

    Multiple named ``ModelConfig`` entries can reference the same provider
    so that, e.g., two Ollama models share a single base_url + api_key.

    Phase 13 (HARD-01 / D-13-01): per-provider ``request_timeout``
    override (None means "use OrchestratorConfig.default_llm_request_timeout").
    Phase 13 (HARD-05 / D-13-06): ollama providers MUST declare
    ``base_url``; the @model_validator below catches the omission at
    config-load and raises ``LLMConfigError``. The hardcoded public
    Ollama fallback in ``runtime.llm`` is removed in the same phase.
    """
    kind: ProviderKind
    base_url: str | None = None       # ollama (REQUIRED via validator)
    api_key: str | None = None        # ollama, azure_openai
    endpoint: str | None = None       # azure_openai (validated lazily in builder)
    api_version: str | None = None    # azure_openai
    request_timeout: float | None = Field(
        default=None, gt=0, le=600,
    )  # NEW Phase 13 (D-13-01) — None -> OrchestratorConfig default

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "ProviderConfig":
        # D-13-06: only ollama is promoted to config-load validation in
        # Phase 13. azure_openai (`endpoint`) and openai_compat
        # (`base_url` + `api_key`) keep their existing first-request
        # ValueError raises in `_build_*_chat`. Promoting them is a
        # potential follow-up; see CONTEXT.md "Deferred Ideas".
        if self.kind == "ollama" and not self.base_url:
            raise LLMConfigError(
                provider="ollama", missing_field="base_url",
            )
        return self


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


class GatePolicy(BaseModel):
    """Phase 11 (FOC-04): declarative HITL gating policy.

    Drives the framework's pure ``should_gate`` boundary. The LLM never
    sees this config -- flow control is a framework decision, not a
    skill-prompt incantation.

    ``confidence_threshold`` is the strict-less-than predicate the gate
    applies to the active turn confidence; tool calls below the
    threshold fire a low_confidence pause for any non-auto-rated tool.

    ``gated_environments`` enumerates Session.environment values that
    automatically gate every non-auto-rated tool call regardless of
    confidence -- lifecycle defence against blast radius in production.

    ``gated_risk_actions`` enumerates GatewayAction Literal values
    (``auto``/``notify``/``approve``) that ALWAYS trigger a gate
    regardless of env or confidence. Default ``{"approve"}`` mirrors
    v1.0 HITL behaviour.

    Phase 11 chooses ``"approve"`` (the actual GatewayAction literal)
    over CONTEXT.md's sketched ``"hitl"`` -- see
    src/runtime/tools/gateway.py:32 for the canonical 3-valued
    GatewayAction Literal.
    """

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    gated_environments: set[str] = Field(
        default_factory=lambda: {"production"},
    )
    gated_risk_actions: set[str] = Field(
        default_factory=lambda: {"approve"},
    )


class RetryPolicy(BaseModel):
    """Phase 12 (FOC-05): declarative retry policy.

    Drives the framework's pure ``should_retry`` boundary. The LLM never
    sees this config -- flow control is a framework decision, not a
    skill-prompt incantation. Mirrors GatePolicy's shape so the
    OrchestratorConfig surface stays uniform.

    ``max_retries`` is the absolute cap on automatic retries (compared
    with ``retry_count`` via ``>=``). 0 disables auto-retry entirely;
    the recommended default 2 mirrors the v1.2 ROADMAP sketch and the
    existing transient-5xx auto-retry budget in graph.py.

    ``retry_on_transient`` lets apps with strict SLOs disable framework
    auto-retry of transient errors entirely (escalate immediately
    instead).

    ``retry_low_confidence_threshold`` is the strict-less-than predicate
    for "the LLM gave up; don't burn budget on a retry". Defaults to
    0.4 -- well below the typical gate_policy 0.7-0.8 threshold so a
    low-confidence escalation triggers HITL intervention before the
    retry path even considers it.
    """

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=2, ge=0, le=10)
    retry_on_transient: bool = True
    retry_low_confidence_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
    )


class OrchestratorConfig(BaseModel):
    model_config = {"extra": "forbid"}

    entry_agent: str = "intake"
    # Signals an agent may emit (via the configured patch tool's ``patch.signal``)
    # that the router will accept and look up against the skill's ``routes`` table.
    # Anything outside this set falls through to ``when: default``. Override
    # in YAML to extend the vocabulary; the default keeps current behaviour.
    signals: list[str] = Field(
        default_factory=lambda: ["success", "failed", "needs_input"],
    )

    # Generic terminal-tool registry — apps declare which tool calls
    # transition the session to which status, plus optional per-rule
    # extra-field extraction. Replaces the v1.0 hardcoded
    # ``_TERMINAL_TOOL_RULES`` table in ``orchestrator.py`` (D-06-01,
    # D-06-02). Empty list = framework cannot infer any terminal
    # status -> every session falls through to
    # ``default_terminal_status``.
    terminal_tools: list[TerminalToolRule] = Field(default_factory=list)

    # Status vocabulary the app exposes. Keys are the status names
    # the app uses (``resolved``, ``escalated``, ``approved``,
    # ``changes_requested``, ...). Empty dict is allowed for the
    # framework default ``OrchestratorConfig()`` so unconfigured apps
    # still validate (real apps populate this in their YAML).
    # D-06-03, D-06-05.
    statuses: dict[str, StatusDef] = Field(default_factory=dict)

    # Status assigned when the graph runs to ``__end__`` and no
    # ``terminal_tools`` rule fires. Required when ``statuses`` is
    # non-empty; must reference a key in ``statuses``. Apps own
    # this name — ``incident_management`` uses ``needs_review``,
    # ``code_review`` uses ``unreviewed`` (D-06-06).
    default_terminal_status: str | None = None

    # Tool names whose ``args.patch`` blob the harvester should fold
    # into agent confidence/signal/rationale (DECOUPLE-02 generalization
    # of the v1.0 single-tool path). Empty default means "no patch
    # tools" so unconfigured apps pay nothing. Apps populate this in
    # YAML alongside ``terminal_tools``; staying off the framework
    # hardcoded path keeps generic-runtime free of app vocabulary
    # leaks.
    patch_tools: list[str] = Field(default_factory=list)

    # Tool names the harvester should treat as "typed-terminal"
    # (carrying flat ``confidence``/``confidence_rationale`` args and
    # implying ``signal=success``) WITHOUT the orchestrator's finalize
    # path firing a status transition for them. Used for tools that
    # mark an agent stage complete but do not themselves end the
    # session. Empty default means "no harvest-only tools". Distinct
    # from ``terminal_tools`` (which both harvest and transition
    # status).
    harvest_terminal_tools: list[str] = Field(default_factory=list)

    # Dotted module paths the orchestrator imports at create()-time and
    # binds via each module's ``register(mcp_app, cfg)`` callable. Empty
    # list = no app MCP servers (framework-only). Order is preserved.
    # Replaces the v1.0 hardcoded framework-internal MCP-server imports
    # plus ``set_environments`` / ``set_escalation_teams`` setter calls
    # in orchestrator.py (DECOUPLE-04 / D-07-02 / D-07-03). Apps declare
    # their per-tool servers under ``orchestrator.mcp_servers`` in YAML;
    # framework no longer hardcodes incident-vocabulary modules.
    mcp_servers: list[str] = Field(default_factory=list)

    # Optional MCP tool the orchestrator invokes when a user clicks
    # ``Escalate`` from the awaiting_input gate. ``None`` (default)
    # means the orchestrator skips the tool call entirely and only
    # transitions the session to the rule-driven status. Apps that
    # want a side-effect (page on-call, file ticket) set this to the
    # bare tool name; the orchestrator looks up the matching rule in
    # ``terminal_tools`` to determine the resulting status.
    escalate_action_tool_name: str | None = None

    # Default team to pass to the escalation tool when the user did
    # not pick one. Only meaningful if ``escalate_action_tool_name``
    # is set. Apps own this default (``incident_management`` defaults
    # to ``platform-oncall``).
    escalate_action_default_team: str | None = None

    # Dotted path to a pydantic BaseModel subclass that validates the
    # ``state_overrides=`` dict passed to ``Orchestrator.start_session``.
    # Format: ``module.path:ClassName`` OR ``module.path.ClassName`` (both
    # accepted; ``:`` is the canonical entry-point form). ``None`` (default)
    # = no validation; ``start_session(state_overrides=...)`` passes the
    # dict through unchanged (D-08-02 backward-compat). Resolved at
    # ``Orchestrator.create()`` via ``importlib.import_module`` + ``getattr``;
    # bad path raises at boot with a useful message (DECOUPLE-05 / D-08-01).
    state_overrides_schema: str | None = None

    # Phase 9 (D-09-02 / FOC-01): map of LLM-visible-arg -> dotted-path
    # on the live Session. Tools whose param name matches a key in this
    # dict get the param stripped from the LLM-visible signature, and
    # the framework supplies the resolved value at _invoke_tool /
    # _GatedTool._run / _arun time. Apps declare what to inject; the
    # framework stays generic. Empty default = no injection (legacy
    # behaviour). Validated at config-load: keys are non-empty
    # identifiers, values are dotted paths starting with "session.".
    injected_args: dict[str, str] = Field(default_factory=dict)

    # Phase 11 (FOC-04): declarative HITL gating policy. Apps tune
    # thresholds in YAML; the framework's should_gate boundary reads
    # this struct and the LLM never sees it. Default keeps v1.1
    # behaviour (production gates "approve"-risk tools, threshold 0.7).
    gate_policy: "GatePolicy" = Field(default_factory=lambda: GatePolicy())

    # Phase 12 (FOC-05): declarative retry policy. Apps tune
    # max_retries / retry_on_transient / low-confidence threshold in
    # YAML; the framework's should_retry boundary reads this struct
    # and the LLM never sees it. Default keeps v1.2 behaviour
    # (max_retries=2, transient retries enabled, confidence floor 0.4).
    retry_policy: "RetryPolicy" = Field(
        default_factory=lambda: RetryPolicy(),
    )

    # Phase 13 (HARD-01 / D-13-02): framework-default LLM HTTP request
    # timeout in seconds. Per-provider ``ProviderConfig.request_timeout``
    # overrides this; ``None`` on the provider means "use this default".
    # Bounded to catch indefinite hangs (CONCERNS C1) while leaving room
    # for slow CPU Ollama runs (e.g., gpt-oss:120b). 600s upper bound
    # prevents accidentally-disabling the protection.
    default_llm_request_timeout: float = Field(
        default=120.0, gt=0, le=600,
    )

    @field_validator("state_overrides_schema")
    @classmethod
    def _validate_state_overrides_schema_format(
        cls, v: str | None,
    ) -> str | None:
        """String-format sanity check for the dotted-path schema reference.

        Real importlib resolution happens at ``Orchestrator.create()``
        time so config-load doesn't drag the schema module into every
        consumer. This validator only catches obviously-malformed
        strings (whitespace, hyphens, missing class component) so the
        actual ImportError/AttributeError is the only reason boot
        ever fails (DECOUPLE-05 / D-08-01).
        """
        if v is None:
            return v
        if not v.strip():
            raise ValueError(
                "state_overrides_schema must be non-empty when set"
            )
        # Accept either ``mod.path:ClassName`` or ``mod.path.ClassName``.
        # Each component must be a Python identifier; the trailing
        # element MUST be a class name (no further dots after the
        # separator).
        if not re.fullmatch(
            r"[A-Za-z_][\w.]*[:.][A-Za-z_]\w*", v,
        ):
            raise ValueError(
                f"state_overrides_schema={v!r} is not a valid dotted "
                f"path (expected `module.path:ClassName` or "
                f"`module.path.ClassName`)"
            )
        return v

    @field_validator("injected_args")
    @classmethod
    def _validate_injected_args(
        cls, v: dict[str, str],
    ) -> dict[str, str]:
        """Phase 9 (D-09-02): config-load validation for injected_args.

        Each entry is ``arg_name -> dotted_path`` where ``arg_name`` must
        be a valid Python identifier (it is the keyword name on a tool
        signature) and ``dotted_path`` must be a non-empty string with at
        least one dot (e.g. ``session.environment``). Real attribute
        resolution happens at injection time in
        :func:`runtime.tools.arg_injection.inject_injected_args` so
        config-load doesn't drag the live ``Session`` into every consumer.
        """
        for key, path in v.items():
            if not key or not key.isidentifier():
                raise ValueError(
                    f"injected_args key {key!r} must be a non-empty "
                    f"Python identifier"
                )
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"injected_args[{key!r}] must be a non-empty dotted path"
                )
            if "." not in path:
                raise ValueError(
                    f"injected_args[{key!r}]={path!r} must be a dotted path "
                    f"(e.g. 'session.environment')"
                )
        return v

    @model_validator(mode="after")
    def _validate_terminal_tool_registry(self) -> "OrchestratorConfig":
        """Cross-field invariants for the terminal-tool registry.

        * If ``statuses`` is non-empty, ``default_terminal_status``
          must be set and reference an existing status name.
        * The status referenced by ``default_terminal_status`` must
          be ``terminal=True`` (a non-terminal default makes no
          sense).
        * Every ``terminal_tools[i].status`` must reference an
          existing status name.

        Empty ``statuses`` (the framework's bare default) skips
        these checks so ``OrchestratorConfig()`` still constructs.
        Apps with ``statuses`` populated cross-validate at boot per
        D-06-03 / D-06-06.
        """
        if not self.statuses:
            # Bare framework default: nothing to cross-validate. If
            # ``default_terminal_status`` is set without ``statuses``
            # the app made a config mistake — flag it.
            if self.default_terminal_status is not None:
                raise ValueError(
                    "default_terminal_status is set but statuses is "
                    "empty; declare the status vocabulary first"
                )
            if self.terminal_tools:
                raise ValueError(
                    "terminal_tools is non-empty but statuses is "
                    "empty; declare the status vocabulary first"
                )
            return self

        if self.default_terminal_status is None:
            raise ValueError(
                "default_terminal_status is required when statuses "
                "is non-empty"
            )
        if self.default_terminal_status not in self.statuses:
            valid = sorted(self.statuses.keys())
            raise ValueError(
                f"default_terminal_status={self.default_terminal_status!r} "
                f"is not a declared status; valid statuses: {valid}"
            )
        default_def = self.statuses[self.default_terminal_status]
        if not default_def.terminal:
            raise ValueError(
                f"default_terminal_status={self.default_terminal_status!r} "
                f"references a non-terminal status (terminal=False); "
                f"the default must be terminal"
            )
        for idx, rule in enumerate(self.terminal_tools):
            if rule.status not in self.statuses:
                valid = sorted(self.statuses.keys())
                raise ValueError(
                    f"terminal_tools[{idx}].status={rule.status!r} "
                    f"(tool_name={rule.tool_name!r}) is not a "
                    f"declared status; valid statuses: {valid}"
                )
        return self


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
    # M7: lesson refresher knobs. ``lesson_refresh_cron`` is a 5-field
    # cron expression evaluated in UTC; default ``0 3 * * *`` runs daily
    # at 03:00 UTC. ``lesson_refresh_window_days`` bounds how far back
    # the refresher walks for terminal-status sessions on each tick.
    lesson_refresh_cron: str = "0 3 * * *"
    lesson_refresh_window_days: int = 7
    # Per-app session-id prefix. Threaded through ``SessionStore`` to
    # ``Session.id_format`` so each app picks its own id namespace
    # (``INC`` for incident management, ``REVIEW`` for code review,
    # ``HR`` for HR cases, ...). Default ``"SES"`` keeps unconfigured
    # apps generic. Validated as 1-16 chars of alphanumerics and
    # hyphens so the resulting id stays scannable.
    session_id_prefix: str = "SES"
    # UI rendering knobs surfaced to the generic runtime UI. Mirrors
    # AppConfig.ui — the FrameworkAppConfig provider can either copy
    # AppConfig.ui or supply its own. Defaults to empty so apps that
    # don't render with the generic UI pay nothing.
    ui: UIConfig = Field(default_factory=UIConfig)

    @field_validator("session_id_prefix")
    @classmethod
    def _validate_session_id_prefix(cls, v: str) -> str:
        if not _SESSION_ID_PREFIX_RE.match(v):
            raise ValueError(
                f"session_id_prefix={v!r} must be 1-16 chars of "
                "alphanumerics and hyphens (no whitespace, no symbols)"
            )
        return v


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


class ApiConfig(BaseModel):
    """API surface knobs surfaced to the React frontend."""

    # CORS origins allowed by the FastAPI CORSMiddleware. Default
    # covers the two common React dev-server URLs (Vite, CRA/Next).
    # Production deployments override via YAML to lock down to their
    # actual frontend origin.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
        ]
    )
    # Allow credentials on cross-origin requests (cookies, auth headers).
    cors_allow_credentials: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: Paths = Field(default_factory=Paths)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
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
            # ``BaseModel.__dict__`` is typed as ``MappingProxyType`` in
            # the pydantic stub; the documented post-validator mutation
            # path is direct ``__dict__`` assignment, which works at
            # runtime (pydantic stores fields in a plain dict).
            self.__dict__["dedup"] = DedupConfig(**self.dedup)  # pyright: ignore[reportIndexIssue]
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
        # the documented way to mutate after validation. (Stub types
        # ``__dict__`` as MappingProxyType; runtime is a plain dict.)
        self.__dict__["triggers"] = coerced  # pyright: ignore[reportIndexIssue]
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
    # Optimistic concurrency token. Incremented on every successful
    # ``SessionStore.save``; reads observe the value at load time. Saves
    # with a stale version raise ``StaleVersionError`` so the caller can
    # reload + retry.
    version: int = 1
    # Phase 11 (FOC-04): transient per-turn confidence hint set by the
    # agent runner (graph.py / responsive.py) AFTER each
    # _harvest_tool_calls_and_patches call so the gateway's should_gate
    # boundary can apply low_confidence gating using whatever
    # confidence the agent has emitted so far. Reset to ``None`` at
    # turn start; never persisted (``Field(exclude=True)``). The
    # framework treats ``None`` as "no signal yet" and does NOT fire a
    # low_confidence gate -- this avoids a false-positive gate on the
    # very first tool call of a turn before any envelope/tool-arg
    # carrying confidence has surfaced.
    turn_confidence_hint: float | None = Field(default=None, exclude=True)

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
    def id_format(cls, *, seq: int, prefix: str = "SES") -> str:
        """Return the canonical session id for the given sequence number.

        ``prefix`` is supplied by ``SessionStore._next_id`` from
        ``FrameworkAppConfig.session_id_prefix`` so each app picks its
        own namespace via plain config (e.g. ``INC`` for incident
        management, ``REVIEW`` for code review, ``HR`` for HR cases,
        ...). Apps with truly bespoke id shapes can still override this
        classmethod on their ``Session`` subclass and ignore ``prefix``.

        ``seq`` is the per-day monotonic sequence supplied by
        ``SessionStore._next_id``; it lets the default format produce
        the expected zero-padded suffix without each subclass
        re-implementing the SQL scan.
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{prefix}-{today}-{seq:03d}"

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

    Phase 10 (FOC-03): also honours
    ``llm.with_structured_output(AgentTurnOutput)`` so stub-driven tests
    survive the runner's envelope contract. The structured response is
    derived from the same canned text + a default 0.85 confidence; tests
    that need a specific envelope shape can override
    ``stub_envelope_confidence`` / ``stub_envelope_rationale`` /
    ``stub_envelope_signal``.

    Phase 15 (LLM-COMPAT-01): ``langchain.agents.create_agent`` with
    ``response_format=AgentTurnOutput`` (via ``AutoStrategy`` ->
    ``ToolStrategy`` for non-native-structured-output models, including
    this stub) injects ``AgentTurnOutput`` as a CALLABLE TOOL. The
    agent loop only terminates when the LLM emits a tool call NAMED
    ``AgentTurnOutput``. ``bind_tools`` records that envelope-tool name
    so ``_generate`` can auto-emit a closing tool call after any
    user-configured ``tool_call_plan`` is exhausted -- preserving the
    pre-Phase-15 stub semantics (canned text + optional pre-scripted
    tool calls) while satisfying the new tool-loop termination
    contract.
    """
    role: str = "default"
    canned_responses: dict[str, str] = Field(default_factory=dict)
    tool_call_plan: list[dict] | None = None
    stub_envelope_confidence: float = 0.85
    stub_envelope_rationale: str = "stub envelope rationale"
    stub_envelope_signal: str | None = None
    _called_once: bool = False
    # Phase 15 (LLM-COMPAT-01): set by ``bind_tools`` when
    # ``langchain.agents.create_agent`` injects a structured-output tool
    # for ``AgentTurnOutput``. Holds the bare tool name (e.g.
    # ``"AgentTurnOutput"``) so ``_generate`` can emit a final
    # envelope-shaped tool call to close the agent loop.
    _envelope_tool_name: str | None = None

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
        elif self._envelope_tool_name is not None:
            # Phase 15 (LLM-COMPAT-01): the tool_call_plan is exhausted
            # (or wasn't configured) AND ``langchain.agents.create_agent``
            # has bound the AgentTurnOutput envelope as a tool. Emit a
            # closing tool call so the loop terminates with a populated
            # ``structured_response``. The args mirror the
            # ``with_structured_output`` path's envelope construction so
            # tests see the same confidence / rationale / signal regardless
            # of whether the new tool-strategy or the legacy structured-
            # output path is in play.
            tool_calls.append({
                "name": self._envelope_tool_name,
                "args": {
                    "content": text or ".",
                    "confidence": self.stub_envelope_confidence,
                    "confidence_rationale": self.stub_envelope_rationale,
                    "signal": self.stub_envelope_signal,
                },
                "id": str(uuid4()),
            })
        msg = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Record the AgentTurnOutput envelope-tool name when present.

        Phase 15 (LLM-COMPAT-01): ``langchain.agents.create_agent`` with
        ``response_format=AgentTurnOutput`` calls ``bind_tools(...)``
        with the user's tools PLUS the envelope-as-a-tool. We scan the
        list for the AgentTurnOutput-shaped tool (matched by ``__name__``
        on Pydantic schemas, ``name`` on ``BaseTool`` instances, or the
        ``"name"`` key on dict-shaped tool specs) and remember it on the
        instance so ``_generate`` can close the agent loop with a
        synthetic envelope tool call after any pre-scripted
        ``tool_call_plan`` is exhausted. Tools bound by the framework
        itself (real BaseTools the agent should call) flow through
        unchanged -- the stub still emits them only via
        ``tool_call_plan``.
        """
        for t in tools or []:
            name = (
                getattr(t, "__name__", None)
                or getattr(t, "name", None)
                or (isinstance(t, dict) and t.get("name"))
            )
            if isinstance(name, str) and name == "AgentTurnOutput":
                self._envelope_tool_name = name
                break
        return self

    # ``BaseChatModel.with_structured_output`` returns ``Runnable[..., dict | BaseModel]``
    # in the langchain stub; this stub override returns a deterministic
    # ``_StructuredRunnable`` so tests can drive structured outputs
    # without a live provider. Functionally a Runnable (it implements
    # ``invoke`` + ``ainvoke``); the stub mismatch is cosmetic.
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Phase 10 (FOC-03): honour the structured-output pass.

        Historically (pre-Phase-15) the deprecated
        ``langgraph.prebuilt.create_react_agent`` factory called this
        after its tool loop completed. The current
        ``langchain.agents.create_agent`` path uses a tool-strategy
        binding instead (see ``bind_tools`` above), but providers and
        test code that call ``with_structured_output`` directly still
        get a deterministic schema instance.

        We return a Runnable-like that yields a valid ``schema``
        instance derived from the stub's canned text and the
        per-instance envelope configuration. Tests can tune
        ``stub_envelope_confidence`` etc. to drive gate / reconcile
        paths.
        """
        text = self.canned_responses.get(self.role, f"[stub:{self.role}] no canned response")
        confidence = self.stub_envelope_confidence
        rationale = self.stub_envelope_rationale
        signal = self.stub_envelope_signal

        class _StructuredRunnable:
            def __init__(self, schema_cls):
                self._schema = schema_cls

            def _build(self):
                # Construct an instance of whatever schema was passed.
                # Common case: AgentTurnOutput; permissive fallback handles
                # other pydantic schemas the test may pass.
                try:
                    return self._schema(
                        content=text or ".",
                        confidence=confidence,
                        confidence_rationale=rationale,
                        signal=signal,
                    )
                except Exception:
                    # Permissive fallback for unfamiliar schemas: try
                    # model_validate on a minimal dict.
                    return self._schema.model_validate({
                        "content": text or ".",
                        "confidence": confidence,
                        "confidence_rationale": rationale,
                        "signal": signal,
                    })

            def invoke(self, *_args, **_kwargs):
                return self._build()

            async def ainvoke(self, *_args, **_kwargs):
                return self._build()

        return _StructuredRunnable(schema)


def _resolve_timeout(
    provider: ProviderConfig, default: float,
) -> float:
    """Resolve effective request timeout for a provider.

    Per-provider override wins; falls back to the framework default
    (typically ``OrchestratorConfig.default_llm_request_timeout``).
    """
    if provider.request_timeout is not None:
        return provider.request_timeout
    return default


def _wrap_chat_with_timeout(
    base: BaseChatModel,
    provider_name: str,
    model_id: str,
    request_timeout: float,
) -> BaseChatModel:
    """Wrap ``base`` so every ``ainvoke`` is bounded by
    ``asyncio.wait_for(..., timeout=request_timeout)`` and raises
    ``LLMTimeoutError(provider, model, elapsed_ms)`` on hang.

    The native langchain timeout knob (``request_timeout=`` on
    openai/azure or ``client_kwargs={'timeout': ...}`` on ollama) is
    honoured at the httpx layer; this wrapper guarantees the
    framework-typed exception AND a hard ceiling even if the
    underlying client hangs in a way httpx misses (e.g., post-headers
    TCP read stall on a slow Ollama). D-13-04: subclassing
    ``TimeoutError`` means ``policy._TRANSIENT_TYPES`` auto-classifies
    the error as transient (zero edits to ``policy.py``).
    """
    base_cls = type(base)

    class _Bounded(base_cls):  # type: ignore[misc, valid-type]
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                return await asyncio.wait_for(
                    super().ainvoke(*args, **kwargs),
                    timeout=request_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError) as e:
                if isinstance(e, LLMTimeoutError):
                    # Already typed; don't double-wrap.
                    raise
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                raise LLMTimeoutError(
                    provider=provider_name,
                    model=model_id,
                    elapsed_ms=elapsed_ms,
                ) from e

    # Reuse the live pydantic instance's state without re-running
    # __init__ (which would re-init the underlying httpx clients).
    bounded = _Bounded.model_construct(**base.model_dump())
    # Some langchain client classes initialise non-pydantic attrs
    # (httpx clients, run_manager, etc.) inside __init__. Copy them
    # through so the wrapped instance shares the same network state.
    for attr_name in (
        "_client", "_async_client",
        "_async_httpx_client", "_sync_httpx_client",
        "client", "async_client",
    ):
        if hasattr(base, attr_name):
            try:
                object.__setattr__(
                    bounded, attr_name, getattr(base, attr_name),
                )
            except (AttributeError, TypeError):
                # Slot-only or read-only attrs on some langchain
                # versions -- the bounded instance will re-init on
                # first use; not a correctness issue.
                pass
    return bounded


def _build_ollama_chat(
    provider: ProviderConfig, model_id: str, temperature: float,
    *, request_timeout: float,
) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # Many Ollama models (gemma*, gpt-oss, ministral, etc.) don't support
    # native function-calling, which is langchain-ollama's default method
    # for ``with_structured_output``. Subclass to force
    # ``method='json_schema'`` (uses Ollama's structured-output API) so
    # Phase 10's ``response_format=AgentTurnOutput`` envelope actually
    # round-trips instead of failing with ``OutputParserException``
    # when the LLM emits prose.
    class _ChatOllamaJsonSchema(ChatOllama):  # type: ignore[misc, valid-type]
        def with_structured_output(self, schema, *, method=None, **kw):
            return super().with_structured_output(
                schema, method=method or "json_schema", **kw,
            )

    # Phase 13 (HARD-01): ChatOllama has NO native ``request_timeout``
    # field; the canonical incantation is ``client_kwargs={"timeout": ...}``,
    # which propagates to the underlying httpx.AsyncClient.
    client_kwargs: dict[str, Any] = {"timeout": request_timeout}
    api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
    if api_key:
        client_kwargs["headers"] = {
            "Authorization": f"Bearer {api_key}",
        }
    # Phase 13 (HARD-05): base_url is now config-load-validated by
    # ProviderConfig._validate_required_fields. NO fallback to a
    # public Ollama URL (air-gap rule violation).
    kwargs: dict[str, Any] = {
        "base_url": provider.base_url,
        "model": model_id,
        "temperature": temperature,
        "client_kwargs": client_kwargs,
    }
    base = _ChatOllamaJsonSchema(**kwargs)
    return _wrap_chat_with_timeout(
        base, "ollama", model_id, request_timeout,
    )


def _build_azure_chat(
    provider: ProviderConfig, model: ModelConfig,
    *, request_timeout: float,
) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI
    if provider.endpoint is None:
        raise ValueError("azure_openai provider requires 'endpoint'")
    if model.deployment is None:
        raise ValueError(
            f"azure_openai model {model.model!r} requires 'deployment'"
        )
    _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
    # ``request_timeout`` is a runtime alias for ``timeout`` on
    # AzureChatOpenAI (langchain-openai > 0.3 declares it via Pydantic
    # ``Field(alias="timeout")``); the langchain stubs only expose
    # ``timeout``, hence the stub gap.
    base = AzureChatOpenAI(
        azure_endpoint=provider.endpoint,
        api_version=provider.api_version or "2024-08-01-preview",
        azure_deployment=model.deployment,
        api_key=SecretStr(_ak) if _ak else None,
        temperature=model.temperature,
        request_timeout=request_timeout,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
    )
    return _wrap_chat_with_timeout(
        base, "azure_openai", model.model, request_timeout,
    )


def get_llm(cfg: LLMConfig, model_name: str | None = None, *,
            role: str = "default",
            stub_canned: dict[str, str] | None = None,
            stub_tool_plan: list[dict] | None = None,
            stub_envelope_confidence: float | None = None,
            stub_envelope_rationale: str | None = None,
            stub_envelope_signal: str | None = None,
            default_llm_request_timeout: float = 120.0,
            ) -> BaseChatModel:
    """Build a chat model by named entry from ``cfg.models``.

    ``model_name`` defaults to ``cfg.default``. Validation that the name
    exists is enforced by ``LLMConfig`` itself (model_validator), so a
    missing name here means caller passed a typo -- raise loudly.

    Phase 10 (FOC-03): stub callers can now tune the canned envelope
    (confidence / rationale / signal) so gate-trigger tests preserve their
    pre-Phase-10 semantics by emitting a low-confidence envelope.

    Phase 13 (HARD-01): non-stub builds are bounded by an effective
    ``request_timeout`` resolved as ``provider.request_timeout`` (per-
    provider override) -> ``default_llm_request_timeout`` (framework
    default; callers pass ``cfg.orchestrator.default_llm_request_timeout``).
    The default keyword value (120.0) matches OrchestratorConfig's default
    so test paths that build LLMs without an OrchestratorConfig in scope
    still get a sane bound.
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
        kwargs: dict[str, Any] = {
            "role": role,
            "canned_responses": stub_canned or {},
            "tool_call_plan": stub_tool_plan,
        }
        if stub_envelope_confidence is not None:
            kwargs["stub_envelope_confidence"] = stub_envelope_confidence
        if stub_envelope_rationale is not None:
            kwargs["stub_envelope_rationale"] = stub_envelope_rationale
        if stub_envelope_signal is not None:
            kwargs["stub_envelope_signal"] = stub_envelope_signal
        return StubChatModel(**kwargs)

    effective = _resolve_timeout(provider, default_llm_request_timeout)

    if provider.kind == "ollama":
        return _build_ollama_chat(
            provider, model.model, model.temperature,
            request_timeout=effective,
        )
    if provider.kind == "azure_openai":
        return _build_azure_chat(
            provider, model, request_timeout=effective,
        )
    if provider.kind == "openai_compat":
        return _build_openai_compat_chat(
            provider, model, request_timeout=effective,
        )
    raise ValueError(f"Unknown provider kind: {provider.kind!r}")


def _build_openai_compat_chat(
    provider: ProviderConfig, model: ModelConfig,
    *, request_timeout: float,
) -> BaseChatModel:
    """Build a ``ChatOpenAI`` pointed at an OpenAI-compatible endpoint
    (OpenRouter, Together, vLLM, etc.). Reuses langchain-openai's
    ``ChatOpenAI`` with ``base_url=`` override and the provider's
    ``api_key`` (resolved from env via the YAML loader).
    """
    from langchain_openai import ChatOpenAI
    if provider.base_url is None:
        raise ValueError(
            "openai_compat provider requires 'base_url' "
            "(e.g. https://openrouter.ai/api/v1)"
        )
    if provider.api_key is None:
        raise ValueError("openai_compat provider requires 'api_key'")
    # See AzureChatOpenAI block above: ``request_timeout`` is a runtime
    # alias for ``timeout`` not in the langchain stubs.
    base = ChatOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=model.model,
        temperature=model.temperature,
        request_timeout=request_timeout,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
    )
    return _wrap_chat_with_timeout(
        base, "openai_compat", model.model, request_timeout,
    )


def get_embedding(
    cfg: LLMConfig, *, default_llm_request_timeout: float = 120.0,
) -> Embeddings:
    """Build the configured embedding model. Raises if ``cfg.embedding`` is None.

    Phase 13 (HARD-01): same per-provider override -> framework default
    timeout resolution as ``get_llm``. Embeddings traffic shares the
    request_timeout knob with chat (see CONTEXT.md "Deferred Ideas" --
    splitting embedding timeout from chat is a future refinement).

    Note (Phase 13 review WR-01): unlike the chat builders -- which apply a
    defence-in-depth ``asyncio.wait_for`` wrapper (``_wrap_chat_with_timeout``)
    that guarantees a structured ``LLMTimeoutError`` with ``elapsed_ms`` even
    on partial-byte stalls -- embeddings rely SOLELY on the underlying
    httpx-layer timeout configured above (``client_kwargs={"timeout": ...}``
    for Ollama, ``request_timeout=`` for Azure). This asymmetry is a
    deliberate scope choice tied to Phase 13 CONTEXT.md "Deferred Ideas" #4
    (splitting embeddings timeout from chat timeout). If embeddings need
    stricter bounds than chat -- or if the httpx-layer timeout proves
    insufficient against post-headers TCP read stalls on the embeddings
    path the same way it can on chat -- a future phase can mirror
    ``_wrap_chat_with_timeout`` for the embeddings public surface
    (``aembed_query`` / ``aembed_documents``).
    """
    if cfg.embedding is None:
        raise ValueError("llm.embedding is not configured")
    provider = cfg.providers[cfg.embedding.provider]
    effective = _resolve_timeout(provider, default_llm_request_timeout)
    if provider.kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        # Phase 13 (HARD-01): OllamaEmbeddings has NO native
        # ``request_timeout`` field; canonical incantation is
        # ``client_kwargs={"timeout": ...}`` (same as ChatOllama).
        client_kwargs: dict[str, Any] = {"timeout": effective}
        api_key = provider.api_key or os.environ.get("OLLAMA_API_KEY")
        if api_key:
            client_kwargs["headers"] = {
                "Authorization": f"Bearer {api_key}",
            }
        # Phase 13 (HARD-05): base_url config-load-validated; NO public fallback.
        return OllamaEmbeddings(
            base_url=provider.base_url,
            model=cfg.embedding.model,
            client_kwargs=client_kwargs,
        )
    if provider.kind == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        if provider.endpoint is None:
            raise ValueError("azure_openai provider requires 'endpoint'")
        deployment = cfg.embedding.deployment or cfg.embedding.model
        _ak = provider.api_key or os.environ.get("AZURE_OPENAI_KEY")
        # See chat builders above: ``request_timeout`` is a runtime
        # alias for ``timeout`` not surfaced in the langchain-openai stub.
        return AzureOpenAIEmbeddings(
            azure_endpoint=provider.endpoint,
            api_version=provider.api_version or "2024-08-01-preview",
            azure_deployment=deployment,
            api_key=SecretStr(_ak) if _ak else None,
            request_timeout=effective,  # pyright: ignore[reportCallIssue]  -- Phase 13 (HARD-01) -- alias for ``timeout`` not in stub
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
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

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


class SessionEventRow(Base):
    """Append-only event log for a session.

    Events are immutable; they record what was observed (tool call,
    status transition, agent run completion) and feed the status
    finalizer's inference logic. Sequence is monotonic per session
    and globally autoincrementing.
    """
    __tablename__ = "session_events"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("incidents.id"), index=True, nullable=False,
    )
    kind: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    ts: Mapped[str] = mapped_column(String, nullable=False)


class SessionLessonRow(Base):
    """M5: distilled "lesson" extracted from one resolved session.

    Each lesson captures (a) the symptom that started the session
    (via ``embedding_text`` which seeds the vector index), (b) the
    tool sequence the framework ran, (c) the final outcome
    (status + confidence + summary), and (d) provenance metadata so
    callers can tell auto-extracted lessons from operator-curated
    ones. The intake runner reads lessons via ``LessonStore
    .find_similar`` and surfaces the top-k as ``findings["lessons"]``
    on each new session.

    Append-only by convention — :class:`LessonStore` provides ``add``
    but no ``update``. M7's nightly refresher writes a fresh row when
    the extractor version changes; older rows stay queryable.
    """
    __tablename__ = "session_lessons"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_session_id: Mapped[str] = mapped_column(
        String, ForeignKey("incidents.id"), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    signals: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    tool_sequence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    outcome_status: Mapped[str] = mapped_column(String, nullable=False)
    outcome_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence_final: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_session_lessons_source_session_id", "source_session_id"),
        Index(
            "ix_session_lessons_outcome_status_created_at",
            "outcome_status", "created_at",
        ),
    )

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
        # AzureOpenAIEmbeddings.api_key is typed as ``SecretStr | None``
        # (pydantic v2). Wrap the env-sourced str so the type matches.
        return AzureOpenAIEmbeddings(
            azure_deployment=cfg.deployment,
            model=cfg.model,
            azure_endpoint=p.endpoint,
            api_version=p.api_version,
            api_key=SecretStr(p.api_key) if p.api_key else None,
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
    # ``langchain_postgres.DistanceStrategy.INNER_PRODUCT`` exists at
    # runtime (verified via the live module) but the langchain-postgres
    # stubs only expose ``COSINE`` / ``EUCLIDEAN``.
    return {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN,
        "inner_product": DistanceStrategy.INNER_PRODUCT,  # pyright: ignore[reportAttributeAccessIssue]
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

StateT = TypeVar("StateT", bound=Session)

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
        with SqlaSession(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def _list_filtered(self, *, filter_kwargs: Mapping[str, Any]) -> list[StateT]:
        """List non-deleted rows matching the given column filters.

        Pure SQL prefilter — used by both vector and keyword paths.
        """
        with SqlaSession(self.engine) as session:
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
        # ``similarity_search_with_score_by_vector`` is provided by the
        # concrete FAISS / pgvector / langchain-postgres backends (and
        # validated by ``runtime.storage.vector.build_vector_store``)
        # but the abstract ``langchain_core.vectorstores.VectorStore``
        # base class does not declare it.
        raw = self.vector_store.similarity_search_with_score_by_vector(vec, k=limit * 4)  # pyright: ignore[reportAttributeAccessIssue]
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
        def _ef(i, key, default: Any = ""):
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

# StateT is bound to ``Session`` (not bare ``BaseModel``) because the
# store body reads typed fields (``id``, ``status``, ``version``,
# ``updated_at`` …) that are declared on ``runtime.state.Session`` and
# not on ``pydantic.BaseModel``. The resolver in
# :mod:`runtime.state_resolver` already enforces a ``Session`` subclass
# at config time, and every existing caller (production + tests) passes
# either bare ``Session`` or a ``Session`` subclass — see
# Phase 19 / HARD-03 for the rationale (was: ``bound=BaseModel`` which
# made pyright flag every typed-field access).
StateT = TypeVar("StateT", bound=Session)


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


class StaleVersionError(RuntimeError):
    """Raised when ``SessionStore.save`` observes that the row has been
    updated since the in-memory copy was loaded.

    Callers should reload from the store and re-apply their mutation.
    """


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
        id_prefix: str = "SES",
    ) -> None:
        self.engine = engine
        self._state_cls = state_cls
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_path = vector_path
        self.vector_index_name = vector_index_name
        self.distance_strategy = distance_strategy
        # Per-app session-id namespace. Threaded into
        # ``state_cls.id_format`` so each app's rows share a stable
        # ``PREFIX-YYYYMMDD-NNN`` shape. Default ``"SES"`` keeps the
        # bare-Session path framework-neutral; apps configure this
        # via ``FrameworkAppConfig.session_id_prefix``.
        self._id_prefix = id_prefix

    # ---------- ID minting ----------
    def _next_id(self, session: SqlSession) -> str:
        """Mint a new session id via ``state_cls.id_format(seq=...)``.

        The per-app id namespace is supplied as ``self._id_prefix`` (from
        ``FrameworkAppConfig.session_id_prefix``); ``id_format`` may
        also be overridden on the state subclass for fully bespoke
        shapes. The store still owns the monotonic sequence — it scans
        for prior rows whose id starts with the same ``PREFIX-YYYYMMDD-``
        stem and returns ``max(seq) + 1``.
        """
        # Probe today's prefix by asking the state class to format seq=1
        # and stripping the ``-001`` suffix. Apps that override
        # ``id_format`` to return a non-``PREFIX-YYYYMMDD-NNN`` shape
        # (e.g. opaque ULIDs) fall through to the simple count path
        # below.
        sample = self._state_cls.id_format(seq=1, prefix=self._id_prefix)
        m = _SESSION_ID_RE.match(sample)
        if m is None:
            # Custom format — count all rows as the sequence base. Apps
            # that want collision-free ids should mint ULIDs in
            # ``id_format`` and ignore ``seq``.
            count = session.execute(
                select(IncidentRow.id)
            ).scalars().all()
            return self._state_cls.id_format(
                seq=len(count) + 1, prefix=self._id_prefix,
            )

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
        return self._state_cls.id_format(
            seq=max_seq + 1, prefix=self._id_prefix,
        )

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
        # ``_iso(_now())`` returns ``str`` here -- the input datetime is
        # never None -- but the helper's signature is the broader
        # ``Optional[str]``. ``or ""`` keeps pyright + the typed
        # ``Session.updated_at: str`` field consistent without changing
        # behaviour (real value is always present).
        incident.updated_at = _iso(_now()) or ""
        sess = incident  # local alias — avoids repeating the domain token in new code
        expected_version = getattr(sess, "version", 1)
        # Bump in-memory BEFORE building the row dict so the persisted
        # row reflects the new version.
        sess.version = expected_version + 1
        with SqlSession(self.engine) as session:
            existing = session.get(IncidentRow, sess.id)
            prior_text = _embed_source_from_row(existing) if existing is not None else ""
            if existing is not None and existing.version != expected_version:
                # Roll back the in-memory bump so the caller can reload + retry.
                sess.version = expected_version
                raise StaleVersionError(
                    f"session {sess.id} version is {existing.version}, "
                    f"expected {expected_version}"
                )
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
        # ``save_local`` is FAISS-specific; the runtime ``hasattr`` guard
        # at the top of this method already ensured this codepath only
        # runs against FAISS (other VectorStores omit the method).
        # ``langchain_core.vectorstores.VectorStore`` doesn't declare it.
        self.vector_store.save_local(  # pyright: ignore[reportAttributeAccessIssue]
            folder_path=str(folder),
            index_name=self.vector_index_name,
        )

    def _add_vector(self, inc: Session) -> None:
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

    def _refresh_vector(self, inc: Session, *, prior_text: str) -> None:
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
        # Optimistic-concurrency token — has its own typed column.
        "version",
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
            "version": row.version if row.version is not None else 1,
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

        # ``kwargs`` is built up from heterogeneous sources (typed row
        # columns + ``extra_fields`` blob) so pyright infers each value
        # as ``object``. At runtime each entry matches the concrete
        # ``state_cls`` field type by construction (the row schema is
        # the source of truth); pydantic's own validation rejects bad
        # shapes at the constructor.
        return self._state_cls(**kwargs)  # pyright: ignore[reportArgumentType]

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
            "version": getattr(inc, "version", 1),
        }

# ====== module: runtime/storage/event_log.py ======

EventKind = Literal[
    "agent_started",
    "agent_finished",
    "tool_invoked",
    "confidence_emitted",
    "route_decided",
    "gate_fired",
    "status_changed",
    "lesson_extracted",
]

_VALID_EVENT_KINDS: frozenset[str] = frozenset(get_args(EventKind))


@dataclass(frozen=True)
class SessionEvent:
    """Immutable view of one row in the event log."""
    seq: int
    session_id: str
    kind: str
    payload: dict
    ts: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """Append-only log of session events.

    Events drive the status finalizer's inference (e.g. a registered
    ``<terminal_tool>`` event appearing in the log -> session reached
    the corresponding terminal status). They are never mutated or
    deleted.
    """

    def __init__(self, *, engine: Engine) -> None:
        self.engine = engine

    def append(self, session_id: str, kind: str, payload: dict) -> None:
        """Append a new event row. Never mutates existing rows."""
        with Session(self.engine) as s:
            with s.begin():
                s.add(SessionEventRow(
                    session_id=session_id,
                    kind=kind,
                    payload=dict(payload),
                    ts=_now(),
                ))

    def record(
        self,
        session_id: str,
        kind: EventKind,
        **payload: Any,
    ) -> None:
        """Convenience over ``append`` for the common kwargs shape.

        ``record(sid, "tool_invoked", tool="x", latency_ms=12)`` is
        equivalent to ``append(sid, "tool_invoked", {"tool": "x",
        "latency_ms": 12})`` but validates ``kind`` against the
        :data:`EventKind` Literal at call time — a typo is a hard
        failure, not a silently-malformed row.
        """
        if kind not in _VALID_EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {kind!r}; allowed: "
                f"{sorted(_VALID_EVENT_KINDS)}"
            )
        self.append(session_id, kind, payload)

    def iter_for(
        self, session_id: str, *, since: int | None = None,
    ) -> Iterator[SessionEvent]:
        """Yield events for ``session_id`` in monotonic insertion order.

        ``since``: optional ``seq`` watermark. When supplied, only events
        with ``seq > since`` are returned — enables SSE / WebSocket
        streaming clients to resume from their last observed seq without
        re-receiving the backlog.
        """
        with Session(self.engine) as s:
            stmt = (
                select(SessionEventRow)
                .where(SessionEventRow.session_id == session_id)
                .order_by(SessionEventRow.seq)
            )
            if since is not None:
                stmt = stmt.where(SessionEventRow.seq > since)
            for row in s.execute(stmt).scalars():
                yield SessionEvent(
                    seq=row.seq,
                    session_id=row.session_id,
                    kind=row.kind,
                    payload=row.payload,
                    ts=row.ts,
                )

# ====== module: runtime/storage/migrations.py ======

_FORWARD_COLUMNS: list[tuple[str, str]] = [
    ("parent_session_id", "VARCHAR"),  # dedup linkage
    ("dedup_rationale", "TEXT"),       # LLM rationale
    ("extra_fields", "JSON"),          # generic round-trip tunnel
]
_FORWARD_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, table, column) — mirrors models.IncidentRow.__table_args__.
    ("ix_incidents_parent_session_id", "incidents", "parent_session_id"),
]

# Default audit fields. Mirrors the Pydantic defaults on
# :class:`runtime.state.ToolCall`. Keep these in sync — a divergence
# means rows hydrated post-migration would carry different defaults
# than rows hydrated via the Pydantic constructor, which would surface
# as subtle test flakes long after the migration ran.
_AUDIT_DEFAULTS: dict[str, Any] = {
    "status": "executed",
    "risk": None,
    "approver": None,
    "approved_at": None,
    "approval_rationale": None,
}


def _fill_audit_fields(tc: dict[str, Any]) -> bool:
    """Mutate ``tc`` in place, filling any missing audit field with its
    default. Returns ``True`` when at least one key was added.

    Existing values (including explicit ``None`` already on the row)
    are left untouched — this is the idempotency guarantee.
    """
    changed = False
    for key, default in _AUDIT_DEFAULTS.items():
        if key not in tc:
            tc[key] = default
            changed = True
    return changed


def _normalise_tool_calls_list(
    tool_calls: Iterable[Any] | None,
) -> tuple[list[Any], bool]:
    """Walk a session's tool_calls JSON list, fill missing audit fields.

    Returns ``(new_list, changed)``. Non-dict entries (corrupt rows)
    are passed through unchanged — the migration is not a validator.
    """
    if not tool_calls:
        return [], False
    new: list[Any] = []
    changed = False
    for tc in tool_calls:
        if isinstance(tc, dict):
            # Copy so we don't mutate caller-owned data accidentally.
            tc_copy = dict(tc)
            if _fill_audit_fields(tc_copy):
                changed = True
            new.append(tc_copy)
        else:
            new.append(tc)
    return new, changed


def migrate_tool_calls_audit(engine: Engine) -> dict[str, int]:
    """Walk every session's ``tool_calls`` and fill missing audit fields.

    Idempotent — running on a freshly-migrated DB is a no-op.

    Returns a small stats dict::

        {"sessions_scanned": N, "sessions_updated": M, "rows_filled": K}

    where ``rows_filled`` is the count of individual ToolCall entries
    that received at least one default. Useful for ops dashboards and
    post-migration verification.
    """
    scanned = 0
    updated = 0
    filled = 0
    with SqlSession(engine) as session:
        rows = session.query(IncidentRow).all()
        for row in rows:
            scanned += 1
            new_list, changed = _normalise_tool_calls_list(row.tool_calls)
            if changed:
                # Count individual entries that gained at least one
                # field. Cheap re-walk — rows.tool_calls is already in
                # memory.
                for old, new in zip(row.tool_calls or [], new_list):
                    if isinstance(old, dict) and isinstance(new, dict):
                        if any(k not in old for k in _AUDIT_DEFAULTS):
                            filled += 1
                row.tool_calls = new_list
                updated += 1
        if updated:
            session.commit()
    return {
        "sessions_scanned": scanned,
        "sessions_updated": updated,
        "rows_filled": filled,
    }


def migrate_add_lesson_table(engine: Engine) -> dict[str, int]:
    """M5: create the ``session_lessons`` table if missing. Idempotent.

    Older databases predating M5 lack this table; we use
    ``Base.metadata.create_all`` scoped to the lesson table so the
    DDL is generated by SQLAlchemy (handles SQLite / Postgres / etc.)
    rather than handwritten ALTER statements. Running on a freshly-
    created database is a no-op (``create_all`` checks existence).

    Returns ``{"tables_added": N}``.
    """


    inspector = inspect(engine)
    if "session_lessons" in inspector.get_table_names():
        return {"tables_added": 0}
    Base.metadata.create_all(
        engine,
        tables=[SessionLessonRow.__table__],  # pyright: ignore[reportArgumentType]
    )
    return {"tables_added": 1}


def migrate_add_session_columns(engine: Engine) -> dict[str, int]:
    """Add post-initial columns to ``incidents`` if missing. Idempotent.

    Older on-disk databases may lack ``extra_fields``,
    ``parent_session_id``, or ``dedup_rationale``; SQLAlchemy's read-side
    query then errors with ``no such column``. This walker uses
    ``PRAGMA table_info`` (via SQLAlchemy's ``inspect``) to detect
    missing columns and adds each one nullable. Running on a freshly-
    migrated DB is a no-op.

    Returns ``{"columns_added": N, "indexes_added": M}``.
    """
    inspector = inspect(engine)
    if "incidents" not in inspector.get_table_names():
        # Fresh DB; ``Base.metadata.create_all`` already produced the
        # full schema. Nothing to backfill.
        return {"columns_added": 0, "indexes_added": 0}
    existing_cols = {c["name"] for c in inspector.get_columns("incidents")}
    existing_idx = {i["name"] for i in inspector.get_indexes("incidents")}
    added_cols = 0
    added_idx = 0
    with engine.begin() as conn:
        for col, sql_type in _FORWARD_COLUMNS:
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE incidents ADD COLUMN {col} {sql_type}"))
                added_cols += 1
        for idx_name, table, col in _FORWARD_INDEXES:
            if idx_name in existing_idx:
                continue
            # If the column itself was just added (or already present)
            # the index is safe to create now.
            cols_after = {c["name"] for c in inspect(conn).get_columns(table)}
            if col in cols_after:
                conn.execute(text(f"CREATE INDEX {idx_name} ON {table} ({col})"))
                added_idx += 1
    return {"columns_added": added_cols, "indexes_added": added_idx}

# ====== module: runtime/storage/lesson_store.py ======

_log = logging.getLogger("runtime.storage.lesson_store")


class LessonStore:
    """Append-only lesson corpus with vector similarity lookup.

    Telemetry / refresher writes through ``add(row)``; the intake
    runner reads through ``find_similar(query=...)``.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        vector_store: Optional[VectorStore] = None,
        distance_strategy: str = "cosine",
        similarity_threshold: float = 0.7,
    ) -> None:
        self.engine = engine
        self.vector_store = vector_store
        self.distance_strategy = distance_strategy
        self.similarity_threshold = similarity_threshold

    def add(self, lesson: SessionLessonRow) -> None:
        """Persist ``lesson`` to the relational table AND vector store.

        Relational write goes first so a vector-store hiccup is
        recoverable from disk. Vector failures are logged at WARNING
        and swallowed — the row is still discoverable via SQL lookup
        and the M7 refresher can re-embed on next pass.
        """
        # Snapshot the fields the vector-store call needs BEFORE the
        # SQL transaction commits — once the session closes, the row
        # detaches and attribute access raises DetachedInstanceError.
        lesson_id = lesson.id
        embedding_text = lesson.embedding_text
        source_session_id = lesson.source_session_id
        outcome_status = lesson.outcome_status

        with SqlaSession(self.engine) as s:
            with s.begin():
                s.add(lesson)

        if self.vector_store is None:
            return
        try:
            self.vector_store.add_documents(
                [
                    Document(
                        page_content=embedding_text,
                        metadata={
                            "id": lesson_id,
                            "source_session_id": source_session_id,
                            "outcome_status": outcome_status,
                        },
                    )
                ],
                ids=[lesson_id],
            )
        except Exception:  # noqa: BLE001 — vector backends raise a variety
            _log.warning(
                "LessonStore.add: vector_store write failed for lesson %s; "
                "row is still queryable via SQL",
                lesson_id, exc_info=True,
            )

    def find_similar(
        self,
        *,
        query: str,
        limit: int = 3,
        threshold: Optional[float] = None,
    ) -> list[tuple[SessionLessonRow, float]]:
        """Return up to ``limit`` lessons whose vector similarity to the
        embedded ``query`` is at or above ``threshold``. Returns an
        empty list when no vector store is configured.

        Result tuples are ``(row, similarity)`` sorted by descending
        similarity. Soft-deleted source sessions are not filtered here
        — the caller decides whether to honour them (M9 e2e covers the
        soft-delete-suppression contract).
        """
        if self.vector_store is None:
            return []
        threshold = (
            self.similarity_threshold if threshold is None else threshold
        )


        try:
            raw = self.vector_store.similarity_search_with_score(
                query, k=limit * 4,
            )
        except Exception:  # noqa: BLE001
            _log.warning(
                "LessonStore.find_similar: vector_store query failed",
                exc_info=True,
            )
            return []
        out: list[tuple[SessionLessonRow, float]] = []
        for doc, distance in raw:
            score = distance_to_similarity(
                float(distance), self.distance_strategy,
            )
            if score < threshold:
                continue
            lid = doc.metadata.get("id")
            if not lid:
                continue
            row = self._load(lid)
            if row is None:
                continue
            out.append((row, score))
            if len(out) >= limit:
                break
        return out

    def _load(self, lesson_id: str) -> Optional[SessionLessonRow]:
        with SqlaSession(self.engine) as s:
            return s.get(SessionLessonRow, lesson_id)

# ====== module: runtime/learning/extractor.py ======

EXTRACTOR_VERSION = "1"


def _project_signals(session: Session) -> dict[str, Any]:
    """Carve a JSON-safe dict of categorical signals out of the
    session's ``extra_fields``. Used as the lesson row's queryable
    ``signals`` column — the intake runner can SQL-filter by these
    later.

    The framework is domain-neutral: every str / int / float /
    bool value in ``extra_fields`` becomes a signal. Apps that
    want richer filterability declare their state-class schema and
    the relevant keys flow through automatically.
    """
    extra = session.extra_fields or {}
    out: dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, (str, int, float, bool)) and v is not None:
            out[k] = v
    return out


def _project_tool_sequence(event_log: EventLog, session_id: str) -> list[dict]:
    """Walk the event log; produce a small ``[{tool, args_summary,
    result_kind}]`` list for every ``tool_invoked`` event in order."""
    seq: list[dict] = []
    for ev in event_log.iter_for(session_id):
        if ev.kind != "tool_invoked":
            continue
        seq.append({
            "tool": ev.payload.get("tool"),
            "args_summary": ev.payload.get("args", {}),
            "result_kind": ev.payload.get("result_kind"),
        })
    return seq


def _compose_embedding_text(
    session: Session,
    status: str,
    tool_sequence: list[dict],
    confidence_final: Optional[float],
) -> str:
    """Canonical embedding source. Same inputs -> identical string.

    Form: ``<to_agent_input>\\n\\nOutcome: <status>\\nKey tools:
    [<t1>, <t2>]\\nConfidence: <conf>``. Kept stable across releases
    so M7 can detect unchanged rows without re-embedding.
    """
    tools = [t.get("tool") for t in tool_sequence if t.get("tool")]
    return (
        f"{session.to_agent_input()}\n\n"
        f"Outcome: {status}\n"
        f"Key tools: {tools}\n"
        f"Confidence: {confidence_final}"
    )


class LessonExtractor:
    """Distills a terminal session into a :class:`SessionLessonRow`.

    Pure-function class — no I/O. The caller (orchestrator M4 hook or
    M7 batch refresher) is responsible for persisting the row via
    :class:`LessonStore.add` and emitting a ``lesson_extracted``
    event.
    """

    @staticmethod
    def extract(
        *,
        session: Session,
        event_log: EventLog,
        terminal_statuses: frozenset[str] | None = None,
    ) -> Optional[SessionLessonRow]:
        """Return a :class:`SessionLessonRow` for a terminal session,
        or ``None`` when the session is not in a terminal status.

        ``terminal_statuses`` is the configured terminal-status set
        (typically every name in ``cfg.orchestrator.statuses`` whose
        ``terminal=True``). When ``None``, no status check is applied
        and the extractor produces a row for any session — useful
        for tests that synthesise a pre-resolved session.
        """
        if terminal_statuses is not None and session.status not in terminal_statuses:
            return None

        tool_sequence = _project_tool_sequence(event_log, session.id)
        signals = _project_signals(session)
        confidence_final: Optional[float] = None
        outcome_summary = ""
        if session.agents_run:
            last_run = session.agents_run[-1]
            confidence_final = last_run.confidence
            outcome_summary = last_run.summary

        embedding_text = _compose_embedding_text(
            session,
            session.status,
            tool_sequence,
            confidence_final,
        )

        row = SessionLessonRow(
            id=str(uuid4()),
            source_session_id=session.id,
            created_at=datetime.now(timezone.utc),
            signals=signals,
            tool_sequence=tool_sequence,
            outcome_status=session.status,
            outcome_summary=outcome_summary,
            confidence_final=confidence_final,
            embedding_text=embedding_text,
            provenance={
                "kind": "auto",
                "model": "bge-m3",
                "extractor_version": EXTRACTOR_VERSION,
            },
        )
        # Emit the lesson_extracted event alongside the row so callers
        # need not duplicate the bookkeeping. Telemetry failures are
        # logged and dropped — the row is still returned.
        try:
            event_log.record(
                session.id, "lesson_extracted",
                lesson_id=row.id,
                outcome_status=row.outcome_status,
            )
        except Exception:  # noqa: BLE001 — telemetry must not block extraction
            import logging
            logging.getLogger("runtime.learning.extractor").debug(
                "event_log.record(lesson_extracted) failed", exc_info=True,
            )
        return row

# ====== module: runtime/learning/scheduler.py ======

_log = logging.getLogger("runtime.learning.scheduler")


@dataclass
class RefreshStats:
    """Outcome of a single :meth:`LessonRefresher.run_once` invocation."""

    sessions_scanned: int = 0
    lessons_added: int = 0
    lessons_skipped: int = 0


class LessonRefresher:
    """Nightly refresher for the lesson corpus.

    Constructor wires the three collaborators (engine, lesson_store,
    event_log) so the cron tick can run without touching the global
    orchestrator. Mirrors the
    :class:`runtime.tools.approval_watchdog.ApprovalWatchdog`
    start/stop shape: ``start(loop)`` is idempotent and returns
    immediately; ``stop()`` is a graceful shutdown.

    The actual work happens in :meth:`run_once`, which tests call
    synchronously. The APScheduler-driven cron job is a thin wrapper
    around the same method.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        lesson_store: LessonStore,
        event_log: EventLog,
        terminal_statuses: frozenset[str],
        cron: str = "0 3 * * *",
        window_days: int = 7,
    ) -> None:
        self.engine = engine
        self.lesson_store = lesson_store
        self.event_log = event_log
        self.terminal_statuses = terminal_statuses
        self.cron = cron
        self.window_days = window_days
        self._scheduler: Optional[object] = None
        # Mirror of ApprovalWatchdog's idempotency flag.
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Scheduler lifecycle (cron entry point).
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start an :class:`AsyncIOScheduler` on ``loop`` that fires
        :meth:`run_once` per :attr:`cron`. Idempotent — a second call
        with the same instance returns immediately.

        Called from ``OrchestratorService.start()`` on the service's
        background loop.
        """
        if self._scheduler is not None:
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        async def _arm() -> None:
            self._stopped = False
            scheduler = AsyncIOScheduler(timezone="UTC", event_loop=loop)
            trigger = CronTrigger.from_crontab(self.cron, timezone="UTC")
            scheduler.add_job(
                self._run_once_async,
                trigger=trigger,
                id="lesson_refresher",
                replace_existing=True,
            )
            scheduler.start()
            self._scheduler = scheduler

        fut = asyncio.run_coroutine_threadsafe(_arm(), loop)
        fut.result(timeout=5.0)

    async def stop(self) -> None:
        """Shut the scheduler down. Idempotent and safe to call before
        :meth:`start` or after a previous :meth:`stop`."""
        if self._stopped:
            return
        self._stopped = True
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is None:
            return
        try:
            # AsyncIOScheduler.shutdown is sync but the underlying job
            # cleanup happens on the loop.
            scheduler.shutdown(wait=False)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:  # noqa: BLE001
            _log.warning(
                "LessonRefresher.stop: scheduler shutdown raised",
                exc_info=True,
            )

    async def close(self) -> None:
        """Alias for :meth:`stop`. Provided so callers using
        ``async with`` patterns read naturally."""
        await self.stop()

    # ------------------------------------------------------------------
    # Work — the cron tick + synchronous test entry point.
    # ------------------------------------------------------------------

    async def _run_once_async(self) -> RefreshStats:
        """APScheduler-callable wrapper around :meth:`run_once`."""
        return self.run_once()

    def run_once(self) -> RefreshStats:
        """One refresh pass.

        Walks ``incidents`` for sessions whose ``status`` is in
        :attr:`terminal_statuses` and whose ``updated_at`` falls within
        the last :attr:`window_days`. For each session:

        * Skip if a SessionLessonRow with the current
          ``EXTRACTOR_VERSION`` already exists for ``source_session_id``.
        * Otherwise call :meth:`LessonExtractor.extract` and persist
          via :meth:`LessonStore.add`.

        Returns a :class:`RefreshStats` summary.
        """
        stats = RefreshStats()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.window_days)

        with SqlaSession(self.engine) as s:
            stmt = (
                select(IncidentRow)
                .where(IncidentRow.deleted_at.is_(None))
                .where(IncidentRow.updated_at >= cutoff)
            )
            for row in s.execute(stmt).scalars():
                if row.status not in self.terminal_statuses:
                    continue
                stats.sessions_scanned += 1
                if self._has_current_lesson(s, row.id):
                    stats.lessons_skipped += 1
                    continue
                try:
                    inc = self._row_to_session(row)
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "LessonRefresher: failed to hydrate session %s; skipping",
                        row.id, exc_info=True,
                    )
                    continue
                lesson = LessonExtractor.extract(
                    session=inc,
                    event_log=self.event_log,
                )
                if lesson is None:
                    continue
                try:
                    self.lesson_store.add(lesson)
                    stats.lessons_added += 1
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "LessonRefresher: lesson_store.add failed for %s; "
                        "row stays unwritten this pass",
                        row.id, exc_info=True,
                    )
        _log.info(
            "lesson refresher tick: scanned=%d added=%d skipped=%d",
            stats.sessions_scanned, stats.lessons_added, stats.lessons_skipped,
        )
        return stats

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _has_current_lesson(
        self, session: SqlaSession, source_session_id: str,
    ) -> bool:
        """True iff a lesson row with the CURRENT extractor_version
        already exists for ``source_session_id``. Older version rows
        do NOT block — the refresher rewrites when the version bumps.
        """
        stmt = (
            select(SessionLessonRow)
            .where(SessionLessonRow.source_session_id == source_session_id)
        )
        for row in session.execute(stmt).scalars():
            prov = row.provenance or {}
            if prov.get("extractor_version") == EXTRACTOR_VERSION:
                return True
        return False

    def _row_to_session(self, row: IncidentRow):
        """Hydrate a minimal :class:`runtime.state.Session` from a row.

        Reuses :class:`SessionStore`'s converter so the extractor sees
        the same shape it would in the orchestrator finalize hook.
        """


        # ``state_cls=None`` lets the converter default to the bare
        # framework ``Session`` — the extractor only reads fields
        # declared on the base class (id, status, agents_run,
        # extra_fields, to_agent_input).
        converter = SessionStore(engine=self.engine)
        return converter._row_to_incident(row)

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

# ====== module: runtime/service.py ======

_log = logging.getLogger("runtime.service")

T = TypeVar("T")


@dataclass
class _ActiveSession:
    """In-memory metadata for an in-flight session.

    Lives in ``OrchestratorService._registry``; mutated only on the
    loop thread so the dict itself needs no thread lock. Snapshots are
    produced via :meth:`OrchestratorService.list_active_sessions`,
    which submits a coroutine to the loop and returns a list of plain
    dicts to the calling thread.
    """

    session_id: str
    started_at: str
    status: str = "running"
    current_agent: str | None = None
    task: asyncio.Task | None = None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionCapExceeded(RuntimeError):
    """Raised by ``start_session`` when the service is already running
    ``max_concurrent_sessions`` sessions.

    Fail fast, do not queue. Callers (Streamlit, FastAPI handlers)
    catch this and surface a clear error — Streamlit shows a toast;
    the HTTP layer translates it to a 429 with ``Retry-After``.
    """

    def __init__(self, cap: int) -> None:
        super().__init__(
            f"OrchestratorService at capacity ({cap} concurrent); "
            f"reject incoming start_session"
        )
        self.cap = cap


class OrchestratorService:
    """Process-singleton orchestrator service.

    Surface: construction, singleton accessor, ``start()`` /
    ``shutdown()``, coroutine submission bridge, and the shared MCP
    client pool.

    Thread-safety (HARD-06): ``get_or_create()`` and
    ``_reset_singleton()`` serialise singleton mutation through a
    class-level ``threading.Lock``. Concurrent first-callers
    (Streamlit warmup + FastAPI startup hook racing during process
    boot) all observe the same instance — the loser of the race blocks
    on the lock briefly, then short-circuits on the
    ``_instance is None`` check inside the critical section.
    """

    # Class-level singleton state. Guarded by ``_lock`` so concurrent
    # ``get_or_create()`` callers can't double-construct the service.
    # Reset on ``shutdown()`` via :meth:`_reset_singleton`.
    _lock: threading.Lock = threading.Lock()
    _instance: "OrchestratorService | None" = None

    def __init__(
        self,
        cfg: AppConfig,
        max_concurrent_sessions: int | None = None,
    ) -> None:
        self.cfg = cfg
        # Resource cap. Prefer the explicit constructor arg; fall back
        # to ``cfg.runtime.max_concurrent_sessions``. Tests mutate this
        # attribute directly to drive cap behaviour deterministically.
        self.max_concurrent_sessions: int = (
            max_concurrent_sessions
            if max_concurrent_sessions is not None
            else cfg.runtime.max_concurrent_sessions
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        # Shared MCP client pool — built lazily on first ``get_mcp_client``
        # so processes that never touch MCP pay zero startup cost. All
        # mutations of ``_mcp_clients`` / ``_mcp_locks`` happen on the
        # background loop, so the dicts themselves don't need a thread
        # lock.
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_clients: dict[str, Any] = {}
        self._mcp_locks: dict[str, asyncio.Lock] = {}
        # Per-server-name asyncio.Lock guarding lazy build. Created on the
        # loop the first time the server is requested.
        self._mcp_build_locks: dict[str, asyncio.Lock] = {}
        # Shared Orchestrator (lazy-built on first session start) and
        # the in-flight session registry. The registry dict itself is
        # only mutated from the loop thread (writers go through
        # ``submit_and_wait``); readers also hop through the loop so the
        # snapshot is point-in-time consistent with concurrent mutators.
        self._orch: Any | None = None
        self._registry: dict[str, _ActiveSession] = {}
        # Lazily-built lock for serialising orchestrator construction
        # under concurrent ``start_session`` calls. Created on the loop.
        self._orch_build_lock: asyncio.Lock | None = None
        # Pending-approval timeout watchdog. Started in ``start()`` iff
        # ``cfg.runtime.gateway`` is configured; otherwise None and the
        # lifecycle hooks are no-ops.
        self._approval_watchdog: Any | None = None
        # M7 nightly lesson refresher. Started in ``start()`` iff the
        # orchestrator has a lesson_store; otherwise None (the lifecycle
        # hooks short-circuit).
        self._lesson_refresher: Any | None = None

    @classmethod
    def get_or_create(cls, cfg: AppConfig) -> "OrchestratorService":
        """Return the process-singleton service, building it on first call.

        Subsequent calls ignore the supplied ``cfg`` and return the
        existing instance — there is exactly one orchestrator service per
        Python process. To rebuild with a new config, call
        ``shutdown()`` first.

        Thread-safe (HARD-06): the check-and-construct pair runs inside
        a class-level ``threading.Lock``. A concurrent second caller
        either blocks until the first caller's ``__init__`` returns and
        then short-circuits on the ``_instance is not None`` check, or
        wins the race and constructs alone — no double construction.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    def start(self) -> None:
        """Spin up the background thread + asyncio loop.

        Idempotent: a no-op if the loop is already running. Blocks until
        the background thread reports the loop is ready (5s timeout) so
        callers can ``submit()`` immediately after ``start()`` returns.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="OrchestratorService",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("OrchestratorService loop failed to start within 5s")
        # Arm the pending-approval watchdog iff a gateway is configured.
        # The watchdog is harmless when no high-risk tool calls ever
        # fire (it scans the empty registry), but skipping the start
        # when the gateway is off keeps process startup quiet for apps
        # that have not opted into HITL.
        gateway_cfg = getattr(self.cfg.runtime, "gateway", None)
        if gateway_cfg is not None:


            timeout_s = getattr(
                gateway_cfg, "approval_timeout_seconds", 3600,
            )
            self._approval_watchdog = ApprovalWatchdog(
                self,
                approval_timeout_seconds=timeout_s,
            )
            self._approval_watchdog.start(self._loop)


    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            # Drain any remaining tasks before closing so no coroutine is
            # left dangling without a chance to clean up.
            try:
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                self._loop.close()

    def submit(
        self, coro: Awaitable[T]
    ) -> concurrent.futures.Future[T]:
        """Submit a coroutine to the background loop from any thread.

        Returns a ``concurrent.futures.Future`` whose ``.result()`` blocks
        the calling thread until the coroutine resolves on the loop. Safe
        to call concurrently from multiple threads.
        """
        if self._loop is None:
            raise RuntimeError(
                "OrchestratorService not started; call start() first"
            )
        if not self._loop.is_running():
            raise RuntimeError("OrchestratorService loop is not running")
        # Public signature accepts ``Awaitable[T]`` for caller flexibility;
        # ``run_coroutine_threadsafe`` requires a ``Coroutine``. Every
        # in-tree caller passes ``async def fn()`` — a Coroutine — so the
        # cast is sound. Outside callers passing a non-coroutine
        # Awaitable would already fail at runtime.
        return asyncio.run_coroutine_threadsafe(
            cast(Coroutine[Any, Any, T], coro), self._loop,
        )

    def submit_and_wait(
        self, coro: Awaitable[T], timeout: float | None = None
    ) -> T:
        """Submit a coroutine and block the caller until it resolves.

        Convenience wrapper for sync callers (Streamlit, FastAPI request
        handlers, CLI). Raises ``concurrent.futures.TimeoutError`` if the
        coroutine doesn't complete within ``timeout`` seconds.

        WARNING: do not call from an async function whose event loop is
        the same loop ``OrchestratorService`` is hosting (e.g. tests using
        ``httpx.AsyncClient + ASGITransport`` against the FastAPI app
        share the same loop the service runs on). The caller would block
        the loop while waiting for work scheduled onto that same loop —
        a deadlock. Use :meth:`submit_async` from async code.
        """
        return self.submit(coro).result(timeout=timeout)

    async def submit_async(self, coro: Awaitable[T]) -> T:
        """Bridge a coroutine onto the service's background loop, awaitable
        from any caller's loop.

        Async equivalent of :meth:`submit_and_wait`. ``asyncio.wrap_future``
        exposes the cross-thread ``concurrent.futures.Future`` returned by
        ``run_coroutine_threadsafe`` as awaitable on the calling loop, so
        the caller yields control while the work runs on the service's
        loop. Safe to call from a request handler whose event loop is the
        same one the service is hosting (no deadlock).
        """
        if self._loop is None:
            raise RuntimeError(
                "OrchestratorService not started; call start() first"
            )
        if not self._loop.is_running():
            raise RuntimeError("OrchestratorService loop is not running")
        # See ``submit`` above for the Awaitable-vs-Coroutine cast.
        fut = asyncio.run_coroutine_threadsafe(
            cast(Coroutine[Any, Any, T], coro), self._loop,
        )
        return await asyncio.wrap_future(fut)

    async def get_mcp_client(self, server_name: str) -> Any:
        """Return the shared FastMCP client for ``server_name``, building
        on first request.

        Lookup is serialised via a per-server ``asyncio.Lock`` so two
        concurrent sessions racing for the same server don't double-build
        the client. The clients themselves are reused across all sessions
        for the lifetime of the service; teardown happens in
        :meth:`shutdown`.

        Raises ``KeyError`` if ``server_name`` is not declared in
        ``cfg.mcp.servers``.
        """
        # Build-lock dict mutation must happen on the loop; we *are* on
        # the loop here (this is an async method).
        if server_name not in self._mcp_build_locks:
            self._mcp_build_locks[server_name] = asyncio.Lock()
        async with self._mcp_build_locks[server_name]:
            if server_name in self._mcp_clients:
                return self._mcp_clients[server_name]
            server_cfg = next(
                (s for s in self.cfg.mcp.servers if s.name == server_name),
                None,
            )
            if server_cfg is None:
                raise KeyError(
                    f"MCP server {server_name!r} not declared in cfg.mcp.servers"
                )
            if self._mcp_stack is None:
                self._mcp_stack = AsyncExitStack()
                await self._mcp_stack.__aenter__()
            client = build_fastmcp_client(server_cfg)
            await self._mcp_stack.enter_async_context(client)
            self._mcp_clients[server_name] = client
            self._mcp_locks[server_name] = asyncio.Lock()
            return client

    def lock_for(self, server_name: str) -> asyncio.Lock:
        """Return the per-server ``asyncio.Lock`` that serialises tool
        calls against a single FastMCP client.

        Must be called after ``get_mcp_client(server_name)`` has built
        the client, otherwise ``KeyError``.
        """
        return self._mcp_locks[server_name]

    # ------------------------------------------------------------------
    # Per-session task scheduling + in-flight registry
    # ------------------------------------------------------------------

    async def _ensure_orchestrator(self) -> Any:
        """Lazily build the shared ``Orchestrator`` on the loop thread.

        Concurrent ``start_session`` calls coordinate through
        ``_orch_build_lock`` so we never build the orchestrator twice.
        Returns the cached instance on subsequent calls.
        """
        # Build-lock construction must happen on the loop. We *are* on
        # the loop here (this is an async method invoked via the bridge).
        if self._orch_build_lock is None:
            self._orch_build_lock = asyncio.Lock()
        async with self._orch_build_lock:
            if self._orch is None:
                # Lazy import to avoid a circular dependency at module
                # load time (orchestrator transitively imports a lot).

                self._orch = await Orchestrator.create(self.cfg)
                # M7: nightly lesson refresher. Wired on first
                # orchestrator build so the engine + lesson_store +
                # event_log handles are already populated.
                self._maybe_start_lesson_refresher(self._orch)
            return self._orch

    def _maybe_start_lesson_refresher(self, orch: Any) -> None:
        """Arm the M7 nightly refresher on first orchestrator build.
        No-op when the orchestrator has no lesson_store / event_log
        (test fixtures, apps that disable the corpus) or when the
        refresher is already armed."""
        if self._lesson_refresher is not None:
            return
        lesson_store = getattr(orch, "lesson_store", None)
        event_log = getattr(orch, "event_log", None)
        if lesson_store is None or event_log is None:
            return


        framework_cfg = getattr(orch, "framework_cfg", None)
        cron = getattr(framework_cfg, "lesson_refresh_cron", "0 3 * * *")
        window_days = getattr(framework_cfg, "lesson_refresh_window_days", 7)
        terminal_statuses = frozenset(
            name for name, sdef in self.cfg.orchestrator.statuses.items()
            if getattr(sdef, "terminal", False)
        )
        if not terminal_statuses or self._loop is None:
            return
        self._lesson_refresher = LessonRefresher(
            engine=orch.store.engine,
            lesson_store=lesson_store,
            event_log=event_log,
            terminal_statuses=terminal_statuses,
            cron=cron,
            window_days=window_days,
        )
        try:
            self._lesson_refresher.start(self._loop)
        except Exception:  # noqa: BLE001 — don't break orch build on cron failure
            _log.warning(
                "LessonRefresher start failed; corpus refresh disabled",
                exc_info=True,
            )
            self._lesson_refresher = None

    def start_session(
        self,
        *,
        query: str = "",
        state_overrides: dict | None = None,
        environment: str | None = None,
        submitter: dict | None = None,
        reporter_id: str | None = None,
        reporter_team: str | None = None,
        trigger: Any | None = None,
    ) -> str:
        """Start a new agent session. Returns the session id immediately.

        The session row is created (and the id minted) synchronously on
        the loop so the caller has a stable handle before this method
        returns. The actual graph run is launched as an ``asyncio.Task``
        on the same loop and runs in the background — the caller does
        **not** block on it. Listen via :meth:`list_active_sessions` and
        per-session state lookups for progress.

        ``state_overrides`` is a free-form dict of domain fields the app
        stamps onto the new session row. The framework only projects
        ``environment`` onto the storage column today; other keys ride
        through to app-specific MCP tools.

        ``submitter`` is a free-form dict the calling app interprets.
        For incident-management it is ``{"id": "...", "team": "..."}``;
        other apps can carry app-specific keys (e.g. code-review's
        ``{"id": "<github-username>", "pr_url": "..."}``). The framework
        only projects ``id``/``team`` onto the row's reporter columns.

        Deprecated kwargs (coerced and warned):
          * ``environment`` -> ``state_overrides={"environment": ...}``
          * ``reporter_id`` / ``reporter_team`` -> ``submitter``

        The registry entry is evicted by a ``Task.add_done_callback`` on
        completion, cancellation, or failure — so a session that crashes
        does not leak a stale entry.
        """



        # Resolve the generic ``submitter`` and ``state_overrides`` once
        # on the caller's thread — the deprecation warnings fire here
        # (in the user's frame), not deep inside the loop's ``_scheduler``.
        resolved_overrides = _coerce_state_overrides(
            state_overrides, environment,
        )
        resolved_submitter = _coerce_submitter(
            submitter, reporter_id, reporter_team
        )
        sub_id = (resolved_submitter or {}).get("id", "user-mock")
        sub_team = (resolved_submitter or {}).get("team", "platform")
        env = (resolved_overrides or {}).get("environment", "")

        async def _scheduler() -> str:
            # Enforce the concurrency cap on the loop thread so the
            # registry size check is race-free. Fail-fast with
            # ``SessionCapExceeded``; the exception propagates through
            # ``submit_and_wait`` -> ``Future.result()`` to the caller.
            if len(self._registry) >= self.max_concurrent_sessions:
                raise SessionCapExceeded(self.max_concurrent_sessions)
            orch = await self._ensure_orchestrator()
            # Allocate the row (and its id) synchronously on the loop
            # so the caller gets a stable id back. The graph then runs
            # in a separate task — registration happens here, before
            # the task is created, so ``list_active_sessions`` sees the
            # entry immediately.
            inc = orch.store.create(
                query=query,
                environment=env,
                reporter_id=sub_id,
                reporter_team=sub_team,
            )
            session_id = inc.id
            # Stamp trigger provenance onto the row before the graph
            # runs so any crash mid-graph still leaves an audit trail.
            # ``inc.findings`` is a JSON dict on the row.
            if trigger is not None:
                try:
                    received_at = trigger.received_at.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                except Exception:  # noqa: BLE001
                    received_at = _utc_iso_now()
                inc.findings["trigger"] = {
                    "name": getattr(trigger, "name", None),
                    "transport": getattr(trigger, "transport", None),
                    "target_app": getattr(trigger, "target_app", None),
                    "received_at": received_at,
                }
                orch.store.save(inc)
            entry = _ActiveSession(
                session_id=session_id,
                started_at=_utc_iso_now(),
            )
            self._registry[session_id] = entry

            async def _run() -> None:
                # Fail-fast on contention (D-03): if another task already
                # holds the session lock, refuse the new turn immediately.
                if orch._locks.is_locked(session_id):

                    raise SessionBusy(session_id)
                # Hold the per-session lock for the full graph turn,
                # including any HITL interrupt() pause (D-01).
                async with orch._locks.acquire(session_id):
                    try:
                        await orch.graph.ainvoke(
                            GraphState(
                                session=inc,
                                next_route=None,
                                last_agent=None,
                                error=None,
                            ),
                            config=orch._thread_config(session_id),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # Phase 11 (FOC-04 / D-11-04): GraphInterrupt is a
                        # pending-approval pause, not a failure. Don't stamp
                        # status='error' on the registry entry -- let
                        # LangGraph's checkpointer hold the paused state
                        # and let the UI's Approve/Reject action drive
                        # resume.
                        try:
                            from langgraph.errors import GraphInterrupt
                            if isinstance(exc, GraphInterrupt):
                                # Propagate so the underlying Task
                                # observer (stop_session etc.) still
                                # sees the exception, but skip the
                                # status='error' write.
                                raise
                        except ImportError:  # pragma: no cover
                            pass
                        # Mark the registry entry so any concurrent snapshot
                        # observes the failure before the done-callback
                        # evicts it. The exception itself is preserved on
                        # the task object for ``stop_session`` and any
                        # other observer that holds a Task reference.
                        e = self._registry.get(session_id)
                        if e is not None:
                            e.status = "error"
                        raise

            task = asyncio.create_task(_run(), name=f"session:{session_id}")
            entry.task = task

            # Eviction is loop-local: ``add_done_callback`` fires on the
            # loop thread, so the dict mutation is single-threaded.
            def _evict(_t: asyncio.Task) -> None:
                self._registry.pop(session_id, None)

            task.add_done_callback(_evict)
            return session_id

        return self.submit_and_wait(_scheduler(), timeout=30.0)

    # ------------------------------------------------------------------
    # stop_session — cancel in-flight task + persist stopped status
    # ------------------------------------------------------------------

    def stop_session(self, session_id: str) -> None:
        """Cancel an in-flight session and mark its row ``status="stopped"``.

        Idempotent: calling on an unknown id, an already-stopped session,
        or a session that completed naturally is a no-op (does not raise).
        Also clears ``pending_intervention`` so a session interrupted
        mid-resume doesn't leave a stale prompt on the row.

        Partial work (recorded ``tool_calls``, ``agents_run``) is
        preserved — they are written as they happen, and stopping is
        not a rollback.
        """

        async def _stop() -> None:
            entry = self._registry.get(session_id)
            task = entry.task if entry is not None else None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception:  # noqa: BLE001
                    # The graph itself may have raised; we still want to
                    # mark the row stopped below. Swallow here, but log
                    # so post-mortem reveals the underlying failure.
                    _log.warning(
                        "stop_session: graph raised during cancel-await for %s",
                        session_id,
                        exc_info=True,
                    )
            # Persist the stopped status. The orchestrator may not have
            # been built yet (caller passed an unknown id before any
            # session ran) — in that case there's nothing to persist.
            orch = self._orch
            if orch is not None:
                try:
                    inc = orch.store.load(session_id)
                except Exception:  # noqa: BLE001
                    # Unknown id: nothing to persist; treat as no-op. A
                    # genuine store failure is still observable via the log.
                    _log.debug(
                        "stop_session: store.load(%s) failed; treating as unknown id",
                        session_id,
                        exc_info=True,
                    )
                    inc = None
                if inc is not None:
                    inc.status = "stopped"
                    inc.pending_intervention = None
                    orch.store.save(inc)
            # Drop the registry entry if the done-callback didn't already
            # evict it (it always does, but be defensive).
            self._registry.pop(session_id, None)

        # If the loop isn't running (caller stopped the service), be a
        # silent no-op rather than raising — keeps idempotency guarantees.
        if self._loop is None or not self._loop.is_running():
            return
        self.submit_and_wait(_stop(), timeout=10.0)

    # ------------------------------------------------------------------
    # Active-session registry snapshot accessor
    # ------------------------------------------------------------------

    def list_active_sessions(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of in-flight sessions.

        The snapshot coroutine runs on the loop thread, so the view is
        point-in-time consistent w.r.t. concurrent registry mutators
        (which also run on the loop). Each entry is a plain ``dict``
        with ``session_id``, ``status``, ``started_at``, and
        ``current_agent`` keys — callers in any thread can pass it
        around without holding any asyncio resources.

        Returns an empty list when the service has never run a session
        or when every previously-started run has completed.
        """

        async def _snapshot() -> list[dict[str, Any]]:
            return [
                {
                    "session_id": e.session_id,
                    "status": e.status,
                    "started_at": e.started_at,
                    "current_agent": e.current_agent,
                }
                for e in self._registry.values()
            ]

        return self.submit_and_wait(_snapshot(), timeout=5.0)

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the loop, tear down MCP clients, join the thread,
        reset the singleton.

        Idempotent: safe to call multiple times, including after the
        loop has already been torn down. Resets the module-level
        singleton so ``get_or_create()`` will rebuild on the next call.
        """
        if self._loop is None:
            self._reset_singleton()
            return
        loop = self._loop
        thread = self._thread
        # Stop the watchdog before draining sessions so its scan
        # doesn't race against the registry teardown below.
        if loop.is_running() and self._approval_watchdog is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._approval_watchdog.stop(), loop,
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: shutdown must continue even if the watchdog
                # refuses to stop cleanly. Surface the cause so it doesn't
                # silently rot.
                _log.warning(
                    "shutdown: approval watchdog stop failed",
                    exc_info=True,
                )
            self._approval_watchdog = None
        # M7: stop the nightly lesson refresher symmetrically with the
        # watchdog. Same best-effort discipline.
        if loop.is_running() and self._lesson_refresher is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._lesson_refresher.stop(), loop,
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "shutdown: lesson refresher stop failed", exc_info=True,
                )
            self._lesson_refresher = None
        # Cancel in-flight session tasks first so they observe a
        # CancelledError before the orchestrator's underlying
        # resources (DB engine, FastMCP transports) are torn down.
        if loop.is_running() and self._registry:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._cancel_all_sessions(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: a stuck task that ignores cancellation must
                # not block the loop teardown below. Surface for diagnosis.
                _log.warning(
                    "shutdown: cancel_all_sessions failed",
                    exc_info=True,
                )
        # Close the shared orchestrator on the loop, releasing its
        # checkpointer connection / MCP exit-stack.
        if loop.is_running() and self._orch is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._close_orchestrator(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: a misbehaving aclose() must not block
                # the loop / thread join below. Surface for diagnosis.
                _log.warning(
                    "shutdown: orchestrator close failed",
                    exc_info=True,
                )
        # Close MCP clients on the loop *before* stopping it.
        if loop.is_running() and self._mcp_stack is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._close_mcp_pool(), loop
                )
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001
                # Best-effort: don't block shutdown on a misbehaving
                # client. Log so diagnostics survive the silent cleanup.
                _log.warning(
                    "shutdown: MCP pool close failed",
                    exc_info=True,
                )
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)
        self._loop = None
        self._thread = None
        self._started.clear()
        self._mcp_stack = None
        self._mcp_clients.clear()
        self._mcp_locks.clear()
        self._mcp_build_locks.clear()
        self._orch = None
        self._orch_build_lock = None
        self._registry.clear()
        self._approval_watchdog = None
        self._reset_singleton()

    async def _cancel_all_sessions(self) -> None:
        """Cancel every in-flight session task and wait for them to exit.

        Runs on the loop thread. Each task gets up to 5s to honour the
        ``CancelledError``; misbehaving tasks that ignore cancellation
        do not block shutdown beyond that — ``run_loop`` will sweep
        them in its final ``gather`` pass.
        """
        tasks = [e.task for e in self._registry.values() if e.task is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._registry.clear()

    async def _close_orchestrator(self) -> None:
        if self._orch is None:
            return
        orch = self._orch
        self._orch = None
        try:
            await orch.aclose()
        except Exception:  # noqa: BLE001
            # Best-effort cleanup: a checkpointer / MCP exit-stack that
            # blew up on close still leaves the process to exit cleanly.
            # Surface so the failure is observable post-mortem.
            _log.warning(
                "_close_orchestrator: orch.aclose() failed",
                exc_info=True,
            )

    async def _close_mcp_pool(self) -> None:
        if self._mcp_stack is None:
            return
        stack = self._mcp_stack
        self._mcp_stack = None
        await stack.__aexit__(None, None, None)
        self._mcp_clients.clear()
        self._mcp_locks.clear()
        self._mcp_build_locks.clear()

    @classmethod
    def _reset_singleton(cls) -> None:
        """Clear the class-level singleton under the same lock that
        ``get_or_create`` uses — so a reset racing with a fresh
        ``get_or_create`` call cannot leak the stale instance.
        """
        with cls._lock:
            cls._instance = None

# ====== module: runtime/agents/turn_output.py ======

_LOG = logging.getLogger("runtime.orchestrator")

# D-10-03 — heuristic tolerance for envelope-vs-tool-arg confidence mismatch.
# Inclusive boundary (|env - tool| <= 0.05 is silent). Documented for future
# tuning; widening is cheap, narrowing requires care because the LLM's
# self-reported turn confidence is naturally ~5pp noisier than its
# tool-call-time confidence.
_DEFAULT_TOLERANCE: float = 0.05


class AgentTurnOutput(BaseModel):
    """Structural envelope every agent invocation MUST emit.

    The framework wires this as ``response_format=AgentTurnOutput`` on both
    ``create_agent`` call sites (``runtime.graph`` and
    ``runtime.agents.responsive``). Pydantic's ``extra="forbid"`` keeps the
    contract narrow — adding fields is a deliberate schema migration, not a
    free-for-all.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        min_length=1,
        description="Final user-facing message text.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence in this turn's output: "
            "0.85+ strong, 0.5 hedged, <0.4 weak."
        ),
    )
    confidence_rationale: str = Field(
        min_length=1,
        description="One-sentence explanation of the confidence value.",
    )
    signal: str | None = Field(
        default=None,
        description=(
            "Optional next-state signal "
            "(e.g. success | failed | needs_input | default). "
            "Routing layer validates the vocabulary."
        ),
    )


class EnvelopeMissingError(Exception):
    """Raised by :func:`parse_envelope_from_result` when neither
    ``result["structured_response"]`` nor a JSON-shaped final AIMessage
    yields a valid :class:`AgentTurnOutput`.

    Carries structured cause attributes (``agent``, ``field``) so the
    runner can mark the agent_run as ``error`` with a precise reason.
    """

    def __init__(self, *, agent: str, field: str, message: str | None = None):
        self.agent = agent
        self.field = field
        super().__init__(message or f"envelope_missing: {field} (agent={agent})")


def parse_envelope_from_result(
    result: dict,
    *,
    agent: str,
) -> AgentTurnOutput:
    """Extract an :class:`AgentTurnOutput` from a ``create_react_agent`` result.

    Three-step defensive fallback (Risk #1 — Ollama may not honor
    ``response_format`` cleanly across all providers):

    1. ``result["structured_response"]`` — preferred path; LangGraph 1.1.x
       populates it when ``response_format`` is set and the LLM honors
       structured output.
    2. ``result["messages"][-1].content`` parsed as JSON, validated against
       :class:`AgentTurnOutput` — covers providers that stuff envelope JSON
       in the AIMessage body instead of a separate structured field.
    3. Both fail → :class:`EnvelopeMissingError` so the runner marks
       agent_run ``error`` with a structured cause.
    """
    # Path 1: structured_response (preferred)
    sr = result.get("structured_response")
    if isinstance(sr, AgentTurnOutput):
        return sr
    if isinstance(sr, dict):
        try:
            return AgentTurnOutput.model_validate(sr)
        except Exception:  # noqa: BLE001
            # Path 1 produced a dict that doesn't match the envelope
            # schema. Fall through to Path 2 (parse last AIMessage), but
            # log so providers shipping malformed structured_response are
            # observable instead of silently degraded.
            _LOG.debug(
                "envelope path 1 (structured_response dict) failed validation; "
                "falling through to AIMessage JSON parse",
                exc_info=True,
            )

    # Path 2: JSON-parse last AIMessage content
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            return AgentTurnOutput.model_validate(payload)
        except Exception:  # noqa: BLE001
            continue
        break

    # Path 3: fail loudly
    raise EnvelopeMissingError(
        agent=agent,
        field="structured_response",
        message=(
            f"envelope_missing: no structured_response or JSON-decodable "
            f"AIMessage envelope found (agent={agent})"
        ),
    )


def reconcile_confidence(
    envelope_value: float,
    tool_arg_value: float | None,
    *,
    agent: str,
    session_id: str,
    tool_name: str | None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    """Reconcile envelope confidence against typed-terminal-tool-arg confidence.

    D-10-03 contract:
    - When ``tool_arg_value`` is None: return envelope value silently.
    - When both present and ``|envelope - tool_arg| <= tolerance``: return
      tool-arg silently (tool-arg wins on the return regardless — it's the
      finer-grained, gated value).
    - When both present and ``|envelope - tool_arg| > tolerance``: log INFO
      with the verbatim format from CONTEXT.md / D-10-03 and return tool-arg.

    Log shape (preserved verbatim for grep-based observability assertions):
        ``runtime.orchestrator: turn.confidence_mismatch agent={a} turn_value={e:.2f} tool_value={t:.2f} tool={tn} session_id={sid}``
    """
    if tool_arg_value is None:
        return envelope_value
    diff = abs(envelope_value - tool_arg_value)
    if diff > tolerance:
        _LOG.info(
            "turn.confidence_mismatch "
            "agent=%s turn_value=%.2f tool_value=%.2f tool=%s session_id=%s",
            agent,
            envelope_value,
            tool_arg_value,
            tool_name,
            session_id,
        )
    return tool_arg_value


__all__ = [
    "AgentTurnOutput",
    "EnvelopeMissingError",
    "parse_envelope_from_result",
    "reconcile_confidence",
]

# ====== module: runtime/tools/gateway.py ======

if TYPE_CHECKING:
    pass
_log = logging.getLogger("runtime.tools.gateway")

GatewayAction = Literal["auto", "notify", "approve"]

_RISK_TO_ACTION: dict[str, GatewayAction] = {
    "low": "auto",
    "medium": "notify",
    "high": "approve",
}

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def effective_action(
    tool_name: str,
    *,
    env: str | None,
    gateway_cfg: GatewayConfig | None,
) -> GatewayAction:
    """Resolve the effective gateway action for a tool invocation.

    Order of evaluation (the prod-override predicate runs FIRST so it can
    only TIGHTEN the action — never relax it):

      1. ``gateway_cfg is None`` -> ``"auto"`` (gateway disabled).
      2. Prod override: if ``cfg.prod_overrides`` is configured AND
         ``env`` is in ``prod_environments`` AND ``tool_name`` matches
         one of the ``resolution_trigger_tools`` globs -> ``"approve"``.
      3. Risk-tier lookup: ``cfg.policy.get(tool_name)`` mapped via
         ``low->auto``, ``medium->notify``, ``high->approve``.
      4. No policy entry -> ``"auto"`` (safe default).

    Tool-name lookups try the fully-qualified name (``<server>:<tool>``,
    as registered by ``runtime.mcp_loader``) FIRST, then the bare
    suffix as a fallback. This lets app config use bare names without
    knowing the server prefix while keeping prefixed-form policy keys
    deterministically more specific. Globs in
    ``resolution_trigger_tools`` are matched against both forms for
    the same reason, prefixed first.

    The function is pure: same inputs always yield the same output and
    no argument is mutated.
    """
    if gateway_cfg is None:
        return "auto"

    bare = tool_name.split(":", 1)[1] if ":" in tool_name else None

    overrides = gateway_cfg.prod_overrides
    if overrides is not None and env and env in overrides.prod_environments:
        for pattern in overrides.resolution_trigger_tools:
            if fnmatchcase(tool_name, pattern):
                return "approve"
            if bare is not None and fnmatchcase(bare, pattern):
                return "approve"

    risk = gateway_cfg.policy.get(tool_name)
    if risk is not None:
        return _RISK_TO_ACTION[risk]
    if bare is not None:
        risk = gateway_cfg.policy.get(bare)
        if risk is not None:
            return _RISK_TO_ACTION[risk]
    return "auto"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_UTC_TS_FMT)


def _find_pending_index(
    tool_calls: list,
    tool_name: str,
    ts: str,
) -> int | None:
    """Locate the index of the ``pending_approval`` ToolCall row that
    matches ``tool_name`` and ``ts``.

    Used by the wrap_tool resume path to update the in-place audit row
    rather than appending a duplicate. The watchdog may have replaced
    the row with a ``timeout`` entry while the graph was paused — in
    that case we return ``None`` and the resume path leaves the audit
    list unchanged (the watchdog already wrote the canonical record).

    Searches from the end of the list because the pending row is
    almost always the most recent ToolCall.
    """
    for idx in range(len(tool_calls) - 1, -1, -1):
        tc = tool_calls[idx]
        if (getattr(tc, "tool", None) == tool_name
                and getattr(tc, "ts", None) == ts
                and getattr(tc, "status", None) == "pending_approval"):
            return idx
    return None


def _find_existing_pending_index(
    tool_calls: list,
    tool_name: str,
) -> int | None:
    """Find the most recent ``pending_approval`` row for ``tool_name``.

    LangGraph's interrupt/resume model re-runs the gated node from the
    top after ``Command(resume=...)``; we re-use the existing pending
    row rather than appending a duplicate every time the closure
    re-enters the approve branch.
    """
    for idx in range(len(tool_calls) - 1, -1, -1):
        tc = tool_calls[idx]
        if (getattr(tc, "tool", None) == tool_name
                and getattr(tc, "status", None) == "pending_approval"):
            return idx
    return None


def _evaluate_gate(
    *,
    session: Session,
    tool_name: str,
    gate_policy: GatePolicy | None,
    gateway_cfg: GatewayConfig | None,
) -> "GateDecision":
    """Phase 11 (FOC-04) bridge: invoke ``should_gate`` from the wrap.

    Constructs a minimal ``ToolCall`` shape for the pure-function
    boundary, and a temporary ``OrchestratorConfig`` shim with the
    in-flight ``gate_policy`` + ``gateway`` so the pure function sees
    a single config object (its declared signature).

    When ``gate_policy`` is ``None`` -- the legacy callers that have
    not yet been threaded -- a default ``GatePolicy()`` is used so
    Phase-11 behaviour applies uniformly. The default mirrors v1.0
    HITL behaviour (``gated_risk_actions={"approve"}``), so existing
    pre-Phase-11 tests keep passing.
    """
    # Local imports (avoid cycle on policy.py importing gateway).
    # ``GateDecision`` is type-only here -- the lazy import sits in the
    # TYPE_CHECKING block at module top.



    effective_policy = gate_policy if gate_policy is not None else GatePolicy()
    # OrchestratorConfig has model_config={"extra": "forbid"} so we
    # cannot stash gateway as a top-level field. We thread gateway via
    # the cfg.gateway lookup that should_gate already performs via
    # ``getattr(cfg, "gateway", None)``. Building a transient cfg with
    # gate_policy and a stashed gateway attr is the smallest-diff
    # pathway -- avoids changing should_gate's signature.
    cfg = OrchestratorConfig(gate_policy=effective_policy)
    object.__setattr__(cfg, "gateway", gateway_cfg)

    minimal_tc = ToolCall(
        agent="",
        tool=tool_name,
        args={},
        result=None,
        ts=_now_iso(),
        risk="low",
        status="executed",
    )
    confidence = getattr(session, "turn_confidence_hint", None)
    decision: GateDecision = should_gate(
        session=session, tool_call=minimal_tc, confidence=confidence, cfg=cfg,
    )
    return decision


class _GatedToolMarker(BaseTool):
    """Marker base class so ``isinstance(t, _GatedToolMarker)`` identifies
    a tool that has already been wrapped by :func:`wrap_tool`. Used to
    short-circuit ``wrap_tool(wrap_tool(t))`` and avoid wrapper recursion.

    Not instantiated directly — every ``_GatedTool`` defined inside
    :func:`wrap_tool` inherits from this.
    """

    name: str = "_gated_marker"
    description: str = "internal — never invoked"

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("marker base — _GatedTool overrides this")


def wrap_tool(
    base_tool: BaseTool,
    *,
    session: Session,
    gateway_cfg: GatewayConfig | None,
    agent_name: str = "",
    store: "SessionStore | None" = None,
    injected_args: dict[str, str] | None = None,
    gate_policy: GatePolicy | None = None,
    event_log: "EventLog | None" = None,
) -> BaseTool:
    """Wrap ``base_tool`` so every invocation passes through the gateway.

    The factory closes over ``session`` and ``gateway_cfg`` so the live
    audit log (``session.tool_calls``) is the same instance the rest of
    the orchestrator reads — no detour through a separate audit table.

    Returned object is a ``BaseTool`` subclass instance whose ``name``
    and ``description`` mirror the underlying tool, so LangGraph's ReAct
    prompt builder still sees the right tool surface.

    Idempotent: wrapping an already-gated tool returns it unchanged so a
    second ``wrap_tool(wrap_tool(t))`` does not nest wrappers (which would
    cause unbounded recursion when ``_run`` calls ``inner.invoke`` and
    that dispatches back into another ``_GatedTool._run``).

    Phase 9 (D-09-01 / D-09-03): when ``injected_args`` is supplied, the
    gateway expands ``kwargs`` with session-derived values BEFORE
    ``effective_action`` is consulted — so the gateway's risk-rating
    sees the canonical ``environment`` (avoiding T-09-05: gateway
    misclassifies prod as auto because env was missing from the LLM
    args).
    """
    if isinstance(base_tool, _GatedToolMarker):
        return base_tool

    env = getattr(session, "environment", None)
    inner = base_tool
    inject_cfg = injected_args or {}

    # Phase 9 (D-09-01): the LLM-visible args_schema on the wrapper must
    # exclude every injected key — otherwise BaseTool's input validator
    # rejects the call when the LLM omits a "required" arg the framework
    # is about to supply. The inner tool keeps its full schema so the
    # downstream invoke still sees every kwarg.
    if inject_cfg:

        _llm_visible_schema = strip_injected_params(
            inner, frozenset(inject_cfg.keys()),
        ).args_schema
    else:
        _llm_visible_schema = inner.args_schema

    # Phase 9 follow-up: compute the set of param names the inner tool
    # actually accepts so injection skips keys the target tool doesn't
    # declare. Without this filter, a config-wide ``injected_args``
    # entry like ``session_id: session.id`` is unconditionally written
    # to every tool's kwargs — tools that don't accept ``session_id``
    # then raise pydantic ``unexpected_keyword`` errors at the FastMCP
    # validation boundary. ``accepted_params_for_tool`` handles both
    # pydantic-model and JSON-Schema-dict ``args_schema`` shapes.

    _accepted_params: frozenset[str] | None = accepted_params_for_tool(inner)

    def _sync_invoke_inner(payload: Any) -> Any:
        """Sync-invoke the inner tool, translating BaseTool's
        default-``_run`` ``NotImplementedError`` into a clearer message
        for native-async-only tools. Without this, callers see a vague
        ``NotImplementedError`` from langchain core with no hint that
        the right path is ``ainvoke``."""
        try:
            return inner.invoke(payload)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Tool {inner.name!r} appears to be async-only "
                f"(``_run`` not implemented). Use ``ainvoke`` / ``_arun`` "
                f"for this tool instead of the sync invoke path."
            ) from exc

    # Tool-naming regex differs across LLM providers — Ollama allows
    # ``[a-zA-Z0-9_.\-]{1,256}``, OpenAI is stricter at
    # ``^[a-zA-Z0-9_-]+$`` (no dots). The framework's internal naming
    # uses ``<server>:<tool>`` for PVC-08 prefixed-form policy lookups,
    # but the LLM only sees the *wrapper*'s ``.name``. Use ``__``
    # (double underscore) as the LLM-visible separator: it satisfies
    # both providers' regexes and is unambiguous (no real tool name
    # contains a double underscore). ``inner.name`` keeps the colon
    # form so ``effective_action`` / ``should_gate`` policy lookups
    # stay PVC-08-compliant.
    _llm_visible_name = inner.name.replace(":", "__")

    # M3 (per-step telemetry): emit `tool_invoked` and `gate_fired` events
    # through the optional EventLog. Telemetry failures never break a
    # tool call — they are logged at DEBUG and dropped.

    def _cap_args(args_dict: Any) -> Any:
        """Cap args payload at 4 KB of JSON; oversized payloads become
        a small ``{"_truncated": True, "preview": ...}`` marker."""
        try:
            blob = json.dumps(args_dict, default=str)
        except (TypeError, ValueError):
            return {"_unencodable": True}
        if len(blob) <= 4096:
            return args_dict
        return {"_truncated": True, "preview": blob[:4096]}

    def _emit_invoked(
        *,
        status: str,
        risk: str,
        args_dict: Any,
        result: Any,
        latency_ms: float,
    ) -> None:
        if event_log is None:
            return
        try:
            event_log.record(
                session.id,
                "tool_invoked",
                tool=inner.name,
                agent=agent_name,
                args=_cap_args(args_dict),
                result_kind=type(result).__name__,
                latency_ms=round(latency_ms, 3),
                risk=risk,
                status=status,
            )
        except Exception:  # noqa: BLE001 — telemetry must not break a tool call
            _log.debug(
                "event_log.record(tool_invoked) failed", exc_info=True,
            )

    def _emit_gate(*, reason: str) -> None:
        if event_log is None:
            return
        try:
            event_log.record(
                session.id,
                "gate_fired",
                tool=inner.name,
                agent=agent_name,
                reason=reason,
            )
        except Exception:  # noqa: BLE001
            _log.debug(
                "event_log.record(gate_fired) failed", exc_info=True,
            )

    class _GatedTool(_GatedToolMarker):
        name: str = _llm_visible_name
        description: str = inner.description
        # The wrapper does its own arg coercion via the inner tool's schema,
        # so no need to copy it here. Keep ``args_schema`` aligned with the
        # LLM-visible (post-strip) schema so BaseTool's input validator
        # accepts the post-strip kwargs the LLM emits. Phase 9 strips
        # injected keys here; pre-Phase-9 callers see the full schema.
        args_schema: Any = _llm_visible_schema  # type: ignore[assignment]

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            # M3 (per-step telemetry): start the latency clock for every
            # tool invocation. _emit_invoked computes ``(now - t0) * 1000``.
            t0 = time.monotonic()
            # Phase 9 (D-09-01 / T-09-05): inject session-derived args
            # BEFORE the gateway risk lookup so risk-rating sees the
            # post-injection environment value. Pure no-op when
            # ``injected_args`` is empty.
            if inject_cfg:

                kwargs = inject_injected_args(
                    kwargs,
                    session=session,
                    injected_args_cfg=inject_cfg,
                    tool_name=inner.name,
                    accepted_params=_accepted_params or None,
                )
            # Phase 11 (FOC-04): pure-policy gating boundary. Call
            # should_gate to decide whether to pause for HITL approval;
            # also call effective_action so the notify-audit branch
            # below still fires for medium-risk tools that should NOT
            # gate but should record an audit row.
            action = effective_action(
                inner.name, env=env, gateway_cfg=gateway_cfg,
            )
            decision = _evaluate_gate(
                session=session,
                tool_name=inner.name,
                gate_policy=gate_policy,
                gateway_cfg=gateway_cfg,
            )
            if decision.gate:
                from langgraph.types import interrupt

                # M3: emit gate_fired BEFORE the interrupt fires so the
                # event ordering in the log matches the runtime causality
                # (gate decision precedes tool execution / pause).
                _emit_gate(reason=decision.reason)

                # Persist a ``pending_approval`` ToolCall row BEFORE
                # raising GraphInterrupt so the approval-timeout watchdog
                # has a record to scan. ``ts`` is the moment the human
                # approval window opened. Stored args mirror the post-
                # decision rows so the audit history reads consistently.
                #
                # On resume, LangGraph re-enters this node and runs us
                # again from the top — so we must re-use the existing
                # pending row instead of appending a duplicate. The most
                # recent ``pending_approval`` row for this tool wins.
                pending_args = dict(kwargs) if kwargs else {"args": list(args)}
                existing_idx = _find_existing_pending_index(
                    session.tool_calls, inner.name,
                )
                if existing_idx is not None:
                    pending_ts = session.tool_calls[existing_idx].ts
                else:
                    pending_ts = _now_iso()
                    session.tool_calls.append(
                        ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result=None,
                            ts=pending_ts,
                            risk="high",
                            status="pending_approval",
                        )
                    )
                    # CRITICAL: persist the pending_approval row BEFORE
                    # raising interrupt() so the approval-timeout
                    # watchdog (which reads from the DB) and the
                    # /approvals UI can see the pending state. Without
                    # this save the in-memory mutation is invisible to
                    # any out-of-process observer.
                    if store is not None:
                        store.save(session)
                payload = {
                    "kind": "tool_approval",
                    "tool": inner.name,
                    "args": kwargs or args,
                    "tool_call_id": kwargs.get("tool_call_id"),
                }
                # First execution: raises GraphInterrupt, checkpointer pauses.
                # Resume: returns whatever Command(resume=...) supplied.
                decision = interrupt(payload)
                # Decision payload may be a string ("approve" / "reject" /
                # "timeout") or a dict {decision, approver, rationale}.
                if isinstance(decision, dict):
                    verdict = decision.get("decision", "approve")
                    approver = decision.get("approver")
                    rationale = decision.get("rationale")
                else:
                    verdict = decision or "approve"
                    approver = None
                    rationale = None
                # Update the pending_approval row in place rather than
                # appending a second audit entry. The watchdog and the
                # /approvals UI both reason about a single audit row per
                # high-risk call.
                pending_idx = _find_pending_index(
                    session.tool_calls, inner.name, pending_ts,
                )
                verdict_str = str(verdict).lower()
                if verdict_str == "reject":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"rejected": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="rejected",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    rejected_result = {"rejected": True, "rationale": rationale}
                    _emit_invoked(
                        status="rejected", risk="high",
                        args_dict=pending_args, result=rejected_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return rejected_result
                if verdict_str == "timeout":
                    # The approval window expired. Do NOT run the tool;
                    # mark the audit row ``status="timeout"`` so
                    # downstream consumers (UI, retraining) can
                    # distinguish operator-initiated rejections from
                    # automatic timeouts.
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"timeout": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="timeout",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    timeout_result = {"timeout": True, "rationale": rationale}
                    _emit_invoked(
                        status="timeout", risk="high",
                        args_dict=pending_args, result=timeout_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return timeout_result
                # Approved -> run the tool, then update the audit row.
                result = _sync_invoke_inner(kwargs if kwargs else args[0] if args else {})
                if pending_idx is not None:
                    session.tool_calls[pending_idx] = ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=pending_args,
                        result=result,
                        ts=pending_ts,
                        risk="high",
                        status="approved",
                        approver=approver,
                        approved_at=_now_iso(),
                        approval_rationale=rationale,
                    )
                _emit_invoked(
                    status="approved", risk="high",
                    args_dict=pending_args, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return result

            # auto / notify both run the tool now.
            result = _sync_invoke_inner(kwargs if kwargs else args[0] if args else {})

            _args_dict = dict(kwargs) if kwargs else {"args": list(args)}
            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=_args_dict,
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
                _emit_invoked(
                    status="executed_with_notify", risk="medium",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            else:
                _emit_invoked(
                    status="executed", risk="low",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return result

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            # M3: start latency clock; mirror of sync ``_run``.
            t0 = time.monotonic()
            # Phase 9 (D-09-01 / T-09-05): inject session-derived args
            # BEFORE the gateway risk lookup. Mirror of the sync ``_run``.
            if inject_cfg:

                kwargs = inject_injected_args(
                    kwargs,
                    session=session,
                    injected_args_cfg=inject_cfg,
                    tool_name=inner.name,
                    accepted_params=_accepted_params or None,
                )
            # Phase 11 (FOC-04): pure-policy gating boundary. Mirror of
            # the sync ``_run`` -- consult should_gate via
            # ``_evaluate_gate``; still call ``effective_action`` to
            # keep the notify-audit branch for medium-risk tools.
            action = effective_action(
                inner.name, env=env, gateway_cfg=gateway_cfg,
            )
            decision = _evaluate_gate(
                session=session,
                tool_name=inner.name,
                gate_policy=gate_policy,
                gateway_cfg=gateway_cfg,
            )
            if decision.gate:
                from langgraph.types import interrupt

                # M3: emit gate_fired BEFORE interrupt.
                _emit_gate(reason=decision.reason)

                # Persist a ``pending_approval`` audit row BEFORE the
                # GraphInterrupt fires so the watchdog can spot stale
                # approvals. See the sync ``_run`` mirror for details.
                pending_args = dict(kwargs) if kwargs else {"args": list(args)}
                existing_idx = _find_existing_pending_index(
                    session.tool_calls, inner.name,
                )
                if existing_idx is not None:
                    pending_ts = session.tool_calls[existing_idx].ts
                else:
                    pending_ts = _now_iso()
                    session.tool_calls.append(
                        ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result=None,
                            ts=pending_ts,
                            risk="high",
                            status="pending_approval",
                        )
                    )
                    # CRITICAL: persist the pending_approval row BEFORE
                    # raising interrupt() so the approval-timeout
                    # watchdog (which reads from the DB) and the
                    # /approvals UI can see the pending state.
                    if store is not None:
                        store.save(session)
                payload = {
                    "kind": "tool_approval",
                    "tool": inner.name,
                    "args": kwargs or args,
                    "tool_call_id": kwargs.get("tool_call_id"),
                }
                decision = interrupt(payload)
                if isinstance(decision, dict):
                    verdict = decision.get("decision", "approve")
                    approver = decision.get("approver")
                    rationale = decision.get("rationale")
                else:
                    verdict = decision or "approve"
                    approver = None
                    rationale = None
                pending_idx = _find_pending_index(
                    session.tool_calls, inner.name, pending_ts,
                )
                verdict_str = str(verdict).lower()
                if verdict_str == "reject":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"rejected": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="rejected",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    rejected_result = {"rejected": True, "rationale": rationale}
                    _emit_invoked(
                        status="rejected", risk="high",
                        args_dict=pending_args, result=rejected_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return rejected_result
                if verdict_str == "timeout":
                    if pending_idx is not None:
                        session.tool_calls[pending_idx] = ToolCall(
                            agent=agent_name,
                            tool=inner.name,
                            args=pending_args,
                            result={"timeout": True, "rationale": rationale},
                            ts=pending_ts,
                            risk="high",
                            status="timeout",
                            approver=approver,
                            approved_at=_now_iso(),
                            approval_rationale=rationale,
                        )
                    timeout_result = {"timeout": True, "rationale": rationale}
                    _emit_invoked(
                        status="timeout", risk="high",
                        args_dict=pending_args, result=timeout_result,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                    return timeout_result
                result = await inner.ainvoke(kwargs if kwargs else args[0] if args else {})
                if pending_idx is not None:
                    session.tool_calls[pending_idx] = ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=pending_args,
                        result=result,
                        ts=pending_ts,
                        risk="high",
                        status="approved",
                        approver=approver,
                        approved_at=_now_iso(),
                        approval_rationale=rationale,
                    )
                _emit_invoked(
                    status="approved", risk="high",
                    args_dict=pending_args, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return result

            result = await inner.ainvoke(kwargs if kwargs else args[0] if args else {})

            _args_dict = dict(kwargs) if kwargs else {"args": list(args)}
            if action == "notify":
                session.tool_calls.append(
                    ToolCall(
                        agent=agent_name,
                        tool=inner.name,
                        args=_args_dict,
                        result=result,
                        ts=_now_iso(),
                        risk="medium",
                        status="executed_with_notify",
                    )
                )
                _emit_invoked(
                    status="executed_with_notify", risk="medium",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            else:
                _emit_invoked(
                    status="executed", risk="low",
                    args_dict=_args_dict, result=result,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return result

    return _GatedTool()

# ====== module: runtime/tools/arg_injection.py ======

_LOG = logging.getLogger("runtime.orchestrator")


def strip_injected_params(
    tool: BaseTool,
    injected_keys: frozenset[str],
) -> BaseTool:
    """Return a ``BaseTool`` whose ``args_schema`` hides every param named
    in ``injected_keys``.

    The LLM only sees the stripped sig; the framework re-adds the real
    values at invocation time via :func:`inject_injected_args` (D-09-01).

    Properties:

    * **Pure.** The original tool is left unchanged — its ``args_schema``
      is not mutated, so tests and in-process callers that hold a direct
      reference keep their full schema.
    * **Idempotent.** Calling twice with the same keys is equivalent to
      calling once. The cloned schema is structurally identical.
    * **Identity short-circuit.** Empty ``injected_keys`` (or no overlap
      between ``injected_keys`` and the tool's params) returns the tool
      unchanged so unconfigured apps and tools without any injectable
      params pay nothing.
    """
    if not injected_keys:
        return tool
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return tool

    # --- dict path: FastMCP / JSON-Schema tools ---------------------------
    # FastMCP exposes ``args_schema`` as a plain JSON-Schema dict rather
    # than a Pydantic model. Strip injected keys directly from the dict.
    if isinstance(schema, dict):
        props = schema.get("properties", {})
        overlap = injected_keys & set(props)
        if not overlap:
            return tool
        new_props = {k: v for k, v in props.items() if k not in injected_keys}
        required = [r for r in schema.get("required", []) if r not in injected_keys]
        new_dict_schema: dict[str, Any] = {**schema, "properties": new_props, "required": required}
        try:
            return tool.model_copy(update={"args_schema": new_dict_schema})
        except Exception:  # pragma: no cover — defensive fallback
            import copy
            stripped = copy.copy(tool)
            stripped.args_schema = new_dict_schema  # type: ignore[attr-defined]
            return stripped

    # --- Pydantic path: BaseModel subclass tools --------------------------
    if not hasattr(schema, "model_fields"):
        return tool
    overlap = injected_keys & set(schema.model_fields.keys())
    if not overlap:
        # No params to strip — preserve identity (no clone).
        return tool

    # Build the kwargs for ``create_model`` from the surviving fields.
    # Pydantic v2's ``create_model`` accepts ``(annotation, FieldInfo)``
    # tuples; FieldInfo carries default + description + alias so the
    # cloned schema is functionally equivalent to the original minus
    # the stripped fields.
    keep: dict[str, tuple[Any, Any]] = {
        name: (f.annotation, f)
        for name, f in schema.model_fields.items()
        if name not in injected_keys
    }
    new_schema = create_model(
        f"{schema.__name__}__StrippedForLLM",
        __base__=BaseModel,
        **keep,  # type: ignore[arg-type]
    )

    # ``BaseTool`` is itself a pydantic BaseModel — ``model_copy`` clones
    # it cheaply and lets us swap ``args_schema`` without touching the
    # original. Tools that are not pydantic models (extremely rare; only
    # custom subclasses) fall back to a regular shallow copy.
    try:
        stripped = tool.model_copy(update={"args_schema": new_schema})
    except Exception:  # pragma: no cover — defensive fallback
        import copy
        stripped = copy.copy(tool)
        stripped.args_schema = new_schema  # type: ignore[attr-defined]
    return stripped


def _resolve_dotted(root: Session, path: str) -> Any | None:
    """Walk ``path`` ('session.foo.bar') against ``root`` and return the
    terminal value or ``None`` if any segment is missing / None.

    ``path`` must start with ``session.``. The leading ``session`` token
    pins the resolution root to the live Session — config-declared paths
    cannot reach into arbitrary modules. Subsequent segments walk
    attributes (``getattr``) — for fields stored under ``extra_fields``
    apps use ``session.extra_fields.foo`` which goes through the dict
    branch below.
    """
    parts = path.split(".")
    if not parts or parts[0] != "session":
        raise ValueError(
            f"injected_args path {path!r} must start with 'session.'"
        )
    cur: Any = root
    for seg in parts[1:]:
        if cur is None:
            return None
        # Support dict-valued attrs (notably ``Session.extra_fields``)
        # transparently — ``session.extra_fields.pr_url`` resolves
        # whether ``extra_fields`` is a real attribute or a dict on
        # the model. Plain attribute walks work for typed Session
        # subclasses (``IncidentState.environment``).
        if isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            cur = getattr(cur, seg, None)
    return cur


def inject_injected_args(
    tool_args: dict[str, Any],
    *,
    session: Session,
    injected_args_cfg: dict[str, str],
    tool_name: str,
    accepted_params: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return a NEW dict with each injected arg resolved from ``session``.

    Behaviour (D-09-03):

    * Mutation-free: ``tool_args`` is never modified. Callers that need
      to keep the LLM's original call shape can compare ``tool_args`` to
      the return value.
    * Framework wins on conflict. When the LLM already supplied a value
      and the resolved framework value differs, the framework value is
      written and a single INFO record is emitted on the
      ``runtime.orchestrator`` logger with the documented payload tokens
      (``tool``, ``arg``, ``llm_value``, ``framework_value``,
      ``session_id``).
    * Missing/None resolutions are skipped. The arg is left absent so
      the tool's own default-handling (or the MCP server's required-arg
      validator) decides what to do — never silently ``None``.
    * When ``accepted_params`` is provided, injected keys not present in
      that set are skipped. Prevents writing kwargs the target tool
      doesn't accept (which would raise pydantic ``unexpected_keyword``
      validation errors at the FastMCP boundary).
    """
    out = dict(tool_args)
    for arg_name, path in injected_args_cfg.items():
        if accepted_params is not None and arg_name not in accepted_params:
            # The tool doesn't declare this injectable param. Strip any
            # LLM-supplied value too — the LLM shouldn't be emitting it
            # (Phase 9 strips injectable keys from the LLM-visible sig)
            # and forwarding it to the tool would raise pydantic
            # ``unexpected_keyword`` at the FastMCP boundary.
            if arg_name in out:
                _LOG.info(
                    "tool_call.injected_arg_dropped tool=%s arg=%s "
                    "llm_value=%r reason=not_accepted_by_tool session_id=%s",
                    tool_name,
                    arg_name,
                    out[arg_name],
                    getattr(session, "id", "?"),
                )
                del out[arg_name]
            continue
        framework_value = _resolve_dotted(session, path)
        if framework_value is None:
            continue
        if arg_name in out and out[arg_name] != framework_value:
            _LOG.info(
                "tool_call.injected_arg_overridden tool=%s arg=%s "
                "llm_value=%r framework_value=%r session_id=%s",
                tool_name,
                arg_name,
                out[arg_name],
                framework_value,
                getattr(session, "id", "?"),
            )
        out[arg_name] = framework_value
    return out


def accepted_params_for_tool(tool: Any) -> frozenset[str] | None:
    """Return the set of parameter names a wrapped tool accepts.

    Handles both shapes ``args_schema`` can take in this codebase:

    * pydantic ``BaseModel`` subclass — read ``model_fields.keys()``
      (used by mock tools and by tests).
    * JSON-Schema ``dict`` — read ``schema["properties"].keys()``
      (used by real FastMCP-derived tools, which expose the underlying
      function's input schema as a JSON Schema rather than a pydantic
      class).

    Returns ``None`` when the tool has no introspectable schema (caller
    should treat this as "skip filtering" — preserves prior behaviour).
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return None
    if hasattr(schema, "model_fields"):
        return frozenset(schema.model_fields.keys())
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return frozenset(props.keys())
    return None


__all__ = [
    "strip_injected_params",
    "inject_injected_args",
    "accepted_params_for_tool",
    "_LOG",
]

# ====== module: runtime/tools/approval_watchdog.py ======

if TYPE_CHECKING:
    pass
logger = logging.getLogger(__name__)

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Sessions whose status is in this set are *not* candidates for the
# watchdog — either they never paused for approval, or they have already
# moved past it. ``awaiting_input`` is the only status produced by
# ``langgraph.types.interrupt()`` while a high-risk gate is open.
_TERMINAL_STATUSES = frozenset({
    "resolved", "stopped", "escalated", "duplicate", "deleted", "error",
})


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 ``YYYY-MM-DDTHH:MM:SSZ`` ts back into UTC.

    Returns ``None`` for malformed values; callers treat that as
    "skip this row" so the watchdog never crashes on a bad audit
    record.
    """
    if not ts:
        return None
    try:
        # Replace trailing 'Z' so ``fromisoformat`` accepts it on
        # Python <3.11. The format is fixed by ``_UTC_TS_FMT`` so this
        # round-trips cleanly.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


class ApprovalWatchdog:
    """Background asyncio task that resumes stale pending-approval sessions.

    Owned by :class:`runtime.service.OrchestratorService`; started in
    ``OrchestratorService.start()`` and stopped in ``shutdown()``. The
    task runs on the service's background loop so it shares the same
    checkpointer / SQLite engine / FastMCP transports the live
    sessions are using.
    """

    def __init__(
        self,
        service: "OrchestratorService",
        *,
        approval_timeout_seconds: int,
        poll_interval_seconds: float = 60.0,
    ) -> None:
        self._service = service
        self._approval_timeout_seconds = approval_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        # HARD-07: ``stop()`` is idempotent. Once a stop has been
        # initiated (or completed), subsequent calls return immediately
        # rather than racing on ``_task`` / ``_stop_event`` which the
        # first caller is already clearing. Mutated only on the loop
        # thread (where ``stop()`` runs), so no extra lock needed.
        self._stopped: bool = False

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule the watchdog onto ``loop``. Idempotent.

        Must be called from a thread that is not the loop's own thread —
        the typical caller is :meth:`OrchestratorService.start`. Returns
        immediately; the polling coroutine runs in the background.
        """
        if self._task is not None and not self._task.done():
            return

        async def _arm() -> None:
            # Re-arm: a previous ``stop()`` may have flipped this; a
            # fresh ``start()`` re-enables ``stop()``.
            self._stopped = False
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(
                self._run(), name="approval_watchdog",
            )

        fut = asyncio.run_coroutine_threadsafe(_arm(), loop)
        fut.result(timeout=5.0)

    async def stop(self) -> None:
        """Signal the polling loop to exit and await termination.

        HARD-07: Idempotent and abrupt-shutdown safe. Safe to call:
          * before ``start()`` (no-op),
          * multiple times (subsequent calls short-circuit on
            ``_stopped`` after the first caller flips it),
          * concurrently from two callers — the first claims ownership
            of ``_task`` and drains it; the second sees the task is
            already gone and returns.

        Cancellation strategy: signal via ``_stop_event`` first so the
        polling loop exits its ``wait_for`` cleanly; then bound the
        drain by ``asyncio.wait_for(task, timeout=1.0)``. If the task
        ignores the event (or the event loop is being torn down under
        us), fall back to ``task.cancel()`` and one final drain.
        ``CancelledError`` and ``TimeoutError`` are suppressed — there
        is no useful recovery from a watchdog that won't die.

        Runs on the loop thread (called from ``OrchestratorService._close_*``
        helpers, or as a graceful no-op cleanup hook).
        """
        # First-call wins. Subsequent callers (and the after-shutdown
        # path) see ``_stopped`` and return without re-running the
        # drain — protects against double-await on ``_task``.
        if self._stopped:
            return
        self._stopped = True
        # Snapshot to LOCAL variables so concurrent ``stop()`` calls
        # never re-await the same task. We do NOT null out ``_task`` /
        # ``_stop_event`` until after the drain because ``_run()``
        # reads ``self._stop_event`` on every loop iteration; clearing
        # it before signalling would crash the polling loop with
        # ``AttributeError: 'NoneType' object has no attribute
        # 'is_set'`` and produce exactly the noisy teardown this fix
        # is meant to prevent.
        task = self._task
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        if task is None or task.done():
            self._task = None
            self._stop_event = None
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Task is wedged or the loop is shutting down under us.
                # The ``cancel()`` call above is enough to flip the task
                # state; ``run_loop`` 's final ``gather`` pass will sweep
                # it during loop teardown. Don't block shutdown further.
                pass
        finally:
            # Always clear the bookkeeping refs so a subsequent
            # ``start()`` arms cleanly and ``is_running`` reports False.
            self._task = None
            self._stop_event = None

    async def close(self) -> None:
        """Alias for :meth:`stop` — symmetric with aiohttp/httpx.

        Idempotent. Provided so callers using a "close-on-cleanup"
        pattern (``async with`` on parent owners) read naturally.
        """
        await self.stop()

    async def _run(self) -> None:
        """Polling loop. Runs until ``_stop_event`` is set.

        We bind ``stop_event`` to a LOCAL variable on entry so a
        concurrent ``stop()`` cannot null out ``self._stop_event``
        from underneath us mid-iteration (HARD-07: that nulling-while-
        running was the original source of ``AttributeError`` at
        teardown).
        """
        stop_event = self._stop_event
        assert stop_event is not None
        while not stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("approval watchdog tick failed")
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                # Expected — wakes the loop every ``poll_interval_seconds``.
                continue

    async def _tick(self) -> None:
        """One scan + resume pass. Visible for tests via ``run_once``."""
        await self.run_once()

    async def run_once(self) -> int:
        """Single scan pass. Returns the number of sessions resumed.

        Exposed publicly so tests can drive the watchdog
        deterministically without waiting on the polling cadence.
        """
        orch = getattr(self._service, "_orch", None)
        if orch is None:
            return 0
        registry = dict(self._service._registry)
        if not registry:
            return 0
        now = datetime.now(timezone.utc)
        resumed = 0
        for session_id in list(registry.keys()):
            try:
                inc = orch.store.load(session_id)
            except Exception:  # noqa: BLE001
                continue
            status = getattr(inc, "status", None)
            if status in _TERMINAL_STATUSES:
                continue
            if status != "awaiting_input":
                # Only sessions paused on a high-risk gate are watchdog
                # candidates. ``in_progress`` / ``new`` are still
                # actively running on the loop.
                continue
            stale = self._find_stale_pending(inc, now)
            if not stale:
                continue
            # No is_locked() peek here — try_acquire (inside
            # _resume_with_timeout) is the single contention check, so
            # there is no TOCTOU window between check and acquire. The
            # SessionBusy handler below fires on real contention.
            try:
                await self._resume_with_timeout(orch, session_id)
                resumed += 1
            except SessionBusy:
                logger.debug(
                    "approval watchdog: session %s SessionBusy at resume, skipping",
                    session_id,
                )
                continue
            except Exception:  # noqa: BLE001
                logger.exception(
                    "approval watchdog: resume failed for session %s",
                    session_id,
                )
        return resumed

    def _find_stale_pending(self, inc: Any, now: datetime) -> list[int]:
        """Return indices of ``pending_approval`` ToolCalls older than the
        configured timeout."""
        out: list[int] = []
        tool_calls = getattr(inc, "tool_calls", []) or []
        threshold = self._approval_timeout_seconds
        for idx, tc in enumerate(tool_calls):
            if getattr(tc, "status", None) != "pending_approval":
                continue
            ts = _parse_iso(getattr(tc, "ts", None))
            if ts is None:
                continue
            age = (now - ts).total_seconds()
            if age >= threshold:
                out.append(idx)
        return out

    async def _resume_with_timeout(
        self, orch: Any, session_id: str,
    ) -> None:
        """Resume the paused graph with a synthetic timeout decision.

        Uses ``Command(resume=...)`` against the same ``thread_id`` the
        approval API would use — the wrap_tool resume path updates the
        audit row to ``status="timeout"`` automatically.

        Per D-18: the ``ainvoke`` call is wrapped in
        ``orch._locks.try_acquire(session_id)`` so a concurrent user-
        driven turn cannot interleave checkpoint writes for the same
        ``thread_id``. If the lock is already held, ``try_acquire``
        raises ``SessionBusy`` immediately (no waiting); the caller
        (``run_once``) catches that and skips the tick — this is how
        the watchdog tolerates a busy session without piling up.
        """
        from langgraph.types import Command  # local: heavy import

        decision_payload = {
            "decision": "timeout",
            "approver": "system",
            "rationale": "approval window expired",
        }
        async with orch._locks.try_acquire(session_id):
            await orch.graph.ainvoke(
                Command(resume=decision_payload),
                config=orch._thread_config(session_id),
            )

# ====== module: runtime/policy.py ======

if TYPE_CHECKING:  # pragma: no cover -- type checking only


    pass  # noqa: PIE790 -- bundle survives even if imports are stripped


GateReason = Literal[
    "auto",
    "high_risk_tool",
    "gated_env",
    "low_confidence",
    "blocked",
]


class GateDecision(BaseModel):
    """Outcome of a single gating evaluation."""

    model_config = ConfigDict(extra="forbid")
    gate: bool
    reason: GateReason


def should_gate(
    session: Any,
    tool_call: "ToolCall",
    confidence: float | None,
    cfg: "OrchestratorConfig",
) -> GateDecision:
    """Decide whether ``tool_call`` should pause for HITL approval.

    Pure -- delegates the per-tool risk lookup to
    :func:`runtime.tools.gateway.effective_action` (so the v1.0 PVC-08
    prefixed-form lookup invariant is preserved) and combines the
    result with ``session.environment`` and ``confidence`` per the
    precedence rules in the module docstring.

    ``session`` is typed as ``Any`` because the framework's base
    :class:`runtime.state.Session` does not own the ``environment``
    field (apps subclass and add it). The function reads
    ``session.environment`` and tolerates a missing attribute by
    treating it as ``None``.

    ``confidence=None`` means "no signal yet" -- treated internally as
    1.0 to avoid a false-positive low_confidence gate before any
    envelope/tool-arg has surfaced for the active turn.
    """
    # Read gateway config off the OrchestratorConfig. The runtime threads
    # it via cfg.gateway today (sibling of cfg.gate_policy in the
    # OrchestratorConfig namespace) -- gracefully tolerate the legacy
    # path where gateway is configured on RuntimeConfig instead.
    gateway_cfg = getattr(cfg, "gateway", None)
    env = getattr(session, "environment", None)

    risk_action = effective_action(
        tool_call.tool,
        env=env,
        gateway_cfg=gateway_cfg,
    )

    # 1. high-risk tool gates first.
    if risk_action in cfg.gate_policy.gated_risk_actions:
        return GateDecision(gate=True, reason="high_risk_tool")

    # 2. gated env: any non-"auto" risk in a gated environment.
    if (env in cfg.gate_policy.gated_environments
            and risk_action != "auto"):
        return GateDecision(gate=True, reason="gated_env")

    # 3. low confidence: only an actionable tool. None == "no signal yet".
    effective_conf = 1.0 if confidence is None else confidence
    if (effective_conf < cfg.gate_policy.confidence_threshold
            and risk_action != "auto"):
        return GateDecision(gate=True, reason="low_confidence")

    return GateDecision(gate=False, reason="auto")


# ---------------------------------------------------------------
# Phase 12 (FOC-05): pure should_retry policy.
# ---------------------------------------------------------------

RetryReason = Literal[
    "auto_retry",
    "max_retries_exceeded",
    "permanent_error",
    "low_confidence_no_retry",
    "transient_disabled",
]


class RetryDecision(BaseModel):
    """Outcome of a single retry-policy evaluation.

    Pure surface: produced by :func:`should_retry` from
    ``(retry_count, error, confidence, cfg)``. The orchestrator's
    ``_retry_session_locked`` consults this BEFORE running the retry;
    the UI consults the same value via
    ``Orchestrator.preview_retry_decision`` to render the button label /
    disabled state.
    """

    model_config = ConfigDict(extra="forbid")
    retry: bool
    reason: RetryReason


# Whitelist of exception types that are NEVER auto-retryable.
# Schema/validation errors -- the LLM produced bad data; retrying
# without addressing root cause burns budget. Adding a new entry is a
# one-line PR (D-12-02 explicit choice -- no new ToolError ABC).
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    _pydantic.ValidationError,
    EnvelopeMissingError,
)

# Whitelist of exception types that are ALWAYS auto-retryable
# (subject to max_retries). Network blips, asyncio timeouts,
# filesystem/socket transients. httpx is NOT imported because the
# runtime does not raise httpx errors today; built-in TimeoutError
# covers asyncio's 3.11+ alias.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    _asyncio.TimeoutError,
    TimeoutError,
    OSError,
    ConnectionError,
)


def _is_permanent_error(error: Exception | None) -> bool:
    if error is None:
        return False
    return isinstance(error, _PERMANENT_TYPES)


def _is_transient_error(error: Exception | None) -> bool:
    if error is None:
        return False
    return isinstance(error, _TRANSIENT_TYPES)


def should_retry(
    retry_count: int,
    error: Exception | None,
    confidence: float | None,
    cfg: "OrchestratorConfig",
) -> RetryDecision:
    """Decide whether the framework should auto-retry a failed turn.

    Pure -- same inputs always yield identical RetryDecision.

    Precedence (descending; first match wins):
      1. ``retry_count >= cfg.retry_policy.max_retries``
         -> ``RetryDecision(retry=False, reason="max_retries_exceeded")``
      2. ``error`` matches ``_PERMANENT_TYPES``
         -> ``RetryDecision(retry=False, reason="permanent_error")``
      3. ``confidence is not None`` AND
         ``confidence < cfg.retry_policy.retry_low_confidence_threshold``
         AND ``error`` is NOT in ``_TRANSIENT_TYPES``
         -> ``RetryDecision(retry=False, reason="low_confidence_no_retry")``
      4. ``error`` matches ``_TRANSIENT_TYPES`` AND
         ``cfg.retry_policy.retry_on_transient is False``
         -> ``RetryDecision(retry=False, reason="transient_disabled")``
      5. ``error`` matches ``_TRANSIENT_TYPES`` AND
         ``cfg.retry_policy.retry_on_transient is True``
         -> ``RetryDecision(retry=True, reason="auto_retry")``
      6. Default fall-through (no match) -> ``RetryDecision(
         retry=False, reason="permanent_error")`` -- fail-closed
         conservative default (D-12-02).

    ``retry_count`` is the count of PRIOR retries (0 on the first
    retry attempt). Caller is responsible for the bump.

    ``error`` may be ``None`` (caller has no exception object); that is
    treated as a permanent error for safety.

    ``confidence`` is the last AgentRun.confidence for the failed turn;
    ``None`` means "no signal recorded" and skips the low-confidence
    gate.
    """
    # 1. absolute cap -- regardless of error class
    if retry_count >= cfg.retry_policy.max_retries:
        return RetryDecision(retry=False, reason="max_retries_exceeded")

    # 2. permanent errors -- never auto-retry
    if _is_permanent_error(error):
        return RetryDecision(retry=False, reason="permanent_error")

    is_transient = _is_transient_error(error)

    # 3. low-confidence -- only when error is NOT transient (transient
    # errors are mechanical; the LLM's confidence in the business
    # decision is still trustworthy on retry).
    if (confidence is not None
            and confidence < cfg.retry_policy.retry_low_confidence_threshold
            and not is_transient):
        return RetryDecision(
            retry=False, reason="low_confidence_no_retry",
        )

    # 4 + 5. transient classification
    if is_transient:
        if not cfg.retry_policy.retry_on_transient:
            return RetryDecision(retry=False, reason="transient_disabled")
        return RetryDecision(retry=True, reason="auto_retry")

    # 6. fail-closed default
    return RetryDecision(retry=False, reason="permanent_error")


__all__ = [
    # Phase 11
    "GateDecision", "GateReason", "should_gate",
    # Phase 12
    "RetryDecision", "RetryReason", "should_retry",
]

# ====== module: runtime/agents/responsive.py ======

if TYPE_CHECKING:
    pass
logger = logging.getLogger(__name__)


def make_agent_node(
    *,
    skill: Skill,
    llm: BaseChatModel,
    tools: list[BaseTool],
    decide_route: Callable[[Session], str],
    store: SessionStore,
    valid_signals: frozenset[str] | None = None,
    gateway_cfg: GatewayConfig | None = None,
    terminal_tool_names: frozenset[str] = frozenset(),
    patch_tool_names: frozenset[str] = frozenset(),
    gate_policy: "GatePolicy | None" = None,
    event_log: "EventLog | None" = None,
):
    """Factory: build a LangGraph node that runs a ReAct agent and decides a route.

    ``valid_signals`` is the orchestrator-wide accepted signal vocabulary
    (``cfg.orchestrator.signals``). When omitted, the legacy
    ``{success, failed, needs_input}`` default is used so older callers and
    tests keep working.

    ``gateway_cfg`` is the optional risk-rated tool gateway config.
    When supplied, every ``BaseTool`` in ``tools`` is wrapped via
    :func:`runtime.tools.gateway.wrap_tool` *inside the node body* so the
    closure captures the live ``Session`` per agent invocation. When
    ``None``, tools are passed through untouched.
    """
    # Imported lazily to avoid an import cycle: ``runtime.graph`` depends
    # on this module via ``_build_agent_nodes``, but the helpers used
    # inside the node body live in ``graph`` so we keep a single
    # implementation for the responsive path. The cycle is benign at
    # call time — both modules are fully imported before ``node()`` runs.


    async def node(state: GraphState) -> dict:
        incident: Session = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # M3: emit agent_started telemetry before any work happens.
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "agent_started",
                    agent=skill.name, started_at=started_at,
                )
            except Exception:  # noqa: BLE001 — telemetry must not break the agent
                logger.debug(
                    "event_log.record(agent_started) failed", exc_info=True,
                )

        # Wrap tools per-invocation so each wrap closes over the
        # live ``Session`` for this run.
        if gateway_cfg is not None:
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store,
                          gate_policy=gate_policy,
                          event_log=event_log)
                for t in tools
            ]
        else:
            run_tools = tools
        # Phase 10 (FOC-03 / D-10-02) + Phase 15 (LLM-COMPAT-01): every
        # responsive agent invocation is wrapped in an AgentTurnOutput
        # envelope. ``langchain.agents.create_agent`` (the non-deprecated
        # successor to ``langgraph.prebuilt.create_react_agent``) accepts a
        # bare schema as ``response_format`` and, by default, wraps it in
        # ``AutoStrategy`` — ProviderStrategy for models with native
        # structured-output (OpenAI-class), falling back to ToolStrategy
        # otherwise (Ollama). ToolStrategy injects AgentTurnOutput as a
        # callable tool: when the LLM ``calls`` it, the loop terminates on
        # the same turn with ``result["structured_response"]`` populated.
        # Eliminates the old two-call structure (loop + separate
        # ``with_structured_output`` pass) that hit recursion_limit=25 on
        # Ollama models without true function-calling.
        agent_executor = create_agent(
            model=llm,
            tools=run_tools,
            system_prompt=skill.system_prompt,
            response_format=AgentTurnOutput,
        )

        # Phase 11 (FOC-04): reset per-turn confidence hint at the
        # start of each agent step so the gateway treats the first
        # tool call of the turn as "no signal yet".
        try:
            incident.turn_confidence_hint = None
        except (AttributeError, ValueError):
            pass

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): HITL pause -- propagate up.
            raise
        except Exception as exc:  # noqa: BLE001
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        # Tools (e.g. registered patch tools) write straight to disk.
        # Reload so the node's own append of agent_run + tool_calls
        # happens against the tool-mutated state.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
        )
        # Phase 11 (FOC-04): update hint so any subsequent in-turn
        # tool call sees the harvested confidence.
        if agent_confidence is not None:
            try:
                incident.turn_confidence_hint = agent_confidence
            except (AttributeError, ValueError):
                pass
        _pair_tool_responses(messages, incident)

        # Phase 10 (FOC-03 / D-10-03): parse envelope; reconcile against
        # any typed-terminal-tool-arg confidence. Envelope failure is a
        # structured agent_run error.
        try:
            envelope = parse_envelope_from_result(result, agent=skill.name)
        except EnvelopeMissingError as exc:
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        terminal_tool_for_log = _first_terminal_tool_called_this_turn(
            messages, terminal_tool_names,
        )
        final_confidence = reconcile_confidence(
            envelope.confidence,
            agent_confidence,
            agent=skill.name,
            session_id=inc_id,
            tool_name=terminal_tool_for_log,
        )
        final_rationale = agent_rationale or envelope.confidence_rationale
        final_signal = agent_signal if agent_signal is not None else envelope.signal

        final_text = envelope.content or _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        # M3: emit confidence_emitted after reconcile_confidence + signal
        # harvest land, before _record_success_run persists the agent_run.
        if event_log is not None and final_confidence is not None:
            try:
                event_log.record(
                    inc_id, "confidence_emitted",
                    agent=skill.name,
                    value=float(final_confidence),
                    rationale=final_rationale or "",
                    signal=final_signal,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(confidence_emitted) failed", exc_info=True,
                )

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=final_confidence, rationale=final_rationale,
            signal=final_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)

        # M3: emit route_decided + agent_finished. agent_finished carries
        # the token_usage harvested by _sum_token_usage above so the
        # session-level telemetry has per-step counts.
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "route_decided",
                    agent=skill.name,
                    signal=next_route_signal,
                    next_node=next_node,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(route_decided) failed", exc_info=True,
                )
            try:
                event_log.record(
                    inc_id, "agent_finished",
                    agent=skill.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(agent_finished) failed", exc_info=True,
                )

        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


__all__ = ["make_agent_node"]

# ====== module: runtime/agents/supervisor.py ======

logger = logging.getLogger(__name__)


def _safe_eval(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a pre-validated safe-eval expression against ``ctx``.

    The expression must already have passed
    :func:`runtime.skill._validate_safe_expr` — that's enforced at
    skill-load time. We re-parse here (cheap) and walk the tree
    against the same allowlist; any non-whitelisted node is treated
    as evaluating to ``False`` so a malformed runtime expression can
    never escalate to arbitrary code execution.
    """

    _validate_safe_expr(expr, source="supervisor.dispatch_rule")
    # ``compile`` + ``eval`` over a built-in-stripped namespace is the
    # cheapest correct evaluator once the AST is whitelisted. The
    # ``__builtins__`` removal blocks ``__import__`` etc. should the
    # AST checker miss something.
    code = compile(expr, "<safe-eval>", "eval")
    return eval(code, {"__builtins__": {}}, ctx)  # noqa: S307 — AST-whitelisted


def _ctx_for_session(incident: Session) -> dict[str, Any]:
    """Build the variable namespace dispatch-rule expressions see.

    Exposes the live session payload as ``session`` plus a few
    ergonomic top-level aliases for fields operators reach for most
    often. Adding new top-level names is a one-liner; the safe-eval
    AST checker already restricts the language so we don't need to
    sandbox the namespace any further.
    """
    payload = incident.model_dump()
    return {
        "session": payload,
        "status": payload.get("status"),
        "agents_run": payload.get("agents_run") or [],
        "tool_calls": payload.get("tool_calls") or [],
    }


def log_supervisor_dispatch(
    *,
    session: Session,
    supervisor: str,
    strategy: str,
    depth: int,
    targets: list[str],
    rule_matched: str | None,
    payload_size: int,
) -> None:
    """Emit one structured ``supervisor_dispatch`` log entry.

    Operators wanting an end-to-end audit join ``agent_runs`` and the
    log stream by ``incident_id``. The audit trail is deliberately a
    different stream from ``agent_runs`` because supervisors don't burn
    tokens — bloating ``agents_run`` with router rows is a known trap
    we explicitly avoid.
    """
    record = {
        "event": "supervisor_dispatch",
        "ts": datetime.now(timezone.utc).strftime(_UTC_TS_FMT),
        "incident_id": session.id,
        "session_id": session.id,
        "supervisor": supervisor,
        "strategy": strategy,
        "depth": depth,
        "targets": targets,
        "rule_matched": rule_matched,
        "dispatch_payload_size": payload_size,
    }
    logger.info("supervisor_dispatch %s", json.dumps(record))


def _llm_pick_target(
    *,
    skill: Skill,
    llm: BaseChatModel,
    incident: Session,
) -> str:
    """One-shot LLM dispatch: ask the model to choose a subordinate.

    The model is asked to reply with **only** the name of one
    subordinate. We accept the first matching name in the response
    (case-insensitive substring match) and fall back to the first
    subordinate when the response is unparseable — keeping the graph
    moving rather than failing outright.
    """
    prompt = (
        f"{skill.dispatch_prompt}\n\n"
        f"Choose ONE of: {', '.join(skill.subordinates)}.\n"
        f"Reply with only the agent name."
    )
    payload = json.dumps(incident.model_dump(), default=str)
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content=payload),
    ]
    try:
        result = llm.invoke(msgs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "supervisor %s: LLM dispatch failed (%s); falling back to %s",
            skill.name, exc, skill.subordinates[0],
        )
        return skill.subordinates[0]
    text = (getattr(result, "content", "") or "").strip().lower()
    for name in skill.subordinates:
        if name.lower() in text:
            return name
    logger.warning(
        "supervisor %s: LLM reply %r did not name a subordinate; "
        "falling back to %s", skill.name, text, skill.subordinates[0],
    )
    return skill.subordinates[0]


def _rule_pick_target(
    *,
    skill: Skill,
    incident: Session,
) -> tuple[str, str | None]:
    """Walk dispatch_rules in order; return (target, matched_when).

    Falls back to the first subordinate when no rule matches; the
    fallback case carries ``matched_when=None`` so the audit log can
    distinguish "default" from "rule X matched".
    """
    ctx = _ctx_for_session(incident)
    for rule in skill.dispatch_rules:
        try:
            if bool(_safe_eval(rule.when, ctx)):
                return rule.target, rule.when
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "supervisor %s: dispatch_rule %r raised %s; skipping",
                skill.name, rule.when, exc,
            )
    return skill.subordinates[0], None


def _normalize_runner_route(value: Any) -> str:
    """Map runner-supplied route aliases to the canonical graph end token.

    Apps writing runners reach for ``"END"`` / ``"end"`` / ``"__end__"``
    interchangeably; LangGraph's conditional edges only recognise
    ``"__end__"``. Normalising here keeps the runner contract permissive
    without spreading the alias check across the graph layer.
    """
    if isinstance(value, str) and value.strip().lower() in {"end", "__end__"}:
        return "__end__"
    return value


def make_supervisor_node(
    *,
    skill: Skill,
    llm: BaseChatModel | None = None,
    framework_cfg: Any | None = None,
):
    """Build the supervisor LangGraph node.

    Pure routing: no ``AgentRun`` row, no tool execution, no token
    accounting beyond what the optional LLM call itself reports. The
    node sets ``state["next_route"]`` to a subordinate name and returns;
    LangGraph's conditional edges fan out to that node from there.

    The optional ``llm`` is only used when ``skill.dispatch_strategy``
    is ``"llm"``. Callers using ``"rule"`` may pass ``None``.

    When ``skill.runner`` is set, the dotted-path callable is resolved
    at build time and invoked at the start of each node call BEFORE the
    routing dispatch. The runner gets the live ``GraphState`` and the
    optional ``framework_cfg`` and may return ``None`` (continue with
    the routing table) or a dict patch that gets merged into state. A
    patch carrying ``"next_route"`` short-circuits the routing table
    entirely (use ``"__end__"`` to terminate the graph).
    """
    # Local import to avoid the circular runtime.graph -> runtime.agents
    # cycle at module-load time.


    if skill.kind != "supervisor":
        raise ValueError(
            f"make_supervisor_node called with non-supervisor skill "
            f"{skill.name!r} (kind={skill.kind!r})"
        )

    runner: Callable[..., Any] | None = None
    if skill.runner is not None:
        if callable(skill.runner):
            # Test stubs and composed runners may supply a live callable
            # directly rather than a dotted-path string. Access via the
            # class __dict__ to avoid Python binding it as an instance
            # method when the skill is a plain object (not a Pydantic model).
            raw = vars(type(skill)).get("runner", skill.runner)
            runner = raw if callable(raw) else skill.runner
        else:
            # Resolved a second time here so a runner that fails to import
            # at graph-build time still surfaces a clear error. The skill
            # validator catches most issues at YAML load; this is belt-and-
            # braces and also gives us the live callable to invoke.
            runner = _resolve_dotted_callable(
                skill.runner, source=f"supervisor {skill.name!r} runner"
            )

    async def node(state: GraphState) -> dict:
        sess: Session = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        # ``dispatch_depth`` is an extension field on GraphState; start
        # at 0 and increment per supervisor entry.
        depth = int(state.get("dispatch_depth") or 0) + 1
        if depth > skill.max_dispatch_depth:
            logger.warning(
                "supervisor %s: dispatch depth %d exceeds limit %d; aborting",
                skill.name, depth, skill.max_dispatch_depth,
            )
            return {
                "session": sess,
                "next_route": "__end__",
                "last_agent": skill.name,
                "dispatch_depth": depth,
                "error": (
                    f"supervisor {skill.name!r}: max_dispatch_depth "
                    f"{skill.max_dispatch_depth} exceeded"
                ),
            }

        # ----- App-supplied runner hook -------------------------------
        runner_patch: dict[str, Any] = {}
        if runner is not None:
            # Build a thin proxy so the runner can reach intake_context
            # (and any other framework_cfg attributes) without needing
            # framework_cfg to be mutable. The proxy exposes intake_context
            # directly and falls back to framework_cfg for all other attrs.
            _app_cfg_proxy = type("_RunnerAppCfg", (), {
                "intake_context": getattr(framework_cfg, "intake_context", None),
                "__getattr__": lambda self, name: getattr(framework_cfg, name),
            })()
            try:
                result = runner(state, app_cfg=_app_cfg_proxy)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "supervisor %s: runner %s raised; aborting to __end__",
                    skill.name, skill.runner,
                )
                return {
                    "session": sess,
                    "next_route": "__end__",
                    "last_agent": skill.name,
                    "dispatch_depth": depth,
                    "error": (
                        f"supervisor {skill.name!r}: runner failed: {exc}"
                    ),
                }
            if isinstance(result, dict):
                runner_patch = dict(result)
            elif result is not None:
                logger.warning(
                    "supervisor %s: runner returned %s (expected dict|None); "
                    "ignoring", skill.name, type(result).__name__,
                )
            override = runner_patch.pop("next_route", None)
            if override is not None:
                # Short-circuit: skip the routing table entirely. Audit
                # log still fires so operators can trace the decision.
                target = _normalize_runner_route(override)
                # Pick up any fresh reference the runner returned.
                sess = runner_patch.get("session", sess)
                try:
                    payload_size = len(
                        json.dumps(sess.model_dump(), default=str)
                    )
                except Exception:  # noqa: BLE001 — defensive
                    payload_size = 0
                log_supervisor_dispatch(
                    session=sess,
                    supervisor=skill.name,
                    strategy=f"runner:{skill.runner}",
                    depth=depth,
                    targets=[target],
                    rule_matched=None,
                    payload_size=payload_size,
                )
                out: dict[str, Any] = {
                    "session": sess,
                    "next_route": target,
                    "last_agent": skill.name,
                    "dispatch_depth": depth,
                    "error": None,
                }
                # Merge any non-route keys the runner returned (e.g.
                # extra GraphState fields apps want to carry forward).
                for k, v in runner_patch.items():
                    if k not in out:
                        out[k] = v
                return out
            # No override: fold any payload mutation back so the
            # routing table sees the up-to-date object.
            if "session" in runner_patch:
                sess = runner_patch["session"]

        rule_matched: str | None = None
        if skill.dispatch_strategy == "rule":
            target, rule_matched = _rule_pick_target(skill=skill, incident=sess)
        else:  # "llm"
            if llm is None:
                logger.warning(
                    "supervisor %s: strategy=llm but no llm provided; "
                    "falling back to first subordinate", skill.name,
                )
                target = skill.subordinates[0]
            else:
                target = _llm_pick_target(skill=skill, llm=llm, incident=sess)

        # Audit: one structured log entry per dispatch.
        try:
            payload_size = len(json.dumps(sess.model_dump(), default=str))
        except Exception:  # noqa: BLE001 — defensive; size is a hint
            payload_size = 0
        log_supervisor_dispatch(
            session=sess,
            supervisor=skill.name,
            strategy=skill.dispatch_strategy,
            depth=depth,
            targets=[target],
            rule_matched=rule_matched,
            payload_size=payload_size,
        )

        out: dict[str, Any] = {
            "session": sess,
            "next_route": target,
            "last_agent": skill.name,
            "dispatch_depth": depth,
            "error": None,
        }
        # Carry through any extra keys the runner emitted that the
        # framework didn't consume itself (e.g. memory snapshots).
        for k, v in runner_patch.items():
            if k not in out:
                out[k] = v
        return out

    return node


__all__ = ["make_supervisor_node", "log_supervisor_dispatch"]

# ====== module: runtime/agents/monitor.py ======

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe-eval evaluator
# ---------------------------------------------------------------------------


class SafeEvalError(Exception):
    """Raised when a supposedly-validated expression fails to evaluate."""


def safe_eval(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate ``expr`` against ``ctx`` after a fresh AST whitelist check.

    The skill loader validates ``emit_signal_when`` at parse time; we
    re-validate here on every call to keep the threat model defensive
    against any future code path that might construct a Skill bypassing
    the loader's validators.
    """
    _validate_safe_expr(expr, source="monitor.emit_signal_when")
    code = compile(expr, "<safe-eval>", "eval")
    try:
        return eval(code, {"__builtins__": {}}, ctx)  # noqa: S307 — AST-whitelisted
    except Exception as exc:  # noqa: BLE001
        raise SafeEvalError(f"emit_signal_when {expr!r} raised: {exc}") from exc


# ---------------------------------------------------------------------------
# Cron parsing (minute-resolution; matches Skill._validate_cron grammar)
# ---------------------------------------------------------------------------


def _expand_cron_field(field: str, lo: int, hi: int) -> set[int]:
    """Expand a single cron field into the set of int values it matches.

    Supports ``*``, ``*/n``, ``a``, ``a-b``, ``a-b/n``, and
    comma-separated combinations of those — the grammar accepted by
    :func:`runtime.skill._validate_cron`.
    """
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            base, _, step_s = part.partition("/")
            step = int(step_s)
        else:
            base = part
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            a, _, b = base.partition("-")
            start, end = int(a), int(b)
        else:
            v = int(base)
            start, end = v, v
        out.update(range(start, end + 1, step))
    return {v for v in out if lo <= v <= hi}


def _cron_matches(expr: str, when: datetime) -> bool:
    """Return True if the given datetime satisfies the 5-field cron expression.

    Fields: minute, hour, day-of-month, month, day-of-week (0=Mon..6=Sun
    — Python's ``datetime.weekday()`` convention; cron itself uses
    0=Sun, but for our minute-resolution scheduler the convention only
    needs to be internally consistent and documented).
    """
    minute, hour, dom, month, dow = expr.split()
    return (
        when.minute in _expand_cron_field(minute, 0, 59)
        and when.hour in _expand_cron_field(hour, 0, 23)
        and when.day in _expand_cron_field(dom, 1, 31)
        and when.month in _expand_cron_field(month, 1, 12)
        and when.weekday() in _expand_cron_field(dow, 0, 6)
    )


# ---------------------------------------------------------------------------
# Monitor callable factory
# ---------------------------------------------------------------------------


def make_monitor_callable(
    *,
    skill: Skill,
    observe_fn: Callable[[str], Any],
    fire_trigger: Callable[[str, dict[str, Any]], None],
) -> Callable[[], None]:
    """Build the callable a :class:`MonitorRunner` runs per tick.

    ``observe_fn(tool_name)`` is the seam through which the runner
    invokes a tool. Production wires this to the orchestrator's MCP
    tool registry; tests wire it to deterministic stubs.

    ``fire_trigger(name, payload)`` is the seam through which the
    runner fires a trigger. Production wires this to the trigger
    registry; tests wire it to a recorder.

    The returned callable is intentionally synchronous and exception-
    safe: a failed ``observe_fn`` or ``fire_trigger`` is logged and
    swallowed so one bad monitor cannot stall the runner.
    """
    if skill.kind != "monitor":
        raise ValueError(
            f"make_monitor_callable called with non-monitor skill "
            f"{skill.name!r} (kind={skill.kind!r})"
        )

    def tick() -> None:
        observation: dict[str, Any] = {}
        for tool_name in skill.observe:
            try:
                observation[tool_name] = observe_fn(tool_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: observe tool %r raised %s; skipping",
                    skill.name, tool_name, exc,
                )
                observation[tool_name] = None
        ctx = {
            "observation": observation,
            "obs": observation,
        }
        try:
            should_emit = bool(safe_eval(skill.emit_signal_when or "False", ctx))
        except SafeEvalError as exc:
            logger.warning("monitor %s: %s", skill.name, exc)
            return
        if not should_emit:
            return
        try:
            fire_trigger(skill.trigger_target or "", {
                "monitor": skill.name,
                "observation": observation,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "monitor %s: fire_trigger(%s) raised %s",
                skill.name, skill.trigger_target, exc,
            )

    return tick


# ---------------------------------------------------------------------------
# MonitorRunner — orchestrator-level singleton
# ---------------------------------------------------------------------------


class _RegisteredMonitor:
    __slots__ = ("skill", "callable_", "next_run_ts")

    def __init__(self, skill: Skill, callable_: Callable[[], None]) -> None:
        self.skill = skill
        self.callable_ = callable_
        # Track the last *scheduled* minute we fired so we never fire
        # twice for the same wall-clock minute even if the scheduler
        # thread oversleeps.
        self.next_run_ts: datetime | None = None


class MonitorRunner:
    """Owns a bounded thread pool and a scheduler thread that ticks
    registered monitor skills on their cron schedules.

    Exactly one ``MonitorRunner`` exists per ``OrchestratorService``
    instance; the runner is built at service startup and shut down at
    service teardown.

    Concurrency: each tick is dispatched to the
    :class:`~concurrent.futures.ThreadPoolExecutor` so the scheduler
    thread itself never blocks on a slow ``observe`` tool. The pool
    size defaults to ``4`` (R6); each tick has a per-monitor timeout
    sourced from the skill's ``tick_timeout_seconds``.
    """

    def __init__(
        self,
        *,
        observe_fn: Callable[[str], Any],
        fire_trigger: Callable[[str, dict[str, Any]], None],
        max_workers: int = 4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._observe_fn = observe_fn
        self._fire_trigger = fire_trigger
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="monitor",
        )
        self._monitors: dict[str, _RegisteredMonitor] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Injection seam for tests; default uses real wall-clock UTC.
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ----- registration -----

    def register(self, skill: Skill) -> None:
        if skill.kind != "monitor":
            raise ValueError(
                f"MonitorRunner.register: skill {skill.name!r} kind="
                f"{skill.kind!r} (expected 'monitor')"
            )
        callable_ = make_monitor_callable(
            skill=skill,
            observe_fn=self._observe_fn,
            fire_trigger=self._fire_trigger,
        )
        with self._lock:
            if skill.name in self._monitors:
                raise ValueError(f"monitor {skill.name!r} already registered")
            self._monitors[skill.name] = _RegisteredMonitor(skill, callable_)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._monitors.pop(name, None)

    def registered(self) -> list[str]:
        with self._lock:
            return sorted(self._monitors.keys())

    # ----- lifecycle -----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MonitorRunner",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """Halt the scheduler thread and shut down the executor.

        ``wait=True`` (default) blocks up to ``timeout`` seconds for
        in-flight ticks to drain. Daemon threads are still joined so
        pytest fixture teardown is deterministic.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and wait:
            thread.join(timeout=timeout)
        self._executor.shutdown(wait=wait)
        self._thread = None

    # ----- test hook -----

    def tick_once(self, when: datetime | None = None) -> None:
        """Fire any monitors whose cron expression matches ``when``.

        Useful in tests where freezing wall-clock time is awkward; the
        production scheduler loop calls this internally too.
        """
        when = when or self._clock()
        # Truncate to the minute so identical seconds within a minute
        # don't fire the same monitor twice.
        minute = when.replace(second=0, microsecond=0)
        with self._lock:
            entries = list(self._monitors.values())
        for entry in entries:
            try:
                if not _cron_matches(entry.skill.schedule or "* * * * *", minute):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: cron parse failed (%s); skipping tick",
                    entry.skill.name, exc,
                )
                continue
            if entry.next_run_ts == minute:
                # Already fired this minute; idempotent on oversleep.
                continue
            entry.next_run_ts = minute
            self._dispatch(entry)

    def _dispatch(self, entry: _RegisteredMonitor) -> None:
        timeout = float(entry.skill.tick_timeout_seconds or 30.0)
        future = self._executor.submit(entry.callable_)

        def _wait_and_log() -> None:
            try:
                future.result(timeout=timeout)
            except FuturesTimeout:
                logger.warning(
                    "monitor %s: tick exceeded %.1fs timeout",
                    entry.skill.name, timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "monitor %s: tick raised %s", entry.skill.name, exc,
                )

        # Watcher runs on a side thread so the scheduler loop never
        # blocks waiting for a slow tick — the executor handles
        # parallelism, the watcher handles per-tick timeout reporting.
        threading.Thread(
            target=_wait_and_log,
            name=f"monitor-watch:{entry.skill.name}",
            daemon=True,
        ).start()

    # ----- scheduler loop -----

    def _run(self) -> None:
        """Single-threaded scheduler. Wakes once per second, fires
        any monitor whose cron expression matches the current minute,
        marks each fired monitor for the minute so we never fire
        twice if we oversleep.
        """
        while not self._stop.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001 — never crash the loop
                logger.warning("MonitorRunner loop error: %s", exc)
            # Sleep with frequent wakeups so stop() returns promptly.
            self._stop.wait(timeout=1.0)


__all__ = [
    "MonitorRunner",
    "SafeEvalError",
    "make_monitor_callable",
    "safe_eval",
]

# ====== module: runtime/graph.py ======

if TYPE_CHECKING:
    pass
# Phase 11 (FOC-04 / D-11-04): GraphInterrupt is the LangGraph
# pending-approval pause signal. It is NOT an error and must NOT route
# through _handle_agent_failure -- the orchestrator's interrupt-aware
# bridge handles the resume protocol via the checkpointer.
from langgraph.errors import GraphInterrupt


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
            # Phase 15 (LLM-COMPAT-01): the recursion_limit=25 workaround
            # introduced in 3ba099f as a safety net is gone — the
            # ``langchain.agents.create_agent`` migration replaces the
            # old two-call structure (loop + separate
            # ``with_structured_output`` pass) with a single tool-loop
            # whose terminal signal is the AgentTurnOutput tool call
            # itself (AutoStrategy → ToolStrategy fallback for non-
            # function-calling Ollama models). The default langgraph
            # recursion bound is now a true upper bound, not a workaround.
            return await executor.ainvoke(input_)
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): never retry a HITL pause.
            # GraphInterrupt is a checkpointed pending_approval signal,
            # not a transient error.
            raise
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


def _harvest_typed_terminal(
    tc_args: dict,
    state: tuple[float | None, str | None, str | None],
    valid_signals: frozenset[str] | None,
) -> tuple[float | None, str | None, str | None]:
    """Apply a typed-terminal tool call's args to the harvest state."""
    conf, rat, sig = state
    new_conf = _coerce_confidence(tc_args.get("confidence"))
    if new_conf is not None:
        conf = new_conf
    new_rat = _coerce_rationale(tc_args.get("confidence_rationale"))
    if new_rat is not None:
        rat = new_rat
    terminal = _coerce_signal("success", valid_signals)
    if terminal is not None:
        sig = terminal
    return conf, rat, sig


def _harvest_patch_tool(
    tc_args: dict,
    state: tuple[float | None, str | None, str | None],
    terminal_locked: bool,
    valid_signals: frozenset[str] | None,
) -> tuple[float | None, str | None, str | None]:
    """Apply a configured patch-tool's ``args.patch`` blob to the
    harvest state.

    When ``terminal_locked`` is True (a typed-terminal call already
    fired this session), confidence/rationale are pinned; only signal
    can flow through. Generalises the v1.0 single-tool path —
    apps register patch-tool names via
    ``OrchestratorConfig.patch_tools``.
    """
    conf, rat, sig = state
    patch = tc_args.get("patch") or {}
    merged_conf, merged_rat, merged_sig = _merge_patch_metadata(
        patch, conf, rat, sig, valid_signals,
    )
    if not terminal_locked:
        conf, rat = merged_conf, merged_rat
    return conf, rat, merged_sig


def _harvest_tool_calls_and_patches(
    messages: list,
    skill_name: str,
    incident: Session,
    ts: str,
    valid_signals: frozenset[str] | None = None,
    terminal_tool_names: frozenset[str] = frozenset(),
    patch_tool_names: frozenset[str] = frozenset(),
) -> tuple[float | None, str | None, str | None]:
    """Iterate agent messages, record ToolCall entries on the session, and
    harvest confidence / confidence_rationale / signal from typed terminal
    tools or app-declared patch tools.

    Typed terminal tools (those whose bare name is in
    ``terminal_tool_names``, supplied by the caller from
    ``OrchestratorConfig.terminal_tools``) carry confidence and rationale
    as flat kwargs; they imply ``signal=success`` since invoking a
    terminal tool is the agent's declaration that *its stage* completed
    cleanly — not that the session itself was successfully resolved.
    The session-level outcome is inferred separately from tool_calls
    history by ``_finalize_session_status``. Non-terminal agents emit
    routing signal via patch-tool args.

    ``patch_tool_names`` lists the bare tool names that ship a
    ``patch:`` arg the harvester should merge. Empty default means
    "no patch tools" so unconfigured apps pay nothing. Generalises
    the v1.0 single-tool path; apps register patch-tool names via
    ``OrchestratorConfig.patch_tools``.

    Once a typed terminal tool has fired, its confidence/rationale are
    authoritative — a same-message patch must not override them.
    Signal still flows from later patches so triage-style routing
    remains expressive.

    Returns ``(agent_confidence, agent_rationale, agent_signal)``.
    """
    state: tuple[float | None, str | None, str | None] = (None, None, None)
    terminal_locked = False
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            tc_name = tc.get("name", "unknown")
            tc_args = tc.get("args", {}) or {}
            # MCP tools follow ``<server>:<tool>`` with exactly one
            # colon; rsplit on the rightmost colon recovers the bare
            # tool name for both prefixed and unprefixed forms.
            tc_original = tc_name.rsplit(":", 1)[-1]
            incident.tool_calls.append(ToolCall(
                agent=skill_name, tool=tc_name, args=tc_args,
                result=None, ts=ts,
            ))
            if tc_original in terminal_tool_names:
                state = _harvest_typed_terminal(tc_args, state, valid_signals)
                terminal_locked = True
            elif tc_original in patch_tool_names:
                state = _harvest_patch_tool(
                    tc_args, state, terminal_locked, valid_signals,
                )
    return state


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


def _first_terminal_tool_called_this_turn(
    messages: list,
    terminal_tool_names: frozenset[str],
) -> str | None:
    """Return the bare name of the first typed-terminal tool called this turn.

    Phase 10 (FOC-03 / D-10-03): used to label the reconciliation log so
    operators can correlate envelope-vs-tool-arg confidence divergences
    against a specific tool. Tool names may be MCP-prefixed
    (``<server>:<tool>``); we rsplit on the rightmost colon to recover the
    bare name and match against the configured ``terminal_tool_names``.
    Returns None when no terminal tool fired this turn.
    """
    if not terminal_tool_names:
        return None
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name", "")
            bare = name.rsplit(":", 1)[-1]
            if bare in terminal_tool_names:
                return bare
    return None


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


def _try_recover_envelope_from_raw(raw: str) -> AgentTurnOutput | None:
    """Attempt to extract an :class:`AgentTurnOutput` from a raw LLM
    string when LangGraph's structured-output pass raised
    ``OutputParserException``.

    Strategy:
    1. Parse the whole string as JSON.
    2. If that fails, scan for the first balanced ``{...}`` substring
       and try parsing that (handles markdown-fenced JSON or trailing
       chatter).
    3. Validate the parsed dict against :class:`AgentTurnOutput`.

    Returns the parsed envelope on success, ``None`` on any failure.
    """
    if not raw or not raw.strip():
        return None
    candidates: list[str] = [raw]
    # Markdown-fenced JSON: ```json\n{...}\n```
    if "```" in raw:
        for chunk in raw.split("```"):
            stripped = chunk.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
            if stripped.startswith("{"):
                candidates.append(stripped)
    # Greedy: first '{' through last '}'
    first = raw.find("{")
    last = raw.rfind("}")
    if 0 <= first < last:
        candidates.append(raw[first:last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            return AgentTurnOutput.model_validate(payload)
        except Exception:  # noqa: BLE001
            continue
    return None


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
    # Mark the session as terminally failed so the UI can render a
    # retry control. The retry path (``Orchestrator.retry_session``)
    # is the only documented way to move out of this state.
    incident.status = "error"
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
    terminal_tool_names: frozenset[str] = frozenset(),
    patch_tool_names: frozenset[str] = frozenset(),
    injected_args: dict[str, str] | None = None,
    gate_policy: "GatePolicy | None" = None,
    event_log: "EventLog | None" = None,
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

    ``terminal_tool_names`` and ``patch_tool_names`` are the bare tool
    names the harvester should treat as typed-terminal /
    patch-emitting (sourced from ``OrchestratorConfig.terminal_tools``
    union ``OrchestratorConfig.harvest_terminal_tools`` /
    ``OrchestratorConfig.patch_tools``). Empty defaults preserve the
    "no harvester recognition" behavior for legacy callers.

    ``injected_args`` (Phase 9 / D-09-01) is the orchestrator-wide
    map of ``arg_name -> dotted_path`` declared in
    :attr:`OrchestratorConfig.injected_args`. Every entry is stripped
    from each tool's LLM-visible signature (so the LLM cannot emit a
    value for it) and re-supplied at invocation time from session
    state. When ``None`` or empty, tools pass through to the LLM
    unchanged — preserves legacy callers and the framework default.
    """

    async def node(state: GraphState) -> dict:
        incident = state["session"]  # pyright: ignore[reportTypedDictNotRequiredAccess] — orchestrator runtime always supplies session
        inc_id = incident.id
        started_at = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # M3 (per-step telemetry): emit agent_started.
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "agent_started",
                    agent=skill.name, started_at=started_at,
                )
            except Exception:  # noqa: BLE001 — telemetry must not break the agent
                logger.debug(
                    "event_log.record(agent_started) failed", exc_info=True,
                )

        # Phase 9 (D-09-01): strip injected-arg keys from every tool's
        # LLM-visible signature BEFORE create_react_agent serialises the
        # tool surface — so the LLM literally cannot emit values for
        # those params. The framework re-supplies them at invocation
        # time inside the gateway (or an inject-only wrapper) below.

        injected_keys = frozenset((injected_args or {}).keys())
        if injected_keys:
            visible_tools = [
                strip_injected_params(t, injected_keys) for t in tools
            ]
        else:
            visible_tools = tools

        # Wrap tools per-invocation so each wrap closes over the live
        # ``Session`` for this run. When the gateway is unconfigured,
        # the original tools pass through untouched and
        # ``create_react_agent`` sees the same surface as before.
        if gateway_cfg is not None:
            # Pass ORIGINAL tools (pre-strip) to wrap_tool — the gateway
            # wrapper strips internally for the LLM-visible schema while
            # keeping ``inner.args_schema`` intact so
            # ``accepted_params_for_tool`` correctly recognises injected
            # keys (e.g. ``environment``) as accepted by the underlying
            # tool. Stripping twice (here AND in wrap_tool) hides those
            # keys from ``accepted_params``, the inject step skips them,
            # and FastMCP rejects the call as missing required arg.
            run_tools = [
                wrap_tool(t, session=incident, gateway_cfg=gateway_cfg,
                          agent_name=skill.name, store=store,
                          injected_args=injected_args or {},
                          gate_policy=gate_policy,
                          event_log=event_log)
                for t in tools
            ]
        elif injected_keys:
            # No gateway, but injected_args is configured — wrap each
            # tool in an inject-only ``StructuredTool`` so the LLM-visible
            # sig matches ``visible_tools`` while the underlying call
            # still receives the framework-supplied values.
            from langchain_core.tools import StructuredTool

            _inject_cfg = injected_args or {}

            def _make_inject_only_wrapper(
                base: BaseTool, llm_visible: BaseTool, sess: Session,
            ) -> BaseTool:
                async def _arun(**kwargs: Any) -> Any:
                    new_kwargs = _inject_args(
                        kwargs,
                        session=sess,
                        injected_args_cfg=_inject_cfg,
                        tool_name=base.name,
                    )
                    return await base.ainvoke(new_kwargs)

                def _run(**kwargs: Any) -> Any:
                    new_kwargs = _inject_args(
                        kwargs,
                        session=sess,
                        injected_args_cfg=_inject_cfg,
                        tool_name=base.name,
                    )
                    return base.invoke(new_kwargs)

                return StructuredTool.from_function(
                    func=_run,
                    coroutine=_arun,
                    name=base.name,
                    description=base.description,
                    args_schema=llm_visible.args_schema,
                )

            run_tools = [
                _make_inject_only_wrapper(orig, vis, incident)
                for orig, vis in zip(tools, visible_tools)
            ]
        else:
            run_tools = visible_tools
        # Phase 10 (FOC-03 / D-10-02) + Phase 15 (LLM-COMPAT-01): every
        # responsive agent invocation is wrapped in an AgentTurnOutput
        # envelope. ``langchain.agents.create_agent`` (the non-deprecated
        # successor to ``langgraph.prebuilt.create_react_agent``) accepts a
        # bare schema as ``response_format`` and, by default, wraps it in
        # ``AutoStrategy`` — ProviderStrategy for models with native
        # structured-output (OpenAI-class), falling back to ToolStrategy
        # otherwise (Ollama). ToolStrategy injects AgentTurnOutput as a
        # callable tool: when the LLM ``calls`` it, the loop terminates on
        # the same turn with ``result["structured_response"]`` populated.
        # Eliminates the old two-call structure (loop + separate
        # ``with_structured_output`` pass) that hit recursion_limit=25 on
        # Ollama models without true function-calling.
        agent_executor = create_agent(
            model=llm,
            tools=run_tools,
            system_prompt=skill.system_prompt,
            response_format=AgentTurnOutput,
        )

        # Phase 11 (FOC-04): reset per-turn confidence hint. The hint
        # is updated below after _harvest_tool_calls_and_patches; on
        # re-entry from a HITL pause the hint resets cleanly so a new
        # turn starts from "no signal yet" (None).
        try:
            incident.turn_confidence_hint = None
        except (AttributeError, ValueError):
            pass

        try:
            result = await _ainvoke_with_retry(
                agent_executor,
                {"messages": [HumanMessage(content=_format_agent_input(incident))]},
            )
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): HITL pause is NOT an error.
            # Re-raise so LangGraph's checkpointer captures the paused
            # state. Session.status is left to the orchestrator's
            # interrupt-aware bridge, NOT _handle_agent_failure.
            raise
        except Exception as exc:  # noqa: BLE001
            # Phase 10 follow-up: when LangGraph's structured-output pass
            # raises ``OutputParserException`` (Ollama / non-OpenAI
            # providers don't always honor ``response_format`` cleanly),
            # try to recover by parsing the raw LLM output ourselves.
            # The exception's ``llm_output`` carries the model's reply
            # verbatim; if it contains JSON matching the envelope schema,
            # build a synthetic ``result`` and continue. On unrecoverable
            # failure, log the raw output for diagnosis and fall through
            # to ``_handle_agent_failure``.
            try:
                from langchain_core.exceptions import OutputParserException
            except ImportError:  # pragma: no cover — langchain always present
                OutputParserException = ()  # type: ignore[assignment]
            if isinstance(exc, OutputParserException):
                raw = getattr(exc, "llm_output", "") or ""
                logger.warning(
                    "agent.structured_output_parse_failure agent=%s "
                    "raw_len=%d raw_preview=%r",
                    skill.name, len(raw), raw[:500],
                )
                recovered = _try_recover_envelope_from_raw(raw)
                if recovered is not None:
                    logger.info(
                        "agent.structured_output_recovered agent=%s",
                        skill.name,
                    )
                    result = {
                        "messages": [],
                        "structured_response": recovered,
                    }
                else:
                    return _handle_agent_failure(
                        skill_name=skill.name, started_at=started_at, exc=exc,
                        inc_id=inc_id, store=store, fallback=incident,
                    )
            else:
                return _handle_agent_failure(
                    skill_name=skill.name, started_at=started_at, exc=exc,
                    inc_id=inc_id, store=store, fallback=incident,
                )

        # Tools (e.g. registered patch tools) write straight to disk.
        # Reload so the node's own append of agent_run + tool_calls
        # happens against the tool-mutated state — otherwise saving
        # the stale in-memory object clobbers the tools' writes.
        incident = store.load(inc_id)

        messages = result.get("messages", [])
        ts = datetime.now(timezone.utc).strftime(_UTC_TS_FMT)

        # Record tool calls and harvest confidence/signal from configured
        # patch / typed-terminal tools.
        agent_confidence, agent_rationale, agent_signal = _harvest_tool_calls_and_patches(
            messages, skill.name, incident, ts, valid_signals,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
        )
        # Phase 11 (FOC-04): update hint so any subsequent in-turn
        # tool call sees the harvested confidence at the gateway.
        if agent_confidence is not None:
            try:
                incident.turn_confidence_hint = agent_confidence
            except (AttributeError, ValueError):
                pass

        # Pair tool responses with their tool calls.
        _pair_tool_responses(messages, incident)

        # Phase 10 (FOC-03 / D-10-03): parse the structural envelope and
        # reconcile its confidence against any typed-terminal-tool arg
        # confidence harvested above. Envelope failure is a hard error —
        # mark the agent_run failed with structured cause.
        try:
            envelope = parse_envelope_from_result(result, agent=skill.name)
        except EnvelopeMissingError as exc:
            return _handle_agent_failure(
                skill_name=skill.name, started_at=started_at, exc=exc,
                inc_id=inc_id, store=store, fallback=incident,
            )

        terminal_tool_for_log = _first_terminal_tool_called_this_turn(
            messages, terminal_tool_names,
        )
        final_confidence = reconcile_confidence(
            envelope.confidence,
            agent_confidence,
            agent=skill.name,
            session_id=inc_id,
            tool_name=terminal_tool_for_log,
        )
        final_rationale = agent_rationale or envelope.confidence_rationale
        final_signal = agent_signal if agent_signal is not None else envelope.signal

        # Final summary text and token usage.
        # Envelope content takes precedence over last AIMessage scrape.
        final_text = envelope.content or _extract_final_text(messages)
        usage = _sum_token_usage(messages)

        # M3: emit confidence_emitted after reconcile lands.
        if event_log is not None and final_confidence is not None:
            try:
                event_log.record(
                    inc_id, "confidence_emitted",
                    agent=skill.name,
                    value=float(final_confidence),
                    rationale=final_rationale or "",
                    signal=final_signal,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(confidence_emitted) failed", exc_info=True,
                )

        _record_success_run(
            incident=incident, skill_name=skill.name, started_at=started_at,
            final_text=final_text, usage=usage,
            confidence=final_confidence, rationale=final_rationale, signal=final_signal,
            store=store,
        )
        next_route_signal = decide_route(incident)
        next_node = route_from_skill(skill, next_route_signal)

        # M3: emit route_decided + agent_finished (carrying token_usage).
        if event_log is not None:
            try:
                event_log.record(
                    inc_id, "route_decided",
                    agent=skill.name,
                    signal=next_route_signal,
                    next_node=next_node,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(route_decided) failed", exc_info=True,
                )
            try:
                event_log.record(
                    inc_id, "agent_finished",
                    agent=skill.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "event_log.record(agent_finished) failed", exc_info=True,
                )

        return {"session": incident, "next_route": next_node,
                "last_agent": skill.name, "error": None}

    return node


def _decide_from_signal(inc: Session) -> str:
    """Return the latest agent's emitted signal, or "default" if absent.

    Agents emit one of {success, failed, needs_input} via the ``signal``
    key of a configured patch tool (see ``_coerce_signal``).
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

# Phase 10 (FOC-03): per-agent default envelope confidence for the stub
# LLM. Pre-Phase-10 the deep_investigator stub emitted no confidence at
# all, so the gate (threshold 0.75) always interrupted on the first
# call. Post-Phase-10 every agent must emit a confidence value — drive
# DI's stub envelope below threshold to preserve gate-pause behavior in
# existing tests. Other agents default to 0.85 (above threshold).
_DEFAULT_STUB_ENVELOPE_CONFIDENCE: dict[str, float] = {
    "deep_investigator": 0.30,
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
                       registry: ToolRegistry,
                       event_log: "EventLog | None" = None) -> dict:
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
    # Phase 11 (FOC-04): thread the orchestrator's gate_policy down to
    # wrap_tool so should_gate can apply the configured per-app
    # confidence threshold + gated environments / risk actions.
    gate_policy = getattr(cfg.orchestrator, "gate_policy", None)
    # Build the harvester's tool-name sets once per graph-build. The
    # union of ``terminal_tools`` (status-transitioning) and
    # ``harvest_terminal_tools`` (harvest-only) gives the full
    # typed-terminal recognition surface for confidence/rationale
    # capture; ``patch_tools`` carries the patch-blob harvester path.
    terminal_tool_names = frozenset(
        r.tool_name for r in cfg.orchestrator.terminal_tools
    ) | frozenset(cfg.orchestrator.harvest_terminal_tools)
    patch_tool_names = frozenset(cfg.orchestrator.patch_tools)
    nodes: dict = {}
    for agent_name, skill in skills.items():
        kind = getattr(skill, "kind", "responsive")
        if kind == "monitor":
            # Monitors are not graph nodes; skip silently.
            continue
        if kind == "supervisor":
            llm = None
            if skill.dispatch_strategy == "llm":
                llm = get_llm(
                    cfg.llm, skill.model, role=agent_name,
                    default_llm_request_timeout=cfg.orchestrator.default_llm_request_timeout,
                )
            nodes[agent_name] = make_supervisor_node(skill=skill, llm=llm)
            continue
        # Default / "responsive" path.
        if skill.stub_response is not None:
            stub_canned: dict[str, str] | None = {skill.name: skill.stub_response}
        elif agent_name in _DEFAULT_STUB_CANNED:
            stub_canned = {agent_name: _DEFAULT_STUB_CANNED[agent_name]}
        else:
            stub_canned = None
        # Phase 10 (FOC-03): wire a per-agent default envelope confidence
        # into the stub so pre-Phase-10 gate-pause-on-DI tests still pass.
        stub_env_conf = _DEFAULT_STUB_ENVELOPE_CONFIDENCE.get(agent_name)
        llm = get_llm(
            cfg.llm,
            skill.model,
            role=agent_name,
            stub_canned=stub_canned,
            stub_envelope_confidence=stub_env_conf,
            default_llm_request_timeout=cfg.orchestrator.default_llm_request_timeout,
        )
        tools = registry.resolve(skill.tools, cfg.mcp)
        decide = _decide_from_signal
        nodes[agent_name] = make_agent_node(
            skill=skill, llm=llm, tools=tools,
            decide_route=decide, store=store,
            valid_signals=valid_signals,
            gateway_cfg=gateway_cfg,
            terminal_tool_names=terminal_tool_names,
            patch_tool_names=patch_tool_names,
            injected_args=cfg.orchestrator.injected_args,
            gate_policy=gate_policy,
            event_log=event_log,
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
                      framework_cfg: FrameworkAppConfig | None = None,
                      event_log: "EventLog | None" = None):
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
    # ``resolve_framework_app_config(None)`` always returns a bare
    # ``FrameworkAppConfig`` (never None), so the chain above is
    # exhaustive — assert for pyright's flow narrowing.
    assert framework_cfg is not None
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store,
                                registry=registry, event_log=event_log)
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
    # ``langgraph-checkpoint-postgres`` is an optional extra (declared
    # under [project.optional-dependencies].postgres in pyproject) so
    # the wheel is not present in CI's SQLite-only install. The module
    # is only imported on the Postgres URL branch in production.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # pyright: ignore[reportMissingImports]
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
    # Apps own the strict signature -- the framework only enforces
    # ``callable``. The cast satisfies the declared return type without
    # adding a runtime wrapper.
    return cast(Callable[..., dict], obj)

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
        # ``IdempotencyRow.__table__`` is a ``Table`` at runtime; the
        # SQLAlchemy stub types it as the wider ``FromClause``.
        Base.metadata.create_all(engine, tables=[IdempotencyRow.__table__])  # pyright: ignore[reportArgumentType]
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
            # ``rowcount`` is exposed on ``CursorResult`` (the concrete
            # return of DML execute); the abstract ``Result`` stub does
            # not declare it.
            return result.rowcount or 0  # pyright: ignore[reportAttributeAccessIssue]

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
            # Plugin transports inherit from the abstract
            # ``TriggerTransport`` (no positional args declared on the
            # ABC) but every concrete subclass loaded via the entry-
            # point registry must accept the plugin's config object.
            # The ABC mismatch is a stub limitation, not a runtime bug.
            transports.append(kind_cls(pcfg))  # pyright: ignore[reportCallIssue]

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
    event_log: Any = None           # Optional[EventLog] — M1 telemetry sink
    lesson_store: Any = None        # Optional[LessonStore] — M6 lesson corpus
    top_k: int = 3
    similarity_threshold: float = 0.7


def _project_prior(session: Session) -> dict[str, Any]:
    """Compact representation suitable for stashing on findings."""
    return {"id": session.id, "status": session.status}


def _source_session_is_live(lesson_store: Any, source_session_id: str) -> bool:
    """M9: True iff the lesson's source IncidentRow exists AND its
    ``deleted_at`` is NULL. Soft-deleted source sessions suppress
    their lessons from downstream intake surfaces.

    Best-effort: any lookup error is treated as "live" so a flaky DB
    doesn't silently hide lessons. ``lesson_store.engine`` is the
    canonical handle — falling back to ``True`` keeps the runner
    permissive when the store has no engine attached (test stubs).
    """
    engine = getattr(lesson_store, "engine", None)
    if engine is None:
        return True
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session as SqlaSession


        with SqlaSession(engine) as s:
            row = s.execute(
                select(IncidentRow.deleted_at).where(
                    IncidentRow.id == source_session_id
                )
            ).first()
        if row is None:
            return False
        return row[0] is None
    except Exception:  # noqa: BLE001
        return True


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

    # M6: stamp findings["lessons"] from the auto-learning corpus. The
    # intake runner surfaces "incidents like this one were resolved by
    # running tools X, Y, Z" as a hypothesis surface for downstream
    # agents — not a verdict. Best-effort: lesson_store failures are
    # logged and skipped so a misconfigured embedding backend never
    # blocks intake.
    #
    # M9 contract: lessons whose source session has been soft-deleted
    # (incidents.deleted_at IS NOT NULL) MUST be filtered out so an
    # operator-deleted prior session no longer biases new intakes.
    if ctx.lesson_store is not None and text:
        try:
            lesson_hits = ctx.lesson_store.find_similar(
                query=text,
                limit=ctx.top_k,
                threshold=ctx.similarity_threshold,
            )
        except Exception:  # noqa: BLE001 — never block intake on a corpus query
            _log.warning(
                "default_intake_runner: lesson_store.find_similar failed; "
                "skipping for session %s", session.id, exc_info=True,
            )
            lesson_hits = []
        live_hits = [
            (lesson, score) for lesson, score in lesson_hits
            if _source_session_is_live(ctx.lesson_store, lesson.source_session_id)
        ]
        session.findings["lessons"] = [
            {
                "id": lesson.id,
                "summary": lesson.outcome_summary,
                "tools": [
                    t.get("tool") for t in lesson.tool_sequence if t.get("tool")
                ],
            }
            for lesson, _score in live_hits
        ]
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

# ====== module: runtime/locks.py ======

class SessionBusy(RuntimeError):
    """Raised when a session is already executing and cannot accept a new turn.

    Callers should surface this as HTTP 429 with a ``Retry-After: 1`` header
    so that clients know the session will become available shortly.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id!r} is already executing")
        self.session_id = session_id


class _Slot:
    """Per-session lock state: the lock plus reentrancy tracking."""

    __slots__ = ("lock", "owner", "depth")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task | None = None
        self.depth = 0


class SessionLockRegistry:
    """In-process registry of per-session task-reentrant asyncio locks.

    TODO(v2): evict idle slots to cap memory usage for long-running servers.
    """

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}  # TODO(v2): add eviction for idle sessions

    def _slot(self, session_id: str) -> _Slot:
        slot = self._slots.get(session_id)
        if slot is None:
            slot = _Slot()
            self._slots[session_id] = slot
        return slot

    def get(self, session_id: str) -> asyncio.Lock:
        """Return the underlying lock for ``session_id``.

        Direct ``async with reg.get(sid):`` does NOT honour reentrancy.
        Prefer ``async with reg.acquire(sid):`` for nested-safe entry.
        """
        return self._slot(session_id).lock

    def is_locked(self, session_id: str) -> bool:
        """Return ``True`` iff ``session_id`` currently holds the lock.

        Non-blocking. Returns ``False`` for unknown / never-seen session ids
        (no slot is created as a side-effect of this call).
        """
        slot = self._slots.get(session_id)
        return slot is not None and slot.lock.locked()

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Acquire the per-session lock for the duration of the block.

        Reentrant on the current ``asyncio.Task``: if this task already
        holds the lock, the call is a no-op (depth is bumped and yields
        immediately). The actual ``Lock.release`` only happens when the
        outermost ``acquire`` exits.
        """
        slot = self._slot(session_id)
        current = asyncio.current_task()
        if slot.owner is current and current is not None:
            slot.depth += 1
            try:
                yield
            finally:
                slot.depth -= 1
            return
        await slot.lock.acquire()
        slot.owner = current
        slot.depth = 1
        try:
            yield
        finally:
            slot.depth -= 1
            if slot.depth == 0:
                slot.owner = None
                slot.lock.release()

    @asynccontextmanager
    async def try_acquire(self, session_id: str) -> AsyncIterator[None]:
        """Acquire-or-fail. TOCTOU-free single-shot.

        Raises :class:`SessionBusy` immediately if the lock is already
        held; otherwise acquires and yields. Releases on exit.

        Not task-reentrant: if the calling task already holds the lock,
        this still raises. Callers that need reentry use :meth:`acquire`.

        TOCTOU note: ``lock.locked()`` then ``lock.acquire()`` would have
        a check/use window in a multi-threaded world, but asyncio is
        single-threaded per loop and there is no ``await`` between the
        check and the acquire — same-loop callers cannot interleave.
        Cross-thread callers must not use this registry.
        """
        slot = self._slot(session_id)
        if slot.lock.locked():
            raise SessionBusy(session_id)
        await slot.lock.acquire()
        slot.owner = asyncio.current_task()
        slot.depth = 1
        try:
            yield
        finally:
            slot.depth -= 1
            if slot.depth == 0:
                slot.owner = None
                slot.lock.release()

# ====== module: runtime/skill_validator.py ======

class SkillValidationError(RuntimeError):
    """Raised when skill YAML references a tool or route that does not
    exist or is malformed. Refuses to start the orchestrator."""


def _build_bare_to_full_map(registered_tools: set[str]) -> dict[str, list[str]]:
    """Map bare tool name → list of fully-qualified ``<server>:<tool>``."""
    bare_to_full: dict[str, list[str]] = {}
    for full in registered_tools:
        bare = full.split(":", 1)[1] if ":" in full else full
        bare_to_full.setdefault(bare, []).append(full)
    return bare_to_full


def _check_tool_ref(
    skill_name: str,
    tool_ref: str,
    registered_tools: set[str],
    bare_to_full: dict[str, list[str]],
) -> None:
    """Raise SkillValidationError if ``tool_ref`` doesn't resolve to a
    registered tool, or resolves ambiguously across multiple servers."""
    if tool_ref in registered_tools:
        return
    resolutions = bare_to_full.get(tool_ref)
    if resolutions is None:
        raise SkillValidationError(
            f"skill {skill_name!r} references tool {tool_ref!r} which "
            f"is not registered. Known tools: {sorted(registered_tools)[:10]}..."
        )
    if len(resolutions) > 1:
        raise SkillValidationError(
            f"skill {skill_name!r} uses bare tool ref {tool_ref!r} but "
            f"it is exposed by multiple servers: {sorted(resolutions)}. "
            f"Use the prefixed form to disambiguate."
        )


def validate_skill_tool_references(
    skills: dict, registered_tools: set[str],
) -> None:
    """Assert every ``tools.local`` entry in every skill resolves to a
    registered MCP tool.

    ``registered_tools`` is the set of fully-qualified ``<server>:<tool>``
    names from the MCP loader. We accept either bare or prefixed forms
    in skill YAML (the LLM-facing call uses prefixed; YAML can use
    either for ergonomics).
    """
    bare_to_full = _build_bare_to_full_map(registered_tools)
    for skill_name, skill in skills.items():
        local = (skill.get("tools") or {}).get("local") or []
        for tool_ref in local:
            _check_tool_ref(skill_name, tool_ref, registered_tools, bare_to_full)


def validate_skill_routes(skills: dict) -> None:
    """Assert every skill has a ``when: default`` route entry.

    Skipped for ``kind: supervisor`` skills — supervisors dispatch via
    ``dispatch_rules`` to subordinates and do not use the ``routes``
    table at all.
    """
    for skill_name, skill in skills.items():
        if skill.get("kind") == "supervisor":
            continue
        routes = skill.get("routes") or []
        if not any((r.get("when") == "default") for r in routes):
            raise SkillValidationError(
                f"skill {skill_name!r} has no ``when: default`` route — "
                f"agents whose signal doesn't match a rule will hang."
            )

# ====== module: runtime/storage/checkpoint_gc.py ======

def gc_orphaned_checkpoints(engine: Engine) -> int:
    """Remove orphaned checkpoint rows; return count removed.

    Returns 0 if the ``checkpoints`` table doesn't exist (fresh DB,
    LangGraph checkpointer has not yet bootstrapped its schema).
    """
    with engine.begin() as conn:
        live_ids = {row[0] for row in conn.execute(
            text("SELECT id FROM incidents")
        )}
        try:
            rows = conn.execute(text(
                "SELECT DISTINCT thread_id FROM checkpoints"
            )).all()
        except OperationalError:
            return 0
        # thread_id may be ``INC-1`` or ``INC-1:retry-N`` — strip suffix.
        orphans = []
        for (tid,) in rows:
            base = tid.split(":")[0] if tid else tid
            if base not in live_ids:
                orphans.append(tid)
        for tid in orphans:
            conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :tid"),
                {"tid": tid},
            )
        return len(orphans)

# ====== module: runtime/orchestrator.py ======

if TYPE_CHECKING:
    # Avoid a runtime circular import — ``runtime.triggers.base`` only
    # defines a dataclass, and the type appears in a method annotation.
    pass






from langgraph.errors import GraphInterrupt
from langgraph.types import Command













_log = logging.getLogger("runtime.orchestrator")


def _assert_envelope_invariant_on_finalize(session: "Session") -> None:
    """Phase 10 (FOC-03) defence-in-depth log sweep.

    Hard rejection of envelope-less turns happens at the agent runner
    (``parse_envelope_from_result`` raises ``EnvelopeMissingError``,
    which the runner converts into an agent_run marked ``error``).
    This finalize hook only logs WARNING for forensics on legacy on-disk
    sessions whose agent_runs predate the envelope contract. Never
    raises.
    """
    for ar in session.agents_run:
        if ar.confidence is None:
            _log.warning(
                "agent_run.envelope_missing agent=%s session_id=%s",
                ar.agent,
                session.id,
            )


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


# ---------------------------------------------------------------------
# M4 (per-step telemetry): status_changed emission helpers. Kept at
# module scope so test shims that build a partial _O class with only
# specific Orchestrator methods attached still drive the finalize
# path without needing every new helper in their attribute list.
# ---------------------------------------------------------------------

def _latest_terminal_tool_for_status(
    rules,
    tool_calls,
    new_status: str,
) -> str | None:
    """Return the bare name of the most recent executed terminal-tool
    that maps to ``new_status``, for use as the ``cause`` field on the
    ``status_changed`` event. Returns ``None`` if no rule matches.
    """
    bare_names = {r.tool_name for r in rules if r.status == new_status}
    executed = [
        tc for tc in tool_calls
        if getattr(tc, "status", None) == "executed"
    ]
    for tc in reversed(executed):
        name = (tc.tool or "").split(":")[-1]
        if name in bare_names:
            return name
    return None


def _emit_status_changed_event(
    *,
    orch,
    inc,
    from_status: str,
    to_status: str,
    cause: str,
) -> None:
    """Emit a ``status_changed`` event through orch.event_log (when
    present) and trigger the M5 lesson-extraction hook on terminal
    statuses (no-op until M5 wires up LessonExtractor).
    Resilient to shim test classes that don't carry ``event_log``.
    """
    event_log = getattr(orch, "event_log", None)
    if event_log is not None:
        try:
            event_log.record(
                inc.id,
                "status_changed",
                **{"from": from_status, "to": to_status, "cause": cause},
            )
        except Exception:  # noqa: BLE001 — telemetry must not break finalize
            _log.debug(
                "event_log.record(status_changed) failed", exc_info=True,
            )

    # M5 hook point: when ``to_status`` is terminal per app config,
    # invoke the lesson extractor. M4 leaves it as a no-op; M5 swaps
    # this body for the real ``LessonExtractor.extract`` call.
    statuses = getattr(getattr(orch, "cfg", None), "orchestrator", None)
    if statuses is None:
        return
    status_def = statuses.statuses.get(to_status)
    if status_def is not None and status_def.terminal:
        _extract_lesson_on_terminal(orch=orch, inc=inc)


def _extract_lesson_on_terminal(*, orch, inc) -> None:
    """M6: run the LessonExtractor against the finalized session and
    persist the row through the LessonStore. No-op when either the
    event log or the lesson store is unavailable (shim test classes,
    apps that disable the corpus, etc).

    Failures here are logged and dropped — terminal-status routing
    must never fail because the corpus write hiccupped.
    """
    event_log = getattr(orch, "event_log", None)
    lesson_store = getattr(orch, "lesson_store", None)
    if event_log is None or lesson_store is None:
        return None
    try:


        row = LessonExtractor.extract(session=inc, event_log=event_log)
        if row is None:
            return None
        lesson_store.add(row)
    except Exception:  # noqa: BLE001 — finalize must never break on corpus write
        _log.warning(
            "lesson extraction failed for session %s; finalize continues",
            inc.id, exc_info=True,
        )
    return None


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
                 event_log: EventLog | None = None,
                 lesson_store: "Any | None" = None,
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
        # M1 (per-step telemetry): append-only event sink. Single instance
        # shared with framework_cfg.intake_context.event_log so module-level
        # supervisor runners can emit via the same handle.
        self.event_log = event_log
        # M5/M6: lesson corpus store. Shared with
        # framework_cfg.intake_context.lesson_store so the default intake
        # runner reads from the same handle the finalize hook writes to.
        self.lesson_store = lesson_store
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
        # Per-session asyncio.Lock keyed off session_id; serializes
        # finalize and retry within a single process so concurrent
        # streams cannot race on terminal-status transitions.
        self._locks = SessionLockRegistry()
        # Membership-tracked rejection of concurrent retry_session calls
        # on the same session id. The set is mutated under self._locks
        # so the in-flight check + add is atomic per session.
        self._retries_in_flight: set[str] = set()
        # Resolved app-declared pydantic schema for the
        # ``state_overrides=`` kwarg of ``start_session``. ``None``
        # (default) means no validation — start_session passes the
        # dict through unchanged (DECOUPLE-05 / D-08-02 backward-
        # compat). Set by ``Orchestrator.create()`` after importlib
        # resolution of ``cfg.orchestrator.state_overrides_schema``.
        self._state_overrides_cls: type[BaseModel] | None = None

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
            # DECOUPLE-05 / D-08-01: resolve the app-declared
            # state-overrides pydantic schema once at boot. ``None``
            # means no validation (D-08-02 backward-compat). The
            # config-load layer already format-validated the dotted
            # path; importlib failures here surface as a clear boot
            # error naming the path AND the underlying cause.
            state_overrides_cls: type[BaseModel] | None = None
            schema_path = cfg.orchestrator.state_overrides_schema
            if schema_path is not None:
                if ":" in schema_path:
                    module_path, _, class_name = schema_path.rpartition(":")
                else:
                    module_path, _, class_name = schema_path.rpartition(".")
                try:
                    schema_module = importlib.import_module(module_path)
                except ImportError as exc:
                    raise RuntimeError(
                        f"state_overrides_schema={schema_path!r}: "
                        f"failed to import module {module_path!r}: {exc}"
                    ) from exc
                try:
                    state_overrides_cls = getattr(schema_module, class_name)
                except AttributeError as exc:
                    raise RuntimeError(
                        f"state_overrides_schema={schema_path!r}: "
                        f"module {module_path!r} has no attribute "
                        f"{class_name!r}"
                    ) from exc
                if not (isinstance(state_overrides_cls, type)
                        and issubclass(state_overrides_cls, BaseModel)):
                    raise RuntimeError(
                        f"state_overrides_schema={schema_path!r}: "
                        f"{class_name!r} is not a pydantic BaseModel "
                        f"subclass"
                    )
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
                id_prefix=framework_cfg.session_id_prefix,
            )
            history = HistoryStore(
                engine=engine,
                state_cls=repo_state_cls,
                embedder=embedder,
                vector_store=vector_store,
                similarity_threshold=framework_cfg.similarity_threshold,
                distance_strategy=cfg.storage.vector.distance_strategy,
            )
            # M1 (per-step telemetry): append-only event sink writing into
            # session_events; shared by AgentRunRecorder, gateway, and the
            # status-finalize hook (M3/M4). One row per agent boundary or
            # tool call — never mutated.
            event_log = EventLog(engine=engine)
            # M5 + M6: lesson corpus + vector index. Reuses the same
            # backend/distance strategy as the main session vector store;
            # collection_name="lessons" produces a sibling FAISS file
            # under the same path (or a separate pgvector row family
            # under collection "lessons").



            migrate_add_lesson_table(engine)
            _lesson_vector_cfg = _VectorConfig(
                backend=cfg.storage.vector.backend,
                path=cfg.storage.vector.path,
                collection_name="lessons",
                distance_strategy=cfg.storage.vector.distance_strategy,
            )
            lesson_vector_store = build_vector_store(
                _lesson_vector_cfg, embedder, engine,
            )
            lesson_store = _LessonStore(
                engine=engine,
                vector_store=lesson_vector_store,
                distance_strategy=cfg.storage.vector.distance_strategy,
                similarity_threshold=framework_cfg.intake_similarity_threshold,
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
                    event_log=event_log,
                    lesson_store=lesson_store,
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
            # Generic app-MCP-server discovery (DECOUPLE-04 / D-07-02 /
            # D-07-03). Each module listed in
            # ``cfg.orchestrator.mcp_servers`` is imported and asked to
            # bind its config-derived state via the
            # ``register(mcp_app, cfg)`` contract. The framework no
            # longer hardcodes incident-vocabulary module paths or
            # setter names. ``mcp_app`` is ``None`` here — modules
            # expose their own per-module FastMCP instance composed by
            # the loader; the parameter exists for contract uniformity
            # and future composition needs.
            for module_path in cfg.orchestrator.mcp_servers:
                mod = importlib.import_module(module_path)
                reg = getattr(mod, "register", None)
                if reg is None:
                    raise RuntimeError(
                        f"orchestrator.mcp_servers entry {module_path!r} does "
                        f"not expose a `register(mcp_app, cfg)` callable"
                    )
                reg(None, cfg)
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

            registered = {e.name for e in registry.entries.values()}
            validate_skill_tool_references(
                {s.name: s.model_dump() for s in skills.values()},
                registered,
            )
            validate_skill_routes(
                {s.name: s.model_dump() for s in skills.values()},
            )
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

            try:
                removed = gc_orphaned_checkpoints(engine)
                if removed:
                    _log.info("checkpoint gc: removed %d orphaned threads", removed)
            except Exception:
                _log.exception("checkpoint gc failed (non-fatal)")
            graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                      registry=registry,
                                      checkpointer=checkpointer,
                                      framework_cfg=framework_cfg,
                                      event_log=event_log)
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
                    _default_timeout_capture = (
                        cfg.orchestrator.default_llm_request_timeout
                    )

                    def _factory():
                        return get_llm(
                            _llm_cfg_capture, _model_name, role="dedup",
                            default_llm_request_timeout=_default_timeout_capture,
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
            # ``intake_context`` was attached via ``object.__setattr__`` ~140
            # lines up; pyright doesn't see dynamic Pydantic attrs, so go
            # via getattr for the type-checker.
            if dedup_pipeline is not None:
                getattr(framework_cfg, "intake_context").dedup_pipeline = dedup_pipeline
            # No bespoke resume graph — resume runs through the main
            # graph via ``Command(resume=...)`` against the same
            # thread_id, with the checkpointer rehydrating paused state.
            # ``repo_state_cls: Type[BaseModel]`` matches the loose
            # bound on ``Orchestrator.StateT`` (also ``BaseModel``) at
            # the call site, but pyright sees the un-narrowed
            # ``StateT`` placeholder. Concrete narrowing happens via
            # the runtime resolver enforced earlier in this method.
            instance = cls(cfg, store, skills, registry, graph,
                           stack, framework_cfg=framework_cfg,
                           state_cls=repo_state_cls,  # pyright: ignore[reportArgumentType]
                           history=history,
                           event_log=event_log,
                           lesson_store=lesson_store,
                           checkpointer=checkpointer,
                           checkpointer_close=checkpointer_close,
                           dedup_pipeline=dedup_pipeline)
            # DECOUPLE-05 / D-08-01: stash the resolved schema class
            # so ``start_session`` can run ``model_validate`` on the
            # ``state_overrides=`` kwarg.
            instance._state_overrides_cls = state_overrides_cls
            return instance
        except BaseException:
            # Best-effort: close the checkpointer connection if it was
            # built before we hit the failure, so we don't leak FDs.
            try:
                await checkpointer_close()  # pyright: ignore[reportPossiblyUnboundVariable]
            except Exception:  # noqa: BLE001
                # The original BaseException is what the caller cares
                # about; this cleanup failure must not mask it. Log so
                # the FD-leak path stays observable.
                _log.warning(
                    "build: checkpointer_close failed during error rollback",
                    exc_info=True,
                )
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
                # Best-effort: the rest of aclose() (exit_stack drain)
                # must still run so MCP transports don't leak. Log so
                # checkpointer-close failures stay observable.
                _log.warning(
                    "aclose: checkpointer close failed",
                    exc_info=True,
                )
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

    def _finalize_session_status(self, session_id: str) -> str | None:
        """Transition a graph-completed session to a terminal status by
        INFERRING from tool-call history.

        Inference walks the configured terminal-tool rules in
        ``self.cfg.orchestrator.terminal_tools`` (D-06-01, D-06-08) and
        the latest executed tool call wins. When no rule fires, the
        session falls through to
        ``self.cfg.orchestrator.default_terminal_status`` — apps own
        this name (D-06-06).

        Per-rule ``extract_fields`` populate ``inc.extra_fields`` with
        whatever metadata the app declared (D-06-02; preserves the
        v1.0 ``team`` capture for the escalation flow). When the
        matched rule's status has ``kind="escalation"``, the
        ``team`` extract is mirrored to ``extra_fields["escalated_to"]``
        so existing UIs reading the v1.0 key continue to work
        (D-06-05 — kind-based dispatch).

        Sessions already in a terminal status are left untouched.
        """
        try:
            inc = self.store.load(session_id)
        except FileNotFoundError:
            return None
        if inc.status not in ("new", "in_progress"):
            return None

        # Phase 10 (FOC-03) defence-in-depth: hard rejection of envelope-less
        # turns happens at the agent runner; this hook only logs WARNING for
        # forensics on legacy on-disk sessions whose agent_runs predate the
        # envelope contract. Never raises.
        _assert_envelope_invariant_on_finalize(inc)

        decision = self._infer_terminal_decision(inc.tool_calls)
        # Capture from-status BEFORE any mutation so the M4 status_changed
        # event carries the correct transition. Both branches below mutate
        # inc.status.
        from_status = inc.status
        if decision is None:
            default = self.cfg.orchestrator.default_terminal_status
            if default is None:
                # App did not declare a default — leave the session
                # alone rather than blind-coerce. Production apps
                # always configure it (cross-validated at config-load
                # whenever ``statuses`` is populated).
                return None
            inc.status = default
            inc.extra_fields["needs_review_reason"] = (
                "graph completed without terminal tool call"
            )
            _emit_status_changed_event(
                orch=self, inc=inc,
                from_status=from_status, to_status=default,
                cause="default_terminal_status",
            )
            return self._save_or_yield(inc, default)

        new_status, extracted = decision
        inc.status = new_status
        for key, value in extracted.items():
            if value:
                inc.extra_fields[key] = value
        # v1.0 compatibility: mirror ``team`` into the historical
        # ``escalated_to`` extra-field key when the matched status is
        # an escalation (D-06-05 kind-based dispatch). Keeps generic-
        # framework code free of escalation vocabulary while preserving
        # the contract existing UIs read.
        status_def = self.cfg.orchestrator.statuses.get(new_status)
        if status_def is not None and status_def.kind == "escalation":
            team = extracted.get("team")
            if team:
                inc.extra_fields["escalated_to"] = team
        # M4: emit status_changed with cause=<matched terminal tool>.
        # The terminal-tool name from the matched rule is the most
        # specific cause label downstream consumers (UI, learner) need.
        cause_tool = _latest_terminal_tool_for_status(
            self.cfg.orchestrator.terminal_tools,
            inc.tool_calls, new_status,
        )
        _emit_status_changed_event(
            orch=self, inc=inc,
            from_status=from_status, to_status=new_status,
            cause=cause_tool or "terminal_tool_match",
        )
        return self._save_or_yield(inc, new_status)

    def _infer_terminal_decision(
        self, tool_calls,
    ) -> tuple[str, dict[str, str]] | None:
        """Walk executed tool_calls latest-first; return
        ``(new_status, extracted_fields_dict)`` for the first matching
        configured rule, or ``None`` if no rule fires.

        Replaces the v1.0 module-level ``_infer_terminal_decision``;
        instance-method shape (D-06-08) lets us read ``self.cfg``
        without constructor plumbing. Empty
        ``self.cfg.orchestrator.terminal_tools`` (the framework
        default) short-circuits to ``None`` so unconfigured apps
        behave as if no rule fires.
        """
        rules = self.cfg.orchestrator.terminal_tools
        if not rules:
            return None
        executed = [
            tc for tc in tool_calls
            if getattr(tc, "status", None) == "executed"
        ]
        for tc in reversed(executed):
            tool_name = tc.tool or ""
            for rule in rules:
                bare = rule.tool_name
                if not (tool_name == bare or tool_name.endswith(f":{bare}")):
                    continue
                # DECOUPLE-07 / D-08-03: optional argument-value
                # discriminator. When ``match_args`` is non-empty
                # the rule applies only if EVERY (key, value) pair
                # matches ``tc.args[key]`` exactly. Empty default
                # matches any args (v1.0 single-rule shape).
                if rule.match_args:
                    args = tc.args if isinstance(tc.args, dict) else {}
                    if not all(
                        args.get(k) == v
                        for k, v in rule.match_args.items()
                    ):
                        continue
                extracted = {
                    dest: self._extract_field(tc, lookup_keys)
                    for dest, lookup_keys in rule.extract_fields.items()
                }
                return rule.status, {
                    k: v for k, v in extracted.items() if v
                }
        return None

    def _extract_field(
        self, tc, lookup_keys: list[str],
    ) -> str | None:
        """Pull a field value from a ToolCall's args/result via
        ``args.X`` / ``result.X`` lookup hints. Returns the first
        non-falsy match, or ``None``. Generalised from v1.0
        ``_extract_team`` (D-06-02, D-06-08) — same lookup syntax,
        no longer pinned to the ``team`` field name.
        """
        args = tc.args if isinstance(tc.args, dict) else {}
        result = tc.result if isinstance(tc.result, dict) else {}
        for key in lookup_keys:
            scope, _, attr = key.partition(".")
            source = args if scope == "args" else result
            value = source.get(attr)
            if value:
                return value
        return None

    def _save_or_yield(self, inc, new_status: str) -> str | None:
        """Save with stale-version protection. Returns ``new_status`` on
        success or ``None`` if a concurrent finalize won the race.
        """
        try:
            self.store.save(inc)
            return new_status
        except StaleVersionError:
            return None

    @staticmethod
    def _is_graph_interrupt(exc: BaseException) -> bool:
        """Phase 11 (FOC-04 / D-11-04): identify a LangGraph HITL pause.

        ``GraphInterrupt`` is NOT an error -- it signals a checkpointed
        ``pending_approval`` state. Real exceptions still flow through
        the normal failure path. Helper kept on the orchestrator so
        callers don't each re-import langgraph internals.
        """
        return isinstance(exc, GraphInterrupt)

    @staticmethod
    def _extract_last_error(inc: "Session") -> Exception | None:
        """Reconstruct the last error from a Session in status='error'.

        The graph runner stores failures as an AgentRun with
        ``summary='agent failed: <repr>'`` (graph.py:_handle_agent_failure).
        We can't recover the original Exception type, so we return a
        synthetic representative whose CLASS matches a _PERMANENT_TYPES
        / _TRANSIENT_TYPES whitelist entry where possible -- that's all
        :func:`runtime.policy.should_retry` needs (it does isinstance
        checks).

        Mapping (first match wins per AgentRun.summary scan, newest
        first):

          - "EnvelopeMissingError" in body -> EnvelopeMissingError
          - "ValidationError"     in body -> pydantic.ValidationError
          - "TimeoutError" / "timed out"  -> TimeoutError
          - "OSError" / "ConnectionError" -> OSError
          - everything else               -> RuntimeError (falls
            through to permanent_error per fail-closed default in
            should_retry)
        """

        import pydantic as _pydantic
        for run in reversed(inc.agents_run):
            summary = (run.summary or "")
            if not summary.startswith("agent failed:"):
                continue
            body = summary.removeprefix("agent failed:").strip()
            if "EnvelopeMissingError" in body:
                return _EnvelopeMissingError(
                    agent=run.agent or "unknown",
                    field="confidence",
                    message=body,
                )
            if "ValidationError" in body or "validation error" in body:
                # Build a synthetic ValidationError; pydantic v2 supports
                # ValidationError.from_exception_data.
                try:
                    return _pydantic.ValidationError.from_exception_data(
                        title="reconstructed", line_errors=[],
                    )
                except Exception:  # pragma: no cover -- pydantic API drift
                    return RuntimeError(body)
            if ("TimeoutError" in body or "timed out" in body
                    or "asyncio.TimeoutError" in body):
                return TimeoutError(body)
            if "OSError" in body or "ConnectionError" in body:
                return OSError(body)
            return RuntimeError(body)
        return None

    @staticmethod
    def _extract_last_confidence(inc: "Session") -> float | None:
        """Return the last recorded turn-level confidence on the session,
        or None if no AgentRun carries one. should_retry treats None as
        'no signal yet' and skips the low-confidence gate.
        """
        for run in reversed(inc.agents_run):
            if run.confidence is not None:
                return run.confidence
        return None

    def preview_retry_decision(
        self, session_id: str,
    ) -> "RetryDecision":
        """Phase 12 (FOC-05 / D-12-04): return the framework's retry
        decision WITHOUT executing anything. The UI calls this to render
        the retry button label + disabled state.

        Pure: same inputs always yield identical RetryDecision. Loads
        the session from store; reads (retry_count, last_error,
        last_confidence) and consults the same policy
        ``runtime.policy.should_retry`` that ``_retry_session_locked``
        uses. No mutation, no thread-id bump, no lock acquired.

        For sessions whose status is not "error" (i.e. nothing to
        retry), returns ``RetryDecision(retry=False,
        reason="permanent_error")`` -- a defensive caller-friendly
        outcome that lets the UI render a "cannot auto-retry" state
        without inventing a new reason value.
        """
        try:
            inc = self.store.load(session_id)
        except FileNotFoundError:
            return RetryDecision(retry=False, reason="permanent_error")
        if inc.status != "error":
            return RetryDecision(retry=False, reason="permanent_error")
        retry_count = int(inc.extra_fields.get("retry_count", 0))
        last_error = self._extract_last_error(inc)
        last_confidence = self._extract_last_confidence(inc)
        return should_retry(
            retry_count=retry_count,
            error=last_error,
            confidence=last_confidence,
            cfg=self.cfg.orchestrator,
        )

    async def _finalize_session_status_async(
        self, session_id: str,
    ) -> str | None:
        """Lock-guarded async wrapper around ``_finalize_session_status``.

        All async call sites must use this one. The per-session lock
        prevents two concurrent flows from each observing
        pre-transition state and racing on the save. The second waiter
        loads after the first commits, sees terminal status, and the
        sync helper returns ``None`` (no transition).
        """
        async with self._locks.acquire(session_id):
            return self._finalize_session_status(session_id)

    def _thread_config(self, incident_id: str) -> dict:
        """Build the LangGraph ``config`` dict for a per-session thread.

        With a checkpointer attached, every ``ainvoke`` / ``astream_events``
        call must carry a ``configurable.thread_id`` so LangGraph can scope
        the durable state. The default thread id is the session id, but
        ``retry_session`` rebinds the session to a fresh thread id (so
        the graph runs from the entry rather than resuming a terminated
        checkpoint). The chosen thread id is persisted on the session
        in ``extra_fields["active_thread_id"]`` so subsequent resume
        calls land on the correct paused checkpoint.
        """
        try:
            inc = self.store.load(incident_id)
            thread_id = (inc.extra_fields or {}).get("active_thread_id") or incident_id
        except FileNotFoundError:
            thread_id = incident_id
        return {"configurable": {"thread_id": thread_id}}

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
        # DECOUPLE-05 / D-08-01: validate against the app-registered
        # pydantic schema if configured. ``None`` (D-08-02) skips
        # validation entirely so the legacy v1.0 free-form dict
        # surface still works for unconfigured apps. ``getattr`` with
        # a default keeps tests that build the orchestrator via
        # ``__new__`` (bypassing ``__init__``) working.
        state_overrides_cls = getattr(self, "_state_overrides_cls", None)
        if state_overrides_cls is not None and state_overrides is not None:
            state_overrides_cls.model_validate(state_overrides)
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
        new_status = await self._finalize_session_status_async(inc.id)
        if new_status:
            yield {"event": "status_auto_finalized", "incident_id": inc.id,
                   "status": new_status, "ts": _event_ts()}
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
            tool_name = self.cfg.orchestrator.escalate_action_tool_name
            default_team = self.cfg.orchestrator.escalate_action_default_team
            team = decision.get("team") or default_team
            allowed = list(self.framework_cfg.escalation_teams)
            # Only enforce roster membership when the framework is
            # actually configured with one — apps without a roster
            # accept any team string.
            if allowed and team is not None and team not in allowed:
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

            # Look up the rule for the configured escalation tool so
            # status assignment and extra-field key are driven by the
            # registry rather than hardcoded vocabulary.
            rule = None
            if tool_name is not None:
                for r in self.cfg.orchestrator.terminal_tools:
                    if r.tool_name == tool_name:
                        rule = r
                        break

            inc_loaded = self.store.load(incident_id)
            if tool_name is not None:
                # App registered a side-effect tool; invoke it and record
                # the tool-call so finalize-style introspection sees it.
                message = (
                    f"Session {incident_id} escalated by user"
                    + (f" — team {team}." if team else ".")
                    + " Confidence below threshold."
                )
                tool_args: dict = {"incident_id": incident_id, "message": message}
                if team is not None:
                    tool_args["team"] = team
                # Phase 9 (D-09-01): expose the live session to
                # _invoke_tool's injection branch via the implicit slot.
                # try/finally so a failed tool call doesn't leak the
                # reference into the next orchestrator-driven call.
                self._current_session_for_invoke = inc_loaded
                try:
                    tool_result = await self._invoke_tool(tool_name, tool_args)
                finally:
                    self._current_session_for_invoke = None
                inc_loaded.tool_calls.append(ToolCall(
                    agent="orchestrator",
                    tool=tool_name,
                    args=tool_args,
                    result=tool_result,
                    ts=_event_ts(),
                ))

            # Status assignment: prefer the rule's declared status; fall
            # back to default_terminal_status when no rule registered.
            if rule is not None:
                new_status = rule.status
            else:
                new_status = (
                    self.cfg.orchestrator.default_terminal_status
                    or inc_loaded.status
                )
            inc_loaded.status = new_status
            # Capture team via the rule's first extract-field destination
            # (typically ``team``). When no rule registered the field but
            # team is present, fall back to ``team`` for stability.
            if team is not None:
                if rule is not None and rule.extract_fields:
                    first_dest = next(iter(rule.extract_fields.keys()))
                    inc_loaded.extra_fields[first_dest] = team
                else:
                    inc_loaded.extra_fields["team"] = team
                # v1.0 compat: mirror to ``escalated_to`` when the new
                # status is kind=escalation, matching finalize semantics.
                status_def = self.cfg.orchestrator.statuses.get(new_status)
                if status_def is not None and status_def.kind == "escalation":
                    inc_loaded.extra_fields["escalated_to"] = team
            inc_loaded.pending_intervention = None
            self.store.save(inc_loaded)
            event_payload: dict = {
                "event": "resume_completed", "incident_id": incident_id,
                "status": new_status, "ts": _event_ts(),
            }
            if team is not None:
                event_payload["team"] = team
            yield event_payload
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

    async def retry_session(self, session_id: str) -> AsyncIterator[dict]:
        """Restart a failed/stopped session on a fresh LangGraph thread.

        Rejects (with retry_rejected event) if a retry is already in
        flight for this session id. The check is fast-fail BEFORE
        acquiring the lock so the rejecting caller is not blocked.
        """
        if session_id in self._retries_in_flight:
            _log.warning("retry_session rejected (fast-fail): %s already in flight",
                         session_id)
            yield {"event": "retry_rejected",
                   "incident_id": session_id,
                   "reason": "retry already in progress",
                   "ts": _event_ts()}
            return
        async with self._locks.acquire(session_id):
            # Re-check inside the lock to close the TOCTOU window
            # between the membership check above and the acquire:
            # task A could have completed its full retry-and-finally
            # discard between this caller's outer check and acquire,
            # but a third concurrent task could have entered and added
            # itself between A's discard and B's acquire.
            if session_id in self._retries_in_flight:
                _log.warning("retry_session rejected (post-acquire): %s",
                             session_id)
                yield {"event": "retry_rejected",
                       "incident_id": session_id,
                       "reason": "retry already in progress",
                       "ts": _event_ts()}
                return
            self._retries_in_flight.add(session_id)
            try:
                async for ev in self._retry_session_locked(session_id):
                    yield ev
            finally:
                self._retries_in_flight.discard(session_id)

    async def _retry_session_locked(self, session_id: str) -> AsyncIterator[dict]:
        """Re-run the graph for a session that failed mid-flight.

        Only sessions in ``status="error"`` are retryable — those are
        the ones a graph node terminated with a recorded
        ``agent failed: ...`` AgentRun (see
        :func:`runtime.graph._handle_agent_failure`). The retry uses a
        fresh LangGraph thread id so the compiled graph runs from the
        entry node rather than resuming the terminated checkpoint.

        Yields the same UI-event shape as ``stream_session`` plus
        ``retry_started`` / ``retry_rejected`` / ``retry_completed``
        envelopes so the UI can render a banner.
        """
        try:
            inc = self.store.load(session_id)
        except FileNotFoundError:
            yield {"event": "retry_rejected", "incident_id": session_id,
                   "reason": "session not found", "ts": _event_ts()}
            return
        if inc.status != "error":
            yield {"event": "retry_rejected", "incident_id": session_id,
                   "reason": f"not in error state (status={inc.status})",
                   "ts": _event_ts()}
            return
        # Phase 12 (FOC-05 / D-12-04): consult the framework's pure
        # retry policy BEFORE mutating session state. The decision is
        # derived from (retry_count, last_error, last_turn_confidence,
        # cfg) -- LLM intent is not consulted. On retry=False, emit
        # retry_rejected with the policy's reason and DO NOT bump the
        # retry_count or thread id (preserves the "not retryable"
        # state on disk for UI re-rendering and retry-budget audits).
        prior_retry_count = int(inc.extra_fields.get("retry_count", 0))
        last_error = self._extract_last_error(inc)
        last_confidence = self._extract_last_confidence(inc)
        decision = should_retry(
            retry_count=prior_retry_count,
            error=last_error,
            confidence=last_confidence,
            cfg=self.cfg.orchestrator,
        )
        if not decision.retry:
            _log.info(
                "retry_session policy-rejected: id=%s reason=%s",
                session_id, decision.reason,
            )
            yield {"event": "retry_rejected", "incident_id": session_id,
                   "reason": decision.reason, "ts": _event_ts()}
            return
        # Drop the failed AgentRun(s) so the timeline only retains
        # successful runs. Retry attempts then append fresh runs.
        inc.agents_run = [
            r for r in inc.agents_run
            if not (r.summary or "").startswith("agent failed:")
        ]
        # Bump retry counter for unique LangGraph thread id (the prior
        # thread's checkpoint sits at a terminal node and would
        # short-circuit a same-thread re-invocation).
        retry_count = int(inc.extra_fields.get("retry_count", 0)) + 1
        inc.extra_fields["retry_count"] = retry_count
        thread_id = f"{session_id}:retry-{retry_count}"
        # Pin the active thread id so any subsequent resume / approval
        # call uses the new checkpoint, not the original session-id
        # thread (which is at the terminated failure node).
        inc.extra_fields["active_thread_id"] = thread_id
        inc.status = "in_progress"
        self.store.save(inc)
        yield {"event": "retry_started", "incident_id": session_id,
               "retry_count": retry_count, "ts": _event_ts()}
        async for ev in self.graph.astream_events(
            GraphState(session=inc, next_route=None, last_agent=None, error=None),
            version="v2",
            config=self._thread_config(session_id),
        ):
            yield self._to_ui_event(ev, session_id)
        new_status = await self._finalize_session_status_async(session_id)
        if new_status:
            yield {"event": "status_auto_finalized", "incident_id": session_id,
                   "status": new_status, "ts": _event_ts()}
        yield {"event": "retry_completed", "incident_id": session_id,
               "ts": _event_ts()}

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
        except GraphInterrupt:
            # Phase 11 (FOC-04 / D-11-04): a resume that re-paused via
            # a fresh HITL gate. Don't restore the prior pending_intervention
            # block (the new pending_approval ToolCall row is the
            # canonical pause record now). Propagate so LangGraph's
            # checkpointer captures the new pause; the UI's
            # _render_pending_approvals_block surfaces the resume target.
            raise
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
        Used for orchestrator-driven tool calls (e.g. an app-registered
        escalation tool invoked from the awaiting_input gate) that aren't
        initiated by an LLM.

        Phase 9 (D-09-01): orchestrator-driven calls also flow through
        injection so the tool gets the canonical session-derived arg set
        even when the orchestrator only passed intent-args. The current
        session is read off ``self._current_session_for_invoke`` (set
        by callers via try/finally) so the public signature stays
        unchanged. When no session is reachable the injection step is
        a no-op — the existing escalation path keeps working unchanged.
        """
        entry = next(
            (e for e in self.registry.entries.values() if e.name == name),
            None,
        )
        if entry is None:
            raise KeyError(f"tool '{name}' not registered")
        session = getattr(self, "_current_session_for_invoke", None)
        cfg_inject = self.cfg.orchestrator.injected_args
        if session is not None and cfg_inject:

            args = inject_injected_args(
                args,
                session=session,
                injected_args_cfg=cfg_inject,
                tool_name=name,
                accepted_params=accepted_params_for_tool(entry.tool),
            )
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

_log = logging.getLogger("runtime.api")


# HTTP status -> structured error code. Used by the global exception
# handler to keep React's error UI from having to switch on every
# integer status code.
_STATUS_TO_CODE: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "unprocessable_entity",
    429: "rate_limited",
    500: "internal_error",
    501: "not_implemented",
    503: "service_unavailable",
}


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


class RetryDecisionPreview(BaseModel):
    """Response from ``GET /sessions/{sid}/retry/preview``."""
    retry: bool
    reason: str


class LessonResponse(BaseModel):
    """Response item for ``GET /sessions/{sid}/lessons``."""
    id: str
    source_session_id: str
    outcome_status: str
    outcome_summary: str
    confidence_final: float | None = None
    tools: list[str] = Field(default_factory=list)
    created_at: str


class EventEnvelope(BaseModel):
    """Single SSE/WS event payload. Wraps M1 :class:`SessionEvent`."""
    seq: int
    session_id: str
    kind: str
    payload: dict
    ts: str


class ErrorDetail(BaseModel):
    """Body of the structured JSON error envelope."""
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """Wire shape for every 4xx/5xx body the API returns. React calls
    can rely on a stable ``{"error": {"code", "message", "details"}}``
    shape regardless of which handler raised."""
    error: ErrorDetail


def _error_envelope(
    *,
    code: str,
    message: str,
    details: dict | None = None,
    status: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a structured JSON error response. ``headers`` preserves
    the original :class:`HTTPException.headers` (e.g. ``Retry-After``
    on a 429) so contract tests + clients see them through the
    global exception handler.
    """
    return JSONResponse(
        status_code=status,
        content=ErrorEnvelope(
            error=ErrorDetail(
                code=code, message=message, details=details or {},
            ),
        ).model_dump(),
        headers=headers,
    )


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
                # Best-effort: a misbehaving trigger transport must not
                # block ``svc.shutdown()`` below. Surface for observability.
                _log.warning(
                    "trigger registry stop_all failed during lifespan teardown",
                    exc_info=True,
                )
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

    # CORS: configure once with the AppConfig-supplied origins so the
    # React dev server (Vite at :5173, CRA/Next at :3000 by default) can
    # call every endpoint, SSE included. Production deployments lock
    # the origin list down via YAML — same shape, narrower allow-list.
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=cfg.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global handler: HTTPException → structured error envelope. React
    # clients can assume every 4xx/5xx body matches the
    # ``{"error":{"code","message","details"}}`` shape regardless of
    # which handler raised. Per-handler ``raise HTTPException(...,
    # detail=...)`` still works; the handler below normalises the body.
    @fastapi_app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _request: Request, exc: StarletteHTTPException,
    ):
        detail = exc.detail
        # Preserve per-exception headers (e.g. Retry-After on 429).
        passthrough_headers = getattr(exc, "headers", None) or None
        if isinstance(detail, dict) and "error" in detail:
            # Caller already structured it; pass through unchanged.
            return JSONResponse(
                status_code=exc.status_code,
                content=detail,
                headers=passthrough_headers,
            )
        code = _STATUS_TO_CODE.get(exc.status_code, "http_error")
        message = detail if isinstance(detail, str) else str(detail)
        return _error_envelope(
            code=code, message=message,
            status=exc.status_code,
            headers=passthrough_headers,
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
            # ``SessionCapExceeded`` and ``SessionBusy`` are matched by class
            # name to avoid a hard import dependency at module-load time.
            if e.__class__.__name__ in ("SessionCapExceeded", "SessionBusy"):
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
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
                # CodeQL py/stack-trace-exposure: never serialise raw
                # str(exc) into a client-bound stream — exception text
                # can carry stack-trace-equivalent details (file paths,
                # internal IDs). Use the exception class name + the
                # structured envelope shape the rest of the API uses.
                err = {
                    "error": {
                        "code": "resume_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

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
            if e.__class__.__name__ in ("SessionCapExceeded", "SessionBusy"):
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
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

            # Per D-20: wrap the ainvoke in the per-session lock so an
            # approval submission cannot interleave checkpoint writes
            # against any other turn on the same thread_id. Uses the
            # blocking ``acquire`` (not ``try_acquire``) — if a turn is
            # mid-flight the approval waits for it to release; the
            # service loop's overall request deadline bounds wait.
            # Future fail-fast switch is a one-line change to
            # try_acquire (the existing 429 handler at L484-489 already
            # routes ``SessionBusy`` to HTTP 429).
            async with orch._locks.acquire(session_id):
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
        try:
            await svc.submit_async(_resume())
        except Exception as e:  # noqa: BLE001
            if e.__class__.__name__ == "SessionBusy":
                raise HTTPException(
                    status_code=429,
                    detail=str(e),
                    headers={"Retry-After": "1"},
                ) from e
            raise
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

    # ==================================================================
    # T2: generic /sessions/* endpoints (React-ready, non-legacy).
    # ==================================================================

    @fastapi_app.get("/sessions/recent")
    async def recent_sessions(request: Request, limit: int = 20) -> list[dict]:
        """List recent sessions of ANY status — closed + active.

        Replaces the legacy session-list route which used a domain-
        flavoured noun. React's history panel calls this.
        """
        orch = request.app.state.orchestrator
        return orch.list_recent_sessions(limit=limit)

    @fastapi_app.get("/sessions/{session_id}")
    async def get_session_detail(session_id: str, request: Request) -> dict:
        """Full session detail. Generic equivalent of the legacy
        domain-flavoured detail route. 404 when the id is unknown."""
        orch = request.app.state.orchestrator
        try:
            return orch.get_session(session_id)
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            raise HTTPException(
                status_code=404, detail="session not found",
            ) from e

    @fastapi_app.post("/sessions/{session_id}/resume")
    async def resume_session_sse(
        session_id: str, req: ResumeRequest, request: Request,
    ) -> StreamingResponse:
        """Generic resume — SSE stream of orchestrator events.

        Mirrors the legacy domain-flavoured resume route but on the
        non-legacy URL the React client will use. Error frames map to
        the structured error envelope; raw exception text never reaches
        the wire.
        """
        orch = request.app.state.orchestrator
        decision: dict = {"action": req.decision}
        if req.user_input is not None:
            decision["input"] = req.user_input

        async def _events():
            try:
                async for ev in orch.resume_investigation(
                    session_id, decision,
                ):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                err = {
                    "error": {
                        "code": "resume_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    @fastapi_app.post("/sessions/{session_id}/retry")
    async def retry_session_sse(
        session_id: str, request: Request,
    ) -> StreamingResponse:
        """Retry a failed session. SSE stream of orchestrator events."""
        orch = request.app.state.orchestrator

        async def _events():
            try:
                async for ev in orch.retry_session(session_id):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                err = {
                    "error": {
                        "code": "retry_failed",
                        "message": exc.__class__.__name__,
                        "details": {},
                    }
                }
                yield f"data: {json.dumps(err, default=str)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    @fastapi_app.get(
        "/sessions/{session_id}/retry/preview",
        response_model=RetryDecisionPreview,
    )
    async def preview_retry(
        session_id: str, request: Request,
    ) -> RetryDecisionPreview:
        """Preview whether a retry would proceed without actually
        running it. Used by the UI to render the retry button's
        enabled/disabled state."""
        orch = request.app.state.orchestrator
        try:
            decision = orch.preview_retry_decision(session_id)
        except (FileNotFoundError, ValueError, KeyError, LookupError) as e:
            raise HTTPException(
                status_code=404, detail="session not found",
            ) from e
        return RetryDecisionPreview(
            retry=bool(decision.retry),
            reason=str(decision.reason),
        )

    @fastapi_app.get(
        "/sessions/{session_id}/lessons",
        response_model=list[LessonResponse],
    )
    async def list_session_lessons(
        session_id: str, request: Request,
    ) -> list[LessonResponse]:
        """List M5 SessionLessonRows whose source_session_id matches
        this session — i.e. the lessons this session contributed to
        the corpus. Empty list when the session never reached a
        terminal status."""
        orch = request.app.state.orchestrator
        lesson_store = getattr(orch, "lesson_store", None)
        if lesson_store is None:
            return []
        from sqlalchemy import select as _select
        from sqlalchemy.orm import Session as _SqlaSession


        with _SqlaSession(lesson_store.engine) as s:
            stmt = (
                _select(SessionLessonRow)
                .where(SessionLessonRow.source_session_id == session_id)
                .order_by(SessionLessonRow.created_at.desc())
            )
            rows = list(s.execute(stmt).scalars())
            out: list[LessonResponse] = []
            for row in rows:
                tools = [
                    t.get("tool") for t in row.tool_sequence
                    if t.get("tool")
                ]
                out.append(LessonResponse(
                    id=row.id,
                    source_session_id=row.source_session_id,
                    outcome_status=row.outcome_status,
                    outcome_summary=row.outcome_summary,
                    confidence_final=row.confidence_final,
                    tools=tools,
                    created_at=row.created_at.isoformat(),
                ))
            return out

    # ==================================================================
    # T3: SSE event stream + T4: WebSocket fallback.
    # ==================================================================

    @fastapi_app.get("/sessions/{session_id}/events")
    async def sse_events(
        session_id: str, request: Request, since: int = 0,
    ) -> StreamingResponse:
        """Server-Sent Events stream of the M1 EventLog for a session.

        Pushes every row whose ``seq > since`` as a JSON
        :class:`EventEnvelope` frame. Polls the EventLog at 250ms
        intervals — simple and reliable; an asyncio-Queue pub/sub layer
        can replace this when perf demands it.

        Disconnect-aware: each iteration checks
        ``request.is_disconnected()`` so the poll loop terminates
        promptly when the client closes the connection. Closes within
        one poll interval (~250ms) of disconnect.
        """
        import asyncio as _asyncio
        orch = request.app.state.orchestrator
        event_log = getattr(orch, "event_log", None)
        if event_log is None:
            raise HTTPException(
                status_code=503, detail="event_log not configured",
            )

        async def _stream():
            last_seq = since
            # Initial drain: replay any backlog past `since` —
            # unconditionally; the disconnect check belongs on the
            # tail-poll loop, not mid-backlog (otherwise an eager
            # disconnect-check ASGI client drops some events).
            for ev in event_log.iter_for(session_id, since=last_seq):
                envelope = EventEnvelope(
                    seq=ev.seq, session_id=ev.session_id,
                    kind=ev.kind, payload=ev.payload, ts=ev.ts,
                )
                last_seq = ev.seq
                yield f"data: {envelope.model_dump_json()}\n\n"
            # Tail: poll for new rows. Bounded by client-disconnect.
            try:
                while not await request.is_disconnected():
                    await _asyncio.sleep(0.25)
                    for ev in event_log.iter_for(session_id, since=last_seq):
                        envelope = EventEnvelope(
                            seq=ev.seq, session_id=ev.session_id,
                            kind=ev.kind, payload=ev.payload, ts=ev.ts,
                        )
                        last_seq = ev.seq
                        yield f"data: {envelope.model_dump_json()}\n\n"
            except _asyncio.CancelledError:
                return

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @fastapi_app.websocket("/ws/sessions/{session_id}/events")
    async def ws_events(websocket: WebSocket, session_id: str) -> None:
        """WebSocket fallback for the SSE event stream. Same payload
        shape (:class:`EventEnvelope`); clients that prefer WS over
        SSE call this instead. ``since`` is read from the
        ``?since=N`` query string."""
        import asyncio as _asyncio
        await websocket.accept()
        orch = websocket.app.state.orchestrator
        event_log = getattr(orch, "event_log", None)
        if event_log is None:
            await websocket.close(code=1011, reason="event_log not configured")
            return
        since_raw = websocket.query_params.get("since", "0")
        try:
            last_seq = int(since_raw)
        except ValueError:
            last_seq = 0
        try:
            # Initial backlog drain.
            for ev in event_log.iter_for(session_id, since=last_seq):
                last_seq = ev.seq
                await websocket.send_json(
                    EventEnvelope(
                        seq=ev.seq, session_id=ev.session_id,
                        kind=ev.kind, payload=ev.payload, ts=ev.ts,
                    ).model_dump()
                )
            # Tail loop.
            while True:
                await _asyncio.sleep(0.25)
                for ev in event_log.iter_for(session_id, since=last_seq):
                    last_seq = ev.seq
                    await websocket.send_json(
                        EventEnvelope(
                            seq=ev.seq, session_id=ev.session_id,
                            kind=ev.kind, payload=ev.payload, ts=ev.ts,
                        ).model_dump()
                    )
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 — close cleanly on any sink error
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass

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
