"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""
from __future__ import annotations
import importlib
import warnings
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncIterator, Generic, Type, TypeVar
from datetime import datetime, timezone

from pydantic import BaseModel

from typing import TYPE_CHECKING

from runtime.checkpointer import make_checkpointer
from runtime.config import (
    AppConfig,
    FrameworkAppConfig,
    MetadataConfig,
    resolve_framework_app_config,
)

if TYPE_CHECKING:
    # Avoid a runtime circular import — ``runtime.triggers.base`` only
    # defines a dataclass, and the type appears in a method annotation.
    pass
    from runtime.triggers.base import TriggerInfo  # noqa: F401
from runtime.dedup import DedupConfig, DedupPipeline, DedupResult
from runtime.intake import IntakeContext
from runtime.llm import get_llm
from runtime.skill import load_all_skills, Skill
from runtime.mcp_loader import load_tools, ToolRegistry
from langgraph.types import Command

from runtime.graph import build_graph, GraphState
from runtime.state import Session, ToolCall
from runtime.state_resolver import resolve_state_class
from runtime.storage.engine import build_engine
from runtime.storage.embeddings import build_embedder
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore
from runtime.storage.vector import build_vector_store


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
            # Cross-cutting framework knobs come from the dotted-path
            # provider on RuntimeConfig.framework_app_config_path. When
            # unset the runtime falls back to a bare FrameworkAppConfig()
            # so unit tests that build an AppConfig without an app-side
            # provider keep working.
            framework_cfg = resolve_framework_app_config(
                cfg.runtime.framework_app_config_path,
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
            # The DedupConfig comes through a generic provider hook
            # (``RuntimeConfig.dedup_config_path``) — the runtime never
            # imports an app-specific config to discover dedup settings.
            dedup_pipeline: DedupPipeline | None = None
            dedup_cfg: DedupConfig | None = _resolve_dedup_config(
                cfg.runtime.dedup_config_path,
            )
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
