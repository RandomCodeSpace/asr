"""Public Orchestrator class — the API consumed by the UI and (future) FastAPI."""
from __future__ import annotations
import importlib
import logging
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
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from runtime.graph import build_graph, GraphState
from runtime.policy import RetryDecision, should_retry
from runtime.state import Session, ToolCall
from runtime.state_resolver import resolve_state_class
from runtime.storage.engine import build_engine
from runtime.storage.embeddings import build_embedder
from runtime.storage.history_store import HistoryStore
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore, StaleVersionError
from runtime.storage.vector import build_vector_store
from runtime.locks import SessionLockRegistry

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
            from runtime.skill_validator import (
                validate_skill_routes,
                validate_skill_tool_references,
            )
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
            from runtime.storage.checkpoint_gc import gc_orphaned_checkpoints
            try:
                removed = gc_orphaned_checkpoints(engine)
                if removed:
                    _log.info("checkpoint gc: removed %d orphaned threads", removed)
            except Exception:
                _log.exception("checkpoint gc failed (non-fatal)")
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
            instance = cls(cfg, store, skills, registry, graph,
                           stack, framework_cfg=framework_cfg,
                           state_cls=repo_state_cls,
                           history=history,
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
        from runtime.agents.turn_output import (
            EnvelopeMissingError as _EnvelopeMissingError,
        )
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
            from runtime.tools.arg_injection import inject_injected_args
            args = inject_injected_args(
                args,
                session=session,
                injected_args_cfg=cfg_inject,
                tool_name=name,
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
