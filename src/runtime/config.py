"""Config schemas for the orchestrator."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from runtime.terminal_tools import StatusDef, TerminalToolRule
from runtime.errors import LLMConfigError   # NEW Phase 13 (D-13-05/06)


# Session-id prefix grammar. The framework mints session ids of the form
# ``{PREFIX}-YYYYMMDD-NNN`` (see ``runtime.state.Session.id_format``);
# the prefix is the only piece an app picks. Allow alphanumerics + hyphens,
# bound the length so the id stays scannable in logs and DB indexes, and
# refuse the empty string so the resulting id never starts with a stray ``-``.
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
            # Aliases for the example apps' per-agent model overrides
            # (e.g. incident_management's intake skill carries
            # ``model: gpt_oss_cheap`` for the v1.5-C / M8 proof point).
            # Tests + ``LLMConfig.stub()`` callers route them to the
            # same stub provider so the skill validator passes without
            # forcing every test to re-declare the registry.
            "gpt_oss": ModelConfig(provider="stub", model="stub-1"),
            "gpt_oss_cheap": ModelConfig(provider="stub", model="stub-1"),
            "workhorse": ModelConfig(provider="stub", model="stub-1"),
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
    """Relational store for session metadata. SQLite (dev) or Postgres (prod)."""
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
        from runtime.dedup import DedupConfig
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
        from runtime.triggers.config import (
            APITriggerConfig,
            PluginTriggerConfig,
            ScheduleTriggerConfig,
            WebhookTriggerConfig,
        )
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
