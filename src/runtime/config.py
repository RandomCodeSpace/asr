"""Config schemas for the orchestrator."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
import yaml


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


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: Paths = Field(default_factory=Paths)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    # Declarative trigger registry. Each entry is one transport-flavoured
    # ``TriggerConfig`` (api/webhook/schedule/plugin). Typed as
    # ``list[Any]`` because Pydantic v2's discriminated-union binding
    # pulls in the trigger module at import time, which would introduce
    # a circular import. The ``_coerce_triggers`` validator below
    # promotes raw dicts to the proper TriggerConfig variants.
    triggers: list[Any] = Field(default_factory=list)

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
