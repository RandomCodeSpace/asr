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
