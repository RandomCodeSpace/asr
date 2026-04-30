"""Config schemas for the orchestrator."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


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


class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    incidents: IncidentConfig = Field(default_factory=IncidentConfig)
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"]
    )
    paths: Paths = Field(default_factory=Paths)


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
    raw = yaml.safe_load(Path(path).read_text())
    resolved = _interpolate(raw)
    return AppConfig(**resolved)
