import pytest
from pydantic import ValidationError
from orchestrator.config import (
    AppConfig, LLMConfig, ProviderConfig, ModelConfig, EmbeddingConfig,
    MCPServerConfig, MCPConfig,
)


def test_default_app_config_is_stub_provider():
    cfg = AppConfig(llm=LLMConfig.stub(), mcp=MCPConfig())
    assert cfg.llm.providers["stub"].kind == "stub"
    assert cfg.llm.default == "stub_default"
    assert cfg.environments == ["production", "staging", "dev", "local"]
    assert cfg.incidents.similarity_threshold == pytest.approx(0.85)
    assert cfg.intervention.confidence_threshold == pytest.approx(0.75)
    assert cfg.intervention.escalation_teams == [
        "platform-oncall", "data-oncall", "security-oncall",
    ]


def test_named_models_resolve_provider_by_name():
    cfg = AppConfig(
        llm=LLMConfig(
            default="workhorse",
            providers={
                "ollama_cloud": ProviderConfig(
                    kind="ollama",
                    base_url="https://ollama.com",
                    api_key="key",
                ),
            },
            models={
                "workhorse": ModelConfig(
                    provider="ollama_cloud",
                    model="llama3.1:70b",
                    temperature=0.0,
                ),
            },
        ),
        mcp=MCPConfig(),
    )
    assert cfg.llm.providers["ollama_cloud"].base_url == "https://ollama.com"
    assert cfg.llm.models["workhorse"].model == "llama3.1:70b"


def test_unknown_default_model_rejected():
    with pytest.raises(ValidationError, match="llm.default"):
        LLMConfig(
            default="ghost",
            providers={"stub": ProviderConfig(kind="stub")},
            models={"workhorse": ModelConfig(provider="stub", model="x")},
        )


def test_unknown_provider_ref_rejected():
    with pytest.raises(ValidationError, match="not found in llm.providers"):
        LLMConfig(
            default="workhorse",
            providers={"stub": ProviderConfig(kind="stub")},
            models={
                "workhorse": ModelConfig(provider="phantom", model="x"),
            },
        )


def test_embedding_provider_must_be_known():
    with pytest.raises(ValidationError, match="llm.embedding.provider"):
        LLMConfig(
            default="workhorse",
            providers={"stub": ProviderConfig(kind="stub")},
            models={"workhorse": ModelConfig(provider="stub", model="x")},
            embedding=EmbeddingConfig(provider="phantom", model="emb-1"),
        )


def test_embedding_optional_and_resolves():
    cfg = LLMConfig(
        default="workhorse",
        providers={
            "ollama": ProviderConfig(kind="ollama", base_url="https://x"),
        },
        models={"workhorse": ModelConfig(provider="ollama", model="m")},
        embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text"),
    )
    assert cfg.embedding is not None
    assert cfg.embedding.model == "nomic-embed-text"


def test_mcp_server_in_process_requires_module():
    server = MCPServerConfig(
        name="local_inc",
        transport="in_process",
        category="incident_management",
        module="orchestrator.mcp_servers.incident",
    )
    assert server.module == "orchestrator.mcp_servers.incident"


def test_mcp_server_http_requires_url():
    server = MCPServerConfig(
        name="external",
        transport="http",
        category="ticketing",
        url="https://example.com/mcp",
        enabled=False,
    )
    assert server.url == "https://example.com/mcp"
    assert server.enabled is False


def test_orchestrator_default_entry_agent():
    cfg = AppConfig(llm=LLMConfig.stub(), mcp=MCPConfig())
    assert cfg.orchestrator.entry_agent == "intake"


def test_orchestrator_explicit_entry_agent():
    from orchestrator.config import OrchestratorConfig
    cfg = AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(entry_agent="diagnostic"),
    )
    assert cfg.orchestrator.entry_agent == "diagnostic"
