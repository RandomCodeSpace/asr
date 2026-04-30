from orchestrator.config import (
    AppConfig, LLMConfig, OllamaConfig, MCPServerConfig, MCPConfig,
)


def test_default_app_config_is_stub_provider():
    cfg = AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
        mcp=MCPConfig(),
    )
    assert cfg.llm.provider == "stub"
    assert cfg.environments == ["production", "staging", "dev", "local"]
    assert cfg.incidents.similarity_threshold == 0.85
    # Intervention defaults: 0.75 threshold + 3 oncall teams.
    assert cfg.intervention.confidence_threshold == 0.75
    assert cfg.intervention.escalation_teams == [
        "platform-oncall", "data-oncall", "security-oncall",
    ]


def test_ollama_provider_requires_ollama_config():
    cfg = AppConfig(
        llm=LLMConfig(
            provider="ollama",
            default_model="llama3.1:70b",
            ollama=OllamaConfig(base_url="https://ollama.com", api_key="key"),
        ),
        mcp=MCPConfig(),
    )
    assert cfg.llm.ollama.base_url == "https://ollama.com"


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
