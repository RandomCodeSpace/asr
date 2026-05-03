from pathlib import Path
import pytest
from orchestrator.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_config.yaml"


def test_loads_yaml_and_resolves_env_vars(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-ollama")
    monkeypatch.setenv("EXTERNAL_MCP_URL", "https://x.example/mcp")
    monkeypatch.setenv("EXT_TOKEN", "ext-tok")
    cfg = load_config(FIXTURE)
    assert cfg.llm.default == "workhorse"
    assert cfg.llm.providers["ollama_cloud"].kind == "ollama"
    assert cfg.llm.providers["ollama_cloud"].api_key == "secret-ollama"
    assert cfg.llm.models["workhorse"].model == "llama3.1:70b"
    assert cfg.mcp.servers[1].url == "https://x.example/mcp"
    assert cfg.mcp.servers[1].headers["Authorization"] == "Bearer ext-tok"
    # incidents.similarity_threshold moved to IncidentAppConfig (P1-E); the
    # YAML's `incidents:` block is now a benign extra ignored by AppConfig
    # and is exercised by tests/test_incident_app_config.py instead.


def test_unset_env_var_raises(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("EXTERNAL_MCP_URL", raising=False)
    monkeypatch.delenv("EXT_TOKEN", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    with pytest.raises(KeyError, match="OLLAMA_API_KEY"):
        load_config(FIXTURE)
