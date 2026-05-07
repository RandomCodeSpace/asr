import pytest
from pydantic import TypeAdapter

from examples.incident_management.mcp_servers.observability import build_environment_validator


def test_environment_validator_accepts_configured():
    Validator = build_environment_validator(["production", "staging", "dev"])
    ta = TypeAdapter(Validator)
    assert ta.validate_python("production") == "production"


def test_environment_validator_rejects_unknown():
    Validator = build_environment_validator(["production", "staging"])
    ta = TypeAdapter(Validator)
    with pytest.raises(Exception):
        ta.validate_python("prod")  # typo close to "production"


def test_environment_validator_lowercases_for_match():
    Validator = build_environment_validator(["production"])
    ta = TypeAdapter(Validator)
    assert ta.validate_python("PRODUCTION") == "production"
