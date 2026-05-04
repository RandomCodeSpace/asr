import pytest
from runtime.skill_validator import (
    SkillValidationError, validate_skill_tool_references,
)


def test_validator_passes_when_all_tools_exist():
    skills = {"intake": {"tools": {"local": ["lookup_similar_incidents", "create_incident"]}}}
    registered_tools = {"local_inc:lookup_similar_incidents", "local_inc:create_incident"}
    validate_skill_tool_references(skills, registered_tools)  # no raise


def test_validator_raises_on_typo():
    skills = {"intake": {"tools": {"local": ["lookup_similar_incidnets"]}}}  # typo
    registered_tools = {"local_inc:lookup_similar_incidents"}
    with pytest.raises(SkillValidationError, match="lookup_similar_incidnets"):
        validate_skill_tool_references(skills, registered_tools)


def test_validator_raises_on_default_route_missing():
    from runtime.skill_validator import validate_skill_routes
    skills = {
        "intake": {
            "routes": [{"when": "success", "next": "triage"}]  # missing default
        }
    }
    with pytest.raises(SkillValidationError, match="when: default"):
        validate_skill_routes(skills)
