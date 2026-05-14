import pytest
from runtime.graph import route_from_skill, AgentRunRecorder


def test_route_from_skill_matches_known_route():
    from runtime.skill import Skill, RouteRule
    s = Skill(
        name="x", description="d",
        routes=[RouteRule(when="matched", next="resolution"),
                RouteRule(when="default", next="triage")],
        system_prompt="",
    )
    assert route_from_skill(s, "matched") == "resolution"
    assert route_from_skill(s, "default") == "triage"


def test_route_from_skill_unknown_route_falls_back_to_default():
    from runtime.skill import Skill, RouteRule
    s = Skill(name="x", description="d",
              routes=[RouteRule(when="default", next="triage")], system_prompt="")
    assert route_from_skill(s, "unknown_signal") == "triage"


def test_route_from_skill_no_routes_raises():
    from runtime.skill import Skill
    s = Skill(name="x", description="d", routes=[], system_prompt="")
    with pytest.raises(ValueError, match="no routes"):
        route_from_skill(s, "default")


def test_recorder_appends_agent_run_and_tool_calls():
    from runtime.state import Session
    inc = Session(
        id="INC-1", status="new", created_at="t", updated_at="t",
        extra_fields={
            "query": "q",
            "environment": "dev",
            "reporter": {"id": "u", "team": "t"},
        },
    )
    rec = AgentRunRecorder(agent="intake", session=inc)
    rec.start()
    rec.record_tool_call("get_user_context", {"user_id": "u"}, {"team": "platform"})
    rec.finish(summary="created INC")
    assert len(inc.agents_run) == 1
    assert inc.agents_run[0].agent == "intake"
    assert len(inc.tool_calls) == 1
    assert inc.tool_calls[0].tool == "get_user_context"
