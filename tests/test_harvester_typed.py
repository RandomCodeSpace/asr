"""When the agent calls a typed terminal tool (mark_resolved, mark_escalated,
submit_hypothesis), the harvester reads confidence/rationale from the flat
tc_args and implies signal=success.

This is the post-Task-3.5 contract: confidence is no longer carried inside
update_incident.patch — it's a required arg on the typed terminal tools, and
the harvester picks it up directly."""
from langchain_core.messages import AIMessage

from runtime.graph import _harvest_tool_calls_and_patches
from runtime.state import Session


def _make_inc(sid: str = "INC-1") -> Session:
    return Session(
        id=sid, status="new",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        extra_fields={},
    )


def test_harvester_reads_confidence_from_submit_hypothesis_return():
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "1", "name": "submit_hypothesis",
                "args": {
                    "incident_id": "INC-1",
                    "hypotheses": "h",
                    "confidence": 0.85,
                    "confidence_rationale": "r",
                },
            }],
        ),
    ]
    conf, rationale, signal = _harvest_tool_calls_and_patches(
        messages, "deep_investigator", inc, ts="2026-01-01T00:00:00Z",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    assert conf == 0.85
    assert rationale == "r"
    assert signal == "success"


def test_harvester_reads_confidence_from_mark_resolved():
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "1", "name": "mark_resolved",
                "args": {
                    "incident_id": "INC-1",
                    "resolution_summary": "done",
                    "confidence": 0.95,
                    "confidence_rationale": "verified",
                },
            }],
        ),
    ]
    conf, rationale, signal = _harvest_tool_calls_and_patches(
        messages, "resolution", inc, ts="t",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    assert conf == 0.95
    assert rationale == "verified"
    assert signal == "success"


def test_harvester_reads_confidence_from_mark_escalated():
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "1", "name": "mark_escalated",
                "args": {
                    "incident_id": "INC-1",
                    "team": "platform-oncall",
                    "reason": "rejected",
                    "confidence": 0.4,
                    "confidence_rationale": "weak",
                },
            }],
        ),
    ]
    conf, rationale, signal = _harvest_tool_calls_and_patches(
        messages, "resolution", inc, ts="t",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    assert conf == 0.4
    assert rationale == "weak"
    assert signal == "success"


def test_harvester_handles_prefixed_typed_tool_name():
    """MCP tool names are prefixed (`local_inc:mark_resolved`); the
    harvester strips the prefix to detect the typed tool."""
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "1", "name": "local_inc:mark_resolved",
                "args": {
                    "incident_id": "INC-1",
                    "resolution_summary": "done",
                    "confidence": 0.9,
                    "confidence_rationale": "r",
                },
            }],
        ),
    ]
    conf, _, signal = _harvest_tool_calls_and_patches(
        messages, "resolution", inc, ts="t",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    assert conf == 0.9
    assert signal == "success"


def test_harvester_still_reads_signal_from_update_incident_patch():
    """Non-terminal agents (triage, intake) emit signal via
    update_incident.patch.signal — that path must keep working."""
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "1", "name": "update_incident",
                "args": {
                    "incident_id": "INC-1",
                    "patch": {"signal": "success", "category": "latency"},
                },
            }],
        ),
    ]
    _, _, signal = _harvest_tool_calls_and_patches(
        messages, "triage", inc, ts="t",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    assert signal == "success"


def test_harvester_typed_tool_with_no_args_returns_none():
    """If the typed-tool args are missing (malformed message), don't crash."""
    inc = _make_inc()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "mark_resolved", "args": {}}],
        ),
    ]
    conf, _, signal = _harvest_tool_calls_and_patches(
        messages, "resolution", inc, ts="t",
        valid_signals=frozenset({"success", "failed", "default"}),
    )
    # Confidence missing → None preserved; signal=success still implied
    # because the call was attempted.
    assert conf is None
    assert signal == "success"
