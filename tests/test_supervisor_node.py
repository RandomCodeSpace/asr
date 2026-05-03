"""Tests for the supervisor agent kind.

Covers:

* Pure routing — no AgentRun row written.
* dispatch_strategy=rule picks the first matching rule and stamps it
  in the audit log entry.
* dispatch_strategy=llm with a stub that names a subordinate routes
  to that subordinate.
* Recursion bound: a supervisor at depth >= max_dispatch_depth aborts
  cleanly with __end__ + an error message.
"""
from __future__ import annotations

import json
import logging

import pytest

from runtime.agents.supervisor import (
    log_supervisor_dispatch,
    make_supervisor_node,
)
from runtime.skill import DispatchRule, Skill
from runtime.state import Session


def _session(**overrides) -> Session:
    base = dict(
        id="SES-1", status="open",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
    )
    base.update(overrides)
    return Session(**base)


def _supervisor(*, strategy: str = "rule", **overrides) -> Skill:
    base = dict(
        name="router", description="d", kind="supervisor",
        subordinates=["analyzer", "responder"],
    )
    base.update(overrides)
    if strategy == "rule":
        base.setdefault("dispatch_strategy", "rule")
        base.setdefault(
            "dispatch_rules",
            [
                DispatchRule(when="status == 'high'", target="responder"),
                DispatchRule(when="True", target="analyzer"),
            ],
        )
    else:
        base.setdefault("dispatch_strategy", "llm")
        base.setdefault("dispatch_prompt", "choose one")
    return Skill(**base)


@pytest.mark.asyncio
async def test_rule_strategy_first_match_wins():
    skill = _supervisor()
    node = make_supervisor_node(skill=skill)
    sess = _session(status="high")
    state = {"session": sess, "dispatch_depth": 0}
    out = await node(state)
    assert out["next_route"] == "responder"
    assert out["last_agent"] == "router"
    assert out["dispatch_depth"] == 1
    assert out["error"] is None


@pytest.mark.asyncio
async def test_rule_strategy_default_falls_through():
    skill = _supervisor()
    node = make_supervisor_node(skill=skill)
    sess = _session(status="open")
    state = {"session": sess}
    out = await node(state)
    assert out["next_route"] == "analyzer"


@pytest.mark.asyncio
async def test_supervisor_writes_no_agent_run_row():
    """The supervisor must not append to ``Session.agents_run``."""
    skill = _supervisor()
    node = make_supervisor_node(skill=skill)
    sess = _session()
    state = {"session": sess}
    out = await node(state)
    # The supervisor returns the same session reference; nothing was
    # appended to agents_run by the node itself.
    assert out["session"].agents_run == []


@pytest.mark.asyncio
async def test_recursion_bound_aborts_cleanly():
    skill = _supervisor(max_dispatch_depth=3)
    node = make_supervisor_node(skill=skill)
    sess = _session()
    state = {"session": sess, "dispatch_depth": 3}  # depth -> 4 on entry
    out = await node(state)
    assert out["next_route"] == "__end__"
    assert "max_dispatch_depth" in (out["error"] or "")


@pytest.mark.asyncio
async def test_audit_log_entry_carries_matched_rule(caplog):
    skill = _supervisor()
    node = make_supervisor_node(skill=skill)
    sess = _session(status="high")
    with caplog.at_level(logging.INFO, logger="runtime.agents.supervisor"):
        await node({"session": sess})
    # Find the structured supervisor_dispatch line.
    record = next(
        r for r in caplog.records
        if r.message.startswith("supervisor_dispatch")
    )
    payload = json.loads(record.message[len("supervisor_dispatch "):])
    assert payload["event"] == "supervisor_dispatch"
    assert payload["supervisor"] == "router"
    assert payload["targets"] == ["responder"]
    assert payload["strategy"] == "rule"
    assert payload["rule_matched"] == "status == 'high'"
    assert payload["depth"] == 1


def test_log_helper_emits_one_line(caplog):
    sess = _session()
    with caplog.at_level(logging.INFO, logger="runtime.agents.supervisor"):
        log_supervisor_dispatch(
            session=sess, supervisor="router", strategy="llm",
            depth=2, targets=["a", "b"], rule_matched=None,
            payload_size=42,
        )
    matching = [r for r in caplog.records if "supervisor_dispatch" in r.message]
    assert len(matching) == 1
    payload = json.loads(matching[0].message[len("supervisor_dispatch "):])
    assert payload["depth"] == 2
    assert payload["targets"] == ["a", "b"]
    assert payload["rule_matched"] is None
    assert payload["dispatch_payload_size"] == 42


@pytest.mark.asyncio
async def test_llm_strategy_picks_named_subordinate():
    """A stub LLM that returns the name of a subordinate routes to that one."""

    class StubLLM:
        def invoke(self, msgs):
            class Result:
                content = "responder"
            return Result()

    skill = _supervisor(strategy="llm")
    node = make_supervisor_node(skill=skill, llm=StubLLM())
    sess = _session()
    out = await node({"session": sess})
    assert out["next_route"] == "responder"


@pytest.mark.asyncio
async def test_llm_strategy_unparseable_falls_back_to_first():
    class StubLLM:
        def invoke(self, msgs):
            class Result:
                content = "no idea"
            return Result()

    skill = _supervisor(strategy="llm")
    node = make_supervisor_node(skill=skill, llm=StubLLM())
    out = await node({"session": _session()})
    assert out["next_route"] == "analyzer"


@pytest.mark.asyncio
async def test_make_supervisor_node_rejects_wrong_kind():
    s = Skill(name="r", description="d", kind="responsive", system_prompt="hi")
    with pytest.raises(ValueError, match="non-supervisor"):
        make_supervisor_node(skill=s)
