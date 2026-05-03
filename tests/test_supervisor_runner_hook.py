"""P9-9h — supervisor runner extension hook.

Covers the framework-level pieces:

* Back-compat: a ``kind: supervisor`` skill *without* a ``runner``
  still works exactly as before (rule / llm dispatch).
* The runner is invoked BEFORE the dispatch table and receives the
  live ``GraphState``; non-route patches are merged into the node's
  return value.
* When the runner returns ``{"next_route": "__end__"}`` the dispatch
  table is skipped entirely.
* A misconfigured dotted path (``module:missing_attr`` /
  ``"not.a.real.module:fn"``) fails at *skill load* time, not at
  graph-build / first-call time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from runtime.agents.supervisor import make_supervisor_node
from runtime.skill import DispatchRule, Skill, load_skill
from runtime.state import Session


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _session(**overrides) -> Session:
    base = dict(
        id="SES-1", status="open",
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
    )
    base.update(overrides)
    return Session(**base)


def _supervisor(*, runner: str | None = None, **overrides) -> Skill:
    """Build a rule-strategy supervisor skill, optionally with a runner."""
    base = dict(
        name="router", description="d", kind="supervisor",
        subordinates=["analyzer", "responder"],
        dispatch_strategy="rule",
        dispatch_rules=[
            DispatchRule(when="status == 'high'", target="responder"),
            DispatchRule(when="True", target="analyzer"),
        ],
    )
    if runner is not None:
        base["runner"] = runner
    base.update(overrides)
    return Skill(**base)


# ---------------------------------------------------------------------------
# Module-level callables that the dotted-path resolver can find.
# ---------------------------------------------------------------------------

# `_calls` is mutated by the runners below so tests can assert that
# the framework actually invoked them.
_calls: list[dict] = []


def runner_noop(state, *, app_cfg=None):  # noqa: ARG001
    _calls.append({"kind": "noop", "state_keys": sorted(state.keys())})
    return None


def runner_mutates_session(state, *, app_cfg=None):  # noqa: ARG001
    sess = state["session"]
    sess.findings = {**sess.findings, "router_note": "hydrated"}
    _calls.append({"kind": "mutate", "incident_id": sess.id})
    return {"session": sess}


def runner_short_circuits(state, *, app_cfg=None):  # noqa: ARG001
    sess = state["session"]
    sess.findings = {**sess.findings, "router_note": "duplicate"}
    _calls.append({"kind": "short_circuit", "incident_id": sess.id})
    return {"session": sess, "next_route": "__end__"}


def runner_short_circuits_friendly_alias(state, *, app_cfg=None):  # noqa: ARG001
    """Apps reach for ``"END"`` interchangeably with ``"__end__"`` — both
    must terminate the graph."""
    return {"session": state["session"], "next_route": "END"}


def runner_returns_extra_keys(state, *, app_cfg=None):  # noqa: ARG001
    """A non-terminal patch with extra keys is merged into the return."""
    return {"session": state["session"], "ad_hoc_flag": "yes"}


def runner_raises(state, *, app_cfg=None):  # noqa: ARG001
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# 1. Back-compat: skills without a runner still work as before
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_with_no_runner_uses_routing_table():
    """A supervisor skill without ``runner`` must behave exactly as before.

    ``status='high'`` should match the first dispatch rule and route
    to ``responder`` — the same expectation P6 tests already cover.
    """
    skill = _supervisor()  # runner is None
    assert skill.runner is None
    node = make_supervisor_node(skill=skill)

    state = {"session": _session(status="high"), "dispatch_depth": 0}
    out = await node(state)

    assert out["next_route"] == "responder"
    assert out["last_agent"] == "router"
    assert out["error"] is None


# ---------------------------------------------------------------------------
# 2. Runner is invoked before routing dispatch and may mutate state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_runner_called_before_routing():
    """The runner runs first; its session mutation is visible to the
    dispatch table and reflected in the node's return value."""
    _calls.clear()
    skill = _supervisor(
        runner="tests.test_supervisor_runner_hook:runner_mutates_session",
    )
    node = make_supervisor_node(skill=skill)

    sess = _session(status="open")
    state = {"session": sess}
    out = await node(state)

    # The runner was invoked exactly once.
    assert len(_calls) == 1
    assert _calls[0]["kind"] == "mutate"
    # Routing still happened (no override) → "True" rule wins → analyzer.
    assert out["next_route"] == "analyzer"
    # Session mutation survived back to the node return.
    assert out["session"].findings.get("router_note") == "hydrated"


@pytest.mark.asyncio
async def test_supervisor_runner_extra_keys_merged_into_state():
    skill = _supervisor(
        runner="tests.test_supervisor_runner_hook:runner_returns_extra_keys",
    )
    node = make_supervisor_node(skill=skill)
    out = await node({"session": _session(status="open")})
    assert out["next_route"] == "analyzer"
    # Non-route, non-session keys flow through to GraphState.
    assert out["ad_hoc_flag"] == "yes"


@pytest.mark.asyncio
async def test_supervisor_runner_returning_none_falls_through():
    _calls.clear()
    skill = _supervisor(runner="tests.test_supervisor_runner_hook:runner_noop")
    node = make_supervisor_node(skill=skill)
    out = await node({"session": _session(status="high")})
    assert out["next_route"] == "responder"  # rule table still wins
    assert _calls and _calls[0]["kind"] == "noop"


# ---------------------------------------------------------------------------
# 3. Runner short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_runner_short_circuits_via_next_route():
    """A runner returning ``next_route='__end__'`` must skip the
    dispatch table and terminate the graph."""
    _calls.clear()
    skill = _supervisor(
        runner="tests.test_supervisor_runner_hook:runner_short_circuits",
    )
    node = make_supervisor_node(skill=skill)

    sess = _session(status="high")  # would normally route to responder
    out = await node({"session": sess})

    assert out["next_route"] == "__end__"
    assert out["last_agent"] == "router"
    assert out["error"] is None
    # Runner ran once.
    assert _calls and _calls[0]["kind"] == "short_circuit"
    # Mutation is preserved on the returned session.
    assert out["session"].findings.get("router_note") == "duplicate"


@pytest.mark.asyncio
async def test_supervisor_runner_accepts_friendly_end_alias():
    skill = _supervisor(
        runner="tests.test_supervisor_runner_hook:runner_short_circuits_friendly_alias",
    )
    node = make_supervisor_node(skill=skill)
    out = await node({"session": _session(status="high")})
    assert out["next_route"] == "__end__"


@pytest.mark.asyncio
async def test_supervisor_runner_exception_routes_to_end_with_error():
    skill = _supervisor(
        runner="tests.test_supervisor_runner_hook:runner_raises",
    )
    node = make_supervisor_node(skill=skill)
    out = await node({"session": _session(status="high")})
    assert out["next_route"] == "__end__"
    assert "boom" in (out["error"] or "")


# ---------------------------------------------------------------------------
# 4. Dotted-path resolution at skill load
# ---------------------------------------------------------------------------


def test_runner_dotted_path_resolution_at_skill_load_bad_module(tmp_path: Path) -> None:
    """A ``runner`` pointing at a module that doesn't exist must raise
    at YAML load (skill validation), not at first call."""
    d = tmp_path / "router"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d",
        "kind": "supervisor",
        "subordinates": ["a"],
        "dispatch_strategy": "rule",
        "dispatch_rules": [{"when": "True", "target": "a"}],
        "runner": "no.such.module:fn",
    }))
    with pytest.raises(ValidationError) as exc:
        load_skill(d)
    msg = str(exc.value)
    assert "runner" in msg
    assert "no.such.module" in msg


def test_runner_dotted_path_resolution_at_skill_load_bad_attr(tmp_path: Path) -> None:
    """A real module + missing attribute also fails at load."""
    d = tmp_path / "router"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({
        "description": "d",
        "kind": "supervisor",
        "subordinates": ["a"],
        "dispatch_strategy": "rule",
        "dispatch_rules": [{"when": "True", "target": "a"}],
        "runner": "tests.test_supervisor_runner_hook:does_not_exist",
    }))
    with pytest.raises(ValidationError) as exc:
        load_skill(d)
    msg = str(exc.value)
    assert "does_not_exist" in msg


def test_runner_dotted_path_resolution_rejects_malformed() -> None:
    """A path missing the attribute component fails at construction."""
    with pytest.raises(ValidationError) as exc:
        Skill(
            name="x", description="d", kind="supervisor",
            subordinates=["a"], dispatch_strategy="rule",
            dispatch_rules=[DispatchRule(when="True", target="a")],
            runner="just_a_token",  # no ":" and no "."
        )
    assert "runner" in str(exc.value)


def test_runner_dotted_path_rejects_non_callable() -> None:
    """A dotted path resolving to a non-callable (e.g. a constant) fails."""
    with pytest.raises(ValidationError) as exc:
        Skill(
            name="x", description="d", kind="supervisor",
            subordinates=["a"], dispatch_strategy="rule",
            dispatch_rules=[DispatchRule(when="True", target="a")],
            runner="tests.test_supervisor_runner_hook:_calls",  # a list, not callable
        )
    assert "non-callable" in str(exc.value)


def test_runner_field_rejected_for_responsive_kind() -> None:
    """``runner`` is supervisor-only; a responsive skill setting it must fail."""
    with pytest.raises(ValidationError) as exc:
        Skill(
            name="x", description="d", kind="responsive",
            system_prompt="hi",
            runner="tests.test_supervisor_runner_hook:runner_noop",
        )
    assert "runner" in str(exc.value)
