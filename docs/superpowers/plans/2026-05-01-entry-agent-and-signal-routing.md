# Entry-Agent + Signal-Driven Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the orchestrator's hard dependence on hardcoded agent names. Make the entry agent configurable, replace the hardcoded `deep_investigator → resolution` gate-injection rule with a per-route `gate:` field, introduce a `{success, failed, needs_input, default}` signal vocabulary that agents emit via `update_incident`, and drop the no-op `_DECIDERS` map in favor of a single generic decider.

**Architecture:**
1. Add `orchestrator.entry_agent` (default `"intake"`) to top-level `AppConfig`. `build_graph` reads it for `set_entry_point`.
2. Add `gate: str | None` to `RouteRule`. `build_graph` collects `gated_edges = {(from, to): gate_type}` from all skill routes; the router intercepts those edges and inserts the gate node. Resume-graph entry is derived as the upstream end of the (single) gated edge.
3. Each agent's prompt — extended via `_common/output.md` — instructs the LLM to include `signal: <success|failed|needs_input>` in its final `update_incident` patch. `make_agent_node` harvests it the same way it currently harvests `confidence`, stamps it on `AgentRun.signal`, and the generic decider returns `inc.agents_run[-1].signal or "default"`.
4. Routes match against signals. Unmatched signals fall through to `when: default`.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph, PyYAML, pytest + pytest-asyncio.

---

## File Structure

**Modified:**
- `src/orchestrator/config.py` — new `OrchestratorConfig` model, `AppConfig.orchestrator` field
- `src/orchestrator/skill.py` — `RouteRule.gate` field
- `src/orchestrator/incident.py` — `AgentRun.signal` field
- `src/orchestrator/graph.py` — drop `_DECIDERS` + `_decide_*`, add `_coerce_signal` and `_decide_from_signal`, harvest signal in `make_agent_node`, replace `set_entry_point("intake")` with config lookup, replace hardcoded gate-injection rule with `gated_edges` map derived from routes; same for `build_resume_graph`
- `config/config.yaml` — add `orchestrator: entry_agent: intake`
- `config/skills/intake/config.yaml` — routes use `success`/`failed`/`default`
- `config/skills/triage/config.yaml` — routes use `success`/`failed`/`default`
- `config/skills/deep_investigator/config.yaml` — routes use `success`/`failed`/`default`, `gate: confidence` on the success edge
- `config/skills/resolution/config.yaml` — routes use `success`/`failed`/`default`
- `config/skills/_common/output.md` — append signal-emission instruction
- `dist/app.py` — regenerated via `scripts/build_single_file.py`

**Test files modified:**
- `tests/test_config.py` — entry_agent default + explicit
- `tests/test_skill.py` — `RouteRule.gate` field
- `tests/test_agent_node.py` — signal harvesting (success / failed / unknown / bool)
- `tests/test_build_graph.py` — entry agent honored, `gated_edges` insertion
- `tests/test_resume.py` — resume entry derived from gated edges (verify still passes)

No new files created.

---

## Task 1: Config schema — `OrchestratorConfig.entry_agent` and `RouteRule.gate`

**Files:**
- Modify: `src/orchestrator/config.py`
- Modify: `src/orchestrator/skill.py`
- Test: `tests/test_config.py`, `tests/test_skill.py`

- [ ] **Step 1: Write the failing test for `RouteRule.gate`**

Append to `tests/test_skill.py`:

```python
def test_route_rule_gate_defaults_to_none():
    from orchestrator.skill import RouteRule
    r = RouteRule(when="default", next="triage")
    assert r.gate is None


def test_route_rule_gate_explicit_value():
    from orchestrator.skill import RouteRule
    r = RouteRule(when="success", next="resolution", gate="confidence")
    assert r.gate == "confidence"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skill.py::test_route_rule_gate_defaults_to_none tests/test_skill.py::test_route_rule_gate_explicit_value -v`
Expected: FAIL — `RouteRule` has no `gate` attribute.

- [ ] **Step 3: Add `gate` field to `RouteRule`**

In `src/orchestrator/skill.py`, replace the `RouteRule` class (currently lines ~54–56):

```python
class RouteRule(BaseModel):
    when: str
    next: str
    gate: str | None = None
```

- [ ] **Step 4: Run RouteRule tests to verify they pass**

Run: `pytest tests/test_skill.py -v`
Expected: PASS — all skill tests including the two new ones.

- [ ] **Step 5: Write the failing tests for `OrchestratorConfig`**

Append to `tests/test_config.py`:

```python
def test_orchestrator_default_entry_agent():
    from orchestrator.config import AppConfig, LLMConfig, MCPConfig
    cfg = AppConfig(llm=LLMConfig(provider="stub", default_model="stub-1"),
                    mcp=MCPConfig())
    assert cfg.orchestrator.entry_agent == "intake"


def test_orchestrator_explicit_entry_agent():
    from orchestrator.config import (
        AppConfig, LLMConfig, MCPConfig, OrchestratorConfig,
    )
    cfg = AppConfig(
        llm=LLMConfig(provider="stub", default_model="stub-1"),
        mcp=MCPConfig(),
        orchestrator=OrchestratorConfig(entry_agent="diagnostic"),
    )
    assert cfg.orchestrator.entry_agent == "diagnostic"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_orchestrator_default_entry_agent tests/test_config.py::test_orchestrator_explicit_entry_agent -v`
Expected: FAIL — `OrchestratorConfig` does not exist.

- [ ] **Step 7: Add `OrchestratorConfig` and register on `AppConfig`**

In `src/orchestrator/config.py`, add a new model just above `AppConfig` (after `InterventionConfig`):

```python
class OrchestratorConfig(BaseModel):
    entry_agent: str = "intake"
```

Then add a field to `AppConfig`:

```python
class AppConfig(BaseModel):
    llm: LLMConfig
    mcp: MCPConfig
    incidents: IncidentConfig = Field(default_factory=IncidentConfig)
    environments: list[str] = Field(
        default_factory=lambda: ["production", "staging", "dev", "local"]
    )
    paths: Paths = Field(default_factory=Paths)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_skill.py -v`
Expected: PASS — all config + skill tests.

- [ ] **Step 9: Commit**

```bash
git add src/orchestrator/config.py src/orchestrator/skill.py tests/test_config.py tests/test_skill.py
git commit -m "feat(config): add OrchestratorConfig.entry_agent and RouteRule.gate"
```

---

## Task 2: Domain model — `AgentRun.signal`

**Files:**
- Modify: `src/orchestrator/incident.py`
- Test: `tests/test_incident_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incident_model.py`:

```python
def test_agent_run_signal_defaults_to_none():
    from orchestrator.incident import AgentRun
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1", summary="ok")
    assert run.signal is None


def test_agent_run_signal_explicit():
    from orchestrator.incident import AgentRun
    run = AgentRun(agent="intake", started_at="t0", ended_at="t1",
                   summary="ok", signal="success")
    assert run.signal == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_incident_model.py::test_agent_run_signal_defaults_to_none tests/test_incident_model.py::test_agent_run_signal_explicit -v`
Expected: FAIL — `AgentRun` has no `signal` attribute.

- [ ] **Step 3: Add `signal` to `AgentRun`**

In `src/orchestrator/incident.py`, replace the `AgentRun` class:

```python
class AgentRun(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    summary: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_rationale: str | None = None
    signal: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_incident_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/incident.py tests/test_incident_model.py
git commit -m "feat(incident): add AgentRun.signal for routing"
```

---

## Task 3: Harvest `signal` from `update_incident` patches in the agent node

**Files:**
- Modify: `src/orchestrator/graph.py`
- Test: `tests/test_agent_node.py`

- [ ] **Step 1: Write failing tests for signal harvesting**

Append to `tests/test_agent_node.py`:

```python
@pytest.mark.asyncio
async def test_agent_node_captures_signal_from_update_incident(incident):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="success", next="deep_investigator"),
                RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "success"}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    triage_runs = [r for r in reloaded.agents_run if r.agent == "triage"]
    assert triage_runs and triage_runs[-1].signal == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    ("success", "success"), ("SUCCESS", "success"),
    ("failed", "failed"), ("Failed", "failed"),
    ("needs_input", "needs_input"),
])
async def test_agent_node_signal_normalises_case(incident, raw, expected):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": raw}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal == expected


@pytest.mark.asyncio
async def test_agent_node_signal_unknown_string_is_none(incident, caplog):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "halfway"}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal is None
    assert any("halfway" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_agent_node_signal_rejects_bool(incident, caplog):
    inc, store = incident
    skill = Skill(
        name="triage", description="d",
        routes=[RouteRule(when="default", next="deep_investigator")],
        system_prompt="You are triage.",
    )
    llm = StubChatModel(
        role="triage",
        canned_responses={"triage": "done"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": True}},
        }],
    )
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=lambda inc: "default",
        store=store,
    )
    with caplog.at_level(logging.WARNING, logger="orchestrator.graph"):
        await node(GraphState(incident=inc, next_route=None, last_agent=None, error=None))
    reloaded = store.load(inc.id)
    assert reloaded.agents_run[-1].signal is None
    assert any("bool" in rec.getMessage().lower() for rec in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_node.py::test_agent_node_captures_signal_from_update_incident -v`
Expected: FAIL — signal not harvested, `agents_run[-1].signal is None`.

- [ ] **Step 3: Add `_VALID_SIGNALS` and `_coerce_signal`**

In `src/orchestrator/graph.py`, just after `_coerce_rationale` (currently ends ~line 64):

```python
_VALID_SIGNALS: frozenset[str] = frozenset({"success", "failed", "needs_input"})


def _coerce_signal(raw) -> str | None:
    """Coerce a raw signal value emitted by an LLM to a canonical lowercase
    string, or None when the value cannot be interpreted.

    Recognised signals are ``success``, ``failed``, and ``needs_input``. Any
    other string emits a warning and yields None — the route lookup then
    falls back to ``when: default``. ``bool`` is rejected explicitly because
    Python treats it as ``int`` and string-coerces to ``"True"``/``"False"``.
    """
    if isinstance(raw, bool):
        logger.warning("signal value is bool (%r); rejecting", raw)
        return None
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning("non-string signal %r (%s); rejecting",
                       raw, type(raw).__name__)
        return None
    key = raw.strip().lower()
    if key in _VALID_SIGNALS:
        return key
    logger.warning("unknown signal %r; treating as None (will fall through to default)", raw)
    return None
```

- [ ] **Step 4: Harvest `signal` in `make_agent_node`**

In `src/orchestrator/graph.py` `make_agent_node`, find the loop that harvests confidence (currently around lines 224–244). Add a third local and one harvest line. Replace the block:

```python
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        agent_confidence: float | None = None
        agent_rationale: str | None = None
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("args", {}) or {}
                incident.tool_calls.append(ToolCall(
                    agent=skill.name,
                    tool=tc_name,
                    args=tc_args,
                    result=None,
                    ts=ts,
                ))
                if tc_name == "update_incident":
                    patch = tc_args.get("patch") or {}
                    if "confidence" in patch:
                        agent_confidence = _coerce_confidence(patch["confidence"])
                    if "confidence_rationale" in patch:
                        agent_rationale = _coerce_rationale(patch["confidence_rationale"])
```

with:

```python
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        agent_confidence: float | None = None
        agent_rationale: str | None = None
        agent_signal: str | None = None
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("args", {}) or {}
                incident.tool_calls.append(ToolCall(
                    agent=skill.name,
                    tool=tc_name,
                    args=tc_args,
                    result=None,
                    ts=ts,
                ))
                if tc_name == "update_incident":
                    patch = tc_args.get("patch") or {}
                    if "confidence" in patch:
                        agent_confidence = _coerce_confidence(patch["confidence"])
                    if "confidence_rationale" in patch:
                        agent_rationale = _coerce_rationale(patch["confidence_rationale"])
                    if "signal" in patch:
                        agent_signal = _coerce_signal(patch["signal"])
```

Then in the same function, find the `incident.agents_run.append(AgentRun(...))` block (currently around lines 277–287) and add `signal=agent_signal,`:

```python
        incident.agents_run.append(AgentRun(
            agent=skill.name, started_at=started_at, ended_at=ended_at,
            summary=final_text or f"{skill.name} completed",
            token_usage=TokenUsage(
                input_tokens=agent_in,
                output_tokens=agent_out,
                total_tokens=agent_total,
            ),
            confidence=agent_confidence,
            confidence_rationale=agent_rationale,
            signal=agent_signal,
        ))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_node.py -v`
Expected: PASS — original confidence tests still green, new signal tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/graph.py tests/test_agent_node.py
git commit -m "feat(graph): harvest signal from update_incident patches onto AgentRun"
```

---

## Task 4: Replace `_DECIDERS` with `_decide_from_signal`

**Files:**
- Modify: `src/orchestrator/graph.py`
- Test: `tests/test_agent_node.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_node.py`:

```python
@pytest.mark.asyncio
async def test_agent_node_routes_on_emitted_signal(incident):
    """When the agent emits signal=failed, the node should pick the route
    rule with when=failed even though when=default also matches."""
    inc, store = incident
    skill = Skill(
        name="intake", description="d",
        routes=[
            RouteRule(when="success", next="triage"),
            RouteRule(when="failed", next="__end__"),
            RouteRule(when="default", next="triage"),
        ],
        system_prompt="You are intake.",
    )
    llm = StubChatModel(
        role="intake",
        canned_responses={"intake": "no luck"},
        tool_call_plan=[{
            "name": "update_incident",
            "args": {"incident_id": inc.id,
                     "patch": {"signal": "failed"}},
        }],
    )
    from orchestrator.graph import _decide_from_signal
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=_decide_from_signal,
        store=store,
    )
    out = await node(GraphState(incident=inc, next_route=None,
                                last_agent=None, error=None))
    assert out["next_route"] == "__end__"


@pytest.mark.asyncio
async def test_agent_node_falls_back_to_default_when_no_signal(incident):
    inc, store = incident
    skill = Skill(
        name="intake", description="d",
        routes=[
            RouteRule(when="success", next="resolution"),
            RouteRule(when="default", next="triage"),
        ],
        system_prompt="You are intake.",
    )
    # Stub emits no signal at all.
    llm = StubChatModel(role="intake", canned_responses={"intake": "ok"})
    from orchestrator.graph import _decide_from_signal
    node = make_agent_node(
        skill=skill, llm=llm, tools=[],
        decide_route=_decide_from_signal,
        store=store,
    )
    out = await node(GraphState(incident=inc, next_route=None,
                                last_agent=None, error=None))
    assert out["next_route"] == "triage"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_node.py::test_agent_node_routes_on_emitted_signal tests/test_agent_node.py::test_agent_node_falls_back_to_default_when_no_signal -v`
Expected: FAIL — `_decide_from_signal` doesn't exist.

- [ ] **Step 3: Replace `_DECIDERS` with `_decide_from_signal`**

In `src/orchestrator/graph.py`, delete the block (currently lines ~301–323):

```python
# Per-agent route decision functions.
def _decide_intake(inc: Incident) -> str:
    return "default"


def _decide_triage(inc: Incident) -> str:
    return "default"


def _decide_deep_investigator(inc: Incident) -> str:
    return "default"


def _decide_resolution(inc: Incident) -> str:
    return "default"


_DECIDERS: dict[str, Callable[[Incident], str]] = {
    "intake": _decide_intake,
    "triage": _decide_triage,
    "deep_investigator": _decide_deep_investigator,
    "resolution": _decide_resolution,
}
```

Replace with:

```python
def _decide_from_signal(inc: Incident) -> str:
    """Return the latest agent's emitted signal, or "default" if absent.

    Agents emit one of {success, failed, needs_input} via the ``signal``
    key of their final ``update_incident`` patch (see ``_coerce_signal``).
    The node harvests it onto ``AgentRun.signal``; this decider then reads
    the *most recent* run (which is the one that just finished, since the
    node has already appended it). If no signal is present we return
    "default" so ``route_from_skill`` picks the fallback rule.
    """
    if not inc.agents_run:
        return "default"
    return inc.agents_run[-1].signal or "default"
```

- [ ] **Step 4: Update `_build_agent_nodes` to use the generic decider**

In `src/orchestrator/graph.py` `_build_agent_nodes` (currently around lines 387–405), replace:

```python
        decide = _DECIDERS.get(agent_name, lambda inc: "default")
```

with:

```python
        decide = _decide_from_signal
```

- [ ] **Step 5: Run all graph + agent tests to verify**

Run: `pytest tests/test_agent_node.py tests/test_build_graph.py tests/test_graph_helpers.py -v`
Expected: PASS — all existing tests plus new signal-routing tests.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/graph.py tests/test_agent_node.py
git commit -m "refactor(graph): drop _DECIDERS, route on harvested AgentRun.signal"
```

---

## Task 5: Configurable entry point + route-driven gate insertion

**Files:**
- Modify: `src/orchestrator/graph.py`
- Test: `tests/test_build_graph.py`

- [ ] **Step 1: Write failing tests for entry-agent and gated-edge plumbing**

Append to `tests/test_build_graph.py`:

```python
@pytest.mark.asyncio
async def test_build_graph_honours_entry_agent_from_config(cfg, tmp_path):
    """Entry node is whichever agent cfg.orchestrator.entry_agent names."""
    from dataclasses import replace
    from orchestrator.config import OrchestratorConfig
    skills = load_all_skills("config/skills")
    store = IncidentStore(tmp_path)
    # Override entry to triage.
    cfg2 = cfg.model_copy(update={
        "orchestrator": OrchestratorConfig(entry_agent="triage"),
    })
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg2.mcp, stack)
        graph = await build_graph(cfg=cfg2, skills=skills, store=store,
                                  registry=registry)
        compiled = graph.get_graph()
        # Find edges leaving __start__; the head should now be triage.
        edges_from_start = [
            e for e in compiled.edges if e.source == "__start__"
        ]
        targets = {e.target for e in edges_from_start}
        assert "triage" in targets


@pytest.mark.asyncio
async def test_build_graph_inserts_gate_for_gated_route(cfg, tmp_path):
    """A skill with a gate-marked route edge should result in the gate node
    being inserted between that agent and its target."""
    skills = load_all_skills("config/skills")
    store = IncidentStore(tmp_path)
    async with AsyncExitStack() as stack:
        registry = await load_tools(cfg.mcp, stack)
        graph = await build_graph(cfg=cfg, skills=skills, store=store,
                                  registry=registry)
        nodes = set(graph.get_graph().nodes.keys())
        assert "gate" in nodes
        # Sanity: deep_investigator's success route (which carries gate)
        # should not have a direct edge to resolution that bypasses the gate.
        # We assert by behaviour: see test_full_graph_runs_to_terminal_with_stub_llm.
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_build_graph.py::test_build_graph_honours_entry_agent_from_config -v`
Expected: FAIL — `build_graph` still calls `set_entry_point("intake")`.

- [ ] **Step 3: Generalise `build_graph` to read entry agent and gated edges**

In `src/orchestrator/graph.py`, replace `build_graph` (currently lines ~416–455) with:

```python
async def build_graph(*, cfg: AppConfig, skills: dict, store: IncidentStore,
                      registry: ToolRegistry):
    """Compile the main LangGraph from configured skills and routes.

    The entry agent is read from ``cfg.orchestrator.entry_agent``. Gate
    insertions are derived from each skill's route rules: a rule with
    ``gate: confidence`` causes the router to redirect ``(this_agent, next)``
    through the ``gate`` node.

    The ``registry`` is provided by the caller — typically the
    :class:`Orchestrator`, which loads MCP tools into an :class:`AsyncExitStack`
    so the underlying FastMCP transports stay alive for the lifetime of the
    compiled graph.
    """
    entry = cfg.orchestrator.entry_agent
    if entry not in skills:
        raise ValueError(
            f"orchestrator.entry_agent={entry!r} is not a known skill "
            f"(known: {sorted(skills.keys())})"
        )
    gated_edges = _collect_gated_edges(skills)

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name, node in nodes.items():
        sg.add_node(agent_name, node)
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))

    sg.set_entry_point(entry)

    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        la = state.get("last_agent")
        if (la, nr) in gated_edges:
            return "gate"
        return nr

    for agent_name in skills.keys():
        possible_targets = {s.name for s in skills.values()} | {END, "gate"}
        target_map = {name: name for name in possible_targets if name != END}
        target_map[END] = END
        sg.add_conditional_edges(agent_name, _router, target_map)

    # Determine where the gate forwards on a "default" pass — there is at
    # most one downstream agent per gated edge; with one gate type today
    # we collapse to a single target. If the user defines multiple gated
    # edges with different downstream agents in the future, ``_gate_router``
    # will need a state-aware lookup; for now we assert "exactly one" and
    # error loudly otherwise.
    gate_targets = {to for (_from, to) in gated_edges}
    if len(gate_targets) > 1:
        raise ValueError(
            f"multiple gated downstream targets {sorted(gate_targets)} not "
            f"yet supported; only one gated edge per graph"
        )
    gate_target = next(iter(gate_targets), None)
    if gate_target is not None:
        sg.add_conditional_edges("gate", _gate_router, {
            gate_target: gate_target, END: END,
        })
    else:
        # No gated edges → gate is unreachable but still a node; wire to END.
        sg.add_edge("gate", END)

    return sg.compile()
```

Then add the helper just above `build_graph` (or at the top of the module after `route_from_skill`):

```python
def _collect_gated_edges(skills: dict) -> dict[tuple[str, str], str]:
    """Return ``{(from_agent, to_node): gate_type}`` for every route rule
    whose ``gate`` is set. Today only ``gate: confidence`` is recognised."""
    edges: dict[tuple[str, str], str] = {}
    for agent_name, skill in skills.items():
        for rule in skill.routes:
            if rule.gate:
                edges[(agent_name, rule.next)] = rule.gate
    return edges
```

Also adjust `_gate_router` (currently lines ~408–413) so its return value uses the new dynamic target — but the existing implementation already returns `"resolution"` literally, which we need to make dynamic. Replace `_gate_router`:

```python
def _gate_router(state: GraphState):
    """Forward to the gate's downstream target (or END if the gate paused).

    The caller wires this with a `target_map` whose keys include the actual
    downstream node name and END, so returning the route signal directly
    works for any topology with a single gate downstream.
    """
    nr = state.get("next_route")
    if nr in (None, "__end__"):
        return END
    # gate emits ``next_route="default"`` on pass; the caller maps that to
    # the actual downstream node via the target_map. We return the *node*
    # name here, not the signal — so we look it up from state instead.
    return state.get("_gate_target") or "default"
```

That is awkward. **Use a closure instead** so the gate target is captured at build time:

Replace the whole gate-wiring section in `build_graph` with:

```python
    if gate_target is not None:
        def _gate_to(state: GraphState):
            nr = state.get("next_route")
            if nr in (None, "__end__"):
                return END
            return gate_target
        sg.add_conditional_edges("gate", _gate_to, {
            gate_target: gate_target, END: END,
        })
    else:
        sg.add_edge("gate", END)
```

And **delete the module-level `_gate_router` function entirely** (currently lines ~408–413) — it's no longer used. The gate-pass router is now a build-time closure.

- [ ] **Step 4: Apply the same generalisation to `build_resume_graph`**

Replace `build_resume_graph` (currently lines ~458–494) with:

```python
async def build_resume_graph(*, cfg: AppConfig, skills: dict,
                             store: IncidentStore, registry: ToolRegistry):
    """Compile a sub-graph that re-runs from the *upstream* end of the
    (single) gated edge through to the gate's downstream target.

    Used by ``Orchestrator.resume_investigation`` after the user supplies
    new context: agents before the gated edge already ran, so we resume
    from the gated agent with the updated incident. Same gate semantics —
    if the new run is still low-confidence, we'll pause again.
    """
    gated_edges = _collect_gated_edges(skills)
    if not gated_edges:
        raise ValueError(
            "build_resume_graph requires at least one route with gate set; "
            "no gated edges were found in the configured skills"
        )
    if len(gated_edges) > 1:
        raise ValueError(
            f"multiple gated edges {sorted(gated_edges.keys())} not yet "
            f"supported; resume entry is ambiguous"
        )
    resume_from, gate_target = next(iter(gated_edges.keys()))

    sg = StateGraph(GraphState)
    nodes = _build_agent_nodes(cfg=cfg, skills=skills, store=store, registry=registry)
    for agent_name in (resume_from, gate_target):
        if agent_name in nodes:
            sg.add_node(agent_name, nodes[agent_name])
    sg.add_node("gate", make_gate_node(cfg=cfg, store=store))
    sg.set_entry_point(resume_from)

    def _router(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        la = state.get("last_agent")
        if (la, nr) in gated_edges:
            return "gate"
        return nr

    for agent_name in (resume_from, gate_target):
        sg.add_conditional_edges(agent_name, _router, {
            resume_from: resume_from,
            gate_target: gate_target,
            "gate": "gate",
            END: END,
        })

    def _gate_to(state: GraphState):
        nr = state.get("next_route")
        if nr in (None, "__end__"):
            return END
        return gate_target
    sg.add_conditional_edges("gate", _gate_to, {
        gate_target: gate_target, END: END,
    })
    return sg.compile()
```

- [ ] **Step 5: Run the full graph + resume + gate test suites**

Run: `pytest tests/test_build_graph.py tests/test_resume.py tests/test_gate.py tests/test_graph_helpers.py -v`
Expected: PASS — including the two new tests added in step 1, and existing resume / gate tests should still hold because the derived behaviour matches the previous hardcoded behaviour when `deep_investigator` carries `gate: confidence`.

If `test_resume.py` fails because the configured skills don't yet have `gate: confidence` on the `deep_investigator → resolution` edge, the failure is expected at this step — that wiring is added in Task 6. Run only `test_resume.py` once Task 6's config edits land, not now.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/graph.py tests/test_build_graph.py
git commit -m "refactor(graph): configurable entry agent + route-driven gate insertion"
```

---

## Task 6: Update agent configs, common prompt, and top-level config

**Files:**
- Modify: `config/config.yaml`
- Modify: `config/skills/intake/config.yaml`
- Modify: `config/skills/triage/config.yaml`
- Modify: `config/skills/deep_investigator/config.yaml`
- Modify: `config/skills/resolution/config.yaml`
- Modify: `config/skills/_common/output.md`

- [ ] **Step 1: Append the signal-emission instruction to `_common/output.md`**

Replace `config/skills/_common/output.md` with:

```markdown
## Output
Your final reply — the message you emit *after* all tool calls have completed — must be a concise summary in **2–4 sentences, ≤150 words total**. The UI already renders the structured fields you wrote via `update_incident` separately; do not restate them. Skip code-fenced blocks unless quoting an actual log line verbatim. Inline bold/italic markdown is fine.

## Signal
In your final `update_incident` patch, **always** include a `signal` field set to one of:

- `success` — you completed your specialty and the workflow should advance normally.
- `failed` — a tool errored, you produced no useful output, or you are confident no further work on this incident makes sense at this stage.
- `needs_input` — you cannot proceed without additional human-supplied context.

The orchestrator routes the workflow to the next node based on this signal. If you omit it, the route falls back to the rule marked `when: default`.
```

- [ ] **Step 2: Update `config/skills/intake/config.yaml`**

Replace its `routes` block with:

```yaml
description: First-line agent — enriches the just-created INC and checks for known prior resolutions
temperature: 0.0
tools:
  - lookup_similar_incidents
  - get_user_context
  - update_incident
routes:
  - when: success
    next: triage
  - when: failed
    next: __end__
  - when: default
    next: triage
```

- [ ] **Step 3: Update `config/skills/triage/config.yaml`**

Replace with:

```yaml
description: Categorize, prioritize, and assess the impact of the incident
temperature: 0.0
tools:
  - update_incident
  - get_service_health
  - check_deployment_history
routes:
  - when: success
    next: deep_investigator
  - when: failed
    next: __end__
  - when: default
    next: deep_investigator
```

- [ ] **Step 4: Update `config/skills/deep_investigator/config.yaml`** — gain `gate: confidence` on the success route

Replace with:

```yaml
description: Perform diagnostic deep-dive — pull logs, metrics, propose hypotheses
temperature: 0.0
tools:
  - get_logs
  - get_metrics
  - update_incident
routes:
  - when: success
    next: resolution
    gate: confidence
  - when: failed
    next: __end__
  - when: default
    next: resolution
    gate: confidence
```

- [ ] **Step 5: Update `config/skills/resolution/config.yaml`**

Replace with:

```yaml
description: Propose and (mock-)apply a fix; close the INC or escalate
temperature: 0.0
tools:
  - propose_fix
  - apply_fix
  - notify_oncall
  - update_incident
routes:
  - when: success
    next: __end__
  - when: failed
    next: __end__
  - when: default
    next: __end__
```

- [ ] **Step 6: Update `config/config.yaml` to declare the orchestrator entry agent**

After the `intervention:` block at the bottom, append:

```yaml
orchestrator:
  entry_agent: intake
```

(Default would already give the same value, but declaring it makes the choice visible at the top-level config.)

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests -v`
Expected: PASS — all suites green. `test_build_graph.py::test_build_graph_inserts_gate_for_gated_route` now resolves because `deep_investigator` declares `gate: confidence`. `test_resume.py` runs cleanly because the gated edge derivation succeeds.

- [ ] **Step 8: Commit**

```bash
git add config/config.yaml config/skills/intake/config.yaml \
        config/skills/triage/config.yaml \
        config/skills/deep_investigator/config.yaml \
        config/skills/resolution/config.yaml \
        config/skills/_common/output.md
git commit -m "feat(skills): adopt success/failed/needs_input signal vocabulary; declare gate on DI"
```

---

## Task 7: Regenerate the single-file bundle

**Files:**
- Regenerate: `dist/app.py` via `scripts/build_single_file.py`

- [ ] **Step 1: Run the bundler**

Run: `python scripts/build_single_file.py`
Expected: `dist/app.py` updated. Bundler prints success and any size delta.

- [ ] **Step 2: Smoke-test the bundle**

Run: `python -c "import ast; ast.parse(open('dist/app.py').read()); print('parse OK')"`
Expected: `parse OK`.

Run: `pytest tests/test_build_single_file.py -v`
Expected: PASS — bundle integrity test still green.

- [ ] **Step 3: Commit**

```bash
git add dist/app.py
git commit -m "chore(dist): regenerate app.py for entry-agent + signal-routing changes"
```

---

## Verification checklist (run after Task 7)

- [ ] `pytest tests -v` — all green, no skips beyond pre-existing ones.
- [ ] `python scripts/build_single_file.py` — clean run.
- [ ] Manually start the app (`python -m orchestrator.api` or the documented entry) and submit one query end-to-end through the Streamlit UI — confirm the incident traverses intake → triage → deep_investigator → gate → resolution as before, and the new `signal` field shows on each AgentRun in the persisted JSON.
- [ ] Inspect a saved INC JSON — `agents_run[*].signal` should be one of `success`, `failed`, `needs_input`, or `null` (the LLM may take a few runs to consistently emit it; confirm `_common/output.md` is concatenated into the final system prompt).
