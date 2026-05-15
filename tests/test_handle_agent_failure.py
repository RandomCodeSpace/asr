"""Coverage tests for ``runtime.graph._handle_agent_failure`` (graph.py:613-644).

This helper is invoked by the agent runner when an agent body raises a
non-pause exception (anything other than ``GraphInterrupt``). It reloads
the session (absorbing partial tool writes), appends a failure
``AgentRun``, marks the session ``status='error'``, persists, and returns
the state dict the LangGraph node yields.
"""
from __future__ import annotations

import pytest

from runtime.config import EmbeddingConfig, MetadataConfig, ProviderConfig
from runtime.graph import _handle_agent_failure
from runtime.state import AgentRun, Session
from runtime.storage.embeddings import build_embedder
from runtime.storage.engine import build_engine
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    eng = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"))
    Base.metadata.create_all(eng)
    embedder = build_embedder(
        EmbeddingConfig(provider="s", model="x", dim=1024),
        {"s": ProviderConfig(kind="stub")},
    )
    return SessionStore(engine=eng, embedder=embedder)


def _seed_session(store: SessionStore, *, agents_run: list[AgentRun] | None = None) -> Session:
    """Create + persist a baseline session and return it."""
    inc = store.create(query="probe", environment="dev",
                       reporter_id="u1", reporter_team="t")
    if agents_run:
        inc.agents_run.extend(agents_run)
        store.save(inc)
        inc = store.load(inc.id)
    return inc


class TestHappyPath:
    def test_failure_run_appended_and_status_set_to_error(self, store):
        inc = _seed_session(store)
        result = _handle_agent_failure(
            skill_name="triage",
            started_at="2026-05-15T00:00:00Z",
            exc=RuntimeError("upstream blew up"),
            inc_id=inc.id,
            store=store,
            fallback=inc,
        )
        # Returned state dict
        assert result["last_agent"] == "triage"
        assert result["next_route"] is None
        assert result["error"] == "upstream blew up"
        assert isinstance(result["session"], Session)
        # Persisted session reflects the failure
        loaded = store.load(inc.id)
        assert loaded.status == "error"
        assert len(loaded.agents_run) == 1
        run = loaded.agents_run[0]
        assert run.agent == "triage"
        assert run.summary == "agent failed: upstream blew up"

    def test_appends_to_existing_run_history(self, store):
        prior = AgentRun(
            agent="intake",
            started_at="2026-05-15T00:00:00Z",
            ended_at="2026-05-15T00:00:01Z",
            summary="completed: routed to triage",
        )
        inc = _seed_session(store, agents_run=[prior])
        _handle_agent_failure(
            skill_name="triage",
            started_at="2026-05-15T00:00:02Z",
            exc=TimeoutError("provider hung"),
            inc_id=inc.id,
            store=store,
            fallback=inc,
        )
        loaded = store.load(inc.id)
        assert [r.agent for r in loaded.agents_run] == ["intake", "triage"]
        assert "agent failed: provider hung" in loaded.agents_run[1].summary

    def test_preserves_partial_tool_writes_via_reload(self, store):
        """If a tool wrote to the session before the agent raised,
        the reload-then-append pattern must keep that tool's write."""
        inc = _seed_session(store)
        # Simulate a tool write that already persisted.
        from runtime.state import ToolCall
        inc.tool_calls.append(ToolCall(
            agent="triage",
            tool="lookup_similar_incidents",
            args={"query": "x"},
            result={"hits": []},
            ts="2026-05-15T00:00:00Z",
        ))
        store.save(inc)
        # Caller's stale `fallback` reference does not have the tool call.
        stale = inc.model_copy(deep=True)
        stale.tool_calls = []
        _handle_agent_failure(
            skill_name="triage",
            started_at="2026-05-15T00:00:02Z",
            exc=RuntimeError("oops"),
            inc_id=inc.id,
            store=store,
            fallback=stale,
        )
        loaded = store.load(inc.id)
        # Tool call survived because _handle_agent_failure reloaded
        # before appending its failure run.
        assert len(loaded.tool_calls) == 1
        assert loaded.tool_calls[0].tool == "lookup_similar_incidents"


class TestFallbackPath:
    def test_uses_fallback_when_session_missing_from_store(self, store):
        # Session never persisted; store.load(inc_id) raises FileNotFoundError.
        from runtime.state import Session
        ghost = Session(
            id="INC-20260515-999",
            status="in_progress",
            created_at="2026-05-15T00:00:00Z",
            updated_at="2026-05-15T00:00:00Z",
        )
        result = _handle_agent_failure(
            skill_name="intake",
            started_at="2026-05-15T00:00:00Z",
            exc=RuntimeError("dropped on the floor"),
            inc_id="INC-20260515-999",
            store=store,
            fallback=ghost,
        )
        # The fallback was used, the failure run was appended,
        # and the now-populated fallback was saved.
        assert result["session"].status == "error"
        loaded = store.load("INC-20260515-999")
        assert loaded.id == "INC-20260515-999"
        assert loaded.status == "error"
        assert len(loaded.agents_run) == 1
        assert loaded.agents_run[0].agent == "intake"
