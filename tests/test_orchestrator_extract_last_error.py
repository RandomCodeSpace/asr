"""Coverage tests for ``Orchestrator._extract_last_error`` (orchestrator.py:945-998).

Pure mapping from a Session's failed-AgentRun summary string to a
representative typed exception (used by :func:`runtime.policy.should_retry`'s
``isinstance`` checks). Table-driven across every branch.
"""
from __future__ import annotations

import pydantic
import pytest

from runtime.agents.turn_output import EnvelopeMissingError
from runtime.orchestrator import Orchestrator
from runtime.state import AgentRun, Session


def _session_with_failures(*summaries: str) -> Session:
    """Build a Session with the given run summaries (oldest first).

    Empty string defaults to a non-failure run (summary 'completed: x').
    """
    runs = []
    for i, summary in enumerate(summaries):
        runs.append(AgentRun(
            agent=f"agent-{i}",
            started_at="2026-05-15T00:00:00Z",
            ended_at="2026-05-15T00:00:01Z",
            summary=summary,
        ))
    return Session(
        id="INC-test",
        status="error",
        created_at="2026-05-15T00:00:00Z",
        updated_at="2026-05-15T00:00:01Z",
        agents_run=runs,
    )


class TestNoFailedRun:
    def test_empty_runs_returns_none(self):
        inc = _session_with_failures()
        assert Orchestrator._extract_last_error(inc) is None

    def test_only_successful_runs_returns_none(self):
        inc = _session_with_failures(
            "completed: triage routed to investigate",
            "completed: investigate found root cause",
        )
        assert Orchestrator._extract_last_error(inc) is None

    def test_run_with_empty_summary_returns_none(self):
        inc = _session_with_failures("")
        assert Orchestrator._extract_last_error(inc) is None


class TestEnvelopeMissingMapping:
    def test_envelope_missing_error_matched(self):
        inc = _session_with_failures(
            "agent failed: EnvelopeMissingError: confidence (agent=intake)",
        )
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, EnvelopeMissingError)
        assert err.agent == "agent-0"
        assert err.field == "confidence"


class TestValidationErrorMapping:
    def test_capitalised_validation_error(self):
        inc = _session_with_failures("agent failed: ValidationError on field foo")
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, pydantic.ValidationError)

    def test_lowercase_validation_error(self):
        inc = _session_with_failures("agent failed: pydantic raised a validation error here")
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, pydantic.ValidationError)


class TestTimeoutMapping:
    @pytest.mark.parametrize("body", [
        "agent failed: TimeoutError: provider hung",
        "agent failed: request timed out after 30s",
        "agent failed: asyncio.TimeoutError",
    ])
    def test_timeout_variants_match(self, body):
        inc = _session_with_failures(body)
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, TimeoutError)


class TestOSErrorMapping:
    @pytest.mark.parametrize("body", [
        "agent failed: OSError: too many open files",
        "agent failed: ConnectionError: refused",
    ])
    def test_oserror_variants_match(self, body):
        inc = _session_with_failures(body)
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, OSError)


class TestRuntimeErrorFallback:
    def test_unknown_failure_returns_runtime_error(self):
        inc = _session_with_failures("agent failed: KeyError: something weird")
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, RuntimeError)


class TestNewestFailureWins:
    def test_reversed_iteration_picks_newest_failure(self):
        # First failed (older) is OSError, second failed (newer) is Timeout.
        # Reversed iteration should hit the timeout first and return it.
        inc = _session_with_failures(
            "agent failed: OSError: stale",
            "agent failed: TimeoutError: fresh",
        )
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, TimeoutError)
        assert "fresh" in str(err)

    def test_skips_non_failure_summaries(self):
        # Only one failure in the middle of successes — should still be found.
        inc = _session_with_failures(
            "completed: triage ok",
            "agent failed: OSError: middle failure",
            "completed: another success",
        )
        err = Orchestrator._extract_last_error(inc)
        assert isinstance(err, OSError)
