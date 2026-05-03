"""MCP-server tests for ``examples/code_review/mcp_server.py``.

Covers tool registration, ``add_review_finding`` appends to the
session, and ``set_recommendation`` persists across reload.

These tests use a small in-memory store that satisfies the ``load`` /
``save`` contract the MCP server depends on without touching
SQLAlchemy.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone

import pytest

from examples.code_review.state import CodeReviewState, PullRequest
from examples.code_review.mcp_server import CodeReviewMCPServer


_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_TODAY = datetime.now(timezone.utc).strftime("%Y%m%d")


class InMemorySessionStore:
    """Drop-in stand-in for ``SessionStore[CodeReviewState]``.

    Implements only the slice the MCP server uses: ``load(id)`` and
    ``save(state)``. Stores deep copies so a caller mutating a returned
    state does not silently mutate the cell — matching the SQLAlchemy
    store's snapshot semantics.
    """

    def __init__(self) -> None:
        self._cells: dict[str, CodeReviewState] = {}

    def save(self, state: CodeReviewState) -> None:
        self._cells[state.id] = copy.deepcopy(state)

    def load(self, session_id: str) -> CodeReviewState:
        if session_id not in self._cells:
            raise FileNotFoundError(session_id)
        return copy.deepcopy(self._cells[session_id])


def _make_state(seq: int = 1) -> CodeReviewState:
    """Build a fully-formed ``CodeReviewState`` with a valid id."""
    return CodeReviewState(
        id=f"INC-{_TODAY}-{seq:03d}",
        status="new",
        created_at=_NOW,
        updated_at=_NOW,
        pr=PullRequest(
            repo="org/repo",
            number=42,
            title="Add caching layer",
            author="alice",
            base_sha="aaaaaaa",
            head_sha="bbbbbbb",
            additions=12,
            deletions=3,
            files_changed=2,
        ),
    )


@pytest.fixture
def store():
    return InMemorySessionStore()


@pytest.fixture
def server(store):
    s = CodeReviewMCPServer()
    s.configure(store=store)  # InMemorySessionStore satisfies the contract
    return s


def test_mcp_server_has_three_tools(server) -> None:
    """The server registers exactly the three documented tools."""
    # FastMCP exposes registered tools via the async ``list_tools`` API,
    # which returns a list of ``FunctionTool`` objects with a ``.name``.
    # Collapse to a name set so the assertion is order-insensitive.
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"fetch_pr_diff", "add_review_finding", "set_recommendation"}, (
        f"unexpected tool registration: {sorted(names)}"
    )


def test_add_finding_appends_to_session(server, store) -> None:
    """``add_review_finding`` grows ``review_findings`` by exactly one."""
    state = _make_state(seq=1)
    store.save(state)
    assert store.load(state.id).extra_fields.get("review_findings", []) == []

    result = asyncio.run(server._tool_add_review_finding(
        session_id=state.id,
        severity="warning",
        file="src/api.py",
        line=42,
        category="performance",
        message="Unbounded HTTP request — add a timeout.",
        suggestion="response = httpx.get(url, timeout=10.0)",
    ))
    assert result["ok"] is True
    assert result["findings_count"] == 1

    reloaded = store.load(state.id)
    findings = reloaded.extra_fields.get("review_findings", [])
    assert len(findings) == 1
    finding = findings[0]
    assert finding["severity"] == "warning"
    assert finding["file"] == "src/api.py"
    assert finding["line"] == 42
    assert finding["category"] == "performance"
    assert finding["suggestion"] is not None


def test_set_recommendation_writes_summary(server, store) -> None:
    """``set_recommendation`` persists both fields across a reload."""
    state = _make_state(seq=2)
    store.save(state)

    result = asyncio.run(server._tool_set_recommendation(
        session_id=state.id,
        recommendation="request_changes",
        summary="Two error-severity issues block merge; see findings.",
    ))
    assert result == {
        "ok": True,
        "recommendation": "request_changes",
        "summary": "Two error-severity issues block merge; see findings.",
    }

    reloaded = store.load(state.id)
    assert reloaded.extra_fields.get("overall_recommendation") == "request_changes"
    assert reloaded.extra_fields.get("review_summary") == "Two error-severity issues block merge; see findings."


def test_set_recommendation_rejects_invalid_value(server, store) -> None:
    """Recommendation outside the allowed literal set raises before storage."""
    state = _make_state(seq=3)
    store.save(state)
    with pytest.raises(ValueError, match="approve/request_changes/comment"):
        asyncio.run(server._tool_set_recommendation(
            session_id=state.id,
            recommendation="merge_now",
            summary="x",
        ))
    # Storage untouched: the load still returns the default None recommendation.
    reloaded = store.load(state.id)
    assert reloaded.extra_fields.get("overall_recommendation") is None


def test_fetch_pr_diff_returns_synthetic_when_no_fixture(server, tmp_path) -> None:
    """With an empty fixtures dir, ``fetch_pr_diff`` falls back to synthetic."""
    server.fixtures_dir = tmp_path  # no JSON under tmp_path -> synthetic path
    diff = asyncio.run(server._tool_fetch_pr_diff(repo="org/repo", number=99))
    assert diff["source"] == "synthetic"
    assert diff["repo"] == "org/repo"
    assert diff["number"] == 99
    assert diff["additions"] >= 1
    assert diff["files_changed"], "synthetic diff should name at least one file"
