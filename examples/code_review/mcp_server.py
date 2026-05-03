"""FastMCP server: code_review tools, backed by ``SessionStore``.

Part of the code-review example application. Framework code does not
import this module.

Mirrors the shape of ``examples/incident_management/mcp_server.py``:

- a ``CodeReviewMCPServer`` dataclass that wraps a :class:`FastMCP` and is
  bound to a ``SessionStore[CodeReviewState]``;
- a module-level default ``mcp`` export so the framework's MCP loader can
  discover the server by name;
- a ``set_state(...)`` shim that wires the default server to a concrete
  store at process bootstrap time;
- direct-call shims (``fetch_pr_diff`` / ``add_review_finding`` /
  ``set_recommendation``) for tests that bypass the MCP transport.

The diff fetcher is a stub — it reads JSON fixtures from
``tests/fixtures/code_review/<repo>/<number>.json`` if present, else
synthesises a tiny diff. Real GitHub fetching is out of scope for the
P8 example app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp import FastMCP

from runtime.storage.session_store import SessionStore
from examples.code_review.state import CodeReviewState, ReviewFinding


# ---------------------------------------------------------------------------
# Fixture discovery for ``fetch_pr_diff``.
# ---------------------------------------------------------------------------

# Repo root: examples/code_review/mcp_server.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures" / "code_review"


def _synthetic_diff(repo: str, number: int) -> dict:
    """Return a tiny canned diff so the example runs offline without fixtures."""
    file_path = "src/example.py"
    diff = (
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        "@@ -1,3 +1,4 @@\n"
        " def hello():\n"
        "-    return 'hi'\n"
        "+    # synthetic change for PR #{n}\n"
        "+    return 'hello'\n"
    ).replace("{n}", str(number))
    return {
        "diff": diff,
        "files_changed": [file_path],
        "additions": 2,
        "deletions": 1,
        "repo": repo,
        "number": number,
        "source": "synthetic",
    }


def _load_fixture_diff(fixtures_dir: Path, repo: str, number: int) -> dict | None:
    """Return the fixture-backed diff for ``<repo>/<number>.json``, if present.

    Repo slugs may contain ``/`` (e.g. ``org/repo``); we treat that as a
    nested directory under ``fixtures_dir``.
    """
    candidate = fixtures_dir / repo / f"{number}.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@dataclass
class CodeReviewMCPServer:
    """FastMCP server bound to a single ``SessionStore[CodeReviewState]``.

    ``store`` may be ``None`` until ``configure(store=...)`` is called —
    this matches the incident-management server, which lets the
    module-level default exist before the runtime wires up storage.

    ``fixtures_dir`` defaults to ``tests/fixtures/code_review/`` relative
    to the repo root; tests override it to a ``tmp_path`` so they do not
    accidentally read committed fixtures.
    """

    store: SessionStore[CodeReviewState] | None = None
    fixtures_dir: Path = field(default_factory=lambda: _DEFAULT_FIXTURES_DIR)
    mcp: FastMCP = field(init=False)

    def __post_init__(self) -> None:
        self.mcp = FastMCP("code_review")
        self.mcp.tool(name="fetch_pr_diff")(self._tool_fetch_pr_diff)
        self.mcp.tool(name="add_review_finding")(self._tool_add_review_finding)
        self.mcp.tool(name="set_recommendation")(self._tool_set_recommendation)

    def configure(
        self,
        *,
        store: SessionStore[CodeReviewState],
        fixtures_dir: Path | str | None = None,
    ) -> None:
        self.store = store
        if fixtures_dir is not None:
            self.fixtures_dir = Path(fixtures_dir)

    def _require_store(self) -> SessionStore[CodeReviewState]:
        if self.store is None:
            raise RuntimeError(
                "code_review server not initialized — call configure() "
                "(or the module-level set_state) first"
            )
        return self.store

    # ---------- tools ----------

    async def _tool_fetch_pr_diff(self, repo: str, number: int) -> dict:
        """Return the unified diff for a PR.

        Reads ``<fixtures_dir>/<repo>/<number>.json`` when present; falls
        back to a synthetic minimal diff otherwise. Real GitHub fetch is
        out of scope.
        """
        fixture = _load_fixture_diff(self.fixtures_dir, repo, number)
        if fixture is not None:
            # Honour the fixture; backfill the keys we always promise.
            fixture.setdefault("repo", repo)
            fixture.setdefault("number", number)
            fixture.setdefault("files_changed", [])
            fixture.setdefault("additions", 0)
            fixture.setdefault("deletions", 0)
            fixture.setdefault("source", "fixture")
            return fixture
        return _synthetic_diff(repo, number)

    async def _tool_add_review_finding(
        self,
        session_id: str,
        severity: str,
        file: str,
        line: int | None,
        category: str,
        message: str,
        suggestion: str | None = None,
    ) -> dict:
        """Append a ``ReviewFinding`` to the session's ``review_findings`` list."""
        store = self._require_store()
        session = store.load(session_id)
        finding = ReviewFinding(
            severity=severity,  # type: ignore[arg-type]  # pydantic validates Literal
            file=file,
            line=line,
            category=category,
            message=message,
            suggestion=suggestion,
        )
        session.review_findings.append(finding)
        store.save(session)
        return {"ok": True, "findings_count": len(session.review_findings)}

    async def _tool_set_recommendation(
        self,
        session_id: str,
        recommendation: str,
        summary: str,
    ) -> dict:
        """Set ``overall_recommendation`` and ``review_summary`` on the session."""
        if recommendation not in {"approve", "request_changes", "comment"}:
            raise ValueError(
                f"recommendation must be approve/request_changes/comment; "
                f"got {recommendation!r}"
            )
        store = self._require_store()
        session = store.load(session_id)
        session.overall_recommendation = recommendation  # type: ignore[assignment]
        session.review_summary = summary
        store.save(session)
        return {
            "ok": True,
            "recommendation": recommendation,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Module-level default server (the MCP loader imports ``mcp`` by name).
# ---------------------------------------------------------------------------

_default_server = CodeReviewMCPServer()
mcp = _default_server.mcp


def set_state(
    *,
    store: SessionStore[CodeReviewState],
    fixtures_dir: Path | str | None = None,
) -> None:
    """Configure the default ``CodeReviewMCPServer`` instance."""
    _default_server.configure(store=store, fixtures_dir=fixtures_dir)


# Direct-call shims kept for tests that bypass the FastMCP transport.
async def fetch_pr_diff(repo: str, number: int) -> dict:
    return await _default_server._tool_fetch_pr_diff(repo, number)


async def add_review_finding(
    session_id: str,
    severity: str,
    file: str,
    line: int | None,
    category: str,
    message: str,
    suggestion: str | None = None,
) -> dict:
    return await _default_server._tool_add_review_finding(
        session_id=session_id,
        severity=severity,
        file=file,
        line=line,
        category=category,
        message=message,
        suggestion=suggestion,
    )


async def set_recommendation(
    session_id: str,
    recommendation: str,
    summary: str,
) -> dict:
    return await _default_server._tool_set_recommendation(
        session_id=session_id,
        recommendation=recommendation,
        summary=summary,
    )
