"""Tests for ``scripts/lint_skill_prompts.py`` — the skill-prompt-vs-schema
linter that enforces SKILL-LINTER-01 (Phase 21).

Two acceptance pillars:

1. **Binary-pass on the live tree** — the linter must exit 0 against the
   current ``examples/`` skill prompts. This is the CI-gate guarantee.
2. **Detection** — fixture prompts injected with known-bad references
   (unknown tool, unknown field, legacy ``findings_<x>`` form, malformed
   non-JSON code blocks) must produce the expected violations without
   crashing.

The tests import the linter as a module rather than spawning subprocesses,
which keeps execution under the 5s budget called for in the phase
acceptance gates.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loader (the linter lives under scripts/ which is not a package)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER_PATH = REPO_ROOT / "scripts" / "lint_skill_prompts.py"


def _load_linter():
    spec = importlib.util.spec_from_file_location("lint_skill_prompts", LINTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lint_skill_prompts"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def linter():
    return _load_linter()


# ---------------------------------------------------------------------------
# Fixture builder — synthesizes a tiny examples tree under tmp_path
# ---------------------------------------------------------------------------


def _build_example_tree(
    root: Path,
    *,
    tools_module: str,
    prompt: str,
    patch_model: str | None = None,
) -> None:
    """Create a minimal ``examples/<app>/{mcp_server.py, skills/x/system.md}``
    layout under *root* so the linter can discover tools + prompts via its
    standard traversal."""
    app = root / "examples" / "demo_app"
    skill_dir = app / "skills" / "x"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (app / "mcp_server.py").write_text(tools_module, encoding="utf-8")
    if patch_model:
        # Append patch model definition for `update_incident` discovery.
        existing = (app / "mcp_server.py").read_text(encoding="utf-8")
        (app / "mcp_server.py").write_text(existing + "\n\n" + patch_model, encoding="utf-8")
    (skill_dir / "system.md").write_text(prompt, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_linter_passes_on_current_skill_prompts(linter):
    """Acceptance gate: the live ``examples/`` tree must lint clean."""
    schemas = linter.discover_tools(REPO_ROOT / "examples")
    patch_fields = linter.discover_patch_fields(REPO_ROOT)
    prompts = linter.iter_skill_prompts(REPO_ROOT / "examples")
    assert prompts, "expected at least one skill prompt under examples/"
    violations: list[str] = []
    for path in prompts:
        violations.extend(linter.lint_prompt(path, schemas, patch_fields))
    assert violations == [], (
        "current skill prompts have schema-drift violations:\n"
        + "\n".join(violations)
    )


def test_linter_discovers_known_tools(linter):
    """Sanity: discovery must find every tool the framework gates today."""
    schemas = linter.discover_tools(REPO_ROOT / "examples")
    expected_subset = {
        "update_incident",
        "lookup_similar_incidents",
        "create_incident",
        "mark_resolved",
        "mark_escalated",
        "submit_hypothesis",
        "get_logs",
        "get_metrics",
        "get_service_health",
        "check_deployment_history",
        "propose_fix",
        "apply_fix",
        "fetch_pr_diff",
        "add_review_finding",
        "set_recommendation",
    }
    missing = expected_subset - schemas.keys()
    assert not missing, f"discovery missed tools: {missing}"


def test_linter_detects_unknown_field(linter, tmp_path: Path):
    """Calling ``mark_resolved`` with a non-existent ``priority`` arg must fail."""
    tools = textwrap.dedent("""
        class DemoServer:
            def __init__(self):
                self.mcp.tool(name="mark_resolved")(self._tool_mark_resolved)

            async def _tool_mark_resolved(self, incident_id, confidence,
                                          confidence_rationale, resolution_summary):
                ...
    """)
    prompt = "Call `mark_resolved(priority=high, resolution_summary='ok')` to close."
    _build_example_tree(tmp_path, tools_module=tools, prompt=prompt)

    schemas = linter.discover_tools(tmp_path / "examples")
    patch_fields = linter.discover_patch_fields(tmp_path)
    prompt_path = tmp_path / "examples" / "demo_app" / "skills" / "x" / "system.md"
    violations = linter.lint_prompt(prompt_path, schemas, patch_fields)
    # The bad arg must be reported as the *subject* of a violation
    # (i.e. `... arg 'priority' not in schema ...`), not just appear in
    # the printed valid-args list.
    assert any("arg 'priority'" in v for v in violations), violations
    # The valid field on the same call must not produce a violation
    # whose subject is itself.
    assert not any("arg 'resolution_summary'" in v for v in violations), violations


def test_linter_detects_legacy_findings_underscore(linter, tmp_path: Path):
    """The deprecated ``findings_<agent>`` form must surface as a violation."""
    tools = textwrap.dedent("""
        class DemoServer:
            def __init__(self):
                self.mcp.tool(name="update_incident")(self._tool_update_incident)

            async def _tool_update_incident(self, incident_id, patch):
                ...
    """)
    # Simulate the typed-patch class so the patch_fields discovery has work
    # to do — the linter relies on its presence to decide whether to flag
    # the underscore form.
    patch_model = textwrap.dedent("""
        class UpdateIncidentPatch:
            severity: str | None = None
            category: str | None = None
            findings: dict | None = None
    """)
    prompt = "Set the trail under `findings_triage` on the next call."
    _build_example_tree(
        tmp_path, tools_module=tools, prompt=prompt, patch_model=patch_model,
    )
    # Adjust PATCH_MODELS to point to the synthesized file for this test.
    original = linter.PATCH_MODELS.copy()
    try:
        linter.PATCH_MODELS["update_incident"] = (
            "examples/demo_app/mcp_server.py", "UpdateIncidentPatch",
        )
        schemas = linter.discover_tools(tmp_path / "examples")
        patch_fields = linter.discover_patch_fields(tmp_path)
        prompt_path = tmp_path / "examples" / "demo_app" / "skills" / "x" / "system.md"
        violations = linter.lint_prompt(prompt_path, schemas, patch_fields)
    finally:
        linter.PATCH_MODELS.clear()
        linter.PATCH_MODELS.update(original)
    assert any("findings_triage" in v for v in violations), violations


def test_linter_honors_lint_ignore_directive(linter, tmp_path: Path):
    """A negative example tagged with ``<!-- lint-ignore -->`` must not flag."""
    tools = textwrap.dedent("""
        class DemoServer:
            def __init__(self):
                self.mcp.tool(name="update_incident")(self._tool_update_incident)

            async def _tool_update_incident(self, incident_id, patch):
                ...
    """)
    patch_model = textwrap.dedent("""
        class UpdateIncidentPatch:
            findings: dict | None = None
    """)
    prompt = "Do NOT pass `findings_triage` to update_incident. <!-- lint-ignore: negative example -->"
    _build_example_tree(
        tmp_path, tools_module=tools, prompt=prompt, patch_model=patch_model,
    )
    original = linter.PATCH_MODELS.copy()
    try:
        linter.PATCH_MODELS["update_incident"] = (
            "examples/demo_app/mcp_server.py", "UpdateIncidentPatch",
        )
        schemas = linter.discover_tools(tmp_path / "examples")
        patch_fields = linter.discover_patch_fields(tmp_path)
        prompt_path = tmp_path / "examples" / "demo_app" / "skills" / "x" / "system.md"
        violations = linter.lint_prompt(prompt_path, schemas, patch_fields)
    finally:
        linter.PATCH_MODELS.clear()
        linter.PATCH_MODELS.update(original)
    assert violations == [], f"lint-ignore should suppress the violation: {violations}"


def test_linter_skips_session_injected_args(linter, tmp_path: Path):
    """Phase 9 session-injected args (``incident_id``, ``environment``,
    ``session_id``) must not be flagged when prose names them — the LLM
    can't pass them but the prompt may legitimately reference them by name."""
    tools = textwrap.dedent("""
        class DemoServer:
            def __init__(self):
                self.mcp.tool(name="get_logs")(self._tool_get_logs)

            async def _tool_get_logs(self, service, environment, minutes):
                ...
    """)
    prompt = "Call `get_logs(service, environment, minutes=15)`. The framework injects environment."
    _build_example_tree(tmp_path, tools_module=tools, prompt=prompt)
    schemas = linter.discover_tools(tmp_path / "examples")
    patch_fields = linter.discover_patch_fields(tmp_path)
    prompt_path = tmp_path / "examples" / "demo_app" / "skills" / "x" / "system.md"
    violations = linter.lint_prompt(prompt_path, schemas, patch_fields)
    # All three args (service, environment, minutes) are on the signature
    # OR in the SESSION_INJECTED set — none should produce a violation.
    assert violations == [], (
        f"session-injected + on-signature args should pass: {violations}"
    )


def test_linter_handles_malformed_call_blocks(linter, tmp_path: Path):
    """Malformed inline calls must be tolerated — no crash, no false hits."""
    tools = textwrap.dedent("""
        class DemoServer:
            def __init__(self):
                self.mcp.tool(name="get_logs")(self._tool_get_logs)

            async def _tool_get_logs(self, service, environment, minutes):
                ...
    """)
    prompt = textwrap.dedent("""
        These should NOT crash the linter:

        - Empty call: `get_logs()`
        - Trailing comma: `get_logs(service,)`
        - Stray text: `get_logs(some prose with spaces and ,, double commas)`
        - Not a tool call: `range(10)` is fine.
    """)
    _build_example_tree(tmp_path, tools_module=tools, prompt=prompt)
    schemas = linter.discover_tools(tmp_path / "examples")
    patch_fields = linter.discover_patch_fields(tmp_path)
    prompt_path = tmp_path / "examples" / "demo_app" / "skills" / "x" / "system.md"
    # Should not raise.
    violations = linter.lint_prompt(prompt_path, schemas, patch_fields)
    # ``range`` isn't a discovered tool so it's silently skipped.
    assert not any("range" in v for v in violations), violations


def test_linter_main_entrypoint_exits_zero_on_clean_tree(linter):
    """Exercises ``main()`` end-to-end — what CI invokes."""
    rc = linter.main(
        [
            "--examples-root", str(REPO_ROOT / "examples"),
            "--repo-root", str(REPO_ROOT),
            "--quiet",
        ]
    )
    assert rc == 0, "linter must exit 0 on the live tree (CI gate guarantee)"
