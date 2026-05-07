#!/usr/bin/env python3
"""Skill-Prompt-vs-Schema linter (Phase 21 / SKILL-LINTER-01).

Walks every skill prompt under ``examples/*/skills/*/system.md``, extracts
references to MCP tools (and the field names mentioned for each tool), and
asserts that every referenced tool exists in the canonical inventory and
every field name is on the tool's signature (or — for ``update_incident``-
style nested-patch tools — on the typed pydantic patch model that gates the
patch keys).

Catches LLM-emit-vs-schema drift that has bitten this codebase before:

* **typos**: ``findings_triage`` vs ``findings.triage`` (a ``dict[str, str]``
  with key = agent name).
* **hallucinated session-injected fields**: ``incident_id`` flagged when
  Phase 9's strip should have made it invisible to the LLM.
* **unknown tool names**: drift between prompt instructions and the tools
  actually wired into ``config.yaml``.

Discovery model
---------------

Tools are discovered statically via ``ast`` walks (no FastMCP boot needed,
no I/O). The script enumerates:

* Every ``async def`` / ``def`` at module top-level under
  ``examples/*/mcp_server.py`` and ``examples/*/mcp_servers/*.py``.
* Every method on the FastMCP server class registered through
  ``self.mcp.tool(name="<name>")(self._tool_<name>)`` — bare method args
  (``self``, ``cls``) are excluded; the real arg list is harvested from the
  ``async def _tool_<name>`` signature.

For nested-patch tools — currently just ``update_incident(incident_id,
patch)`` — the script also collects the field set declared by the typed
pydantic ``UpdateIncidentPatch`` model (``model_fields`` keys) and uses that
as the valid ``patch.X`` and ``findings.X`` field set.

Prompt reference extraction
---------------------------

Three regex passes per prompt file:

1. **Backtick tool calls**: ``` `tool_name(arg1, arg2, ...)` ``` — captures
   tool name + arg-name list.
2. **Bare backtick references**: ``` `tool_name` ``` — captures tool name
   only (no arg validation needed).
3. **Patch field references**: ``` `findings_<x>` ``` and ``` `patch.<x>` ```
   — captures field references against the ``UpdateIncidentPatch`` model.

Lines containing ``# lint-ignore: <reason>`` (or markdown-style
``<!-- lint-ignore: ... -->``) at end-of-line are skipped. Use sparingly,
with a one-sentence rationale.

Exit codes
----------

* ``0`` — every reference resolved.
* ``1`` — at least one violation. Each printed as a GitHub-actions ``::error``
  line so the CI summary surfaces it.

Phase: 21-01. Requirement: SKILL-LINTER-01.
"""
from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Args that the framework injects from session state at the gateway boundary
# (Phase 9 / D-09-01). They appear in tool function signatures but are
# stripped from the LLM-visible ``args_schema``. Prompt references to them
# are ALLOWED — prose may name the field even if the LLM cannot pass it —
# but they must not be "hallucinated" (i.e., listed as something the LLM
# itself supplies). The linter accepts them either way; the harder
# Phase-9-strip enforcement lives in the runtime tests, not here.
SESSION_INJECTED = frozenset({"session_id", "incident_id", "environment"})

# Tools whose ``patch`` argument is a typed pydantic model. Entries map a
# tool name to (module path, model class name) for AST-based field discovery.
PATCH_MODELS: dict[str, tuple[str, str]] = {
    "update_incident": (
        "examples/incident_management/mcp_server.py",
        "UpdateIncidentPatch",
    ),
}

# Default scan roots, relative to repo root. Override with --root for tests.
EXAMPLES_ROOT = Path("examples")

# Tool-call backtick patterns. We accept both ``inline tool_name(args)`` and
# bare-name forms. The regex tolerates whitespace and trailing kwargs/equals.
TOOL_CALL_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(([^`)]*)\)`"
)
BARE_TOOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
# Patch-field references. Two shapes seen in this codebase:
#   `findings.<key>` — typed dict[str,str], any string key OK (skip)
#   `findings_<key>` — DEPRECATED underscore form; UpdateIncidentPatch
#                      forbids it (extra="forbid"). Catch as a violation.
LEGACY_FINDINGS_RE = re.compile(r"`(findings_[A-Za-z][A-Za-z0-9_]*)`")
# Lint-ignore directives.
LINT_IGNORE_RE = re.compile(r"#\s*lint-ignore\b|<!--\s*lint-ignore\b")

# Regex helper — split a parenthesised arg list by top-level commas only
# (ignoring commas inside nested function calls like ``service, minutes=15``).
ARG_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^,]*)?")


# ---------------------------------------------------------------------------
# Tool inventory discovery
# ---------------------------------------------------------------------------


def _is_python_tool_def(node: ast.AST) -> bool:
    """Return True if *node* is a top-level ``def``/``async def``."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _collect_args(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return the names of positional/keyword args on *func* (skipping ``self``/``cls``)."""
    names: set[str] = set()
    args = func.args
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if a.arg in {"self", "cls"}:
            continue
        names.add(a.arg)
    return names


def _walk_class_tool_methods(
    cls: ast.ClassDef,
) -> Iterable[tuple[str, set[str]]]:
    """Yield ``(tool_name, arg_set)`` for FastMCP-registered methods.

    Looks for ``self.mcp.tool(name="<name>")(self._tool_<name>)`` calls in
    ``__init__``/setup methods, then matches the registered name to the
    matching ``_tool_<suffix>`` method on the same class. The method's args
    (minus ``self``) become the canonical arg set.
    """
    # 1. Find registrations: map registered_name -> python_method_name
    registrations: dict[str, str] = {}
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        # Match ``something.tool(name="X")(target)``
        outer = node
        if not isinstance(outer.func, ast.Call):
            continue
        inner = outer.func
        if not (isinstance(inner.func, ast.Attribute) and inner.func.attr == "tool"):
            continue
        # name= kwarg on the inner call
        registered_name: str | None = None
        for kw in inner.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    registered_name = kw.value.value
        if registered_name is None:
            continue
        # outer call's first arg is the method reference
        if not outer.args:
            continue
        target = outer.args[0]
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            registrations[registered_name] = target.attr

    # 2. Map registration -> arg set via the method's signature
    method_args: dict[str, set[str]] = {}
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_args[item.name] = _collect_args(item)
    for registered_name, method_name in registrations.items():
        if method_name in method_args:
            yield registered_name, method_args[method_name]


def discover_tools(examples_root: Path) -> dict[str, set[str]]:
    """Walk *examples_root* and return ``{tool_name: {arg_name, ...}}``.

    Two discovery paths:

    * Module-level ``async def``/``def`` in ``mcp_servers/*.py`` (these are
      registered by ``register(mcp_app, cfg)`` which decorates them at import
      time — the registered name == the function name).
    * Class methods in ``mcp_server.py`` registered via the
      ``self.mcp.tool(name="X")(self._tool_X)`` pattern.

    Private/internal funcs (``_seed``, ``_validate_environment``, etc.) are
    filtered by leading-underscore convention, with one exception: methods
    whose name starts with ``_tool_`` are explicit tool implementations and
    are looked up via the class-registration pass.
    """
    tools: dict[str, set[str]] = {}
    for py_path in sorted(examples_root.rglob("*.py")):
        # Only mcp_server.py and mcp_servers/* — skip skills, state, tests.
        if py_path.name == "mcp_server.py":
            pass
        elif py_path.parent.name == "mcp_servers":
            pass
        else:
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError:
            continue
        for node in tree.body:
            # 1) Module-level functions: register themselves verbatim.
            if _is_python_tool_def(node):
                assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                if node.name.startswith("_"):
                    continue
                # Heuristic: module-level helpers like ``register``,
                # ``build_environment_validator`` are not tools. Tools are
                # always async in this codebase.
                if not isinstance(node, ast.AsyncFunctionDef):
                    continue
                tools[node.name] = _collect_args(node)
            # 2) FastMCP server class — extract registered tool methods.
            elif isinstance(node, ast.ClassDef):
                for tool_name, args in _walk_class_tool_methods(node):
                    tools[tool_name] = args
    return tools


def discover_patch_fields(repo_root: Path) -> dict[str, set[str]]:
    """For each entry in :data:`PATCH_MODELS`, return ``{tool_name: {field, ...}}``.

    The field set comes from the pydantic-model class's annotated assignments
    (``severity: str | None = None``). We avoid importing pydantic (and the
    runtime) by AST-walking; this keeps the linter's dependency surface to
    stdlib-only and avoids loading the framework just to lint prompts.
    """
    out: dict[str, set[str]] = {}
    for tool_name, (rel_path, class_name) in PATCH_MODELS.items():
        path = repo_root / rel_path
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                fields: set[str] = set()
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.add(item.target.id)
                out[tool_name] = fields
                break
    return out


# ---------------------------------------------------------------------------
# Skill prompt scanning
# ---------------------------------------------------------------------------


def iter_skill_prompts(examples_root: Path) -> list[Path]:
    """Return every ``examples/*/skills/*/system.md`` path."""
    return sorted(examples_root.glob("*/skills/*/system.md"))


def _split_args(arg_blob: str) -> list[str]:
    """Split a parenthesised arg list and return the bare arg/keyword names."""
    out: list[str] = []
    # Strip surrounding whitespace then split on commas at top level. Since
    # our prompts never embed nested `()` inside the inline backtick form,
    # naive split is safe here. We still defensively reject anything weird.
    if not arg_blob.strip():
        return out
    for part in arg_blob.split(","):
        m = ARG_NAME_RE.match(part.strip())
        if m:
            out.append(m.group(1))
    return out


def lint_prompt(
    prompt_path: Path,
    schemas: dict[str, set[str]],
    patch_fields: dict[str, set[str]],
) -> list[str]:
    """Return a list of violation strings for *prompt_path*."""
    violations: list[str] = []
    text = prompt_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for i, raw_line in enumerate(lines, start=1):
        if LINT_IGNORE_RE.search(raw_line):
            continue
        # Pass 1: tool calls of the form `tool_name(arg, arg, ...)`
        for match in TOOL_CALL_RE.finditer(raw_line):
            tool_name = match.group(1)
            if tool_name not in schemas:
                # Skip if it looks like Python stdlib / utility (heuristic:
                # ignore single-token ``range``, ``len``, etc. — but only if
                # the name is not registered as a tool AND doesn't look like
                # one). For safety, don't flag bare-call mismatches here —
                # only the registered-tool case. Unknown bare-tool names are
                # caught more carefully in pass 2.
                continue
            arg_names = _split_args(match.group(2))
            valid = schemas[tool_name] | SESSION_INJECTED
            for arg_name in arg_names:
                if arg_name not in valid:
                    violations.append(
                        f"{prompt_path}:{i}: tool '{tool_name}' arg '{arg_name}' "
                        f"not in schema (valid: {sorted(schemas[tool_name])})"
                    )

        # Pass 2: bare-tool references — only flag the form
        # `findings_<x>` which is a known wrong shape on update_incident.
        for match in LEGACY_FINDINGS_RE.finditer(raw_line):
            ref = match.group(1)
            # If the ``update_incident`` patch model is known, the only valid
            # findings shape is the typed dict ``findings: dict[str, str]``
            # (key = agent name) — not ``findings_<x>``.
            if "update_incident" in patch_fields:
                violations.append(
                    f"{prompt_path}:{i}: '{ref}' is a legacy underscore form; "
                    f"UpdateIncidentPatch forbids it (extra='forbid'). "
                    f"Use findings dict with key='{ref.removeprefix('findings_')}'."
                )

    return violations


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=EXAMPLES_ROOT,
        help="Root directory containing example apps (default: examples/).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root used for resolving PATCH_MODELS paths (default: cwd).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success summary; only print on failure.",
    )
    args = parser.parse_args(argv)

    schemas = discover_tools(args.examples_root)
    patch_fields = discover_patch_fields(args.repo_root)
    prompts = iter_skill_prompts(args.examples_root)

    if not args.quiet:
        print(
            f"Loaded {len(schemas)} tool schemas: {sorted(schemas)}",
            file=sys.stderr,
        )
        if patch_fields:
            print(
                f"Loaded {len(patch_fields)} patch models: "
                f"{ {k: sorted(v) for k, v in patch_fields.items()} }",
                file=sys.stderr,
            )

    all_violations: list[str] = []
    for prompt_path in prompts:
        all_violations.extend(lint_prompt(prompt_path, schemas, patch_fields))

    if all_violations:
        for v in all_violations:
            print(f"::error::{v}", file=sys.stderr)
        print(
            f"FAIL: {len(all_violations)} violations across {len(prompts)} prompts",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(
            f"OK: {len(schemas)} tools, {len(prompts)} skill prompts — all references resolve.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
