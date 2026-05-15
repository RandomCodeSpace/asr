"""Build deployment bundles from src/runtime/ + examples/{incident_management,code_review}/.

Four artifacts are produced:

- ``dist/app.py`` — framework-only bundle (``src/runtime/`` flattened).
  No ``examples.*`` source code; intra-bundle imports pointing at any
  example package are stripped, leaving the runtime to look up example
  symbols dynamically. Framework is generic; apps live in
  ``examples/``.
- ``dist/apps/incident-management.py`` — self-contained incident-management
  app bundle: runtime modules + ``examples/incident_management/{config,state,
  mcp_server}.py``. Ship target.
- ``dist/apps/code-review.py`` — self-contained code-review app bundle
  with the same runtime modules + ``examples/code_review/{config,state,
  mcp_server}.py``. Demonstrates that the framework is generic by
  producing a second app bundle from the same runtime.
- ``dist/ui.py`` — Streamlit UI only; ``examples/incident_management/ui.py``
  with intra-bundle ``from {orchestrator,runtime,examples.*}`` imports
  rewritten to ``from app import ...``.

Layout choice: the user's deployment workflow is a 7-file copy-only
payload. Historically a single ``dist/app.py`` (framework + app combined)
and ``dist/ui.py``. Now the framework and the example are split, so we
emit a runtime-only ``dist/app.py`` for the framework story AND a fully
self-contained ``dist/apps/incident-management.py`` for the deployment
story. Deploy-time contract: copy ``dist/apps/incident-management.py``
as ``app.py`` into the corporate env (along with ``dist/ui.py`` and the
config tail).

Run: python scripts/build_single_file.py
"""
from __future__ import annotations
import re
from pathlib import Path

RUNTIME_ROOT = Path("src/runtime")
EXAMPLES_ROOT = Path("examples")
# The generic UI lives at src/runtime/ui.py — config-driven badges
# and detail fields keep the shell domain-agnostic so a single bundle
# can ship beside any app entry-point.
UI = Path("src/runtime/ui.py")
OUT_APP = Path("dist/app.py")
OUT_UI = Path("dist/ui.py")
OUT_INCIDENT_APP = Path("dist/apps/incident-management.py")
# Second app bundle, demonstrating the framework is genuinely generic.
OUT_CODE_REVIEW_APP = Path("dist/apps/code-review.py")

# Order matters — emit modules in dependency order.
# Canonical module bodies live under src/runtime/; the bundler reads
# from there for runtime modules. examples/incident_management/* modules
# are included only in the incident-management app bundle (not in the
# runtime-only bundle).
RUNTIME_MODULE_ORDER: list[tuple[Path, str]] = [
    # Phase 13 (HARD-01/HARD-05): typed runtime errors. Leaf module
    # (no runtime.* imports). MUST precede config.py because
    # config.py imports LLMConfigError for the ProviderConfig
    # @model_validator (D-13-05/06).
    (RUNTIME_ROOT, "errors.py"),
    # Phase 16 (BUNDLER-01): generic terminal-tool registry types
    # (StatusDef, TerminalToolRule). Imported at the top of config.py
    # (line 10), so MUST precede config.py — otherwise the bundled
    # config.py raises NameError at module-execution time.
    (RUNTIME_ROOT, "terminal_tools.py"),
    (RUNTIME_ROOT, "config.py"),
    (RUNTIME_ROOT, "state.py"),
    (RUNTIME_ROOT, "state_resolver.py"),
    (RUNTIME_ROOT, "similarity.py"),
    (RUNTIME_ROOT, "skill.py"),
    (RUNTIME_ROOT, "llm.py"),
    (RUNTIME_ROOT, "storage/models.py"),
    (RUNTIME_ROOT, "storage/engine.py"),
    (RUNTIME_ROOT, "storage/embeddings.py"),
    (RUNTIME_ROOT, "storage/vector.py"),
    (RUNTIME_ROOT, "storage/history_store.py"),
    (RUNTIME_ROOT, "storage/session_store.py"),
    # Phase 16 (BUNDLER-01): event-log + idempotent migrations. Both
    # depend only on storage/models.py (already above). event_log is
    # required by orchestrator.py's status finalizer; migrations is
    # invoked at startup (storage/__init__.py wires it but __init__
    # files aren't bundled, so the orchestrator path is the surviving
    # caller).
    (RUNTIME_ROOT, "storage/event_log.py"),
    (RUNTIME_ROOT, "storage/migrations.py"),
    # M5 (per-step telemetry): lesson corpus store + auto-extractor.
    # lesson_store depends on storage/vector.py (already above) and
    # storage/models.py for SessionLessonRow. Bundled before
    # orchestrator.py so it can instantiate the store at boot.
    (RUNTIME_ROOT, "storage/lesson_store.py"),
    (RUNTIME_ROOT, "learning/extractor.py"),
    # M7: nightly lesson refresher (APScheduler cron). Depends on
    # extractor + lesson_store (both above).
    (RUNTIME_ROOT, "learning/scheduler.py"),
    # NOTE: the per-tool mcp_server modules
    # (observability/remediation/user_context) were relocated under
    # ``examples/incident_management/mcp_servers/`` in Phase 7
    # (DECOUPLE-04 / D-07-01). They no longer live under
    # ``src/runtime/`` and are bundled into the incident-management app
    # via ``INCIDENT_APP_MODULE_ORDER`` below — NOT into the
    # framework-only ``dist/app.py`` bundle. ``dist/apps/code-review.py``
    # consequently boots without any incident-vocabulary MCP servers
    # (its ``orchestrator.mcp_servers`` list is empty).
    (RUNTIME_ROOT, "mcp_loader.py"),
    # Phase 16 (BUNDLER-01): long-lived OrchestratorService — the
    # Streamlit UI's `from app import OrchestratorService` import is
    # the headline ImportError this phase fixes. Depends only on
    # config.py and mcp_loader.py (both above). Lazy-imports
    # tools.approval_watchdog at start-up (added below).
    (RUNTIME_ROOT, "service.py"),
    # Phase 10 (FOC-03): AgentTurnOutput envelope + EnvelopeMissingError.
    # Phase 12 (FOC-05) bundles policy.py with a module-level reference
    # to EnvelopeMissingError in _PERMANENT_TYPES, so turn_output MUST
    # precede policy.py in the bundle. (Pre-Phase-12 dists referenced
    # EnvelopeMissingError only inside function bodies, where the strip-
    # plus-rebuild order didn't surface a NameError at import time.)
    (RUNTIME_ROOT, "agents/turn_output.py"),
    # Phase 16 (BUNDLER-01): risk-rated tool gateway. Imported at
    # module level by policy.py, graph.py, agents/responsive.py — so
    # gateway.py MUST precede policy.py. Depends only on config.py +
    # state.py (both already above). arg_injection is its sibling and
    # is lazy-imported from gateway / orchestrator / graph.
    (RUNTIME_ROOT, "tools/gateway.py"),
    (RUNTIME_ROOT, "tools/arg_injection.py"),
    # Phase 16 (BUNDLER-01): pending-approval timeout watchdog,
    # lazy-imported by service.py:189. Bundled here (after gateway, so
    # gateway-related approval state is in scope) but before any module
    # that might trigger the lazy import path.
    (RUNTIME_ROOT, "tools/approval_watchdog.py"),
    # Phase 11 (FOC-04): pure-policy HITL gating boundary. Imported by
    # tools.gateway, which graph.py uses -- so policy.py must precede
    # graph.py in the bundle.
    (RUNTIME_ROOT, "policy.py"),
    # Phase 16 (BUNDLER-01): agent-kind node builders, used by graph.py
    # at construction time. Each depends on skill.py + state.py (both
    # already above) and on gateway.py / turn_output.py / session_store.py
    # for responsive. Bundled BEFORE graph.py so the symbols are in
    # module scope when graph.py's body executes.
    (RUNTIME_ROOT, "agents/responsive.py"),
    (RUNTIME_ROOT, "agents/supervisor.py"),
    (RUNTIME_ROOT, "agents/monitor.py"),
    (RUNTIME_ROOT, "graph.py"),
    (RUNTIME_ROOT, "checkpointer_postgres.py"),
    (RUNTIME_ROOT, "checkpointer.py"),
    # Trigger registry — bundled before orchestrator.py / api.py so
    # ``TriggerInfo`` and ``TriggerRegistry`` are already in module
    # scope when those modules' bodies execute.
    (RUNTIME_ROOT, "triggers/base.py"),
    (RUNTIME_ROOT, "triggers/config.py"),
    (RUNTIME_ROOT, "triggers/resolve.py"),
    (RUNTIME_ROOT, "triggers/idempotency.py"),
    (RUNTIME_ROOT, "triggers/auth.py"),
    (RUNTIME_ROOT, "triggers/transports/api.py"),
    (RUNTIME_ROOT, "triggers/transports/webhook.py"),
    (RUNTIME_ROOT, "triggers/transports/schedule.py"),
    (RUNTIME_ROOT, "triggers/transports/plugin.py"),
    (RUNTIME_ROOT, "triggers/registry.py"),
    # Dedup pipeline — bundled before orchestrator so the symbols
    # (DedupConfig, DedupPipeline) are already module-scoped when
    # orchestrator.py executes its imports.
    (RUNTIME_ROOT, "dedup.py"),
    # 2026-05-03: framework intake runner — used as the default Skill.runner
    # for kind=supervisor skills. Bundled before orchestrator/api/skill so
    # the dotted-path resolver finds default_intake_runner.
    (RUNTIME_ROOT, "intake.py"),
    # Generic memory layers (ASR L2/L5/L7). Lifted from
    # examples/incident_management/asr in Wave 1 of the strip-down.
    # ``session_state`` (L2/L5/L7 pydantic slots) is referenced by the
    # three stores so it must come first.
    (RUNTIME_ROOT, "memory/session_state.py"),
    (RUNTIME_ROOT, "memory/knowledge_graph.py"),
    (RUNTIME_ROOT, "memory/release_context.py"),
    (RUNTIME_ROOT, "memory/playbook_store.py"),
    (RUNTIME_ROOT, "memory/hypothesis.py"),
    (RUNTIME_ROOT, "memory/resolution.py"),
    # Per-session task-reentrant asyncio locks + SessionBusy exception.
    # Must precede orchestrator.py which instantiates SessionLockRegistry.
    (RUNTIME_ROOT, "locks.py"),
    # Phase 16 (BUNDLER-01): load-time skill validator + checkpoint GC.
    # Both lazy-imported from orchestrator.py (lines 447, 472). Bundled
    # before orchestrator.py so the lazy import resolves to in-bundle
    # symbols rather than failing with ModuleNotFoundError after the
    # intra-import stripper removes the original `from runtime.X` line.
    (RUNTIME_ROOT, "skill_validator.py"),
    (RUNTIME_ROOT, "storage/checkpoint_gc.py"),
    (RUNTIME_ROOT, "orchestrator.py"),
    (RUNTIME_ROOT, "api.py"),
    # Retraction routes are a side-car router so they don't bloat
    # api.py. Bundled after api.py so register_dedup_routes can be
    # invoked against the FastAPI app at the bottom of the bundle.
    (RUNTIME_ROOT, "api_dedup.py"),
    # Bootstrap bundle endpoint — single round-trip the React UI hits
    # on session open. Side-car module mounted on api_v1 inside
    # ``api.build_app``; bundled after api.py for the same reason as
    # api_dedup.py.
    (RUNTIME_ROOT, "api_session_full.py"),
    # UI hints endpoint — read once at React boot for the topbar brand
    # block, env switcher list, and approval-rationale dropdown. Same
    # side-car pattern as api_session_full.py.
    (RUNTIME_ROOT, "api_ui_hints.py"),
    # App-overlay UI views endpoint — Approach C extensibility surface
    # the framework UI's Selected-detail panel queries to render
    # "App-specific views →" links. Same side-car pattern.
    (RUNTIME_ROOT, "api_apps_overlay.py"),
]

# Example app modules — flattened *after* the runtime modules in the
# incident-management app bundle. Order matters: config and state must
# precede mcp_server / ui which import from them.
#
# All memory-layer foundations (memory_state, KG/Release/Playbook
# stores, hypothesis_loop, resolution_helpers) were lifted to
# ``runtime.memory`` in Wave 1 of the strip-down and are bundled in
# RUNTIME_MODULE_ORDER above. The per-app supervisor runner that wires
# the framework helpers to incident_management's stores + active-session
# lookup now lives inside ``mcp_server.py``.
INCIDENT_APP_MODULE_ORDER: list[tuple[Path, str]] = [
    # state.py — pydantic ``IncidentStateOverrides`` schema
    # (DECOUPLE-05 / D-08-01). Bundled FIRST so any later module's
    # ``from examples.incident_management.state import …`` resolves
    # against the in-bundle definition (Phase 8).
    (EXAMPLES_ROOT, "incident_management/state.py"),
    # Per-tool MCP servers — relocated under
    # ``examples/incident_management/`` in Phase 7 (DECOUPLE-04 /
    # D-07-01). Bundled before mcp_server.py so the dotted-path
    # discovery loop in orchestrator.py (``cfg.orchestrator.mcp_servers``)
    # imports them out of the bundle without falling back to the
    # now-deleted framework-internal location.
    (EXAMPLES_ROOT, "incident_management/mcp_servers/observability.py"),
    (EXAMPLES_ROOT, "incident_management/mcp_servers/remediation.py"),
    (EXAMPLES_ROOT, "incident_management/mcp_servers/user_context.py"),
    (EXAMPLES_ROOT, "incident_management/mcp_server.py"),
]

# Code-review app modules — same shape as the incident bundle but
# pulls from ``examples/code_review/`` so a second self-contained app
# bundle drops out of the same build script. Per-app ``config.py``
# files were removed in the framework/dedup/environments YAML
# migration; cross-cutting knobs now live on AppConfig directly.
CODE_REVIEW_APP_MODULE_ORDER: list[tuple[Path, str]] = [
    # state.py — pydantic ``CodeReviewStateOverrides`` schema
    # (DECOUPLE-05 / D-08-01). Bundled FIRST so any later module's
    # ``from examples.code_review.state import …`` resolves against
    # the in-bundle definition (Phase 8).
    (EXAMPLES_ROOT, "code_review/state.py"),
    (EXAMPLES_ROOT, "code_review/mcp_server.py"),
]

# Matches both single-line and parenthesized multi-line intra-package imports.
# Intra-bundle prefixes — symbols from these modules land in the bundle and
# their import lines must be stripped to avoid `ModuleNotFoundError` at import
# time (the bundle is `app.py` standalone, not a package). The
# ``examples.code_review`` prefix is included so its intra-package
# imports also get stripped from the code-review bundle.
_INTRA_PREFIXES = (
    r"(?:orchestrator|runtime|examples\.incident_management|examples\.code_review)"
)
INTRA_IMPORT_RE = re.compile(
    rf"^\s*from\s+{_INTRA_PREFIXES}(\.[\w.]+)?\s+import\s+"
    r"(?:\([^)]*\)|.*?)$",
    re.MULTILINE | re.DOTALL,
)
# Also strip ``import X`` and ``import X as Y`` forms for intra-bundle modules.
INTRA_IMPORT_NAME_RE = re.compile(
    rf"^\s*import\s+{_INTRA_PREFIXES}(\.[\w.]+)?(?:\s+as\s+\w+)?\s*$",
    re.MULTILINE,
)
PACKAGE_INIT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$", re.MULTILINE)
# For UI rewriting: capture the symbols, handling parenthesized form.
# Matches every intra-bundle prefix the UI may import from:
# legacy ``orchestrator.X`` plus the canonical ``runtime.X`` and
# ``examples.incident_management.X`` paths.
_FROM_ORCH_RE = re.compile(
    rf"^\s*from\s+{_INTRA_PREFIXES}(?:\.[\w.]+)?\s+import\s+"
    r"(?:\(([^)]*)\)|(.*?))\s*$",
    re.MULTILINE | re.DOTALL,
)


def _read(path: Path) -> str:
    return path.read_text()


# Phase 16 (BUNDLER-01): after stripping intra-imports, ``if TYPE_CHECKING:``
# blocks whose only body line was a ``from runtime.X import Y`` end up as a
# naked ``if`` with no suite — IndentationError at module load. Neutralize
# any orphaned ``if TYPE_CHECKING:`` (followed by blank lines and then a
# dedented top-level statement) by giving it a ``pass`` body. We only target
# top-level ``if TYPE_CHECKING:`` (no leading whitespace) because nested
# guards are rare in this codebase and a wider rewrite risks corrupting
# function-body conditionals.
# NOTE: the inner alternation uses ``[ \t]*\n`` (NOT ``\s*\n``).
# Using ``\s`` would let the inner pattern match the newline anchor
# itself, making ``(\s*\n)*`` a textbook polynomial-backtracking
# trap on long blank-line runs (CodeQL py/redos). ``[ \t]*\n``
# matches exactly one blank-line per iteration with no overlap, so
# the engine takes O(n).
_ORPHANED_TYPE_CHECKING_RE = re.compile(
    r"^if\s+TYPE_CHECKING\s*:\s*\n([ \t]*\n)*(?=\S)",
    re.MULTILINE,
)


def _strip_intra_imports(src: str) -> str:
    src = INTRA_IMPORT_RE.sub("", src)
    src = INTRA_IMPORT_NAME_RE.sub("", src)
    src = _ORPHANED_TYPE_CHECKING_RE.sub("if TYPE_CHECKING:\n    pass\n", src)
    return src


def _rewrite_intra_imports_for_ui(src: str) -> str:
    """Replace all `from orchestrator(.X)? import A, B` with `from app import A, B`.

    Collects all symbols from every such import line, deduplicates, and emits
    a single consolidated `from app import ...` line just after the
    `from __future__ import annotations` line (or at the top if absent).
    """
    symbols: set[str] = set()

    def _collect(m: re.Match) -> str:
        raw = m.group(1) if m.group(1) is not None else (m.group(2) or "")
        for s in raw.split(","):
            s = s.strip()
            if s:
                symbols.add(s)
        return ""  # remove the original line

    src = _FROM_ORCH_RE.sub(_collect, src)
    if symbols:
        emit = "from app import " + ", ".join(sorted(symbols))
        future_pattern = re.compile(
            r"^(from\s+__future__\s+import\s+.*)$", re.MULTILINE
        )
        if future_pattern.search(src):
            src = future_pattern.sub(r"\1\n" + emit, src, count=1)
        else:
            src = emit + "\n" + src
    return src


def _split_imports_and_body(src: str) -> tuple[list[str], str]:
    """Return (top-of-file imports, rest).

    Treats multi-line module docstrings as atomic — an opening ``\"\"\"`` on
    one line keeps every subsequent line in *imports* until the closing
    ``\"\"\"`` is seen, even if the body of the docstring contains lines that
    don't otherwise look like imports. Without this, the opening triple-quote
    landed in *imports* while the closing one fell through to *body*, leaving
    the bundled file with an unterminated triple-quote and Python parsing the
    rest of the world as one giant string.
    """
    imports: list[str] = []
    body_lines: list[str] = []
    in_imports = True
    in_docstring = False  # set True between an unmatched """ / ''' pair

    def _odd_triple_quotes(s: str) -> bool:
        return ((s.count('"""') + s.count("'''")) % 2) == 1

    for line in src.splitlines():
        if in_docstring:
            # Inside a multi-line docstring — every line is import-zone until
            # the matching triple-quote arrives.
            imports.append(line)
            if _odd_triple_quotes(line):
                in_docstring = False
            continue
        stripped = line.strip()
        if in_imports:
            opens_docstring = (stripped.startswith('"""')
                               or stripped.startswith("'''"))
            if (stripped.startswith("import ") or stripped.startswith("from ")
                    or stripped == "" or stripped.startswith("#")
                    or opens_docstring):
                imports.append(line)
                # If the line has an odd number of triple-quotes, it leaves a
                # docstring open across the line boundary.
                if opens_docstring and _odd_triple_quotes(line):
                    in_docstring = True
            else:
                in_imports = False
                body_lines.append(line)
        else:
            body_lines.append(line)
    return imports, "\n".join(body_lines)


def _dedup_and_sort_future(all_imports: list[str]) -> list[str]:
    """Deduplicate imports and hoist __future__ lines to the top."""
    seen: set[str] = set()
    deduped: list[str] = []
    for line in all_imports:
        key = line.strip()
        if key.startswith(("import ", "from __future__", "from ")):
            if key in seen:
                continue
            seen.add(key)
        deduped.append(line)
    future_lines = [ln for ln in deduped if ln.strip().startswith("from __future__")]
    other_lines = [ln for ln in deduped if not ln.strip().startswith("from __future__")]
    return future_lines + other_lines


def _flatten_modules(modules: list[tuple[Path, str]]) -> tuple[list[str], list[str]]:
    """Read, strip intra-imports from, and split each module into imports + body.

    Returns (all_imports, bodies) — the caller dedupes imports and joins.
    """
    all_imports: list[str] = []
    bodies: list[str] = []
    for root, rel in modules:
        path = root / rel
        src = _read(path)
        src = _strip_intra_imports(src)
        future_imports = PACKAGE_INIT_RE.findall(src)
        src = PACKAGE_INIT_RE.sub("", src)
        imports, body = _split_imports_and_body(src)
        label = f"{root.name}/{rel}"
        all_imports.append(f"# ----- imports for {label} -----")
        all_imports.extend(imports)
        for fut in future_imports:
            all_imports.insert(0, fut)
        bodies.append(f"\n# ====== module: {label} ======\n")
        bodies.append(body)
    return all_imports, bodies


def build_runtime_app() -> None:
    """Build dist/app.py — framework only (src/runtime/ flattened).

    Excludes ``examples/incident_management/*`` modules. The runtime references
    the example MCP server by string (``examples.incident_management.mcp_server``)
    for dynamic import, so the path string survives in the bundle, but no
    example *source code* is inlined.
    """
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    all_imports, bodies = _flatten_modules(RUNTIME_MODULE_ORDER)
    final_imports = _dedup_and_sort_future(all_imports)
    OUT_APP.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(f"wrote {OUT_APP} ({OUT_APP.stat().st_size:,} bytes)")


def build_incident_app() -> None:
    """Build dist/apps/incident-management.py — runtime + incident app.

    Self-contained app bundle: runtime modules followed by the
    ``examples/incident_management/{config,state,mcp_server}.py`` modules,
    all flattened with intra-bundle imports stripped. This is the deployment
    ship target — copied to the corporate env (typically renamed to
    ``app.py``) along with ``dist/ui.py``.
    """
    OUT_INCIDENT_APP.parent.mkdir(parents=True, exist_ok=True)
    # Runtime modules must come first so the example modules can reference
    # framework symbols already in scope.
    runtime_imports, runtime_bodies = _flatten_modules(RUNTIME_MODULE_ORDER)
    app_imports, app_bodies = _flatten_modules(INCIDENT_APP_MODULE_ORDER)
    all_imports = runtime_imports + app_imports
    bodies = runtime_bodies + app_bodies
    final_imports = _dedup_and_sort_future(all_imports)
    OUT_INCIDENT_APP.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(
        f"wrote {OUT_INCIDENT_APP} ({OUT_INCIDENT_APP.stat().st_size:,} bytes)"
    )


def build_code_review_app() -> None:
    """Build dist/apps/code-review.py — runtime + code-review app.

    Self-contained app bundle. Same runtime modules as
    ``dist/apps/incident-management.py`` (loaded first so example
    modules can reference framework symbols), then
    ``examples/code_review/{config,state,mcp_server}.py`` flattened
    with intra-bundle imports stripped.
    """
    OUT_CODE_REVIEW_APP.parent.mkdir(parents=True, exist_ok=True)
    runtime_imports, runtime_bodies = _flatten_modules(RUNTIME_MODULE_ORDER)
    app_imports, app_bodies = _flatten_modules(CODE_REVIEW_APP_MODULE_ORDER)
    all_imports = runtime_imports + app_imports
    bodies = runtime_bodies + app_bodies
    final_imports = _dedup_and_sort_future(all_imports)
    OUT_CODE_REVIEW_APP.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(
        f"wrote {OUT_CODE_REVIEW_APP} "
        f"({OUT_CODE_REVIEW_APP.stat().st_size:,} bytes)"
    )


def build_ui() -> None:
    """Build dist/ui.py — Streamlit UI only; imports from sibling app module.

    Designed to sit alongside ``dist/apps/incident-management.py`` (renamed
    to ``app.py`` at deploy time), since the UI imports app-config symbols
    (``load_app_config``, ``framework_app_config_provider``) that live in
    the example app, not the framework runtime.
    """
    OUT_UI.parent.mkdir(parents=True, exist_ok=True)
    all_imports: list[str] = []
    bodies: list[str] = []

    ui_src = _read(UI)
    # Rewrite `from orchestrator.X import Y` → `from app import Y`
    # (must happen before stripping __future__ so the future line anchor works)
    ui_src = _rewrite_intra_imports_for_ui(ui_src)
    future_ui = PACKAGE_INIT_RE.findall(ui_src)
    ui_src = PACKAGE_INIT_RE.sub("", ui_src)
    ui_imports, ui_body = _split_imports_and_body(ui_src)
    all_imports.append(f"# ----- imports for {UI.as_posix()} -----")
    all_imports.extend(ui_imports)
    for fut in future_ui:
        all_imports.insert(0, fut)
    bodies.append(f"\n# ====== module: {UI.as_posix()} ======\n")
    bodies.append(ui_body)

    final_imports = _dedup_and_sort_future(all_imports)
    OUT_UI.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(f"wrote {OUT_UI} ({OUT_UI.stat().st_size:,} bytes)")


def main() -> None:
    # ``build_ui()`` produces the framework-only UI bundle and reads
    # exclusively from ``src/runtime/``, so it stays green even when an
    # example-app source tree is mid-rename. Run it first so dist/ui.py
    # is always emitted regardless of the per-app build outcome.
    build_runtime_app()
    build_ui()
    build_incident_app()
    build_code_review_app()


if __name__ == "__main__":
    main()
