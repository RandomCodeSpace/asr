"""Build two deployment bundles from src/orchestrator/ + ui/streamlit_app.py.

- dist/app.py  — orchestrator core + FastAPI service (uvicorn --factory)
- dist/ui.py   — Streamlit UI only; rewrites `from orchestrator.X import Y`
                  to `from app import Y` (sibling module reference)

Run: python scripts/build_single_file.py
"""
from __future__ import annotations
import re
from pathlib import Path

SRC_ROOT = Path("src/orchestrator")
UI = Path("ui/streamlit_app.py")
OUT_APP = Path("dist/app.py")
OUT_UI = Path("dist/ui.py")

# Order matters — emit modules in dependency order.
CORE_MODULE_ORDER = [
    "config.py",
    "incident.py",
    "similarity.py",
    "skill.py",
    "llm.py",
    "storage/models.py",
    "storage/engine.py",
    "storage/embeddings.py",
    "storage/vector.py",
    "storage/repository.py",
    "mcp_servers/incident.py",
    "mcp_servers/observability.py",
    "mcp_servers/remediation.py",
    "mcp_servers/user_context.py",
    "mcp_loader.py",
    "graph.py",
    "orchestrator.py",
    "api.py",
]

# Matches both single-line and parenthesized multi-line intra-package imports.
INTRA_IMPORT_RE = re.compile(
    r"^\s*from\s+orchestrator(\.[\w.]+)?\s+import\s+"
    r"(?:\([^)]*\)|.*?)$",
    re.MULTILINE | re.DOTALL,
)
PACKAGE_INIT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$", re.MULTILINE)
# For UI rewriting: capture the symbols, handling parenthesized form.
_FROM_ORCH_RE = re.compile(
    r"^\s*from\s+orchestrator(?:\.[\w.]+)?\s+import\s+"
    r"(?:\(([^)]*)\)|(.*?))\s*$",
    re.MULTILINE | re.DOTALL,
)


def _read(path: Path) -> str:
    return path.read_text()


def _strip_intra_imports(src: str) -> str:
    return INTRA_IMPORT_RE.sub("", src)


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
    future_lines = [l for l in deduped if l.strip().startswith("from __future__")]
    other_lines = [l for l in deduped if not l.strip().startswith("from __future__")]
    return future_lines + other_lines


def build_app() -> None:
    """Build dist/app.py — orchestrator core + FastAPI, no Streamlit."""
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    all_imports: list[str] = []
    bodies: list[str] = []

    for rel in CORE_MODULE_ORDER:
        path = SRC_ROOT / rel
        src = _read(path)
        src = _strip_intra_imports(src)
        future_imports = PACKAGE_INIT_RE.findall(src)
        src = PACKAGE_INIT_RE.sub("", src)
        imports, body = _split_imports_and_body(src)
        all_imports.append(f"# ----- imports for {rel} -----")
        all_imports.extend(imports)
        for fut in future_imports:
            all_imports.insert(0, fut)
        bodies.append(f"\n# ====== module: orchestrator/{rel} ======\n")
        bodies.append(body)

    final_imports = _dedup_and_sort_future(all_imports)
    OUT_APP.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(f"wrote {OUT_APP} ({OUT_APP.stat().st_size:,} bytes)")


def build_ui() -> None:
    """Build dist/ui.py — Streamlit UI only; imports from sibling app module."""
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
    all_imports.append("# ----- imports for ui/streamlit_app.py -----")
    all_imports.extend(ui_imports)
    for fut in future_ui:
        all_imports.insert(0, fut)
    bodies.append("\n# ====== module: ui/streamlit_app.py ======\n")
    bodies.append(ui_body)

    final_imports = _dedup_and_sort_future(all_imports)
    OUT_UI.write_text(
        "\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n"
    )
    print(f"wrote {OUT_UI} ({OUT_UI.stat().st_size:,} bytes)")


def main() -> None:
    build_app()
    build_ui()


if __name__ == "__main__":
    main()
