"""Concatenate src/orchestrator/**/*.py + ui/streamlit_app.py into dist/app.py.

Rewrites `from orchestrator...` imports by removing them — the symbols are inlined.
External imports (stdlib + 3rd-party) are deduplicated and hoisted to the top.
"""
from __future__ import annotations
import re
from pathlib import Path

SRC_ROOT = Path("src/orchestrator")
UI = Path("ui/streamlit_app.py")
OUT = Path("dist/app.py")

# Order matters — emit modules in dependency order.
MODULE_ORDER = [
    "config.py",
    "incident.py",
    "similarity.py",
    "skill.py",
    "llm.py",
    "mcp_servers/incident.py",
    "mcp_servers/observability.py",
    "mcp_servers/remediation.py",
    "mcp_servers/user_context.py",
    "mcp_loader.py",
    "graph.py",
    "orchestrator.py",
    "api.py",
]

INTRA_IMPORT_RE = re.compile(r"^\s*from\s+orchestrator(\.[\w.]+)?\s+import\s+.*$", re.MULTILINE)
PACKAGE_INIT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text()


def _strip_intra_imports(src: str) -> str:
    return INTRA_IMPORT_RE.sub("", src)


def _split_imports_and_body(src: str) -> tuple[list[str], str]:
    """Return (top-of-file imports, rest)."""
    imports: list[str] = []
    body_lines: list[str] = []
    in_imports = True
    for line in src.splitlines():
        stripped = line.strip()
        if in_imports:
            if (stripped.startswith("import ") or stripped.startswith("from ")
                    or stripped == "" or stripped.startswith("#")
                    or stripped.startswith('"""') or stripped.startswith("'''")):
                imports.append(line)
            else:
                in_imports = False
                body_lines.append(line)
        else:
            body_lines.append(line)
    return imports, "\n".join(body_lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_imports: list[str] = []
    bodies: list[str] = []

    for rel in MODULE_ORDER:
        path = SRC_ROOT / rel
        src = _read(path)
        src = _strip_intra_imports(src)
        # __future__ imports must be first; collect all and emit once at top.
        future_imports = PACKAGE_INIT_RE.findall(src)
        src = PACKAGE_INIT_RE.sub("", src)
        imports, body = _split_imports_and_body(src)
        all_imports.append(f"# ----- imports for {rel} -----")
        all_imports.extend(imports)
        for fut in future_imports:
            all_imports.insert(0, fut)
        bodies.append(f"\n# ====== module: orchestrator/{rel} ======\n")
        bodies.append(body)

    # UI
    ui_src = _strip_intra_imports(_read(UI))
    future_ui = PACKAGE_INIT_RE.findall(ui_src)
    ui_src = PACKAGE_INIT_RE.sub("", ui_src)
    ui_imports, ui_body = _split_imports_and_body(ui_src)
    all_imports.append("# ----- imports for ui/streamlit_app.py -----")
    all_imports.extend(ui_imports)
    for fut in future_ui:
        all_imports.insert(0, fut)
    bodies.append("\n# ====== module: ui/streamlit_app.py ======\n")
    bodies.append(ui_body)

    # Deduplicate external imports while preserving first-seen order.
    seen: set[str] = set()
    deduped: list[str] = []
    for line in all_imports:
        key = line.strip()
        if key.startswith(("import ", "from __future__", "from ")):
            if key in seen:
                continue
            seen.add(key)
        deduped.append(line)

    # __future__ imports must be the very first executable statement; ensure
    # nothing precedes them except an optional module docstring (we don't emit one).
    future_lines = [line for line in deduped if line.strip().startswith("from __future__")]
    other_lines = [line for line in deduped if not line.strip().startswith("from __future__")]
    final_imports = future_lines + other_lines

    OUT.write_text("\n".join(final_imports) + "\n\n" + "\n".join(bodies) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
