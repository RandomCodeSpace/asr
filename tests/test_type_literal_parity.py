from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_literal_values(alias: str) -> set[str]:
    tree = ast.parse((ROOT / "src/runtime/state.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if not any(isinstance(t, ast.Name) and t.id == alias for t in node.targets):
                continue
            value = node.value
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == "Literal"
            ):
                items = value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
                return {
                    item.value
                    for item in items
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
    raise AssertionError(f"literal alias {alias!r} not found")


def _ts_union_values(interface: str, field: str) -> set[str]:
    text = (ROOT / "web/src/api/types.ts").read_text()
    match = re.search(
        rf"export interface {interface} \{{(?P<body>.*?)\n\}}",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, f"interface {interface!r} not found"
    field_match = re.search(rf"\b{field}:\s*(?P<union>[^;]+);", match.group("body"))
    assert field_match is not None, f"field {interface}.{field} not found"
    return set(re.findall(r"'([^']+)'", field_match.group("union")))


def test_session_status_literals_match_typescript():
    assert _ts_union_values("Session", "status") == _python_literal_values(
        "SessionStatus"
    )


def test_tool_call_status_literals_match_typescript():
    assert _ts_union_values("ToolCall", "status") == _python_literal_values(
        "ToolStatus"
    )
