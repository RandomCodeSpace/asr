"""Phase 16 (BUNDLER-01): defensive ratchet on RUNTIME_MODULE_ORDER.

Walks every ``src/runtime/**/*.py`` module and asserts each one is either
present in :data:`scripts.build_single_file.RUNTIME_MODULE_ORDER` or
explicitly listed in ``_INTENTIONAL_EXCLUSIONS`` below. This catches the
class of bug Phase 16 was created to fix: a new ``src/runtime`` module
shipped without a corresponding bundler entry, leaving the deploy bundle
silently missing the symbol it provides until the operator hits an
``ImportError`` at deploy time.

If you add a new ``src/runtime/*.py``:
  - Add a tuple ``(RUNTIME_ROOT, "<relpath>")`` to ``RUNTIME_MODULE_ORDER``
    in ``scripts/build_single_file.py`` at the correct topological position
    (after every module it imports at the top of file).
  - Regenerate the bundles: ``python scripts/build_single_file.py``.
  - Commit the regenerated ``dist/*`` so the CI staleness gate stays green.

If you genuinely don't want the module bundled (e.g. a CLI entry point or
a separately-bundled UI), add it to ``_INTENTIONAL_EXCLUSIONS`` with a
one-line comment explaining why.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_ROOT = _REPO_ROOT / "src" / "runtime"

# Modules under src/runtime that are deliberately NOT in RUNTIME_MODULE_ORDER.
# Every entry needs a justification — the test fails closed if a new
# unlisted module appears.
_INTENTIONAL_EXCLUSIONS: dict[str, str] = {
    # __main__.py is the python -m runtime entry point; the bundle is
    # imported as a flat module, so an entry guard is not needed.
    "__main__.py": "module entry point — not used by bundle consumers",
    # ui.py is built into a separate dist/ui.py bundle by build_ui();
    # bundling it into dist/app.py would duplicate symbols.
    "ui.py": "bundled separately as dist/ui.py via build_ui()",
}


def _load_runtime_module_order() -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "build_single_file",
        _REPO_ROOT / "scripts" / "build_single_file.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {rel for (_root, rel) in mod.RUNTIME_MODULE_ORDER}


def _enumerate_runtime_modules() -> list[str]:
    """All .py files under src/runtime/, relative to src/runtime, no __init__."""
    found: list[str] = []
    for p in sorted(_RUNTIME_ROOT.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        found.append(p.relative_to(_RUNTIME_ROOT).as_posix())
    return found


def test_every_runtime_module_is_bundled_or_excluded() -> None:
    """Every src/runtime/*.py is either in RUNTIME_MODULE_ORDER or excluded."""
    order = _load_runtime_module_order()
    actual = _enumerate_runtime_modules()

    missing: list[str] = []
    for rel in actual:
        if rel in order:
            continue
        if rel in _INTENTIONAL_EXCLUSIONS:
            continue
        missing.append(rel)

    if missing:
        bullet_list = "\n".join(f"  - {m}" for m in missing)
        pytest.fail(
            "src/runtime/*.py modules NOT in RUNTIME_MODULE_ORDER (and not in "
            "_INTENTIONAL_EXCLUSIONS):\n"
            f"{bullet_list}\n\n"
            "Either add each one to RUNTIME_MODULE_ORDER in "
            "scripts/build_single_file.py at the correct topological "
            "position, OR add it to _INTENTIONAL_EXCLUSIONS in "
            "tests/test_bundle_completeness.py with a justification.\n"
            "After bundling, regenerate: python scripts/build_single_file.py"
        )


def test_intentional_exclusions_actually_exist() -> None:
    """Every entry in _INTENTIONAL_EXCLUSIONS must point to a real file —
    catches stale exclusions left behind after a rename or delete."""
    actual = set(_enumerate_runtime_modules())
    stale = [k for k in _INTENTIONAL_EXCLUSIONS if k not in actual]
    assert not stale, (
        f"Stale entries in _INTENTIONAL_EXCLUSIONS — file no longer "
        f"exists at src/runtime/: {stale}"
    )


def test_runtime_module_order_paths_actually_exist() -> None:
    """RUNTIME_MODULE_ORDER must reference only files that exist on disk."""
    order = _load_runtime_module_order()
    missing = [rel for rel in order if not (_RUNTIME_ROOT / rel).exists()]
    assert not missing, (
        f"RUNTIME_MODULE_ORDER references non-existent files: {missing}"
    )
