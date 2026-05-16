"""Package the air-gap deploy payload.

Composes the 'copy-this-folder' release artifact for the v2.0 deploy
shape. Output layout (mirrors :func:`runtime.api_static.mount_static_assets`
defaults so the deployed FastAPI process can find the SPA without env-var
gymnastics):

::

    out/
      app.py        # symlink/copy of dist/apps/incident-management.py
      ui.py         # dist/ui.py (legacy Streamlit; deprecation banner in v2)
      web/          # web/dist/* — built React SPA (Vite output)
        index.html
        assets/...
        fonts/...

Deploy contract:

- Backend reads ``ASR_WEB_DIST`` (set to ``./web`` by the launcher) or
  falls back to ``<bundle>/web/dist`` (legacy dev layout). Either resolves
  to the SPA produced by ``cd web && npm run build``.
- The five YAML configs + ``.env`` ship alongside the bundle out of band.

Run:

::

    cd web && npm ci && npm run build
    uv run python scripts/build_single_file.py    # framework + app .py bundles
    uv run python scripts/package_airgap.py       # composes out/

The output directory is overwritten on each run (no incremental merge).
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
APP_BUNDLE = DIST_DIR / "apps" / "incident-management.py"
UI_BUNDLE = DIST_DIR / "ui.py"
WEB_BUILT = REPO_ROOT / "web" / "dist"


def package(out_dir: Path, *, strict: bool) -> int:
    """Compose the air-gap payload at ``out_dir``. Returns process-exit code."""
    missing: list[str] = []
    if not APP_BUNDLE.is_file():
        missing.append(f"  - {APP_BUNDLE.relative_to(REPO_ROOT)} (run scripts/build_single_file.py first)")
    if not UI_BUNDLE.is_file():
        missing.append(f"  - {UI_BUNDLE.relative_to(REPO_ROOT)} (run scripts/build_single_file.py first)")
    if not (WEB_BUILT / "index.html").is_file():
        missing.append(
            f"  - {WEB_BUILT.relative_to(REPO_ROOT)}/index.html"
            " (run: cd web && npm ci && npm run build)"
        )
    if missing:
        print("✗ air-gap inputs missing:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        if strict:
            return 2

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if APP_BUNDLE.is_file():
        shutil.copy2(APP_BUNDLE, out_dir / "app.py")
    if UI_BUNDLE.is_file():
        shutil.copy2(UI_BUNDLE, out_dir / "ui.py")
    if (WEB_BUILT / "index.html").is_file():
        shutil.copytree(WEB_BUILT, out_dir / "web")

    (out_dir / "README.txt").write_text(
        "ASR v2.0 air-gap deploy payload\n"
        "-------------------------------\n"
        "Contents:\n"
        "  app.py   — flattened framework + incident-management app\n"
        "  ui.py    — legacy Streamlit prototype (deprecated in v2)\n"
        "  web/     — built React SPA, served by uvicorn at /\n\n"
        "Run:\n"
        "  ASR_WEB_DIST=./web uv run uvicorn app:get_app --factory --port 8000\n"
        "or with the host's pinned Python:\n"
        "  ASR_WEB_DIST=./web python -m uvicorn app:get_app --factory --port 8000\n\n"
        "Open: http://<host>:8000/\n"
    )

    print(f"✓ wrote {out_dir}")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            try:
                size = p.stat().st_size
                print(f"  {p.relative_to(out_dir).as_posix():<40} {size:>10,} b")
            except OSError:
                pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the v2.0 air-gap deploy folder.")
    parser.add_argument(
        "--out",
        default="dist/airgap",
        help="Output directory (default: dist/airgap). Wiped each run.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Emit whatever inputs are present instead of failing on missing pieces.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    return package(out_dir, strict=not args.allow_partial)


if __name__ == "__main__":
    raise SystemExit(main())
