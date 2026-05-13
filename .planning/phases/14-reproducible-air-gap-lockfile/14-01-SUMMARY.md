---
status: completed
phase: 14-reproducible-air-gap-lockfile
plan: 01
subsystem: build / ci / dependencies
tags: [hardening, air-gap, build, ci, lockfile]
requires: [phase-13-llm-provider-hardening]
provides: [uv.lock-CI-install, uv-lock-check-freshness-gate, docs/AIRGAP_INSTALL.md]
affects: [pyproject.toml, .github/workflows/ci.yml, .gitignore, docs/AIRGAP_INSTALL.md, uv.lock]
tech-stack:
  added: [uv (Apache-2.0/MIT, single static binary, Astral)]
  patterns: [pin+hash transitive lockfile, --frozen install, lockfile-drift CI gate]
key-files:
  created:
    - docs/AIRGAP_INSTALL.md
  modified:
    - .github/workflows/ci.yml
    - .gitignore
  unchanged-but-canonical:
    - pyproject.toml         # already PEP 621; no [tool.uv] needed
    - uv.lock                # already in sync (uv lock --check exit 0)
decisions:
  - "Tool: uv 0.11.7 (Apache-2.0/MIT). Picked over pip-tools (loses uv.lock investment, no per-marker pinning) and poetry (would require [project] -> [tool.poetry] rewrite, violates minimal diff)."
  - "uv.lock already exists (171 packages, 4430 lines, in sync per `uv lock --check`); Phase 14 wires CI to install from it, adds the freshness gate, and documents the offline path. No new lockfile generation required."
  - "CI install: `uv sync --frozen --extra dev` (replaces `pip install -e .[dev]`). `--frozen` forbids re-resolving."
  - "CI lockfile-drift gate: `uv lock --check` runs as the FIRST step inside the job (before install) so a stale uv.lock fails the build before anything else."
  - "Tools (ruff, pyright, pytest) run via `uv run` so they execute against the locked virtualenv."
  - "Pinned uv version 0.11.7 in CI (matches local) — bumps are deliberate, not silent."
  - "Documented offline path in `docs/AIRGAP_INSTALL.md` (38 lines): clone -> UV_INDEX_URL=internal-mirror -> `uv sync --frozen [--offline]`. Negation rule added to .gitignore so docs/AIRGAP_INSTALL.md is the single shipped doc."
  - "Single atomic commit per phase precedent (Phase 9-13)."
metrics:
  duration: "~15 min"
  tasks-completed: 8
  files-touched: 4    # (1 new, 2 modified, 1 planning .md whitelisted)
  tests-added: 0       # pure infra, no new test surface
  tests-total: 1044    # (1044 passed, 3 skipped — same as Phase 13)
  ratchet-status: green
  bundle-determinism: deterministic (`git diff --exit-code dist/` clean after regen)
gates:
  uv-lock-check: "Resolved 171 packages in 2ms — exit 0"
  yaml-valid: "9 steps, parses clean"
  ollama-grep-src: "0 matches (HARD-05 ratchet preserved)"
  ruff: "13 errors (pre-Phase-14 baseline, unchanged)"
  pyright-runtime: "54 errors (pre-Phase-14 baseline, unchanged)"
  pyright-full: "329 errors (pre-Phase-14 baseline, unchanged)"
  dist-regen-diff: "clean (exit 0)"
  pytest: "1044 passed, 3 skipped"
---

# Phase 14 Plan 01 Summary — Reproducible Air-Gap Dependency Lockfile

## One-liner

Wired the existing in-repo `uv.lock` into CI via `uv sync --frozen`, added a `uv lock --check` lockfile-freshness gate that fails the build on `pyproject.toml`/`uv.lock` drift, and documented the offline install path in `docs/AIRGAP_INSTALL.md` so an engineer behind a corporate firewall can reproduce the exact dependency graph from an internal mirror without public-internet access. Closes HARD-02 (CONCERNS C2).

## What changed

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | Added `astral-sh/setup-uv@v6` (uv 0.11.7); added `uv lock --check` gate as first job step; replaced `pip install -e ".[dev]"` with `uv sync --frozen --extra dev`; rewrote `ruff` / `pyright` / `pytest` invocations as `uv run …` so they hit the locked venv. |
| `docs/AIRGAP_INSTALL.md` (new) | 38-line offline-install recipe: clone → set `UV_INDEX_URL` → `uv sync --frozen [--offline]` → `uv run pytest tests/ -x`. |
| `.gitignore` | Added `!docs/AIRGAP_INSTALL.md` negation so the air-gap install doc ships while the rest of `docs/` (Claude artefacts) stays ignored. |
| `pyproject.toml` | Unchanged — already PEP 621; uv reads `[project]` natively, no `[tool.uv]` block required. |
| `uv.lock` | Unchanged — already present, 4430 lines, 171 packages, in sync. Verified by `uv lock --check` exit 0. |

## Acceptance gates (all green)

```
uv lock --check                                          : EXIT 0 (171 pkgs, 2 ms)
python -c 'import yaml; yaml.safe_load(open(ci.yml))'    : 9 steps, parses
git grep -nE 'https://ollama\.com|ollama\.com/api' src/  : 0 matches  (HARD-05 ratchet)
ruff check src tests                                     : 13 errors  (pre-existing baseline)
pyright src/runtime                                      : 54 errors  (pre-existing baseline)
pyright                                                  : 329 errors (pre-existing baseline)
python scripts/build_single_file.py && git diff dist/    : clean (exit 0)
pytest tests/ -x                                         : 1044 passed, 3 skipped
```

## Out of scope (deferred)

- A vendored-wheels tarball (truly `--no-index` install kit) — separate phase.
- Pyright / ruff baseline cleanup — pre-existing baselines, not Phase 14 territory.
- `Makefile` `make bootstrap` shim — `uv sync --frozen [--offline]` is the documented equivalent (ROADMAP SC-2 wording allows "or equivalent").
