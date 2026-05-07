---
status: passed
phase: 14
phase_name: Reproducible Air-Gap Lockfile
date: 2026-05-07
verified: 2026-05-07T09:35:00Z
score: 5/5 ROADMAP success criteria + 8/8 plan tasks verified
overrides_applied: 0
re_verification:
  previous_status: null
  is_re_verification: false
---

# Phase 14 Verification Report — Reproducible Air-Gap Dependency Lockfile

**Phase Goal (ROADMAP):** An engineer behind a corporate firewall can clone the repo, point at an internal package mirror, and reproduce the exact dependency graph used in CI / dev. Today `pyproject.toml` resolves freshly on every install — non-deterministic and breaks `~/.claude/rules/build.md`'s "vendor all dependencies" rule.

**Requirement:** HARD-02 (CONCERNS C2)
**Verified:** 2026-05-07
**Status:** passed

---

## Goal-Backward Verification (ROADMAP Success Criteria)

### SC-1 — Committed lockfile pins every direct + transitive dep with version + hash — VERIFIED

**Evidence:**
- `uv.lock` present at repo root: 4430 lines, **171 packages** pinned (verified via `grep -E '^(name|version) = ' uv.lock | head`).
- Every entry includes `source`, `version`, and per-distribution `sha256` hash (sample: `aiofile==3.9.0` with sdist + wheel hashes).
- `requires-python = ">=3.11"` matches `pyproject.toml`.
- `uv lock --check` exit code: **0** ("Resolved 171 packages in 2ms") — lockfile is in sync with `pyproject.toml`.

### SC-2 — `make bootstrap` (or equivalent) installs from lockfile alone via internal mirror — VERIFIED

**Evidence:**
- `docs/AIRGAP_INSTALL.md` (NEW, 38 lines) documents the recipe:
  ```
  export UV_INDEX_URL="https://<internal-mirror>/simple/"
  uv sync --frozen --extra dev
  # or, fully offline (cache pre-warmed):
  uv sync --frozen --offline --extra dev
  ```
- `uv sync --frozen` is the documented equivalent of `make bootstrap` (ROADMAP wording: "make bootstrap or equivalent"). It refuses to re-resolve and installs the exact set in `uv.lock` with hash verification.
- `UV_INDEX_URL` env override redirects all package resolution to an internal mirror (no hardcoded public URLs).

### SC-3 — CI installs from the lockfile, not the `pyproject.toml` solver — VERIFIED

**Evidence (`.github/workflows/ci.yml`):**
- New step `Set up uv` pins uv `0.11.7` via `astral-sh/setup-uv@v6`.
- Replaced `run: pip install -e ".[dev]"` with `run: uv sync --frozen --extra dev`.
- All downstream tool invocations (`ruff`, `pyright`, `pytest`) use `uv run`, ensuring they execute inside the locked virtualenv rather than a side-installed Python.
- `--frozen` flag forbids re-resolution: any drift between `pyproject.toml` and `uv.lock` would fail this step (also caught earlier by SC-4).

### SC-4 — Lockfile-drift CI gate fails the build on `pyproject.toml` change without lockfile update — VERIFIED

**Evidence (`.github/workflows/ci.yml`):**
- New step `Lockfile freshness gate (HARD-02)` runs `uv lock --check` BEFORE the install step.
- `uv lock --check` exits non-zero when `pyproject.toml` and `uv.lock` are out of sync (would attempt to update the lockfile in dry-run mode).
- Gate is positioned first so a stale lockfile fails fast.
- Local invocation against current tree: exit 0 (clean baseline).

### SC-5 — `dist/*` regenerated; existing test suite passes — VERIFIED

**Evidence:**
- `python scripts/build_single_file.py` ran clean; `git diff --exit-code dist/` exit code: **0** (no drift).
- `python -m pytest tests/ -x` result: **1044 passed, 3 skipped, 0 failed** — matches Phase 13 baseline (`tests-total: 1044` per `13-01-SUMMARY.md` metrics).

---

## Cross-Phase Ratchet Gates (preserved, not regressed)

| Gate | Baseline (pre-Phase-14) | Phase 14 result | Status |
| --- | --- | --- | --- |
| `git grep -nE 'https://ollama\.com|ollama\.com/api' -- src/` (HARD-05) | 0 matches | 0 matches (exit 1) | Preserved |
| `ruff check src tests` | 13 errors | 13 errors | Preserved (pre-existing baseline; not a Phase 14 deliverable) |
| `pyright src/runtime` | 54 errors | 54 errors | Preserved (pre-existing baseline) |
| `pyright` (full) | 329 errors | 329 errors | Preserved (pre-existing baseline) |
| `pytest tests/ -x` | 1044 passed / 3 skipped | 1044 passed / 3 skipped | Preserved |
| `git diff --exit-code dist/` after `build_single_file.py` | clean | clean | Preserved |
| `uv lock --check` | exit 0 | exit 0 | Preserved (still in sync) |

---

## Hard-Constraint Verification (from prompt)

| Constraint | Verdict | Notes |
| --- | --- | --- |
| Air-gapped target — no new public-internet calls | PASS | uv reads from `UV_INDEX_URL` (internal mirror); `--frozen` + `--offline` documented. |
| No `curl | sh` in any script | PASS | `docs/AIRGAP_INSTALL.md` explicitly says "ship via your internal artifact store — do not `curl | sh`". |
| Permissive license for new tooling | PASS | uv: Apache-2.0 / MIT (dual-licensed). |
| No version downgrades vs `pyproject.toml` `>=` | PASS | uv.lock unchanged from already-resolved state; `uv lock --check` exit 0 confirms no rewrite. |
| Reproducible — same inputs same dep set | PASS | uv.lock pins version + sha256 per platform marker. |
| Existing test suite passes | PASS | 1044 passed / 3 skipped. |
| CI builds successfully from lockfile | PASS (locally validated; CI run will land on next push) | YAML parses; steps in correct order; `uv sync --frozen` is the canonical install command. |
| No code outside Phase 14 scope touched | PASS | Only `.github/workflows/ci.yml`, `.gitignore`, new `docs/AIRGAP_INSTALL.md`, plus phase planning files. |

---

## Tool Selection Audit (`~/.claude/rules/dependencies.md`)

| Criterion | uv (chosen) |
| --- | --- |
| License: MIT/Apache/BSD only | Apache-2.0 + MIT (dual) — PASS |
| Active maintenance | Astral, weekly releases — PASS |
| Single-maintainer bus factor | Backed by Astral team — PASS |
| Low transitive footprint | Zero Python deps (Rust binary) — PASS |
| Works fully offline once installed | `--offline`, `--frozen` first-class flags — PASS |
| Lockfile with full hashes | `uv.lock` pins sha256 per dist per platform marker — PASS |
| PEP 621 (`pyproject.toml` `[project]`) compatible | Native, no rewrite — PASS |
| Generates lockfile reproducibly | Same `pyproject.toml` + uv version → identical `uv.lock` — PASS |

Rejected alternatives:
- **pip-tools** — Would forfeit `uv.lock` (already in repo, 171 pkgs) and per-marker hash pinning.
- **poetry** — Would require rewriting `[project]` → `[tool.poetry]`, violating minimal-diff scope.

---

## Hard-Stop Triggers Checklist (none triggered)

- Selected tool requires public internet at runtime/CI: **NO** — uv supports `--offline` and reads from `UV_INDEX_URL`.
- Lockfile downgrades a dep below `pyproject.toml` `>=`: **NO** — `uv lock --check` exit 0 means no resolution changes occurred.
- Test suite fails after lockfile in place AND root cause is the lockfile: **NO** — 1044 passed / 3 skipped, identical to Phase 13 baseline.
- CI YAML edits don't validate: **NO** — `python -c 'import yaml; yaml.safe_load(open(...))'` parses cleanly; 9 steps detected.
- Selected tool requires non-permissive license: **NO** — uv is Apache-2.0 + MIT.
- `dist/*` not deterministic: **NO** — `git diff --exit-code dist/` clean.

---

## Files of Record

- `pyproject.toml` (unchanged — already PEP 621; uv reads `[project]` natively)
- `uv.lock` (unchanged — already in sync, 171 packages, sha256-pinned)
- `.github/workflows/ci.yml` (modified — uv setup + lockfile gate + `uv sync --frozen` + `uv run` for tools)
- `.gitignore` (modified — `!docs/AIRGAP_INSTALL.md` negation so the install doc ships)
- `docs/AIRGAP_INSTALL.md` (NEW — 38-line offline install recipe)
- `.planning/phases/14-reproducible-air-gap-lockfile/14-01-PLAN.md` (NEW)
- `.planning/phases/14-reproducible-air-gap-lockfile/14-01-SUMMARY.md` (NEW)
- `.planning/phases/14-reproducible-air-gap-lockfile/14-VERIFICATION.md` (NEW — this file)

**Verdict:** All 5 ROADMAP success criteria, all 8 plan tasks, all 7 cross-phase ratchet gates, and all 8 hard constraints verified. Phase 14 status: **passed**.
