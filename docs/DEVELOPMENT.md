# Development workflow

This document covers the day-to-day contributor loop. Air-gapped install
instructions live in `docs/AIRGAP_INSTALL.md`.

## Setup

```bash
# 1. Clone and create the venv with the lockfile.
git clone <repo>
cd asr
uv sync --frozen --extra dev

# 2. Verify by running the suite.
uv run pytest tests/ -x
```

## Editing source

Source layout:

- `src/runtime/` — framework code, the only thing the bundler reads to
  produce `dist/app.py`.
- `examples/incident_management/`, `examples/code_review/` — example
  apps; bundled into `dist/apps/incident-management.py` and
  `dist/apps/code-review.py` respectively.
- `scripts/build_single_file.py` — the bundler. Reads
  `RUNTIME_MODULE_ORDER` (and per-app order lists), flattens every
  module, strips intra-bundle imports, emits four self-contained `.py`
  files in `dist/`.

## After ANY change to `src/runtime/` or `examples/` — regenerate `dist/`

```bash
uv run python scripts/build_single_file.py
git add dist/
```

Then re-run the test suite. The CI gate `Bundle staleness gate
(HARD-08)` rebuilds the bundles from your source and fails the build if
they don't match the committed `dist/*`. This keeps the air-gap deploy
bundle repaired by construction — every PR that changes the runtime or
the bundler must commit fresh bundles, so the `dist/*` artifacts on
`main` can always be deployed without re-running the bundler on the
target host.

## Adding a new `src/runtime/*.py` module

1. Add a tuple `(RUNTIME_ROOT, "<relpath>")` to `RUNTIME_MODULE_ORDER`
   in `scripts/build_single_file.py`. Place it AFTER every module it
   imports at the top of file (the bundler concatenates in the order
   listed; later module bodies see earlier modules' symbols already in
   scope).

2. Regenerate the bundles:

   ```bash
   uv run python scripts/build_single_file.py
   ```

3. Run the suite — `tests/test_bundle_completeness.py` will fail loudly
   if you forgot step 1.

4. Smoke-test the bundles boot from a fresh tmpdir without the
   `PYTHONPATH=src:.` override that `pytest` sets:

   ```bash
   mkdir /tmp/bundle-check
   cp dist/apps/incident-management.py /tmp/bundle-check/app.py
   cp dist/ui.py /tmp/bundle-check/
   cd /tmp/bundle-check
   unset PYTHONPATH
   uv run python -c "import app; print('app boots')"
   ```

5. Commit `scripts/build_single_file.py` and the regenerated `dist/*`
   in a single change.

## Why two app bundles + a separate UI bundle?

- `dist/app.py` — framework only, no example code. Used to demonstrate
  that the runtime stands on its own.
- `dist/apps/incident-management.py` — the deployment ship target for
  the incident-management app; copied into the corporate environment
  as `app.py` (renamed at deploy).
- `dist/apps/code-review.py` — second app bundle, demonstrating the
  framework is genuinely generic (a second example builds from the
  same runtime).
- `dist/ui.py` — Streamlit UI; sits next to whichever `app.py` you
  deployed and `from app import …` reaches into the deploy bundle's
  flattened namespace.

The deployment workflow is a 7-file copy-only payload (the bundle
files plus a small set of YAML configs and a `.env`). The bundler
turns the multi-file source tree into the smallest possible deploy
payload.
