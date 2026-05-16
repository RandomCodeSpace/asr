# Release process — v2.0

The React UI lives at `web/`. The Python framework lives at `src/runtime/`.
A v2.0.0-rc1 release ships **both** as a single tag.

## Cut a release candidate

1. Verify `web/package.json` version matches the tag you intend to cut.
   ```bash
   jq -r .version web/package.json
   ```
   For v2.0.0-rc1 the answer should be `2.0.0-rc1`.

2. Regenerate the framework bundles (HARD-08 requirement):
   ```bash
   uv run python scripts/build_single_file.py
   git diff --exit-code dist/    # must be clean
   ```

3. Build the React SPA:
   ```bash
   cd web && npm ci && npm run build
   ```

4. Compose the air-gap deploy payload:
   ```bash
   uv run python scripts/package_airgap.py --out dist/airgap
   ```
   Output: `dist/airgap/{app.py, ui.py, web/, README.txt}` — local-only,
   never committed. Drop it on the deploy host.

5. Sanity-check the bundle gates locally:
   ```bash
   uv run pytest -x
   uv run ruff check src/ tests/
   uv run pyright src/runtime
   (cd web && npm run lint && npm run typecheck && npm run test:unit && npm run build && npm run check:size)
   ```

6. Tag the release. Tagging always happens on `main` after the final PR
   merge, never on a feature branch:
   ```bash
   git switch main
   git pull --ff-only
   git tag v2.0.0-rc1 -m "v2.0.0-rc1: React UI"
   git push origin v2.0.0-rc1
   ```

## Bump for the next RC

When cutting `v2.0.0-rc2` (or `v2.0.0`):

1. `web/package.json` — update the `version` field.
2. Repeat the build + tag steps above.

Avoid editing `package.json` mid-PR for an unrelated change; the version
field changes only in a release-prep PR.

## Streamlit prototype

`dist/ui.py` (legacy Streamlit) ships in the deploy folder until the
React UI hits 100% parity (see `docs/REACT_UI_PARITY.md`). Streamlit
displays a deprecation banner; the React UI is the supported path.
