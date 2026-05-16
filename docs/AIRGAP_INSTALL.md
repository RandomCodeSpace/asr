# Air-Gap / Internal-Mirror Install

Reproduce the exact dependency graph that CI uses, behind a corporate firewall,
without any public-internet access.

## Prerequisites

- Python 3.11 available on the target host.
- `uv` `>= 0.11.7` available on the target host (single static binary;
  ship via your internal artifact store — do **not** `curl | sh`).
- An internal PEP 503 / PEP 691 package mirror (Artifactory, Nexus, devpi,
  or `pip download`-populated wheel cache) that contains every distribution
  pinned in `uv.lock`.

## Install

```bash
# 1. Clone (or unpack the source tarball shipped to the air-gapped host).
git clone <internal-git-url>/asr.git
cd asr

# 2. Point uv at the internal mirror (overrides https://pypi.org/simple).
export UV_INDEX_URL="https://<internal-mirror>/simple/"
# Optional: extra index for private wheels.
# export UV_EXTRA_INDEX_URL="https://<internal-mirror>/private/simple/"

# 3. Install from the lockfile only — no resolver, no public-internet calls.
#    Drop --offline if the mirror is reachable; keep it if you have pre-warmed
#    uv's cache and want a hard-fail on any network attempt.
uv sync --frozen --extra dev          # connected to mirror
# uv sync --frozen --offline --extra dev   # fully offline (cache pre-warmed)

# 4. Verify.
uv run pytest tests/ -x
```

## Drift detection

The CI gate `uv lock --check` fails the build whenever `pyproject.toml`
changes without a matching `uv.lock` regeneration. Run the same check
locally before pushing:

```bash
uv lock --check    # exit 0 = in sync; non-zero = regenerate with `uv lock`
```

## Notes

- `uv.lock` pins every direct + transitive dependency to a specific version
  with sha256 hashes per platform marker; identical inputs produce identical
  installs on any host (HARD-02 / CONCERNS C2).
- Ship vendored wheels as a separate tarball if your host has no mirror at
  all; populate `~/.cache/uv` (or `UV_CACHE_DIR`) before running step 3.

## v2.0 — React UI in the air-gap payload

The React SPA in `web/` is built ahead of time and shipped as static
assets alongside the Python bundle. The backend serves it from `/`
via `runtime.api_static.mount_static_assets` so there is no separate
Node process on the deploy host.

```bash
# On a host with internet (one time, or each release):
cd web && npm ci && npm run build
cd ..
uv run python scripts/build_single_file.py    # framework + app bundles
uv run python scripts/package_airgap.py       # composes dist/airgap/

# Ship dist/airgap/ to the air-gap host. Layout:
#   dist/airgap/
#     app.py           # flattened framework + incident-management
#     ui.py            # legacy Streamlit (deprecation banner)
#     web/             # built React SPA (Vite output)
#     README.txt
```

On the air-gap host:

```bash
ASR_WEB_DIST=./web \
  python -m uvicorn app:get_app --factory --port 8000
# open http://<host>:8000/
```

`ASR_WEB_DIST` is read by `runtime.api_static.mount_static_assets`.
If unset, the backend falls back to `<bundle>/web/dist` (legacy dev
layout). When neither resolves, the SPA mount is skipped silently and
the API still serves on `/api/v1/*`.
