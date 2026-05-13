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
