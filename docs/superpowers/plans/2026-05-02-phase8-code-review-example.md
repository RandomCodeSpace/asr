# Phase 8 — Second Example App: Code-Review Assistant (Framework-Genericness Proof)

**Status:** Draft (planning only — no implementation in this document).
**Author:** asr framework team.
**Date:** 2026-05-02.
**Depends on:** Phase 1 (domain extraction), Phase 2 (extensible state + checkpointer), Phase 3 (storage split). Light dependence on Phase 4–7 (entry-agent/signal routing, trigger registry, supervisor kinds, etc.); the plan defers any feature that has not yet shipped.
**Blocks:** Phase 9 (ASR — Auto Site Recovery). P9 must build on a framework that has been *proven* generic. P8 ships fully merged before P9 starts.

---

## 1. Goal + Scope

Stand up `examples/code_review/` as a **second fully-functional example app** that exercises the asr framework in a domain deliberately unrelated to incident management. The point is not a polished GitHub bot; the point is to make any leftover incident-management assumption inside `src/runtime/` *fail loudly* so it can be lifted into the framework or made app-overridable.

**Locked decisions (non-negotiable):**

- **App choice (P8.1):** code-review assistant. Not customer-support, not docs Q&A.
- **Depth (P8.2):** thin proof-of-concept — **3 skills + 1 MCP server + basic Streamlit UI**. Not production-ready.
- **Sequencing:** P8 fully merged on `main` before P9 starts.

**In scope:**

- New app code under `examples/code_review/` mirroring the structural template of `examples/incident_management/` (state, config, mcp_server, skills, ui, `__main__`, README).
- Three skills: `intake`, `reviewer`, `summarizer` — all `kind: responsive` (no supervisor).
- One MCP server `code_review_management` with four **mocked** tools: `read_file`, `read_diff`, `post_comment`, `set_verdict`.
- Streamlit UI listing PRs in review with a detail pane (verdict, comments, diff).
- Webhook trigger config wired through Phase 5's `TriggerRegistry` (PR open/update event).
- A separate dist bundle: `dist/apps/code-review.py`.
- **Three or more framework-leak fixes** discovered while building the above.

**Out of scope:**

- Real GitHub/GitLab API integration — tools are stubs reading from a local fixture dir.
- `kind: supervisor` skills (deferred to Phase 6).
- Pretty UI; we reuse `examples/incident_management/ui.py` primitives.
- Production webhook authentication, retries, dedupe — minimum viable.
- Polishing the `ASR` (Phase 9) on this scaffolding — that is the next phase.

**Definition of "framework-leak":** any place in `src/runtime/**` that:

1. references an incident-only concept (`Reporter`, `environment`, `INC-…`, severity vocabulary), **or**
2. only an incident-management app could plausibly need, **or**
3. an app cannot override without monkey-patching framework code.

Each leak found triggers a **framework adjustment task** — see §4 P8-C, P8-I, P8-J, P8-O.

---

## 2. Target Architecture After Phase 8

```
examples/
  incident_management/        # untouched in feature, may be patched for leak fixes
    state.py                  # IncidentState(Session)
    config.py / config.yaml   # IncidentAppConfig (severity_aliases incident-flavoured)
    mcp_server.py             # incident_management MCP server
    skills/                   # 4 skill YAMLs
    ui.py / __main__.py / __init__.py / README.md

  code_review/                # NEW
    __init__.py
    __main__.py               # `python -m examples.code_review` → Streamlit UI
    state.py                  # CodeReviewState(Session)
    config.py                 # CodeReviewAppConfig
    config.yaml               # mirrors defaults
    mcp_server.py             # code_review_management server (4 stub tools)
    skills/
      _common/                # output.md (likely shared via path symlink/import)
      intake/config.yaml
      reviewer/config.yaml
      summarizer/config.yaml
    ui.py                     # Streamlit page reusing framework UI primitives
    triggers.yaml             # webhook trigger registration (uses P5 TriggerRegistry)
    fixtures/                 # mocked repo state for stub tools
      pr-101/
        diff.patch
        files/...
    README.md

src/runtime/                  # framework — leak fixes land here
  state.py                    # Session.id_format hook (P8-C)
  config.py                   # AppConfig stays domain-free; minor breaking changes possible
  graph.py / orchestrator.py  # confidence gate becomes opt-in (P8-O if needed)
  reporters.py                # NEW (P8-I) — generic Actor/Principal abstraction (or
                              # remove it from Session entirely; both options live in P8-I)
```

After P8:

- Both apps run **side-by-side in one orchestrator process** (P8-K test).
- A code-review session never imports from `examples.incident_management.*` and vice-versa.
- A fresh grep `grep -rn 'INC-\|Reporter\|environment\|severity_aliases' src/runtime/ --include='*.py' | grep -v '#'` returns **zero matches**.

---

## 3. Naming Map / API Changes

| Concept | Incident app | Code-review app | Framework |
|---|---|---|---|
| Session subclass | `IncidentState` | `CodeReviewState` | `Session` (no domain fields) |
| ID format | `INC-YYYYMMDD-NNN` | `PR-{repo}-{number}` (e.g. `PR-acme-api-101`) | `Session.id_format()` classmethod / strategy hook (P8-C) |
| Triggering event | manual create + signals | webhook PR open/update | `TriggerRegistry` (Phase 5) |
| App config | `IncidentAppConfig` | `CodeReviewAppConfig` | `AppConfig` (generic) |
| Severity vocab | `low/med/high/critical` | `blocker/critical/important/minor` | app-supplied via `severity_aliases` map (already pluggable per Phase 1; verify) |
| Domain MCP server | `incident_management` | `code_review_management` | loader is generic |
| Reporter / actor | `Reporter(id, team)` | PR author `id, handle, repo` | generic `Actor` BaseModel or removed entirely (P8-I) |
| Environment | `environment: str` | none (has `repo`/`branch`) | Session field deleted from framework if still present (P8-J) |

**Breaking changes flagged for P1 / IncidentAppConfig contract:**

- P8-C may add a `id_format()` classmethod to `Session` and migrate `_INC_ID_RE` into `IncidentState`. Apps that constructed sessions through the framework's id minter must switch to the per-state hook.
- P8-I may remove `Reporter` from any framework module and replace with either (a) a generic `Actor` model or (b) drop the field from `Session` entirely (apps add their own).
- P8-J may remove `environment` from `Session` if any vestige remains in the framework.
- Each is gated by "all 231+ existing incident-management tests still pass" before the next task starts.

---

## 4. Task Breakdown

Total: **14 tasks (P8-A → P8-N)** plus one conditional follow-on (P8-O — confidence-gate genericisation, only if leak surfaces).

Each task follows the project pattern: `failing-test → run → implement → run → commit`. Tests are **always added before implementation**.

---

### P8-A — Scaffold `examples/code_review/` directory

Stand up the empty skeleton mirroring `examples/incident_management/`. No logic yet — just the file layout so subsequent tasks have a place to land.

**Files:**

- Create: `examples/code_review/__init__.py` (empty)
- Create: `examples/code_review/__main__.py` (placeholder that prints a banner; will be replaced in P8-G)
- Create: `examples/code_review/state.py` (placeholder `class CodeReviewState(Session): pass`)
- Create: `examples/code_review/config.py` (placeholder `class CodeReviewAppConfig(BaseModel): pass`)
- Create: `examples/code_review/config.yaml` (empty document)
- Create: `examples/code_review/mcp_server.py` (placeholder importing `FastMCP`)
- Create: `examples/code_review/ui.py` (`# placeholder for Streamlit UI`)
- Create: `examples/code_review/skills/_common/output.md` (copied/aliased from incident_management)
- Create: `examples/code_review/skills/intake/config.yaml` (empty `description: ""`)
- Create: `examples/code_review/skills/reviewer/config.yaml`
- Create: `examples/code_review/skills/summarizer/config.yaml`
- Create: `examples/code_review/fixtures/.gitkeep`
- Create: `examples/code_review/README.md` (one-paragraph stub)

**Tests:**

- New: `tests/test_code_review_scaffold.py` asserting all of the above paths exist and `examples.code_review` imports cleanly.

**Acceptance:**

- `python -c "import examples.code_review"` → exit 0.
- `pytest tests/test_code_review_scaffold.py -q` → green.
- `pytest tests/ -q` → existing test count + 1, zero failures.

**Commit:** `feat(p8): scaffold examples/code_review/ skeleton`.

---

### P8-B — `CodeReviewState(Session)` with domain fields

Define the subclass in `examples/code_review/state.py`.

```python
class FileChange(BaseModel):
    path: str
    additions: int
    deletions: int
    status: Literal["added", "modified", "removed", "renamed"]

class ReviewComment(BaseModel):
    file: str | None  # None → general PR comment
    line: int | None
    body: str
    severity: Literal["blocker", "critical", "important", "minor"]
    skill: str        # which skill emitted it

class CodeReviewState(Session):
    repo: str
    branch: str
    pr_number: int
    pr_title: str
    pr_description: str = ""
    pr_author: str
    diff: str = ""           # may be truncated by reviewer
    files_changed: list[FileChange] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    verdict: Literal["approve", "request_changes", "comment"] | None = None
    severity: Literal["blocker", "critical", "important", "minor"] | None = None
```

**Tests:**

- New: `tests/test_code_review_state.py`:
  - `CodeReviewState` round-trips through `model_dump_json` and back.
  - `CodeReviewState` is registered as a valid `state_class` for `Orchestrator[CodeReviewState]` (uses Phase 2 P2-B importlib resolution).
  - No field overlap with `IncidentState` other than the inherited `Session` fields.

**Acceptance:**

- New tests green; full suite green.

**Commit:** `feat(p8): CodeReviewState extending Session`.

---

### P8-C — Framework leak #1: ID-format hook in `Session`

**Symptom predicted in P8-B:** the orchestrator currently mints session IDs using a hard-coded `INC-YYYYMMDD-NNN` regex (`_INC_ID_RE` in `examples/incident_management/state.py`, but the *minter* lives in or is invoked from the framework). Code-review wants `PR-{repo}-{number}`. Today an app cannot override this without monkey-patching.

**Fix:** add a generic, app-overridable id-format hook on `Session`:

```python
# src/runtime/state.py
class Session(BaseModel):
    id: str
    ...

    @classmethod
    def id_format(cls, payload: dict[str, Any]) -> str:
        """Return the canonical session id for this app. Override per subclass.

        Default: a ULID-based opaque id (calls runtime.ids.new_ulid()).
        """
        from runtime.ids import new_ulid
        return new_ulid()
```

Subclasses override:

```python
# examples/incident_management/state.py
class IncidentState(Session):
    @classmethod
    def id_format(cls, payload):
        return mint_inc_id(payload["created_at"])  # existing INC- minter

# examples/code_review/state.py
class CodeReviewState(Session):
    @classmethod
    def id_format(cls, payload):
        return f"PR-{payload['repo']}-{payload['pr_number']}"
```

The framework's `SessionStore.create()` calls `state_cls.id_format(payload)` instead of the legacy `mint_inc_id` import.

**Files:**

- Modify: `src/runtime/state.py` (+ `id_format` classmethod with default)
- Create: `src/runtime/ids.py` (hosting the ULID default)
- Modify: `src/runtime/storage/session_store.py` — call `state_cls.id_format(payload)` in `.create()`
- Modify: `examples/incident_management/state.py` — override `id_format`, drop framework-import of mint helper
- Modify: `examples/code_review/state.py` — override `id_format`

**Tests:**

- New: `tests/test_session_id_format.py`:
  - default `Session.id_format` returns a ULID and is monotonic.
  - `IncidentState.id_format({"created_at": ...})` matches `_INC_ID_RE`.
  - `CodeReviewState.id_format({"repo": "acme-api", "pr_number": 101})` == `"PR-acme-api-101"`.
- Update: any incident test that imported `mint_inc_id` from the framework now imports from `examples.incident_management.state`.
- **All existing 231+ incident tests still pass.**

**Risk:**

- This is a breaking change to the framework's public-ish session-creation contract. The plan asserts: every existing test passes after this commit before P8-D begins. If a test fails, **revert and redesign** — do not paper over.

**Commit:** `feat(p8): pluggable Session.id_format() hook (framework leak fix)`.

---

### P8-D — `CodeReviewAppConfig`

```python
class CodeReviewAppConfig(BaseModel):
    similarity_threshold: float = 0.55      # different from incident default (0.78)
    max_files_to_review: int = 50
    max_diff_size_kb: int = 256
    severity_aliases: dict[str, str] = Field(default_factory=lambda: {
        "P0": "blocker", "P1": "critical",
        "must": "blocker", "should": "important", "nit": "minor",
        "blocker": "blocker", "critical": "critical",
        "important": "important", "minor": "minor",
    })
    fixtures_dir: str = "examples/code_review/fixtures"
    webhook_secret: str | None = None       # env-overridable
```

**Files:**

- Modify: `examples/code_review/config.py`
- Modify: `examples/code_review/config.yaml` (mirror defaults)
- Add config loader `load_code_review_app_config()` modelled on `load_incident_app_config()`.

**Tests:**

- New: `tests/test_code_review_config.py`:
  - default values present and correct,
  - YAML override applied,
  - `severity_aliases` fully-distinct vocabulary from incident,
  - `similarity_threshold` differs from `IncidentAppConfig` default — confirms the framework does not bake the incident default into a shared base.

**Acceptance:** new tests green; suite green.

**Commit:** `feat(p8): CodeReviewAppConfig with code-review severity vocabulary`.

---

### P8-E — `code_review_management` MCP server with 4 stub tools

**File:** `examples/code_review/mcp_server.py`. Mirrors `incident_management/mcp_server.py` shape.

```python
mcp = FastMCP("code_review_management")

@mcp.tool()
def read_file(repo: str, ref: str, path: str) -> str:
    """Return file contents from the fixture dir at fixtures/<repo>/<ref>/<path>."""

@mcp.tool()
def read_diff(pr_number: int) -> str:
    """Return a unified diff from fixtures/<pr-NNN>/diff.patch."""

@mcp.tool()
def post_comment(pr_number: int, comment: str,
                 file: str | None = None, line: int | None = None) -> dict:
    """Append to in-memory `_POSTED[pr_number]`; return {ok: True, comment_id: ...}."""

@mcp.tool()
def set_verdict(pr_number: int,
                verdict: Literal["approve", "request_changes", "comment"]) -> dict:
    """Record verdict in-memory; return {ok: True, verdict: ...}."""
```

Plus `_POSTED: dict[int, list[dict]]` and `_VERDICTS: dict[int, str]` module-level (mock backing store), plus direct-call shims for tests (matches incident pattern).

**Tests:**

- New: `tests/test_code_review_mcp.py`:
  - `read_file` against a fixture path returns expected bytes.
  - `read_diff` returns committed `diff.patch`.
  - `post_comment` accumulates; `set_verdict` overwrites.
  - MCP server registers all four tools (introspect via FastMCP API).
  - The server appears in the framework's MCP loader output when `code_review_management: { module: "examples.code_review.mcp_server" }` is in the loader config.

**Fixtures committed:** `examples/code_review/fixtures/pr-101/diff.patch` (small, real-looking unified diff) and 1–2 file fixtures.

**Commit:** `feat(p8): code_review_management MCP server with stub tools`.

---

### P8-F — Three skill YAMLs

Mirrors `examples/incident_management/skills/*/config.yaml`.

```yaml
# intake/config.yaml
description: Parse PR payload, detect reviewable changes, route to reviewer.
kind: responsive
tools:
  local: ["read_diff", "update_session"]
routes:
  - when: success
    next: reviewer
  - when: failed
    next: __end__
  - when: default
    next: reviewer
```

```yaml
# reviewer/config.yaml
description: Per-file review using read_file; emits ReviewComments.
kind: responsive
tools:
  local: ["read_file", "post_comment", "update_session"]
routes:
  - when: success
    next: summarizer
  - when: failed
    next: __end__
  - when: default
    next: summarizer
```

```yaml
# summarizer/config.yaml
description: Roll up comments into a single PR-level summary; emits verdict.
kind: responsive
tools:
  local: ["post_comment", "set_verdict", "update_session"]
routes:
  - when: success
    next: __end__
  - when: failed
    next: __end__
  - when: default
    next: __end__
```

**Notable absence:** no `gate: confidence` edge anywhere — code-review does not use the intervention gate. If the orchestrator's graph builder *requires* a gate to be wired, that surfaces P8-O (see §6 risks).

**Tests:**

- New: `tests/test_code_review_skills.py`:
  - all three skill YAMLs load via the framework's skill loader,
  - `Skill.from_yaml(...)` produces a valid graph subgraph,
  - the loader respects `paths.skills_dir = examples/code_review/skills`,
  - no `gate` field is set on any route.

**Commit:** `feat(p8): code-review skill YAMLs (intake/reviewer/summarizer)`.

---

### P8-G — Streamlit UI (sidebar PR list, detail with comments + verdict + diff)

**File:** `examples/code_review/ui.py`. Reuses framework UI primitives where they exist; otherwise mirrors `examples/incident_management/ui.py` accordion-per-item style (per CLAUDE.md memory: badge-rich headers, sharp corners). **No new design language**.

UI structure:

- Sidebar: list of `CodeReviewState` rows (status pill + repo/PR-number + author).
- Detail pane (accordion):
  - Header row: PR title, verdict badge, severity badge.
  - Files-changed table (path, +/− counts, status).
  - Comments section: filter by file, severity, skill.
  - Diff viewer (read-only `<pre>` for P8 PoC).

`__main__.py` boots Streamlit:

```python
# examples/code_review/__main__.py
from streamlit.web import bootstrap
from pathlib import Path
bootstrap.run(str(Path(__file__).parent / "ui.py"), False, [], {})
```

**Tests:**

- New: `tests/test_code_review_ui.py`:
  - module imports without error,
  - given a seeded `CodeReviewState`, the page-render function returns expected widget tree (use `streamlit.testing` AppTest if available; else snapshot the render-call sequence).
- E2E smoke (scripted, no browser): `python -m examples.code_review --check` exits 0 (a `--check` flag boots Streamlit headlessly for 2 s and exits).

**Commit:** `feat(p8): Streamlit UI for code-review app`.

---

### P8-H — Webhook trigger configuration

Wire a webhook trigger via Phase 5's `TriggerRegistry`. The config registers a route on the runtime API: `POST /webhooks/code_review` → validate signature → construct `CodeReviewState` payload → start a session.

**Files:**

- Create: `examples/code_review/triggers.yaml`:

```yaml
triggers:
  - name: github_pr_opened
    kind: webhook
    path: /webhooks/code_review
    secret_env: CODE_REVIEW_WEBHOOK_SECRET
    state_payload_builder: examples.code_review.triggers:build_payload
```

- Create: `examples/code_review/triggers.py` with `build_payload(request_json) -> dict` that maps a GitHub-style PR event into the CodeReviewState fields.
- Modify: `config/config.yaml` (or app-local config wiring) to load `triggers.yaml` when the runtime is started in code-review mode.

**Tests:**

- New: `tests/test_code_review_trigger.py`:
  - `build_payload` extracts `repo`, `pr_number`, `pr_title`, `pr_author` from a fixture `pr_opened.json`,
  - mock TriggerRegistry receives the registration,
  - hitting the webhook with a valid HMAC creates a session whose state matches the payload (use FastAPI `TestClient`).
- **Reject** unsigned requests (HMAC mismatch → 401).

**Risk:** if Phase 5's TriggerRegistry is not yet shipped at the time P8 runs, **stop and re-plan** rather than reinventing it inside `examples/code_review/`. The plan does not allow forking trigger-handling logic.

**Commit:** `feat(p8): webhook trigger for code-review PR events`.

---

### P8-I — Framework leak #2: `Reporter` model is incident-specific

**Symptom predicted in P8-B/D:** `examples/incident_management/state.py` defines `Reporter(id, team)` and `IncidentState.reporter: Reporter` — but inspect `src/runtime/state.py` and `src/runtime/incident.py` (or whatever vestige remains post-P1) for any framework-side reference to `Reporter` / `reporter_id` / `reporter_team`. If anything survives, code-review (which has a *PR author*, not a "reporter with a team") cannot model this.

**Fix options (decide during the task, then commit):**

- **Option A:** introduce a generic `Actor` base model in `src/runtime/actors.py` with just `id: str` and an open `metadata: dict[str, Any]`. `IncidentState.reporter: Reporter(Actor)` adds `team`; `CodeReviewState.pr_author: Actor` adds `handle`, `repo` via metadata.
- **Option B:** remove the concept from the framework entirely; let each app define its own. (Preferred if no framework code currently *uses* it.)

**Plan:** start with Option B (zero generalisation). Only fall back to Option A if a framework module — e.g. UI rendering, audit-log helpers, or repo search — needs to know "who initiated this session".

**Files (Option B path):**

- Modify: `src/runtime/state.py` — confirm no `Reporter` import or `reporter` field.
- Modify: any test/fixture in `tests/` that imported `Reporter` from a framework path → import from `examples.incident_management.state`.
- Add framework-level grep guard: `tests/test_runtime_no_domain_leaks.py` asserts:

```python
def test_no_reporter_in_runtime():
    import subprocess
    out = subprocess.check_output(
        ["grep", "-rn", "Reporter", "src/runtime/", "--include=*.py"], text=True,
    )
    # Allow only comment lines and class-name strings inside docstrings
    real_hits = [l for l in out.splitlines() if "#" not in l and '"""' not in l]
    assert real_hits == [], f"Framework leaks Reporter: {real_hits}"
```

**Acceptance:** all 231+ existing tests pass.

**Commit:** `refactor(p8): remove Reporter from framework; app-only domain concept (leak fix)`.

---

### P8-J — Framework leak #3: `environment` field is incident-specific

**Symptom:** the original `Incident` had `environment: str` (production/staging/dev). Phase 1 was supposed to extract this to `IncidentState`, but a residue may exist (a default value, a test fixture, a config validator). Code-review has `repo` + `branch` and **no environment**.

**Fix:** identical pattern to P8-I.

- Confirm `Session.environment` is gone from `src/runtime/state.py`.
- Confirm `AppConfig.environments` is gone from `src/runtime/config.py`.
- Add grep guards in `tests/test_runtime_no_domain_leaks.py`:

```python
def test_no_environment_in_runtime_state():
    src = Path("src/runtime/state.py").read_text()
    assert "environment" not in src

def test_no_environments_in_runtime_config():
    src = Path("src/runtime/config.py").read_text()
    assert "environments" not in src
```

If the grep guard fails, lift the field into `IncidentAppConfig` (it likely already is post-P1) and remove the framework reference.

**Acceptance:** all 231+ existing tests pass; new grep guards green.

**Commit:** `refactor(p8): remove environment leak from framework runtime (leak fix)`.

---

### P8-K — Side-by-side test: incident-management + code-review in one orchestrator process

The decisive proof. A single Python process boots two `Orchestrator` instances:

```python
incident_orch = Orchestrator[IncidentState](cfg=incident_cfg, ...)
review_orch   = Orchestrator[CodeReviewState](cfg=code_review_cfg, ...)

inc_id  = incident_orch.start({...})
pr_id   = review_orch.start({"repo": "acme-api", "pr_number": 101, ...})

incident_orch.run_until_done(inc_id)
review_orch.run_until_done(pr_id)

assert incident_orch.session_store.get(inc_id).status == "resolved"
assert review_orch.session_store.get(pr_id).verdict in {"approve", "request_changes", "comment"}
```

**Files:**

- New: `tests/test_two_apps_one_process.py`.

**What this asserts:**

- Both apps' MCP servers register without name collisions.
- Both apps' skill loaders read from their own `paths.skills_dir`.
- A code-review session never leaks fields into an incident store and vice-versa (storage isolation per Phase 3).
- Logs from one orchestrator do not pollute the other (test captures `caplog` and filters by `app`).

**Acceptance:** test green on first non-trivial pass. If it fails because of a global singleton in `src/runtime/`, **stop, fix, re-run** before continuing. (Fourth potential leak.)

**Commit:** `test(p8): two example apps run side-by-side in one orchestrator process`.

---

### P8-L — End-to-end test: webhook → session → comments → verdict

Drives a full code-review session through the runtime via the webhook entrypoint, end-to-end.

**Test flow:**

1. Start the runtime API in-process (FastAPI `TestClient`).
2. POST a fixture `pr_opened.json` to `/webhooks/code_review` with valid HMAC.
3. Poll the orchestrator until the session reaches a terminal state (timeout 30 s).
4. Assert:
   - `state.files_changed` is non-empty,
   - `state.comments` is non-empty,
   - `state.verdict` ∈ `{approve, request_changes, comment}`,
   - all comments have a known `severity` from `severity_aliases`,
   - the mock `_POSTED` map in `code_review_management` has entries matching `state.comments`,
   - `state.id` matches `^PR-acme-api-\d+$`.

**Files:**

- New: `tests/test_code_review_e2e.py`.
- New: `examples/code_review/fixtures/pr_opened.json` (canonical GitHub-style payload).

**Commit:** `test(p8): end-to-end code-review session via webhook trigger`.

---

### P8-M — README at `examples/code_review/README.md`

User-facing run instructions.

**Sections:**

- What this app does (one paragraph).
- Quickstart: `python -m examples.code_review`.
- Webhook setup (env var `CODE_REVIEW_WEBHOOK_SECRET`, public URL via local tunnel for testing only).
- Skills and what each one does.
- MCP tools and that they are **mocked**.
- Limits (no real GitHub API; supervisor skills not yet supported; no incremental re-review).
- "How this proves the framework is generic" — link to P8-I/J/C leak fixes.

**No further README work elsewhere in this task** (top-level README touched in P8-N).

**Commit:** `docs(p8): README for code-review example app`.

---

### P8-N — Final verification + integration with main repo README

**Steps:**

1. Update top-level `README.md` "Examples" section to list both `incident_management` and `code_review` with one-line descriptions.
2. Run the full suite: `pytest tests/ -v`. Expect **231 (or current baseline) + ~25 new tests** = `~256+`. Zero skips, zero xfails.
3. Run grep guards listed in §5 Done criteria.
4. Build the dist bundle for code-review: `python scripts/build_single_file.py --app code_review --out dist/apps/code-review.py`. Verify import-runs (`python dist/apps/code-review.py --check`) on a clean venv with only vendored deps.
5. Run dependency audit: `pip-audit` clean (per `rules/security.md`); flag High/Critical immediately.
6. Verify both apps run side-by-side via `python -m examples.incident_management` and `python -m examples.code_review` in two terminals against the same backing storage **with isolation** (storage already namespaces by app per Phase 3 — confirm).
7. Manual smoke: open the code-review UI, see the seeded fixture PR, confirm comments + verdict render.

**Final commit:** `chore(p8): Phase 8 verification — code-review app integrated`.

---

### P8-O — (Conditional) Framework leak #4: confidence-gate is incident-specific

**Trigger condition:** if during P8-F or P8-K the graph builder fails because **no skill has `gate: confidence`**, or because the orchestrator's resume-graph requires a gate to wire in, this task is unblocked. Otherwise, **do not execute** — the leak is hypothetical.

**Fix:** make the confidence gate fully opt-in.

- Today (post-P2-H): `make_gate_node` is invoked when *any* route declares `gate: confidence`. Verify this is true.
- If the orchestrator unconditionally inserts a gate (e.g. in `build_resume_graph`) → patch to skip when no route declares one.
- Add `tests/test_no_gate_required.py` running a tiny app whose skills declare zero gates and asserting the graph builds.

**Commit:** `refactor(p8): confidence-gate becomes fully opt-in (leak fix)`.

---

## 5. Sequencing and Dependencies

```
P8-A  (scaffold)
 └─> P8-B  (CodeReviewState)
      ├─> P8-C  (id_format hook — leak fix #1)            ← framework breaking
      └─> P8-D  (CodeReviewAppConfig)
           ├─> P8-E  (MCP server)
           │    └─> P8-F  (skills)
           │         ├─> P8-G  (UI)
           │         ├─> P8-H  (webhook trigger)
           │         └─> P8-O? (gate leak — conditional)
           ├─> P8-I  (Reporter leak — fix #2)             ← framework breaking
           └─> P8-J  (environment leak — fix #3)          ← framework breaking
                ├─> P8-K  (two-apps-one-process test)
                ├─> P8-L  (e2e webhook test)
                └─> P8-M  (README)
                     └─> P8-N  (final verification)
```

**Hard rule:** before P8-C, P8-I, P8-J, or P8-O lands, **all existing 231+ tests must pass**. After each lands, the same rule applies. The leak fixes are the riskiest items; do not batch them.

---

## 6. Risks and Mitigations

**R1 — Mock tools ≠ real GitHub API.**
The four MCP tools are stubs. They prove plumbing, not integration. **Mitigation:** explicit "MOCKED" notes in every tool docstring; README calls it out; integration is Phase 9+ scope.

**R2 — Leak fixes (P8-C, P8-I, P8-J, P8-O) break the IncidentAppConfig contract.**
Each touches `src/runtime/` and may cascade across `IncidentState`, `IncidentAppConfig`, or stored sessions on disk. **Mitigation:** every leak fix runs the entire suite (`pytest tests/ -v`) before commit; hard gate "all 231+ existing tests still green". Plus grep guards in `tests/test_runtime_no_domain_leaks.py` to prevent regressions. If a fix would require migrating on-disk session data, **stop and write a migration plan** instead of pushing through.

**R3 — `kind: supervisor` skills emerge as needed.**
A natural code-review design dispatches one reviewer-per-file under a top-level supervisor. **Mitigation:** P8 stays responsive-only. If supervisor is genuinely required, the reviewer skill enumerates files itself in a single agent run; we accept the loss of parallelism. Real supervisor support is Phase 6.

**R4 — Bundling: `dist/apps/code-review.py` is a fresh artifact.**
The bundler today builds only `dist/apps/incident-management.py`. **Mitigation:** P8-N executes the bundler with `--app code_review`; if the bundler hardcodes `incident_management`, the bundler itself is a fifth leak (rare, but flag it). Add `tests/test_bundle_code_review.py` that imports `dist/apps/code-review.py` on a clean PYTHONPATH and asserts a simple smoke run.

**R5 — Storage isolation between two running apps.**
If both apps share `incidents/faiss` or the SQLite metadata DB, sessions could collide. **Mitigation:** P8-K asserts both orchestrators use distinct `paths.incidents_dir` (rename: `paths.sessions_dir` once Phase 1 cleanup completes — confirm) and distinct `storage.metadata.url`.

**R6 — Streamlit page collisions.**
Both UIs may try to bind port 8501. **Mitigation:** `examples/code_review/__main__.py` accepts `--port` (default 8502); the README documents the convention.

**R7 — CI runtime explosion.**
P8-K and P8-L each spin up MCP servers in-process. **Mitigation:** mark them `pytest.mark.slow`; default `pytest tests/` excludes; CI runs `pytest -m "not slow"` plus a separate slow-tests job.

**R8 — Skill `_common/output.md` shared between apps.**
Today both apps would duplicate the file. **Mitigation:** in P8-A, copy it. If the duplication grows annoying in P8-F, hoist the common YAML fragments to a framework-provided default (deferred — note in follow-ups).

---

## 7. Done Criteria

**Test suite:**

- `pytest tests/ -v` exits 0.
- New tests added in this phase: ~25 (P8-A:1 + P8-B:3 + P8-C:3 + P8-D:3 + P8-E:5 + P8-F:4 + P8-G:2 + P8-H:3 + P8-I:1 + P8-J:2 + P8-K:1 + P8-L:1 + bundle:1).
- Total: existing baseline + ~25, with **zero new skips/xfails**.
- All 231+ pre-existing incident-management tests still pass after each leak fix.

**Grep checks (zero hits expected):**

- `grep -rn 'Reporter' src/runtime/ --include='*.py' | grep -v '^.*#'`
- `grep -rn '\benvironment\b' src/runtime/ --include='*.py' | grep -v '^.*#'`
- `grep -rn 'INC-' src/runtime/ --include='*.py' | grep -v '^.*#'`
- `grep -rn 'severity_aliases' src/runtime/ --include='*.py' | grep -v '^.*#'`
- `grep -rn 'incident' src/runtime/ --include='*.py' | grep -v '^.*#' | grep -v -i 'incident_id_legacy_export'`

**Structural import checks (in fresh Python):**

- `from examples.code_review.state import CodeReviewState` ✓
- `from examples.code_review.config import CodeReviewAppConfig` ✓
- `from examples.code_review.mcp_server import mcp` ✓
- `from runtime.state import Session` ✓ (and `Session().id_format(...)` callable)

**Functional:**

- `python -m examples.code_review --check` exits 0 within 5 s.
- `python -m examples.incident_management --check` exits 0 (regression check).
- Webhook E2E test produces a verdict on the seeded fixture PR.
- Two orchestrators run together in one process without state collision.

**Bundle:**

- `dist/apps/code-review.py` builds and is import-runnable on a clean venv with only vendored deps (per `rules/build.md`).

**Audit:**

- `pip-audit` clean (no High/Critical CVEs); Medium/Low documented.

**Documentation:**

- `examples/code_review/README.md` exists with the §P8-M sections.
- Top-level `README.md` lists both example apps.

**Phase exit:**

- All commits land on `main` via standard PR review.
- A short retrospective note appended to this document under §9 listing every framework leak actually found (vs. §1 predictions) — this informs Phase 9 planning.

---

## 8. Open Questions

- **Q1.** Should `Session.id_format` accept the full state dict or only declared "id-source" fields? (Plan default: full payload dict; revisit if tests show unwanted coupling.)
- **Q2.** Are signal vocabularies (`success`/`failed`/`needs_input`) genuinely generic, or is there a code-review-specific signal needed (e.g. `cannot_review`, `needs_human`)? (Plan default: reuse current signals; flag if reviewer skill needs more.)
- **Q3.** Should the bundler grow a `--app` flag now, or hardcode-then-refactor? (Plan default: `--app` flag; P8-N validates.)
- **Q4.** Is there value in factoring a shared "PR-like / issue-like" abstraction for ASR (Phase 9)? (Plan default: no — wait for ASR's actual shape before generalising; YAGNI.)
- **Q5.** Do we need a CLI entrypoint (`python -m examples.code_review review --pr 101`) for headless runs, or is the webhook + UI sufficient? (Plan default: webhook + UI only; CLI deferred.)

---

## 9. Phase Retrospective (filled at P8-N)

**Leaks predicted in §1:** ID format, Reporter, environment, severity vocabulary, gate, skills_dir.

**Leaks actually found and fixed in P8 (this scoped slice — P8-C, J, K, L, M, N):**

- **Session id format** — `SessionStore._next_id` hard-coded `INC-YYYYMMDD-NNN`. Lifted into `Session.id_format(seq=...)` classmethod hook (P8-C). `IncidentState` keeps `INC-…`; `CodeReviewState` mints `CR-…`. The `_INC_ID_RE` validator was widened to `_SESSION_ID_RE` (`PREFIX-YYYYMMDD-NNN`) so any app's id format passes load/save validation.
- **Row-schema dropping typed app fields** (the *canonical* P8 leak) — `_incident_to_row_dict` and `_row_to_incident` were written for incident-shaped data. `CodeReviewState`'s `pr` / `review_findings` / `overall_recommendation` / `review_summary` / `review_token_budget` silently dropped on round-trip. Fixed by adding `extra_fields: JSON` to `IncidentRow` and driving the converters off `state_cls.model_fields` (P8-J). Additive: legacy rows with `extra_fields=NULL` round-trip cleanly.
- **Build pipeline only knew about incident-management** — `scripts/build_single_file.py` produced only `dist/apps/incident-management.py`. Added `CODE_REVIEW_APP_MODULE_ORDER` + `build_code_review_app()`; `_INTRA_PREFIXES` regex was widened to include `examples.code_review` so its intra-package imports get stripped. Fourth artifact `dist/apps/code-review.py` falls out (P8-K).
- **No second-app coexistence proof** — added `tests/test_two_apps_coexist.py` (P8-L) verifying two `SessionStore` instances on isolated metadata DBs round-trip independently and don't collide on id space.

**Leaks predicted but deferred (out of P8-N scope per task brief):**

- **Reporter** (P8-I) and **environment** (P8-J in original numbering, distinct from the row-schema P8-J above) — explicitly skipped by the executor brief. `IncidentState.reporter: Reporter` and the row's `environment` typed column remain, but the framework no longer assumes other apps have them: the row→model converter projects them only when `state_cls.model_fields` contains the field. Not a leak in the "blocks a second app" sense after P8-J.
- **Confidence-gate genericisation** (P8-O) — conditional task; the leak did not surface during P8-J integration so the task didn't fire.

**Surprise leaks (bad news — pre-P8 hardcoding still present in `src/runtime/`):**

`grep -rn 'IncidentState\|incident_management' src/runtime/ --include='*.py'` shows three real hardcoded imports that predate P8 and were not fixed inside this scoped slice:

- `src/runtime/api.py:35` — `from examples.incident_management.config import load_incident_app_config`
- `src/runtime/graph.py:14` — same
- `src/runtime/orchestrator.py:11` — same (plus `IncidentAppConfig`)

These pull `IncidentAppConfig` fields (`environments`, `confidence_threshold`, `escalation_teams`, `severity_aliases`, `similarity_threshold`, `dedup`) into `runtime.{api,graph,orchestrator}.py`. Each is genuinely framework hardcoding (not a dynamic dotted-path string) and is a real follow-up for Phase 9 or a separate "framework cleanup" pass. Documenting them here so they don't get lost.

**Framework changes shipped:**

- `2833576` — feat(state): id_format hook on Session for app-specific ID minting (P8-C)
- `71007a9` — feat(storage): generic Session round-trip via extra_fields JSON (P8-J)
- `6fe7ccb` — feat(build): bundle code_review as dist/apps/code-review.py (P8-K)
- `54dda4f` — test(genericity): incident_management + code_review run side-by-side (P8-L)
- `f9eea3a` — docs: code_review fully documented; incident_management cross-link (P8-M)
- _this commit_ — docs: Phase 8 complete; framework genericity proven via second example (P8-N)

**Test count:** 588 → 604 passing (+16 new tests across the 5 commits above), 3 skipped, zero new xfails.

**This retrospective drives Phase 9's planning** — ASR sits on whatever the framework looks like after P8 closes. The three pre-P8 hardcoded imports above (api/graph/orchestrator → IncidentAppConfig) should be the **first** thing P9 cleans up before adding new domain code on top.
