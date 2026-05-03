# Code Review — Example Application

Second example app for the `runtime` framework. Built in Phase 8 to *prove* the framework is genuinely generic — every framework leak that surfaced while this app was being built (id format, row schema, build pipeline, intra-bundle imports) was lifted into the framework rather than worked around.

## Run

```bash
python -m examples.code_review
```

Launches the Streamlit UI for browsing in-flight and historical PR reviews.

## Architecture

A 3-skill responsive pipeline (`intake → analyzer → recommender`) that consumes a PR description, walks the diff, files structured `ReviewFinding`s, and emits an approve / request-changes / comment recommendation. The framework owns session lifecycle, agent dispatch, and tool gateway; this example owns domain shape, skill prompts, and MCP tools.

```
examples/code_review/
├── state.py             CodeReviewState(Session) + PullRequest + ReviewFinding
├── config.py            CodeReviewAppConfig + load_code_review_app_config
├── config.yaml          severity_categories, auto_request_changes_on, repos_in_scope
├── mcp_server.py        CodeReviewMCPServer with 3 tools
├── skills/              3 agent YAML configs + _common/ shared style prompt
│   ├── _common/style.md
│   ├── intake/
│   ├── analyzer/
│   └── recommender/
├── ui.py                Streamlit read-only viewer (mirrors incident UI patterns)
├── __main__.py          Entry point
└── README.md            this file
```

## State Model

`CodeReviewState(Session)` extends the framework's generic `Session` with:

| Field | Type | Purpose |
|---|---|---|
| `pr` | `PullRequest` | repo, PR number, title, author, base/head SHAs, line counts |
| `review_findings` | `list[ReviewFinding]` | severity, file, line, category, message, optional suggestion |
| `overall_recommendation` | `"approve" \| "request_changes" \| "comment" \| None` | final verdict |
| `review_summary` | `str` | rolled-up narrative for the human reviewer |
| `review_token_budget` | `int` | telemetry — running token spend on this review |

The framework only reads/writes the inherited `Session` lifecycle/telemetry fields (`id`, `status`, `created_at`, `agents_run`, `tool_calls`, `findings`, `pending_intervention`, `token_usage`). Every domain field above lands in the row's `extra_fields` JSON column on save and is hydrated back into the model on load — no incident-shaped row schema leaks here (P8-J).

## ID Format

Session ids look like `CR-YYYYMMDD-NNN` (e.g. `CR-20260503-001`). The format is owned by `CodeReviewState.id_format(seq=...)` (P8-C) so the code-review id namespace is disjoint from incident-management's `INC-...` namespace — both apps can share the same metadata DB without collisions.

## Configuration

Two layers, in order of precedence:

| Layer | File | What it owns |
|---|---|---|
| Framework | `config/config.yaml` | LLM providers + models, MCP servers, storage URL, paths, `runtime.state_class` |
| App | `examples/code_review/config.yaml` | `severity_categories`, `auto_request_changes_on`, `repos_in_scope`, `review_max_diff_kb` |

Set `runtime.state_class: examples.code_review.state.CodeReviewState` in the framework config so row hydration produces `CodeReviewState` instances and `id_format` is called on the right class.

## MCP Tools

`CodeReviewMCPServer` (FastMCP, name `"code_review"`) exposes three tools to the agents:

- `fetch_pr_diff(repo, number)` — returns `{diff, files_changed, additions, deletions}`. Reads from `tests/fixtures/code_review/<repo>/<number>.json` if present; otherwise synthesises a tiny canned diff so the example runs offline. **Mock — not a real GitHub fetch.**
- `add_review_finding(session_id, severity, file, line, category, message, suggestion=None)` — append a structured finding to `state.review_findings`. Validated against `severity_categories` from `CodeReviewAppConfig`.
- `set_recommendation(session_id, recommendation, summary)` — set `state.overall_recommendation` + `state.review_summary` and finalize the review.

The MCP loader picks this server up via `mcp.servers[*].module = examples.code_review.mcp_server` in the framework config.

## Skills

| Skill | Kind | Tools | Routes (success / default → fail) |
|---|---|---|---|
| `intake` | responsive | `fetch_pr_diff` | `→ analyzer` / `→ analyzer` / `→ __end__` |
| `analyzer` | responsive | `fetch_pr_diff`, `add_review_finding` | `→ recommender` / `→ recommender` / `→ __end__` |
| `recommender` | responsive | `set_recommendation` | `→ __end__` |

All three are `kind: responsive` (no supervisor / monitor) — Phase-6 supervisor support is not exercised here. Common prompt fragments (severity calibration, output shape) live in `skills/_common/style.md` and are inherited by every skill.

## Bundle

Like incident-management, code-review ships as a single self-contained file: `dist/apps/code-review.py`. Build via:

```bash
python scripts/build_single_file.py
```

This produces `dist/app.py` (framework-only), `dist/apps/incident-management.py`, **and** `dist/apps/code-review.py` from the same flattening pipeline (P8-K). All three are `ast.parse`-clean and runnable on a clean venv with only vendored deps.

## Limits / Out of Scope

- Tools are **mocked** — there is no real GitHub or GitLab integration. `fetch_pr_diff` reads a JSON fixture or returns synthetic data; `add_review_finding` and `set_recommendation` write only to the in-process session state.
- No incremental re-review — re-firing the trigger creates a new session.
- No supervisor skills — the diff is walked sequentially by the analyzer agent.
- No PR-author identity model — the framework does not ship a generic `Reporter` / `Actor` concept; each app names its own (`pr.author` here, `Reporter(id, team)` for incident-management).

## How This Proves the Framework Is Generic

Phase 8 was written *to surface and fix* framework leaks. The fixes that landed because this app needed them:

- **P8-C** — `Session.id_format()` classmethod hook. Every `Session` subclass mints its own id format (`INC-...` for incidents, `CR-...` here, anything for future apps). `SessionStore._next_id` no longer hard-codes the incident shape.
- **P8-J** — `extra_fields: JSON` column on the row schema. Round-trip is driven by `state_cls.model_fields`; typed-column fields stay typed, everything else round-trips through the JSON bag. Incident round-trip is preserved; code-review's `pr` / `review_findings` / `overall_recommendation` / `review_summary` / `review_token_budget` now persist losslessly.
- **P8-K** — bundler emits `dist/apps/code-review.py` from the same flattening pipeline as `dist/apps/incident-management.py`.
- **P8-L** — integration test: both apps run side-by-side on isolated metadata DBs without colliding on id space, leaking field shapes, or sharing state.

Phase 9 (ASR) builds on the framework as it stands after these fixes.

## Testing

```bash
pytest tests/test_code_review_*.py tests/test_two_apps_coexist.py tests/test_generic_round_trip.py tests/test_session_id_format.py tests/test_bundle_code_review.py -q --no-cov
```

App-level pin tests live alongside `tests/test_code_review_*.py`; the Phase-8 framework-leak fixes are pinned by `tests/test_session_id_format.py`, `tests/test_generic_round_trip.py`, `tests/test_bundle_code_review.py`, and `tests/test_two_apps_coexist.py`.
