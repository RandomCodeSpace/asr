# Code Review — Example Application

Second example app for the framework. A 3-skill PR review pipeline
(intake → analyzer → recommender) that walks a diff, files structured
findings, and emits an approve / request-changes / comment verdict.

This app exists to **prove the framework is genuinely generic** —
it was built specifically to surface every incident-shaped
assumption that hadn't yet been lifted out of `src/runtime/`. Each
leak became a framework PR rather than an app workaround.

For framework-wide design + decisions, see
[`docs/DESIGN.md`](../../docs/DESIGN.md). This README only covers
the bits specific to this app.

## Run

```bash
uv run python -m runtime --config config/code_review.yaml
ASR_LOG_LEVEL=INFO uv run streamlit run src/runtime/ui.py --server.port 37777
```

## Layout

```
examples/code_review/
├── state.py             CodeReviewState(Session) + PullRequest + ReviewFinding
├── config.py            CodeReviewAppConfig + load_code_review_app_config
├── config.yaml          severity_categories, auto_request_changes_on, repos_in_scope
├── mcp_server.py        CodeReviewMCPServer + 3 tools
├── skills/              3 agent YAML configs + _common/ shared style prompt
│   ├── _common/style.md
│   ├── intake/
│   ├── analyzer/
│   └── recommender/
├── ui.py                Streamlit read-only viewer
└── __main__.py          entry point
```

## Domain shape

`CodeReviewState(Session)` adds `pr: PullRequest`,
`review_findings: list[ReviewFinding]`, `overall_recommendation`,
`review_summary`, `review_token_budget`. Session ids look like
`CR-YYYYMMDD-NNN`.

`PullRequest` carries repo / number / title / author / base+head SHAs
/ line counts. `ReviewFinding` carries severity / file / line /
category / message / optional suggestion. Both are pydantic models
declared in this app's `state.py`.

## MCP tools

`CodeReviewMCPServer` exposes:

- `fetch_pr_diff(repo, number)` — **mock**: reads from
  `tests/fixtures/code_review/<repo>/<number>.json` if present,
  otherwise returns a small canned diff so the example runs offline.
- `add_review_finding(session_id, severity, file, line, category,
  message, suggestion=None)` — append a structured finding to
  `state.review_findings`. Severity is validated against
  `severity_categories` from `CodeReviewAppConfig`.
- `set_recommendation(session_id, recommendation, summary)` —
  finalize the review. Sets `state.overall_recommendation` +
  `state.review_summary`.

No real GitHub/GitLab integration; tools are mocks for demonstration.

## Skills

| Skill | Tools | Routes |
|---|---|---|
| `intake` | `fetch_pr_diff` | → analyzer |
| `analyzer` | `fetch_pr_diff`, `add_review_finding` | → recommender |
| `recommender` | `set_recommendation` | → __end__ |

All three are `kind: responsive`. Common prompt fragments live in
`skills/_common/style.md` and are inherited.

## Limits / Out of scope

- Tools are mocked (no real GitHub/GitLab API calls).
- No incremental re-review (re-firing the trigger creates a new
  session).
- No supervisor / monitor skills exercised.
- No PR-author identity model — each app names its own
  (`pr.author` here, `Reporter(id, team)` in incident-management).
