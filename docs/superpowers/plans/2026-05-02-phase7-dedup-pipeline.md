# Phase 7 — Two-Stage Dedup Pipeline (Embedding + LLM)

**Status:** Plan only. No code. Implementation begins on explicit go-ahead.
**Owner:** ASR framework
**Date:** 2026-05-02
**Depends on:** P1 (storage split), P2 (Generic[StateT]), P3+ (LLM registry / named models)

---

## 1. Goal

Stop re-running full agent pipelines for incidents we've already seen. After intake finishes
and we have *some* signal (symptoms, environment, summary), run a two-stage dedup check:

1. **Stage 1** — embedding similarity over closed sessions in the same environment.
2. **Stage 2** — LLM confirmation on the top-K candidates.

If confirmed, link the new session to the prior one (`parent_session_id`), set
`status="duplicate"`, and stop the agent graph. Never destroy the new session — duplicates
are first-class records that can be retracted.

## 2. Locked decisions (NON-NEGOTIABLE — restated)

- **Merge semantics**: non-destructive link via `parent_session_id`. Always create the new
  session, mark `status="duplicate"`, set the link. Never delete, never overwrite.
- **Lifecycle timing**: post intake-finish (after first agent pass collects symptoms /
  initial context). Stage 2 LLM has full context. Brief two-session co-existence is
  acceptable; we need a clean "demote to duplicate" transition.
- **Model + retraction**: single named model from the global registry (default `cheap`).
  Retraction recorded in a dedicated table but **NOT** fed back to the prompt in P7 MVP.
  Retraction just flips the link/status.

## 3. Non-goals (P7)

- No active learning loop. Retraction logs are recorded for P8+, not used to retrain
  prompts or thresholds in P7.
- No pre-creation dedup (`run_at: pre_creation`) — config field exists but only
  `post_intake` is implemented. `passive` mode (background sweep) is also out of scope.
- No multi-parent / cluster merging. A duplicate has exactly one parent. If the parent is
  later marked a duplicate of a third session, we leave the chain intact (linked list);
  the UI follows links to the root on display.
- No cross-environment dedup. Stage 1 already filters by environment; we don't relax it.
- No automatic re-run of dedup after edits. Dedup runs once per session, post-intake.

## 4. Source material

- ASR roadmap, **Phase 7** section (dedup pipeline).
- `src/runtime/storage/history_store.py` — post-P1 split. Existing `find_similar` does
  embedding-then-keyword fallback. Stage 1 wraps this; we keep the keyword fallback off
  the dedup path (it's for *retrieval*, not *dedup*).
- `examples/incident_management/mcp_server.py` — `lookup_similar_incidents` MCP tool.
  Currently embedding-only. After P7 it can optionally chain stage 2 (out of scope here;
  the dedup pipeline is reusable but the MCP tool change is left for P7+ follow-up).
- `src/runtime/orchestrator.py` — where the lifecycle hook plugs in (post-intake node).
- `src/runtime/state.py` — `IncidentState` (lives under `examples/`, not `runtime/`).
  P7 must not import it from the framework. See §11 R4.

## 5. Architecture overview

```
                       intake agent finishes
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Orchestrator: post-intake hook│
                  └──────────────┬───────────────┘
                                 │ if config.dedup.enabled
                                 ▼
                       DedupPipeline.run(session,
                                          history_store,
                                          llm_registry,
                                          text_extractor)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
   Stage 1: embedding             (no candidates ≥ stage1_threshold?)
   top-K closed sessions                   │
   in same environment                     ▼
              │                       continue normal flow
              ▼
   Stage 2: LLM confirm
   for each candidate (cap K=3)
   structured output:
     {is_duplicate: bool,
      confidence: float,
      rationale: str}
              │
   any (is_duplicate AND confidence ≥ stage2_min_confidence)?
              │
       yes ───┴─── no
        │           │
        ▼           ▼
 Mark duplicate    continue normal flow
 stop graph
```

`DedupPipeline` is framework-level (`runtime/dedup.py`). It is generic over state via a
`text_extractor: Callable[[StateT], str]` supplied by the app. The framework does **not**
import `IncidentState`.

## 6. Config surface

Extend `IncidentAppConfig` (or framework `AppConfig` if we have one post-P2) with a
`dedup` block. Off by default at the framework level; the incident-management example
turns it on.

```yaml
dedup:
  enabled: true            # framework default: false; example default: true
  stage1_threshold: 0.6    # cosine similarity floor for Stage 1 candidates
  stage1_top_k: 5          # how many Stage 1 hits to consider
  stage2_top_k: 3          # how many of those to actually run Stage 2 on (cost cap)
  stage2_model: cheap      # name from the LLM registry
  stage2_min_confidence: 0.7
  run_at: post_intake      # only valid value in P7; field reserved for future
  scope:
    same_environment: true # filter Stage 1 candidates to matching env
    only_closed: true      # ignore in-flight sessions (default true)
```

Validation:

- `0 ≤ stage1_threshold ≤ 1`
- `0 ≤ stage2_min_confidence ≤ 1`
- `1 ≤ stage1_top_k ≤ 20`, `1 ≤ stage2_top_k ≤ stage1_top_k`
- `stage2_model` resolves in the LLM registry at config-load time (fail fast).
- `run_at == "post_intake"` (anything else: fail fast with clear error).

## 7. Data model

### 7.1 Session / IncidentRow changes

Add to the persisted incident row and in-memory session:

| Field                | Type             | Notes                                                  |
|----------------------|------------------|--------------------------------------------------------|
| `parent_session_id`  | `str \| None`    | NULL by default; set when this session is a duplicate. |
| `status`             | `IncidentStatus` | Literal extended with `"duplicate"`.                   |

Migration: add a nullable column `parent_session_id TEXT` to `incidents`, indexed.
Existing rows: NULL. No backfill needed.

### 7.2 IncidentStatus literal

Extend the `Literal` type used for `IncidentState.status` (and the DB CHECK if any) with
`"duplicate"`. Update any switch / match sites — there are a small number; enumerate them
in P7-B.

### 7.3 New table: `dedup_retractions`

```
dedup_retractions
─────────────────
id                INTEGER PRIMARY KEY
session_id        TEXT NOT NULL          -- the session that was un-duplicated
retracted_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
retracted_by      TEXT NULL              -- user id / name if available, else NULL
original_match_id TEXT NOT NULL          -- the parent_session_id at time of retraction
note              TEXT NULL              -- optional free-text reason
```

Index on `session_id`. No FK — keep it append-only and tolerant of session deletion.

## 8. Component changes

### 8.1 `runtime/dedup.py` (NEW)

```python
class DedupDecision(BaseModel):
    is_duplicate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

class DedupResult(BaseModel):
    matched: bool
    parent_session_id: str | None
    candidate_id: str | None
    decision: DedupDecision | None
    stage1_score: float | None

class DedupPipeline(Generic[StateT]):
    def __init__(self, config: DedupConfig, llm_registry: LLMRegistry,
                 text_extractor: Callable[[StateT], str]): ...

    async def run(self, session: Session[StateT],
                  history_store: HistoryStore) -> DedupResult: ...
```

Stage 1: call `history_store.find_similar(text, env, top_k=stage1_top_k,
threshold=stage1_threshold, only_closed=True)`. Filter out the current session's own id.

Stage 2: for the top `stage2_top_k` candidates, call the registered model with a
structured-output prompt (see §8.2). Short-circuit on the **first** candidate that returns
`is_duplicate=True AND confidence ≥ stage2_min_confidence` — return that as the match.
We do not score-rank Stage 2 results; first confirm wins. Stage 1 ordering already
prioritizes by similarity.

Returns `DedupResult(matched=True, ...)` if confirmed; otherwise
`DedupResult(matched=False, ...)`.

The pipeline does not mutate the session. The orchestrator owns mutation. (Easier to
test, easier to dry-run.)

### 8.2 Stage 2 prompt + structured output (P7-E)

Prompt skeleton (kept compact — `cheap` model, latency matters):

```
System: You are deduplicating incident reports for an SRE platform.
Two reports are duplicates iff they describe the same root cause AND the same
service / environment AND overlap in time-of-occurrence. Surface-level keyword overlap
is NOT enough.

User:
[INCIDENT A — existing, id=<id>, opened=<ts>, env=<env>]
<text_extractor(prior_session)>

[INCIDENT B — new, id=<id>, opened=<ts>, env=<env>]
<text_extractor(new_session)>

Decide: is B a duplicate of A?

Respond with JSON matching schema:
  is_duplicate: bool
  confidence: float in [0, 1]
  rationale: 1-2 sentences
```

Use the LLM client's structured-output / JSON-mode path (Pydantic schema =
`DedupDecision`). On parse failure: log, treat as `is_duplicate=False`, do not retry —
budget protection.

### 8.3 `Orchestrator` lifecycle hook (P7-F)

Add a node `dedup_check` between `intake` and the next agent. Pseudocode:

```
async def dedup_check(state):
    if not self.config.dedup.enabled:
        return state                               # passthrough
    result = await self.dedup_pipeline.run(
        session=self.current_session,
        history_store=self.history_store,
    )
    if result.matched:
        state.status = "duplicate"
        state.parent_session_id = result.parent_session_id
        await self.session_store.update(state)
        return Command(goto=END)                   # stop graph
    return state
```

Wire ordering: `START → intake → dedup_check → (existing pipeline)`. The existing
pipeline is unchanged; only the entry point shifts.

### 8.4 `SessionStore.list_recent` (P7-G)

Default behaviour: filter out `status="duplicate"`. New parameter
`include_duplicates: bool = False`. UI uses `include_duplicates=True` to render the
collapsed duplicate row; main list stays clean.

Also expose `list_children(parent_session_id) -> list[Session]` for the parent detail
pane.

### 8.5 Retraction API (P7-H)

`POST /sessions/{id}/un-duplicate`

Body: `{"note": "<optional free text>"}`. Auth: same as the rest of the session API.

Behaviour:

1. Load session by id; 404 if missing.
2. If `status != "duplicate"`, return `409 Conflict` with `{error: "not a duplicate"}`.
3. Capture `original_match_id = session.parent_session_id`.
4. Set `status = "new"` (the canonical post-intake status before dedup ran — see note),
   clear `parent_session_id`.
5. Insert a row in `dedup_retractions` (`session_id`, `retracted_at=now`,
   `retracted_by=auth user or null`, `original_match_id`, `note`).
6. Return `200 OK` with the updated session.

> **Note on restored status**: P7 picks `"new"` as the post-retraction status because the
> session was demoted to duplicate immediately after intake. We do **not** re-run the
> agent pipeline on retraction — operator must trigger that explicitly. Doc this in the
> API response and changelog.

### 8.6 UI surface (P7-I)

Sidebar:

- Default list excludes duplicates (matches `list_recent` default).
- A single collapsed row "N duplicates (click to expand)" pinned at the bottom of the
  list. Expanding shows the duplicate sessions inline, visually subdued (lower opacity,
  small "duplicate of …" caption).

Detail pane — duplicate session:

- Banner: "Marked duplicate of <parent id> on <retraction-eligible timestamp>".
- Link: parent session.
- Button: "Unmark duplicate" → calls retraction API, refreshes view.
- Stage-2 rationale shown in a collapsed "Why?" disclosure (read from the row we logged
  on dedup; storage TBD — simplest: stash on the session row as `dedup_rationale TEXT`).
- Following user's locked Streamlit style: accordion-per-item, badge-rich header
  ("DUPLICATE" badge + parent id + confidence pct), sharp corners.

Detail pane — parent session:

- Children section: count + list of links. Empty when no children.

### 8.7 Bundle regeneration

Per user memory: edits live in `src/`, but `dist/app.py` is the ship target. After P7
edits, regenerate `dist/app.py` and verify the 7-file payload still installs cleanly in
the corporate copy-only env.

## 9. Tests (P7-J)

Use `pytest`. Stub the LLM with a deterministic fake.

### 9.1 Unit — Stage 1 wrapper

- `find_similar` returns N candidates ordered by score → pipeline filters by
  `stage1_threshold`, drops self, caps to `stage1_top_k`.
- No candidates above threshold → returns `DedupResult(matched=False)` immediately
  without calling the LLM (assert LLM stub never invoked).
- `only_closed=True` → in-flight sessions excluded.

### 9.2 Unit — Stage 2 alone

- LLM stub returns `is_duplicate=True, confidence=0.9` → matched.
- `is_duplicate=True, confidence=0.5` with `min_confidence=0.7` → not matched.
- `is_duplicate=False` → not matched, even at `confidence=0.99`.
- LLM returns malformed JSON → caught, treated as not-duplicate, no retry, logged.
- Pipeline calls Stage 2 at most `stage2_top_k` times even if Stage 1 returns more.
- Short-circuit: first confirmed match stops further Stage 2 calls.

### 9.3 Integration — pipeline end-to-end

Fixture: two near-identical "payments-latency" incidents, second opened minutes after
first. Run intake on both; second should land in `dedup_check` and be flagged.

- `session_2.status == "duplicate"`.
- `session_2.parent_session_id == session_1.id`.
- `session_2`'s graph stops; downstream agent nodes never run (assert via call counts).
- `list_recent()` returns `[session_1]` only.
- `list_recent(include_duplicates=True)` returns both.
- `list_children(session_1.id)` returns `[session_2]`.

### 9.4 Integration — retraction

- `POST /sessions/{session_2.id}/un-duplicate` →
  - 200, body shows updated session.
  - `status == "new"`, `parent_session_id is None`.
  - One row in `dedup_retractions` with `original_match_id == session_1.id`.
- Calling retraction on a non-duplicate → 409.
- Calling retraction on a missing id → 404.

### 9.5 Config / disabled behaviour

- `dedup.enabled=false` → `dedup_check` node is a passthrough; LLM stub never called;
  no DB writes related to dedup.
- Invalid config (e.g., `stage2_min_confidence=1.5`) → fails at config load.
- Unknown `stage2_model` → fails at config load.

### 9.6 Threshold edges

- Stage 1 hit just over `stage1_threshold`, Stage 2 says no → not a duplicate.
- Stage 1 hit just under threshold → Stage 2 never called.
- Stage 1 hits at exactly `stage1_threshold` → included (≥, not >).

### 9.7 Generic decoupling

- Compile-time / lint check (or simple grep test) asserting `runtime/dedup.py` does
  **not** import from `examples.incident_management`.

## 10. Tasks

| Task  | Title                                                  | LoC est. | Depends on |
|-------|--------------------------------------------------------|----------|------------|
| P7-A  | Add `parent_session_id` to Session and IncidentRow     | ~80      | —          |
| P7-B  | Add `"duplicate"` to `IncidentStatus` literal          | ~40      | P7-A       |
| P7-C  | Define `DedupConfig` in `IncidentAppConfig`            | ~80      | —          |
| P7-D  | Implement `runtime/dedup.py` — `DedupPipeline`         | ~180     | P7-A, P7-C |
| P7-E  | Stage 2 LLM prompt + Pydantic structured output        | ~80      | P7-D       |
| P7-F  | Lifecycle hook in `Orchestrator` (post-intake)         | ~100     | P7-D, P7-E |
| P7-G  | `SessionStore.list_recent` excludes duplicates         | ~60      | P7-B       |
| P7-H  | Retraction API + `dedup_retractions` table             | ~150     | P7-B, P7-G |
| P7-I  | UI: collapsed duplicate rows + parent/child links      | ~220     | P7-G, P7-H |
| P7-J  | Tests (units + integration) + final verification       | ~400     | all above  |

Total: ~1390 LoC including tests. Implementation order: A → B → C → D → E → F → G → H →
I → J. P7-A and P7-C can be parallelized; everything else is sequential.

### 10.1 Per-task acceptance criteria (concise)

- **P7-A**: column exists, model exposes field, migration runs idempotently, existing
  tests pass.
- **P7-B**: literal updated, all callers compile; no `match` / `if` chain misses the new
  case (verify by grep + type-check).
- **P7-C**: invalid configs rejected at load; `stage2_model` resolves against registry.
- **P7-D**: pipeline returns correct `DedupResult` for the four canonical cases (no
  candidates / candidates but Stage 2 declines / Stage 2 confirms / Stage 2 errors). No
  imports from `examples.*`.
- **P7-E**: prompt + Pydantic schema; malformed output gracefully handled.
- **P7-F**: post-intake call sites mutate state and short-circuit graph on match; no
  effect when disabled.
- **P7-G**: `list_recent` defaults exclude duplicates; opt-in flag returns them.
  `list_children` returns ordered list.
- **P7-H**: API contract per §8.5; retraction row written; idempotent (calling twice on
  a non-duplicate returns 409, never corrupts state).
- **P7-I**: passes the locked Streamlit style (accordion, badges, sharp corners).
  `prefers-reduced-motion` honoured; keyboard-navigable.
- **P7-J**: full suite green; coverage on `runtime/dedup.py` ≥ 90%; integration test from
  §9.3 included.

## 11. Risks & mitigations

- **R1 — Race condition**: two near-simultaneous similar incidents both fire intake
  before either runs dedup. Both run intake; the second is flagged when its dedup_check
  fires. *Acceptable* — intake is cheap; we explicitly accept the brief co-existence.
- **R2 — Stage 2 false positives**: LLM says yes but humans disagree. Retraction API
  handles it (P7-H). No auto-feedback in P7. We log retractions for P8+ analysis.
- **R3 — Stage 2 cost**: calling the LLM for every new incident adds latency. Mitigate
  by (a) only calling Stage 2 when Stage 1 finds candidates; (b) cap at `stage2_top_k=3`
  (configurable); (c) short-circuit on first confirmed match; (d) default model is
  `cheap` from the registry.
- **R4 — Framework / app coupling**: `runtime.dedup` must not import `IncidentState`
  from `examples.incident_management`. Use `Generic[StateT]` from P2 and a
  `text_extractor(state) -> str` callable supplied by the app. Enforced by the test in
  §9.7.
- **R5 — Status transition gaps**: existing call sites that switch on `IncidentStatus`
  may not handle `"duplicate"`. Audit with grep/type-check during P7-B; fix anything
  surfaced before moving on.
- **R6 — Embedding scope drift**: Stage 1 hitting closed sessions in *other* envs would
  poison results. Enforced by `scope.same_environment=true` default, exercised in §9.1.
- **R7 — Bundle regen forgotten**: per memory, `dist/app.py` is the ship target. P7-J's
  final verification step explicitly regenerates and re-tests the bundled artifact.

## 12. Open questions (resolve before implementation, not blockers for the plan)

1. Where do we persist the Stage 2 rationale for the UI? Inline column on the session row
   (`dedup_rationale TEXT`) or a small `dedup_decisions` table? Default: inline column —
   simpler, one rationale per session, no join.
2. Should the parent session's detail pane show a count of *retracted* duplicates (i.e.
   sessions that were once children but have been un-duplicated)? Default: no — out of
   scope; revisit in P8.
3. Should we expose Stage 1 / Stage 2 telemetry (counts, hit rate) on a metrics endpoint?
   Default: no in P7; cheap to add later via the retraction table + log scrape.

## 13. Done criteria

- All ten tasks complete with their acceptance criteria met.
- Tests in §9 all pass.
- Disabled-by-default behaviour verified at the framework level; example app turns it on.
- `dist/app.py` regenerated and the 7-file payload installs and runs in the copy-only
  corporate target.
- Self-review of the full diff per `rules/testing.md`.
- Short summary delivered: changed files, follow-ups, known limitations (active
  learning + retraction feedback deferred to P8).
