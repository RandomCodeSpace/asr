# ADR 0001: Current architecture

**Status:** Accepted (snapshot of `main` as of v1.5, post-PR #11)

**Date:** 2026-05-14

**Context:** This ADR captures the architectural baseline that
v1.5 ships. It is a synthesis of the twelve numbered decisions in
`docs/DESIGN.md` § 12 (DEC-001 through DEC-012). Future ADRs
should be written for new decisions that supersede or refine this
baseline.

---

## Decision

The framework's architecture composes three external layers
(LangGraph, LangChain, FastMCP) with a generic runtime + two
example apps, deployed as a single-file bundle into air-gapped
corporate environments.

### Layer composition

| Layer | Provided by | Owned by us |
|---|---|---|
| Provider clients | `langchain-openai`, `langchain-ollama` | NO |
| Agent factory (per-skill ReAct loop) | `langchain.agents.create_agent` (which is itself a langgraph subgraph) | NO |
| Graph orchestration / checkpointing / `interrupt()` | `langgraph` 1.x | NO |
| MCP tool servers | `fastmcp` | NO |
| **Framework abstractions** (`Session`, `Skill`, `Orchestrator`, gateway, telemetry, storage, bundling, HITL plumbing) | THIS REPO (`src/runtime/`) | YES |
| **Apps** (state subclass, MCP servers, skill prompts) | THIS REPO (`examples/`) or external | YES (examples) / external (downstream apps) |

### Decision summary

Reference: each is detailed in `docs/DESIGN.md` § 12.

| ID | Decision | Why |
|---|---|---|
| DEC-001 | LangGraph as orchestration engine | Out-of-the-box Pregel-style step boundaries + checkpointing + first-class HITL `interrupt()` |
| DEC-002 | `langchain.agents.create_agent` as the per-agent loop (Phase 15) | Single tool-loop; AutoStrategy → ToolStrategy fallback; removed the `recursion_limit=25` workaround |
| DEC-003 | Markdown turn-output contract over `response_format` JSON (Phase 22) | JSON schema brittleness across providers; markdown is what every chat model writes well; parse leniency under our control |
| DEC-004 | Pure-policy HITL gating (Phase 11) | One source of truth (`should_gate`); auditing what gates is one grep |
| DEC-005 | Generic `Session` base + `extra_fields` JSON (v1.1) | Apps extend without schema migrations; framework stays domain-agnostic |
| DEC-006 | Per-agent `skill.model` override (v1.5-C / M8) | Cheap models for cheap agents; one config knob |
| DEC-007 | Single-file bundle for air-gap deploy (BUNDLER-01) | Copy-only deploy; no `pip install` at deploy time |
| DEC-008 | Concept-leak ratchet (v1.5-B) | CI-enforced framework genericity; downward-only count |
| DEC-009 | 429 separate retry regime (v1.5-D) | Free upstream tiers (OpenRouter `…:free`) need 30-60s windows; 5xx default backoff exhausts in 9s |
| DEC-010 | Inner agent checkpointer + reload-on-entry (PR #6) | langgraph 1.x `__interrupt__` semantics + outer Pregel step-boundary checkpointing → reload defends against stale state |
| DEC-011 | Two example apps to prove genericity | Without a second app, "is the framework generic?" is unanswerable |
| DEC-012 | Bundle staleness CI gate (HARD-08) | dist drift = deploy-time bugs; CI rebuilds + diff every PR |

---

## Consequences

### Positive

- **Air-gap deployable** — copy-only 7-file payload; no runtime
  internet dependencies; reproducible installs via `uv.lock`.
- **Genuinely generic** — two distinct example apps prove the
  decoupling; CI ratchet keeps it that way.
- **HITL is first-class** — risk-rated gateway, durable pause via
  langgraph checkpointer, two approval surfaces (UI + API), watchdog
  for stale approvals.
- **Per-step observability** — `EventLog` rows for every
  meaningful boundary, drives the auto-learning lesson store and
  any external observability stack.
- **Provider-agnostic** — Ollama / Azure / OpenAI-compatible via
  one config knob; per-skill override.
- **Resilient to provider quirks** — markdown contract + Path 5/6
  synthesis fallbacks; 429 backoff regime; provider timeout +
  retry on 5xx.

### Negative

- **Two heavy upstream dependencies** (`langgraph`, `langchain`)
  with histories of breaking semantic changes (PR #6 caught one;
  more likely on future major bumps).
- **Single-process model** — `OrchestratorService` is one asyncio
  loop on one host. Multi-host / multi-tenant deploys need
  separate orchestrators per tenant.
- **No built-in auth on the FastAPI surface** — relies on corporate
  network controls. Webhook triggers have bearer auth only.
- **Schema migrations are ad-hoc** — no Alembic. Additive changes
  use `Base.metadata.create_all`; destructive changes need
  hand-rolled scripts.
- **Concept-leak residue** — 39 tokens still on the `incident` /
  `severity` / `reporter` axis after v1.5-B, mostly schema-coupled
  columns + legacy `/incidents/*` URL routes that would require
  destructive migration to remove. Documented in
  `docs/DESIGN.md` § 12 DEC-008.
- **Bundle files are large** (~660-700KB each). Code review on
  `dist/*` is impractical; reviewers focus on `src/runtime/`
  diffs and trust the bundle gate.
- **Streamlit UI is a prototype** — slated for replacement by a
  React UI (v2.0, not started). Adds a transitional cost.

### Neutral

- **No queue / messaging integration shipped** — trigger registry
  + plugin transport ABC exists, but no SQS/Kafka/NATS in-tree.
- **No container Dockerfile** — Inference: bare-VM / systemd
  deploy assumed.
- **No semver tags** — `pyproject.toml` declares `0.1.0`; the
  v1.0 → v1.5 milestone labels are documentation-level, not git
  tags. Squash SHAs in `docs/DESIGN.md` § 13 are the canonical
  references.

---

## Alternatives considered

### Build a graph engine ourselves

Rejected (DEC-001 implicitly). LangGraph's Pregel + checkpointer +
interrupt semantics are exactly what HITL needs. Owning the
orchestration engine would cost us a year of work for a similarly-
shaped result.

### Stay on `langgraph.prebuilt.create_react_agent`

Rejected in Phase 15 (DEC-002). The prebuilt was deprecated; the
`recursion_limit=25` workaround we needed to avoid infinite loops
was a symptom of the prebuilt's interaction with our structured-
output post-pass. `langchain.agents.create_agent` runs a single
tool-loop with native ToolStrategy fallback, removing the workaround.

### Stay on `response_format=AgentTurnOutput` JSON envelope

Rejected in Phase 22 (DEC-003). `response_format` triggered three
classes of brittleness: model-specific JSON drift, tool-strategy +
React END interaction, recursion-limit ceilings. Markdown is the
native format every chat model writes well; the parse step now
happens in our code where leniency is in our control.

### Keep `IncidentState` as the only state class

Rejected in v1.1 (DEC-005). Adding a second app (code_review) was
the forcing function — every "incident-shaped" leak that surfaced
during code-review's build moved into the framework rather than
becoming an app workaround. The concept-leak ratchet (DEC-008,
v1.5-B) keeps this honest.

### Multi-file deploy (zip / tarball / wheel + venv)

Rejected for BUNDLER-01 (DEC-007). Air-gap target is copy-only;
multi-file `pip install` at deploy time is out of scope. The
bundler turns the multi-file source tree into the smallest
possible deploy payload (7 files).

### Use Alembic for schema migrations

Considered, rejected (Inference). Schema changes have been purely
additive so far. When a destructive change becomes necessary,
adding Alembic at that point is straightforward. Until then, the
pydantic + JSON-bag pattern keeps schema rare.

### Multi-agent supervisor as the entry point (instead of intake)

Considered (Phase 6 introduced `kind: supervisor`). The
incident-management example app uses a supervisor for intake (rule-
based dispatch); other apps use a `responsive` skill at entry
(`code_review` does). The framework supports both patterns equally.

---

## Open questions to revisit in future ADRs

These are decisions the v1.5 baseline does NOT take a strong
position on:

1. **Multi-host orchestration.** When does the single-process model
   stop scaling? Does the answer involve a shared lock service, a
   queue between orchestrators, or just "shard by app"?
2. **Authentication on the FastAPI surface.** Air-gap defers this;
   if v2.0 React UI is hosted on a corporate intranet with SSO,
   we'll need at least a JWT verification layer. ADR 0002?
3. **Postgres CI coverage.** The `asr[postgres]` extra ships but
   no CI test exercises it. A postgres container in CI would
   close the gap; cost is CI time + workflow complexity.
4. **Trigger fan-in transports.** SQS / Kafka / NATS plugin
   transports exist as scaffold — no production user yet. When
   the first arrives, the plugin transport ABC may need refining.
5. **React UI architecture.** Stack pick (Next.js? Vite +
   React Router?), state management (TanStack Query?), API codegen
   from a generated OpenAPI spec? ADR 0003 territory.
6. **Lesson-store pruning.** `LessonRow` is append-only; soft delete
   exists but there's no automatic GC. At what corpus size do
   intake's relevance lookups slow down enough to need pruning?
7. **Dual-write inconsistency between IncidentRow.pending_intervention
   and the langgraph checkpointer.** Currently both are written
   when a gate pauses; race-window between the two writes is
   tolerated (operator dashboards may briefly disagree). Worth a
   focused test or a transactional wrapper?

---

## Related documents

- `docs/DESIGN.md` — long-form architecture narrative + decision
  rationale + milestone history
- `docs/00-project-overview.md` — what / who / status
- `docs/02-architecture.md` — quick-scan summary of the layers +
  data flow
- `docs/04-main-flows.md` — entry points + failure modes per flow
- `docs/06-data-model.md` — entities + relationships +
  persistence assumptions
- `docs/10-known-risks-and-todos.md` — what's pending
- `docs/11-agent-handoff.md` — action card for AI agents
