PHASE: P9
SUMMARY: ASR.md is missing from disk (gitignored, untracked, never committed) — only the roadmap describes Phase 9. Per roadmap §1 and §2, P9 is `examples/incident-management/` build-out into the ASR flagship; there is no parallel `examples/asr/`. P9 spans 13 sub-phases (9a-9m): one schema enrichment, five memory-layer MCP servers (L1/L2/L4/L5/L7), four agent kinds, dedup, and UI. Most decisions hinge on missing ASR.md content, so user input is the gate.

DECISIONS:
- header: "ASR.md spec is missing — recover or reconstruct"
  question: "ASR.md is referenced 14 times across the roadmap as the authoritative source for `IncidentState` schema (9a), 7-layer memory taxonomy (9b-9f), agent topology (9g-9k), and risk policy. It is not on disk, not in git, and listed in .gitignore. Without it, sub-phases 9a–9f cannot be plan-detailed because the data shapes (hypothesis array, L1 hot-buffer schema, L2 graph nodes, L5 release-note structure, L7 playbook step model) are undefined. How do we proceed?"
  options:
    - "User pastes / re-creates ASR.md from notes before P9 planning starts (highest fidelity, blocks until done)"
    - "Reconstruct ASR.md inline from roadmap hints + clarifying Q&A; treat reconstruction as the new source of truth (faster, may miss intent)"
    - "Defer ASR.md; plan only sub-phases that don't require it (9h supervisor-router, 9l dedup wiring, 9m UI shell) — limits P9 to ~3 of 13 sub-phases"

- header: "Directory layout: rename or coexist"
  question: "Roadmap §2 shows `examples/incident-management/` (hyphen) as the ASR target; current disk has `examples/incident_management/` (underscore) with only state.py / config.py / mcp_server.py / config.yaml. The user prompt mentions a possible `examples/asr/` parallel. Which physical layout does P9 land in?"
  options:
    - "Rename `incident_management` -> `incident-management` and build P9 there (matches roadmap, breaks Python import; needs package alias)"
    - "Keep `incident_management` as the ASR app — it IS the flagship; no `examples/asr/` ever (matches roadmap intent, zero rename churn)"
    - "Create new `examples/asr/`, leave `incident_management` as a thin starter, migrate piece-by-piece (cleanest naming, double maintenance during transition)"

- header: "MVP scope: which sub-phases define 'ASR works end-to-end'"
  question: "Roadmap estimates 2-3 months for full P9 and explicitly recommends incremental delivery. Thirteen sub-phases is too wide a front to sequence at once. Which slice constitutes the MVP demo (everything else is post-MVP)?"
  options:
    - "Thin vertical slice: 9a (schema) + 9d (L7 playbooks) + 9h (supervisor) + 9i (triage with evidence) + 9k (resolution with risk-gate) + 9m (UI). Defers L1/L2/L4/L5 hot context and the monitor loop. Demo is a manually-triggered incident through full agent flow."
    - "Knowledge-first slice: 9a + 9b (L2 graph) + 9e (L4 docs) + 9i (triage) + 9m. Demonstrates retrieval-augmented triage but no monitoring or resolution. Strong for a 'smart triage' demo."
    - "Monitor-first slice: 9a + 9f (L1 hot buffer) + 9g (monitor agent) + 9h + 9l (dedup). Demonstrates anomaly-driven session creation with dedup, but no investigation depth. Strong for a 'self-driving' demo."
    - "Big-bang: sequence all 13 sub-phases in dependency order, ship when done. Highest risk of stalling mid-flight."

- header: "Optional infra commitment: Neo4j, Redis, Postgres+pgvector"
  question: "Sub-phases 9b/9c/9f introduce three new infra services. Roadmap risk register flags these as needing 'vendored install paths for air-gapped deploy' and notes 'framework runs without them'. Each adds operational surface. Which infra do we commit to in P9?"
  options:
    - "All three: Neo4j (9b) + Postgres-pgvector (9c) + Redis (9f). Full ASR fidelity, full air-gapped vendor effort (3 new docker images, 3 install scripts, 3 backup stories)"
    - "Filesystem fallbacks only: 9b file-system markdown topology, 9c Postgres reusing existing engine, 9f in-process LRU cache instead of Redis. Single new piece of infra (pgvector tables), maximum portability"
    - "Postgres+pgvector only (9c full, 9b/9f deferred or stubbed). Reuses existing Postgres dependency, air-gapped story unchanged"
    - "Defer all three to post-MVP; ship 9a/9d/9h/9i/9k/9m on existing storage. Decide infra after the agent flow is validated"
