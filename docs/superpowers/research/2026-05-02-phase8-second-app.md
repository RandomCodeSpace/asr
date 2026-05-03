PHASE: P8

SUMMARY: Phase 8 ships a second example app under `examples/<name>/` (mirroring `examples/incident_management/`'s state.py + mcp_server.py + skills/ + ui.py + config.{py,yaml}) with the explicit goal of *forcing* the framework to prove it is not incident-flavored. The app choice is the highest-leverage call: pick something too close to IM (renamed-fields support triage) and P8 validates nothing; pick something that exercises a *different* state shape, *different* agent kinds, and *different* trigger source, and every leaked assumption surfaces as a framework fix before P9 (the ASR.md flagship build-out) sits on top. P8 must land before P9 starts.

DECISIONS:

- header: "App choice (the load-bearing decision)"
  question: "Which app should be the framework's second example, given the goal is to stress *every* abstraction (state schema, agent kinds, triggers, tool gateway, dedup, memory pattern) — not produce a polished product?"
  options:
    - name: "Code-review assistant (PR reviewer + commit-watcher)"
      stresses: "State = PR + diff hunks + threaded comments (no severity/environment/reporter); two agent kinds in one app — *responsive* on PR-open + *monitor-loop* (Phase 6) polling for new commits on open PRs; trigger = webhook (different from IM's API + cron); tool gateway exercises a real high-risk class (post-review-comment, request-changes, auto-merge — not 'apply_fix'); dedup = 'is this the same PR thread' (URL-keyed) which is a *different* dedup signal than IM's embedding similarity. Hits Phases 1, 2, 3, 4, 5, 6, 7 simultaneously. Strongest framework stress-test."
      effort: "Upper end of estimate (1.5-2 wk) — but every leak found here is a leak P9 would have hit later"
      verdict: "RECOMMENDED — exercises the most distinct framework surfaces in one app"
    - name: "Customer-support triage (email/Slack → ticket)"
      stresses: "Per-customer memory pattern (history_store keyed on customer_id, not by content similarity); trigger = inbound email/Slack event (new plugin trigger source — Phase 5); escalation target = humans-by-ticket-type (different HITL semantics than apply_fix approval); state has *thread* shape (turns of conversation), not single investigation. Good Phase 4/5 stress."
      effort: "Mid-estimate (1-1.5 wk)"
      verdict: "STRONG ALTERNATE — closer to IM than code-review (still 'inbound problem → resolve'), but the per-customer memory pattern and email trigger are genuinely novel"
    - name: "Documentation Q&A bot (one-shot retrieval)"
      stresses: "Genericity proof for *non-multi-step* workflows — does the framework support a single retrieve-and-answer node graph, or has it baked in 'investigation = N agents'? State is trivial (query + answer + sources); no HITL; no monitor loop; no dedup. *Inverse* stress test — finds where the framework over-assumes complexity."
      effort: "Lowest (3-5 days)"
      verdict: "VIABLE BUT WEAK — surfaces only one class of leak (over-engineering); doesn't exercise gateway, monitor-loop, or dedup; cheap proof but limited evidence"
    - name: "Code refactor agent (file-edit executor)"
      stresses: "Heaviest tool-gateway stress — every file edit is a high-risk tool, every git commit is high-risk, every push is critical. Phase 4 (risk-rated gateway) gets exercised hard. But state shape and agent kinds are similar to IM (one investigation → one resolution). Narrow stress."
      effort: "Mid (1-1.5 wk)"
      verdict: "VIABLE BUT NARROW — best Phase 4 stress test, but doesn't exercise Phase 5/6/7 distinctly"

- header: "Depth / scope"
  question: "Thin proof-of-concept or production-shaped?"
  options:
    - name: "Thin proof (3 skills, 1 MCP server, basic Streamlit UI, tests prove both apps coexist)"
      rationale: "Phase 8's stated goal is 'forcing the framework to demonstrate it doesn't leak' — not shipping a product. Plan estimates 1-2 wk *for that thin scope*. Anything deeper inflates effort without adding framework signal."
      verdict: "RECOMMENDED — matches the plan's stated deliverables (state, skills YAML, MCP server, config, README, tests)"
    - name: "Production-shaped (full UI, observability, deployment story)"
      rationale: "Doesn't add framework-validation signal; defers P9. Save production polish for the IM app in P9."
      verdict: "DEFER"

- header: "Sequencing vs P9"
  question: "Should P8 land before P9 (ASR build-out) starts, or run in parallel?"
  options:
    - name: "P8 fully merged before P9 starts"
      rationale: "Plan explicitly says 'each leak found = framework fix needed' and P9 has 13 sub-phases all sitting on top of the framework. Building P9's L2/L5/L7 MCP servers on a framework that hasn't been stress-tested means rework when P8 surfaces leaks. Sequential is the safer ordering."
      verdict: "RECOMMENDED"
    - name: "P8 and P9a-9c in parallel"
      rationale: "Faster on paper, but every framework fix forced by P8 invalidates P9 work in flight. Only viable if P8 is a thin doc-Q&A bot (which we're not picking)."
      verdict: "REJECT for code-review/support choices; viable only if P8 is the doc-Q&A option"
    - name: "Skip P8, go straight to P9"
      rationale: "Defeats the entire genericification thesis from Phases 1-7 — we'd ship a framework whose 'genericness' was never independently exercised."
      verdict: "REJECT"

KEY FRAMEWORK SURFACES THAT NEED EXERCISING (for grading any candidate):
1. State schema extension (Phase 1/2): does the second app's TypedDict cleanly extend `Session` without touching framework code?
2. Trigger plurality (Phase 5): does the second app introduce a *new* trigger source (webhook, email, file-watch) without modifying `framework/triggers/`?
3. Agent-kind plurality (Phase 6): does the second app use a *different mix* of responsive/supervisor-router/monitor-loop than IM?
4. Tool gateway risk policy (Phase 4): does the second app's high-risk tools (post-comment, send-email, file-write) flow through gateway HITL without bespoke code?
5. Dedup signal (Phase 7): does the second app define a non-embedding dedup key (URL, customer_id, thread_id) that the dedup pipeline accommodates?
6. History store keying: does `HistoryStore` work for a non-incident memory pattern (per-customer history vs per-content similarity)?

The code-review assistant hits 5 of 6 distinctly. Customer-support hits 4. Doc Q&A hits 1-2. Refactor hits 2. That's the case for code-review.
