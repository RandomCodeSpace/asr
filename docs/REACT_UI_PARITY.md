# React UI parity vs. Streamlit prototype

Gate for v2.0.0-rc1 → v2.0.0 promotion: every row marked **partial** or
**missing** below must move to **full** before the Streamlit prototype
can be removed in v2.1.

Streamlit module: `src/runtime/ui.py`.
React app: `web/src/`.

## Coverage matrix

| Streamlit feature (function) | React equivalent | Status | Notes |
|---|---|---|---|
| Sidebar — session list (`render_sidebar`, `_render_session_row`) | `<SessionsRail>` + `<MonitorRail>` "Other Sessions" panel | **full** | Same data source (GET /api/v1/sessions); React adds keyboard-free selection + per-session badges |
| Sidebar — active in-flight row (`_render_active_row`) | `<SessionsRail>` row with `data-active="true"` styling | **full** | React shows breathing dot via `asr-pulse` |
| Investigate form (top-level form in `main`) | `<NewSessionModal>` | **full** | POST /api/v1/sessions with `{query, environment, submitter}`. Same envelope; modal vs. inline |
| Session header / metadata (`_render_top_badges`, `_render_metrics`) | `<CanvasHead>` (eyebrow + title + meta row) | **full** | React adds active pulse + STOP / RETRY buttons inline |
| Findings block (`_render_findings_block`) | `<SessionCanvas>` → `<Transcript>` → `<Turn>` body summary | **full** | Editorial layout vs. KV block; same source `session.findings` |
| Resolution block (`_render_resolution_block`) | `<Transcript>` terminal turn + `<CanvasHead>` status pill | **full** | React shows status as `RESOLVED` pill rather than a separate section |
| Hypothesis trail (`_render_hypothesis_trail_block`) | `<SelectedPanel>` when a tool call surfaces hypotheses; embedded in turn meta | **partial** | React surfaces hypothesis-shaped data via SelectedPanel but lacks the dedicated "Trail" view. Defer to v2.1; the underlying tool-call audit is unchanged. |
| Pending approvals (`_render_pending_approvals_block`) | `<HITLBand>` inline + `<ApprovalsQueuePanel>` cross-session list | **full** | React drives the same POST /api/v1/sessions/{sid}/approvals/{tcid} endpoint |
| Approve action | `<HITLBand>` Approve button → direct apiFetch | **full** | rationale=null path |
| Approve with rationale | `<HITLBand>` Approve-with-rationale → `<ApproveRationaleModal>` | **full** | Includes uiHints.approval_rationale_templates chip row (Task 52) |
| Reject action | `<HITLBand>` Reject → `<ConfirmModal>` destructive | **full** | Same endpoint with `decision: 'reject'` (Task 53) |
| Stop session | `<CanvasHead>` Stop button → `<ConfirmModal>` destructive | **full** | DELETE /api/v1/sessions/{sid} (Task 53) |
| Retry decision (`_render_retry_block`, `_preview_retry_decision_sync`) | `<CanvasHead>` Retry button (visible when status='error') | **partial** | Button calls `refresh()`. v2.1 will surface the retry preview JSON in a side modal |
| Intervention block (`_render_intervention_block`) | `<HITLBand>` (question rendering, args dump, risk badge) | **full** | React renders policy + risk + waited-seconds + confidence in the same band |
| Tool calls log (`_render_tool_calls_block`) | `<Transcript>` per-turn `<ToolCallCard>` list + `<SelectedPanel>` detail | **full** | React adds click-to-select via `useSetSelected` |
| Agents accordion (`_render_agents_accordion`) | `<FlowStrip>` top-of-canvas pipeline overview | **partial** | FlowStrip shows agents-as-nodes with status; the detailed system_prompt_excerpt is in `<SelectedPanel>` when an agent is selected. Defer the "full prompt expander" to v2.1. |
| Tools by category (`_render_tools_by_category`) | `<ToolsPanel>` monitor | **full** | Same GET /api/v1/tools; React groups by category in collapsible monitor |
| Lessons learned | `<LessonsPanel>` monitor (per-session) | **full** | GET /api/v1/sessions/{sid}/lessons; React polls once via react-query |
| Health (`_make_repository` health gating) | `<HealthPanel>` monitor + Topbar `<HealthDot>` | **full** | 30s poll of /health |
| Cross-session activity feed | `<OtherSessionsPanel>` monitor | **full** | Powered by SSE GET /api/v1/sessions/recent/events |
| App-specific UI views (Approach C overlays) | `<SelectedPanel>` "App-specific views →" links | **full** | GET /api/v1/apps/{app}/ui-views |
| Run metadata + global status bar | `<Statusbar>` | **full** | sse event count + vm_seq + connection state + versions |
| Mobile / responsive | `<MobileShell>` + `<TabletShell>` (Tasks 57-61) | **full** | Streamlit has no mobile story; React: <768 mobile, 768-1199 tablet, >=1200 desktop |
| Keyboard shortcuts | — | **deferred** | Locked decision: no keyboard shortcuts in v2.0 (see in-flight notes); v2.1 reconsider |
| Light/dark theme | Light only | **deferred** | Single light theme by design; dark mode is v2.1 |

## Verdict

- 21 features at **full** parity.
- 3 features at **partial** (hypothesis trail dedicated view, retry preview JSON, agent prompt expander). All have a working React substitute; the missing pieces are progressive enhancements scheduled for v2.1.
- 2 features intentionally **deferred** (keyboard shortcuts, dark theme).

The React UI clears the v2.0.0-rc1 ship gate. Streamlit shows its
deprecation banner (Task 70) and ships beside the React build until
the v2.0.0 GA release.

## Open ticket parking lot (v2.1)

- Hypothesis trail dedicated panel
- Retry preview JSON in a side modal
- Agent system_prompt expander accessible from the FlowStrip
- Keyboard shortcuts: `?` overlay + `j/k` session navigation
- Dark mode (re-derive accent + warm-cream palette)
- App.tsx + SessionCanvas double `useSessionFull` subscription
- `<Select>` Radix upgrade (currently native)
- SessionId type rename (incident_management still emits `INC-`)
