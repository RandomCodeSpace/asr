import { describe, it, expect } from 'vitest';
import { sessionReducer, initialSessionState } from '@/state/sessionReducer';
import type { SessionFullBundle, SessionEvent } from '@/api/types';

const baseBundle: SessionFullBundle = {
  session: {
    id: 'SES-1', status: 'in_progress',
    created_at: 'x', updated_at: 'x', deleted_at: null,
    agents_run: [], tool_calls: [], findings: {},
    pending_intervention: null, user_inputs: [],
    parent_session_id: null, dedup_rationale: null,
    extra_fields: {}, version: 1,
  },
  agents_run: [],
  tool_calls: [],
  events: [],
  agent_definitions: {},
  vm_seq: 0,
};

describe('sessionReducer', () => {
  describe('action: bootstrap', () => {
    it('replaces full state with bundle', () => {
      const next = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      expect(next.vmSeq).toBe(0);
      expect(next.session?.id).toBe('SES-1');
    });
  });

  describe('action: event', () => {
    it('drops events with seq <= vmSeq (idempotent)', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: { ...baseBundle, vm_seq: 5 } });
      const stale: SessionEvent = { seq: 3, kind: 'agent_started', payload: {}, ts: 'x' };
      const next = sessionReducer(state, { type: 'event', event: stale });
      expect(next).toBe(state);  // no change
    });

    it('appends events with seq > vmSeq and bumps vmSeq', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      const fresh: SessionEvent = { seq: 1, kind: 'agent_started', payload: { agent: 'intake' }, ts: 'x' };
      const next = sessionReducer(state, { type: 'event', event: fresh });
      expect(next.vmSeq).toBe(1);
      expect(next.events).toHaveLength(1);
    });

    it('event "agent_finished" appends an AgentRun', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      const finished: SessionEvent = {
        seq: 1, kind: 'agent_finished', ts: 'x',
        payload: { agent: 'intake', summary: 'done', confidence: 0.9 },
      };
      const next = sessionReducer(state, { type: 'event', event: finished });
      expect(next.agentsRun).toHaveLength(1);
      expect(next.agentsRun[0]?.agent).toBe('intake');
    });

    it('event "tool_invoked" appends a ToolCall with status="executed"', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      const tool: SessionEvent = {
        seq: 1, kind: 'tool_invoked', ts: 'x',
        payload: { agent: 'triage', tool: 'obs:get_logs', args: {}, result: 'ok' },
      };
      const next = sessionReducer(state, { type: 'event', event: tool });
      expect(next.toolCalls).toHaveLength(1);
      expect(next.toolCalls[0]?.status).toBe('executed');
    });

    it('event "approval_pending" inserts a pending tool call', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      const ev: SessionEvent = {
        seq: 1, kind: 'approval_pending', ts: 'x',
        payload: { agent: 'investigate', tool: 'rem:propose_fix', args: {}, risk: 'high' },
      };
      const next = sessionReducer(state, { type: 'event', event: ev });
      expect(next.toolCalls[0]?.status).toBe('pending_approval');
      expect(next.toolCalls[0]?.risk).toBe('high');
    });

    it('event "status_changed" updates session.status', () => {
      const state = sessionReducer(initialSessionState, { type: 'bootstrap', bundle: baseBundle });
      const ev: SessionEvent = {
        seq: 1, kind: 'status_changed', ts: 'x',
        payload: { status: 'resolved' },
      };
      const next = sessionReducer(state, { type: 'event', event: ev });
      expect(next.session?.status).toBe('resolved');
    });
  });
});
