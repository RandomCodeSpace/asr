import { describe, it, expect } from 'vitest';
import { questionFromToolCall } from '@/lib/hitl/questionFromToolCall';
import type { ToolCall } from '@/api/types';

const baseTc: ToolCall = {
  agent: 'investigate', tool: 'rem:restart_service',
  args: { service: 'payments-svc', environment: 'production' },
  result: null, ts: '2026-05-15T14:18:00Z', risk: 'high',
  status: 'pending_approval', approver: null, approved_at: null, approval_rationale: null,
};

describe('questionFromToolCall', () => {
  it('interpolates app-provided template with args + agent', () => {
    const q = questionFromToolCall(baseTc, {
      'rem:restart_service': 'Restart {service} in {environment}?',
    });
    expect(q).toBe('Restart payments-svc in production?');
  });

  it('falls back to the generic template when no app override matches', () => {
    const q = questionFromToolCall(baseTc, {});
    expect(q).toMatch(/investigate/);
    expect(q).toMatch(/rem:restart_service/);
  });

  it('handles missing args gracefully', () => {
    const tc: ToolCall = { ...baseTc, args: {} };
    const q = questionFromToolCall(tc, {
      'rem:restart_service': 'Restart {service}?',
    });
    // missing arg interpolation keeps the placeholder visible — UI should surface this
    expect(q).toContain('{service}');
  });
});
