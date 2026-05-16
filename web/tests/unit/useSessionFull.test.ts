import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { useSessionFull } from '@/state/useSessionFull';
import type { SessionFullBundle } from '@/api/types';
import { MockEventSource } from '../_helpers/MockEventSource';

const bundle: SessionFullBundle = {
  session: {
    id: 'SES-1', status: 'in_progress',
    created_at: 't0', updated_at: 't0', deleted_at: null,
    agents_run: [], tool_calls: [], findings: {},
    pending_intervention: null, user_inputs: [],
    parent_session_id: null, dedup_rationale: null,
    extra_fields: {}, version: 1,
  },
  agents_run: [], tool_calls: [], events: [],
  agent_definitions: {}, vm_seq: 0,
};

describe('useSessionFull', () => {
  beforeEach(() => {
    MockEventSource.reset();
    // @ts-expect-error global override
    global.EventSource = MockEventSource;
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(bundle), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('fetches bootstrap then sets state.session.id', async () => {
    const { result } = renderHook(() => useSessionFull('SES-1'));
    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.state.session?.id).toBe('SES-1');
    expect(result.current.error).toBeNull();
  });

  it('opens SSE stream after bootstrap and applies events', async () => {
    const { result } = renderHook(() => useSessionFull('SES-1'));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    await waitFor(() => expect(MockEventSource.lastInstance()).toBeDefined());
    expect(MockEventSource.lastInstance()!.url).toBe('/api/v1/sessions/SES-1/events');

    act(() => {
      MockEventSource.lastInstance()!.emit(
        JSON.stringify({ seq: 1, kind: 'status_changed', payload: { status: 'resolved' }, ts: 't1' }),
      );
    });
    await waitFor(() => expect(result.current.state.session?.status).toBe('resolved'));
  });

  it('captures fetch error in error state', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ error: { code: 'not_found', message: 'gone', details: {} } }),
        { status: 404, headers: { 'content-type': 'application/json' } },
      ),
    );
    const { result } = renderHook(() => useSessionFull('SES-x'));
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.code).toBe('not_found');
    expect(result.current.isLoading).toBe(false);
  });
});
