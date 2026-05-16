import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { useSessionList } from '@/state/useSessionList';
import { MockEventSource } from '../_helpers/MockEventSource';

const sessions = [
  { id: 'SES-1', status: 'in_progress', label: 'foo', created_at: 't0', updated_at: 't0' },
  { id: 'SES-2', status: 'resolved', label: 'bar', created_at: 't0', updated_at: 't1' },
];

describe('useSessionList', () => {
  beforeEach(() => {
    MockEventSource.reset();
    // @ts-expect-error global override
    global.EventSource = MockEventSource;
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(sessions), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('fetches /sessions and exposes the list', async () => {
    const { result } = renderHook(() => useSessionList());
    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.sessions).toHaveLength(2);
    expect(result.current.sessions[0]?.id).toBe('SES-1');
  });

  it('opens the recent-events SSE stream and prepends session.created', async () => {
    const { result } = renderHook(() => useSessionList());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    await waitFor(() => expect(MockEventSource.lastInstance()).toBeDefined());
    expect(MockEventSource.lastInstance()!.url).toBe('/api/v1/sessions/recent/events');

    act(() => {
      MockEventSource.lastInstance()!.emit(
        JSON.stringify({
          seq: 1, kind: 'session.created', ts: 't2',
          payload: { id: 'SES-3', status: 'new', label: 'baz', created_at: 't2', updated_at: 't2' },
        }),
      );
    });
    await waitFor(() => expect(result.current.sessions).toHaveLength(3));
    expect(result.current.sessions[0]?.id).toBe('SES-3');
  });

  it('updates an existing session status on session.status_changed', async () => {
    const { result } = renderHook(() => useSessionList());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    await waitFor(() => expect(MockEventSource.lastInstance()).toBeDefined());

    act(() => {
      MockEventSource.lastInstance()!.emit(
        JSON.stringify({
          seq: 1, kind: 'session.status_changed', ts: 't3',
          payload: { id: 'SES-1', status: 'resolved' },
        }),
      );
    });
    await waitFor(() => {
      const s = result.current.sessions.find((x) => x.id === 'SES-1');
      expect(s?.status).toBe('resolved');
    });
  });
});
