import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useApprovalsQueue } from '@/state/useApprovalsQueue';
import { MockEventSource } from '../_helpers/MockEventSource';

const sessions = [
  { id: 'SES-1', status: 'in_progress', label: 'foo', created_at: 't0', updated_at: 't0' },
  { id: 'SES-2', status: 'awaiting_input', label: 'bar', created_at: 't0', updated_at: 't2' },
  { id: 'SES-3', status: 'awaiting_input', label: 'baz', created_at: 't0', updated_at: 't1' },
  { id: 'SES-4', status: 'resolved', label: 'qux', created_at: 't0', updated_at: 't0' },
];

describe('useApprovalsQueue', () => {
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

  it('returns only sessions with status="awaiting_input"', async () => {
    const { result } = renderHook(() => useApprovalsQueue());
    await waitFor(() => expect(result.current.queue.length).toBeGreaterThan(0));
    expect(result.current.queue.every((s) => s.status === 'awaiting_input')).toBe(true);
    expect(result.current.queue).toHaveLength(2);
  });

  it('sorts oldest-waiting-first by updated_at ascending', async () => {
    const { result } = renderHook(() => useApprovalsQueue());
    await waitFor(() => expect(result.current.queue).toHaveLength(2));
    expect(result.current.queue[0]?.id).toBe('SES-3');  // updated_at = t1
    expect(result.current.queue[1]?.id).toBe('SES-2');  // updated_at = t2
  });

  it('exposes total count via result.count', async () => {
    const { result } = renderHook(() => useApprovalsQueue());
    await waitFor(() => expect(result.current.count).toBe(2));
  });
});
