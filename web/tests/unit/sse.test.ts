import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useEventSource } from '@/api/sse';
import { MockEventSource } from '../_helpers/MockEventSource';

describe('useEventSource', () => {
  beforeEach(() => {
    MockEventSource.reset();
    // @ts-expect-error global override
    global.EventSource = MockEventSource;
  });

  it('opens an EventSource at the given URL', async () => {
    renderHook(() => useEventSource('/api/v1/sessions/SES-1/events', () => {}));
    await Promise.resolve();
    expect(MockEventSource.lastInstance()?.url).toBe('/api/v1/sessions/SES-1/events');
  });

  it('calls onMessage with parsed JSON payload per data: line', async () => {
    const onMessage = vi.fn();
    renderHook(() => useEventSource('/x', onMessage));
    await Promise.resolve();
    act(() => {
      MockEventSource.lastInstance()!.emit('{"seq":1,"kind":"agent_started","payload":{},"ts":"x"}');
    });
    expect(onMessage).toHaveBeenCalledWith(
      expect.objectContaining({ seq: 1, kind: 'agent_started' }),
    );
  });
});
