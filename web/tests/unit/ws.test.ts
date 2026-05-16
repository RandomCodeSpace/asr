import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWebSocket } from '@/api/ws';
import { MockWebSocket } from '../_helpers/MockWebSocket';

describe('useWebSocket', () => {
  beforeEach(() => {
    MockWebSocket.reset();
    // @ts-expect-error global override
    global.WebSocket = MockWebSocket;
  });

  it('opens a WebSocket at the given URL', async () => {
    renderHook(() => useWebSocket('ws://test/api/v1/sessions/SES-1/ws', () => {}));
    await Promise.resolve();
    expect(MockWebSocket.lastInstance()?.url).toBe('ws://test/api/v1/sessions/SES-1/ws');
  });

  it('calls onMessage with parsed JSON payload per message event', async () => {
    const onMessage = vi.fn();
    renderHook(() => useWebSocket('ws://x', onMessage));
    await Promise.resolve();
    act(() => {
      MockWebSocket.lastInstance()!.emit('{"seq":2,"kind":"tool_invoked","payload":{},"ts":"y"}');
    });
    expect(onMessage).toHaveBeenCalledWith(
      expect.objectContaining({ seq: 2, kind: 'tool_invoked' }),
    );
  });

  it('skips messages with malformed JSON', async () => {
    const onMessage = vi.fn();
    renderHook(() => useWebSocket('ws://x', onMessage));
    await Promise.resolve();
    act(() => {
      MockWebSocket.lastInstance()!.emit('not json');
    });
    expect(onMessage).not.toHaveBeenCalled();
  });

  it('does not connect when enabled=false', async () => {
    renderHook(() => useWebSocket('ws://x', () => {}, false));
    await Promise.resolve();
    expect(MockWebSocket.lastInstance()).toBeUndefined();
  });
});
