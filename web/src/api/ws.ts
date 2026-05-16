import { useEffect } from 'react';
import type { SessionEvent } from './types';

export function useWebSocket(
  url: string,
  onMessage: (ev: SessionEvent) => void,
  enabled: boolean = true,
) {
  useEffect(() => {
    if (!enabled) return;
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as SessionEvent;
        onMessage(data);
      } catch {
        // skip malformed
      }
    };
    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
    // onMessage intentionally omitted from deps to avoid reconnect on every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]);
}
