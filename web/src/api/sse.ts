import { useEffect } from 'react';
import type { SessionEvent } from './types';

export function useEventSource(
  url: string,
  onMessage: (ev: SessionEvent) => void,
  enabled: boolean = true,
) {
  useEffect(() => {
    if (!enabled) return;
    const es = new EventSource(url);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as SessionEvent;
        onMessage(data);
      } catch {
        // skip malformed
      }
    };
    return () => es.close();
    // onMessage intentionally omitted from deps to avoid reconnect on every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]);
}
