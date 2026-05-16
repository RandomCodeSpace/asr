import { useEffect, useState, useCallback } from 'react';
import { apiFetch, ApiClientError } from '@/api/client';
import { useEventSource } from '@/api/sse';
import type { SessionEvent, SessionId } from '@/api/types';

export interface SessionSummary {
  id: SessionId;
  status: string;
  label?: string;
  created_at: string;
  updated_at: string;
  active_agent?: string | null;
}

export interface UseSessionListResult {
  sessions: SessionSummary[];
  isLoading: boolean;
  error: ApiClientError | null;
}

export function useSessionList(): UseSessionListResult {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<ApiClientError | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // Hit /sessions/recent (history) instead of /sessions (in-flight only)
    // so the rail shows past sessions even when nothing is currently
    // running. The cross-session SSE below still pushes live status +
    // agent_running deltas onto the same list.
    apiFetch<SessionSummary[]>('/sessions/recent?limit=50')
      .then((list) => {
        if (cancelled) return;
        setSessions(list);
        setIsLoading(false);
        setLoaded(true);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiClientError) {
          setError(err);
        } else {
          setError(new ApiClientError(0, 'network_error', String(err), {}));
        }
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onEvent = useCallback((ev: SessionEvent) => {
    if (ev.kind === 'session.created') {
      const summary = ev.payload as unknown as SessionSummary;
      setSessions((prev) => {
        if (prev.find((s) => s.id === summary.id)) return prev;
        return [summary, ...prev];
      });
    } else if (ev.kind === 'session.status_changed') {
      const p = ev.payload as { id: string; status: string };
      setSessions((prev) => prev.map((s) => (s.id === p.id ? { ...s, status: p.status } : s)));
    } else if (ev.kind === 'session.agent_running') {
      const p = ev.payload as { id: string; agent: string | null };
      setSessions((prev) => prev.map((s) => (s.id === p.id ? { ...s, active_agent: p.agent } : s)));
    }
  }, []);

  useEventSource('/api/v1/sessions/recent/events', onEvent, loaded);

  return { sessions, isLoading, error };
}
