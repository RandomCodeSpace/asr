import { useEffect, useReducer, useState, useCallback } from 'react';
import { apiFetch, ApiClientError } from '@/api/client';
import { useEventSource } from '@/api/sse';
import { sessionReducer, initialSessionState, type SessionState } from './sessionReducer';
import type { SessionFullBundle, SessionEvent, SessionId } from '@/api/types';

export interface UseSessionFullResult {
  state: SessionState;
  isLoading: boolean;
  error: ApiClientError | null;
  refresh: () => void;
}

export function useSessionFull(sid: SessionId | null): UseSessionFullResult {
  const [state, dispatch] = useReducer(sessionReducer, initialSessionState);
  const [isLoading, setIsLoading] = useState<boolean>(sid !== null);
  const [error, setError] = useState<ApiClientError | null>(null);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    if (!sid) {
      setIsLoading(false);
      setBootstrapped(false);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setBootstrapped(false);
    apiFetch<SessionFullBundle>(`/sessions/${sid}/full`)
      .then((bundle) => {
        if (cancelled) return;
        dispatch({ type: 'bootstrap', bundle });
        setBootstrapped(true);
        setIsLoading(false);
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
  }, [sid, refreshTick]);

  const onEvent = useCallback((ev: SessionEvent) => {
    dispatch({ type: 'event', event: ev });
  }, []);

  useEventSource(
    sid ? `/api/v1/sessions/${sid}/events` : '',
    onEvent,
    Boolean(sid) && bootstrapped,
  );

  const refresh = useCallback(() => {
    setRefreshTick((n) => n + 1);
  }, []);

  return { state, isLoading, error, refresh };
}
