import { useMemo } from 'react';
import { useSessionList, type SessionSummary } from './useSessionList';
import type { ApiClientError } from '@/api/client';

export interface UseApprovalsQueueResult {
  queue: SessionSummary[];
  count: number;
  isLoading: boolean;
  error: ApiClientError | null;
}

export function useApprovalsQueue(): UseApprovalsQueueResult {
  const { sessions, isLoading, error } = useSessionList();
  const queue = useMemo(
    () =>
      sessions
        .filter((s) => s.status === 'awaiting_input')
        .sort((a, b) => a.updated_at.localeCompare(b.updated_at)),
    [sessions],
  );
  return { queue, count: queue.length, isLoading, error };
}
