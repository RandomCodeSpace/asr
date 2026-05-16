import { useQuery } from '@tanstack/react-query';
import { apiFetch, ApiClientError } from '@/api/client';

export interface RetryDecisionPreview {
  retry: boolean;
  reason: string;
}

/** GET /api/v1/sessions/{sid}/retry/preview — drives the Retry button's
 *  enabled state. Returns `{ retry: false, reason: '...' }` for any
 *  4xx error (e.g. 404 unknown id, 409 not retryable) so the UI can
 *  reflect "no retry available" without crashing the panel. */
export function useRetryPreview(sid: string | null) {
  return useQuery<RetryDecisionPreview>({
    queryKey: ['retry-preview', sid],
    enabled: !!sid,
    queryFn: async () => {
      if (!sid) return { retry: false, reason: 'no session selected' };
      try {
        return await apiFetch<RetryDecisionPreview>(`/sessions/${sid}/retry/preview`);
      } catch (e) {
        if (e instanceof ApiClientError) {
          return { retry: false, reason: e.message };
        }
        throw e;
      }
    },
    staleTime: 5_000,
    gcTime: 60_000,
  });
}
