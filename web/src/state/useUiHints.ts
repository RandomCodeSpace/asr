import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/api/client';
import type { UiHints } from '@/api/types';

export function useUiHints() {
  return useQuery({
    queryKey: ['ui-hints'],
    queryFn: () => apiFetch<UiHints>('/config/ui-hints'),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
