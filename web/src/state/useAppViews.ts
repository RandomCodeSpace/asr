import { useQuery } from '@tanstack/react-query';
import { apiFetch, ApiClientError } from '@/api/client';
import type { AppView } from '@/api/types';

/** GET /api/v1/apps/{appName}/ui-views — Approach C extensibility.
 *  Returns the app-registered overlay views. Empty list on any 4xx
 *  (the app may not register any views, in which case the SelectedPanel
 *  section silently omits). Cached aggressively because the registry is
 *  immutable for the app's lifetime. */
export function useAppViews(appName: string | null) {
  return useQuery<AppView[]>({
    queryKey: ['app-views', appName],
    enabled: !!appName,
    queryFn: async () => {
      if (!appName) return [];
      try {
        return await apiFetch<AppView[]>(`/apps/${appName}/ui-views`);
      } catch (e) {
        if (e instanceof ApiClientError && e.status >= 400 && e.status < 500) return [];
        throw e;
      }
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });
}

/** Filter app views by `applies_to`:
 *  - 'always' matches any selection
 *  - 'agent:NAME' matches when an agent with that name is selected
 *  - 'tool:NAME' matches when a tool call with that tool is selected
 */
export function filterAppViews(
  views: AppView[],
  selection: { kind: string | null; id?: string | null },
): AppView[] {
  if (!views.length) return [];
  return views.filter((v) => {
    if (v.applies_to === 'always') return true;
    if (v.applies_to.startsWith('agent:')) {
      const agentName = v.applies_to.slice('agent:'.length);
      return selection.kind === 'agent' && selection.id === agentName;
    }
    if (v.applies_to.startsWith('tool:')) {
      const toolName = v.applies_to.slice('tool:'.length);
      if (selection.kind !== 'tool_call') return false;
      // tool_call id shape is `{tool}@{ts}` (per SessionCanvas + Transcript)
      const tcTool = (selection.id ?? '').split('@')[0];
      return tcTool === toolName;
    }
    return false;
  });
}
