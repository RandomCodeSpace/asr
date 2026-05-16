import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/api/client';
import type { AgentDefinition } from '@/api/types';

export interface AgentDefinitions {
  list: AgentDefinition[];
  byName: Record<string, AgentDefinition>;
}

export function useAgentDefinitions() {
  return useQuery<AgentDefinitions>({
    queryKey: ['agent-definitions'],
    queryFn: async () => {
      const list = await apiFetch<AgentDefinition[]>('/agents');
      const byName: Record<string, AgentDefinition> = {};
      for (const a of list) {
        byName[a.name] = a;
      }
      return { list, byName };
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
