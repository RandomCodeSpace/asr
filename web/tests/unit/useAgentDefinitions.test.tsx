import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useAgentDefinitions } from '@/state/useAgentDefinitions';

const agents = [
  { name: 'intake', kind: 'responsive', model: 'gpt', tools: [], routes: {}, system_prompt_excerpt: '...' },
  { name: 'triage', kind: 'gated', model: 'claude', tools: ['obs:get_logs'], routes: {}, system_prompt_excerpt: '...' },
];

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('useAgentDefinitions', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(agents), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('fetches /api/v1/agents and exposes the list', async () => {
    const { result } = renderHook(() => useAgentDefinitions(), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.list).toHaveLength(2);
    expect(result.current.data?.list[0]?.name).toBe('intake');
  });

  it('exposes a byName map keyed by agent name', async () => {
    const { result } = renderHook(() => useAgentDefinitions(), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.byName.triage?.kind).toBe('gated');
    expect(result.current.data?.byName.intake?.tools).toEqual([]);
  });

  it('hits /api/v1/agents', async () => {
    renderHook(() => useAgentDefinitions(), { wrapper });
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/agents',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});
