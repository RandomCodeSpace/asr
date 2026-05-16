import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useUiHints } from '@/state/useUiHints';

const hints = {
  brand_name: 'Acme Agents',
  brand_logo_url: null,
  approval_rationale_templates: ['Looks safe', 'Verified'],
  hitl_question_templates: { 'rem:restart_service': 'Restart {service}?' },
  environments: ['dev', 'prod'],
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('useUiHints', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(hints), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('fetches ui-hints once and exposes data', async () => {
    const { result } = renderHook(() => useUiHints(), { wrapper });
    await waitFor(() => expect(result.current.data?.brand_name).toBe('Acme Agents'));
    expect(result.current.isLoading).toBe(false);
  });

  it('hits /api/v1/config/ui-hints', async () => {
    renderHook(() => useUiHints(), { wrapper });
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/config/ui-hints',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});
