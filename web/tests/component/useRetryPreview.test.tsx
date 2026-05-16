import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRetryPreview } from '@/state/useRetryPreview';
import { ReactNode } from 'react';

const ORIGINAL_FETCH = globalThis.fetch;

function makeFetch(response: { ok: boolean; status?: number; body?: unknown }) {
  return vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve({
    ok: response.ok,
    status: response.status ?? (response.ok ? 200 : 400),
    statusText: response.ok ? 'OK' : 'Bad Request',
    clone() { return this; },
    json: () => Promise.resolve(response.body ?? {}),
  } as unknown as Response));
}

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: ReactNode }) =>
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe('useRetryPreview', () => {
  beforeEach(() => { globalThis.fetch = makeFetch({ ok: true, body: { retry: true, reason: 'previous error' } }); });
  afterEach(() => { globalThis.fetch = ORIGINAL_FETCH; });

  it('disabled when sid is null (no fetch fires)', () => {
    const fetchMock = makeFetch({ ok: true });
    globalThis.fetch = fetchMock;
    const { result } = renderHook(() => useRetryPreview(null), { wrapper: wrapper() });
    expect(result.current.data).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns { retry, reason } from the preview endpoint', async () => {
    const { result } = renderHook(() => useRetryPreview('SES-1'), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.data).toEqual({ retry: true, reason: 'previous error' }));
  });

  it('translates 4xx into a non-throwing { retry: false, reason: <msg> }', async () => {
    globalThis.fetch = makeFetch({
      ok: false,
      status: 404,
      body: { error: { code: 'not_found', message: 'session not found', details: {} } },
    });
    const { result } = renderHook(() => useRetryPreview('SES-1'), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.data).toEqual({ retry: false, reason: 'session not found' }));
  });
});
