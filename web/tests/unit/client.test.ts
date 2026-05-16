import { describe, it, expect, beforeEach, vi } from 'vitest';
import { apiFetch, ApiClientError } from '@/api/client';

describe('apiFetch', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('prepends /api/v1 to relative paths', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } }),
    );
    global.fetch = fetchMock;

    await apiFetch('/sessions');
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/sessions',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('attaches Authorization header when token in localStorage', async () => {
    localStorage.setItem('asr.token', 'tk-abc');
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    global.fetch = fetchMock;

    await apiFetch('/sessions');
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    const headers = new Headers(init.headers);
    expect(headers.get('Authorization')).toBe('Bearer tk-abc');
  });

  it('throws ApiClientError on 4xx with structured error body', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ error: { code: 'not_found', message: 'session not found', details: {} } }),
        { status: 404, headers: { 'content-type': 'application/json' } },
      ),
    );

    await expect(apiFetch('/sessions/SES-x')).rejects.toThrow(ApiClientError);
    await expect(apiFetch('/sessions/SES-x')).rejects.toMatchObject({
      status: 404,
      code: 'not_found',
    });
  });
});
