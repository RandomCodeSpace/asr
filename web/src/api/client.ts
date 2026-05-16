const API_PREFIX = '/api/v1';

export class ApiClientError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;
  constructor(status: number, code: string, message: string, details: Record<string, unknown>) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

interface FetchOptions extends RequestInit {
  json?: unknown;
}

export async function apiFetch<T = unknown>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const url = path.startsWith('http') ? path : `${API_PREFIX}${path}`;
  const headers = new Headers(options.headers);
  const token = localStorage.getItem('asr.token');
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.json !== undefined) {
    headers.set('Content-Type', 'application/json');
    options.body = JSON.stringify(options.json);
  }
  const res = await fetch(url, { ...options, headers, method: options.method ?? 'GET' });
  if (!res.ok) {
    let body: { error?: { code?: string; message?: string; details?: Record<string, unknown> } } = {};
    try { body = await res.clone().json(); } catch { /* not JSON */ }
    throw new ApiClientError(
      res.status,
      body.error?.code ?? 'unknown',
      body.error?.message ?? res.statusText,
      body.error?.details ?? {},
    );
  }
  if (res.status === 204) return undefined as T;
  return await res.json() as T;
}
