import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '../_helpers/render';
import { UnDuplicateModal } from '@/modals/UnDuplicateModal';

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

describe('<UnDuplicateModal>', () => {
  beforeEach(() => { globalThis.fetch = makeFetch({ ok: true, body: { session_id: 'SES-1', status: 'new' } }); });
  afterEach(() => { globalThis.fetch = ORIGINAL_FETCH; });

  it('does not render when closed', () => {
    render(
      <UnDuplicateModal
        open={false} onOpenChange={() => {}}
        sessionId="SES-1" parentSessionId="SES-0" onSuccess={() => {}}
      />,
    );
    expect(screen.queryByText(/retract the duplicate flag/i)).not.toBeInTheDocument();
  });

  it('shows session id + parent id in the body when open', () => {
    render(
      <UnDuplicateModal
        open onOpenChange={() => {}}
        sessionId="SES-42" parentSessionId="SES-7" onSuccess={() => {}}
      />,
    );
    expect(screen.getByText(/SES-42/)).toBeInTheDocument();
    expect(screen.getByText(/SES-7/)).toBeInTheDocument();
  });

  it('POSTs un-duplicate with note + retracted_by, then closes', async () => {
    const fetchMock = makeFetch({ ok: true, body: { session_id: 'SES-1' } });
    globalThis.fetch = fetchMock;
    const onSuccess = vi.fn();
    const onOpenChange = vi.fn();
    render(
      <UnDuplicateModal
        open onOpenChange={onOpenChange}
        sessionId="SES-1" parentSessionId="SES-0"
        retractedBy="alice" onSuccess={onSuccess}
      />,
    );
    fireEvent.change(screen.getByLabelText(/note/i), { target: { value: 'false positive' } });
    fireEvent.click(screen.getByRole('button', { name: 'Un-duplicate' }));
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    const call = fetchMock.mock.calls[0];
    if (!call) throw new Error('expected fetch call');
    const [url, init] = call;
    expect(url).toBe('/api/v1/sessions/SES-1/un-duplicate');
    expect((init as RequestInit).method).toBe('POST');
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      retracted_by: 'alice',
      note: 'false positive',
    });
  });

  it('sends note=null when the textarea is empty', async () => {
    const fetchMock = makeFetch({ ok: true });
    globalThis.fetch = fetchMock;
    render(
      <UnDuplicateModal
        open onOpenChange={() => {}}
        sessionId="SES-1" parentSessionId={null} onSuccess={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Un-duplicate' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const call = fetchMock.mock.calls[0];
    if (!call) throw new Error('expected fetch call');
    const [, init] = call;
    expect(JSON.parse((init as RequestInit).body as string).note).toBeNull();
  });

  it('surfaces 409 conflict envelope when backend rejects', async () => {
    globalThis.fetch = makeFetch({
      ok: false,
      status: 409,
      body: { error: { code: 'conflict', message: 'not a duplicate', details: {} } },
    });
    render(
      <UnDuplicateModal
        open onOpenChange={() => {}}
        sessionId="SES-1" parentSessionId="SES-0" onSuccess={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Un-duplicate' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/conflict: not a duplicate/));
  });
});
