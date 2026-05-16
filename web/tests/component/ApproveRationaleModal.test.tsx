import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '../_helpers/render';
import { ApproveRationaleModal } from '@/modals/ApproveRationaleModal';

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

describe('<ApproveRationaleModal>', () => {
  beforeEach(() => { globalThis.fetch = makeFetch({ ok: true, body: { ok: true } }); });
  afterEach(() => { globalThis.fetch = ORIGINAL_FETCH; });

  it('does not render when open=false', () => {
    render(
      <ApproveRationaleModal
        open={false} onOpenChange={() => {}}
        sessionId="SES-1" toolCallId="0" onApproved={() => {}}
      />,
    );
    expect(screen.queryByText(/approve this tool call/i)).not.toBeInTheDocument();
  });

  it('renders title, rationale textarea, and disabled approve button initially', () => {
    render(
      <ApproveRationaleModal
        open onOpenChange={() => {}}
        sessionId="SES-1" toolCallId="0" onApproved={() => {}}
      />,
    );
    expect(screen.getByText(/approve this tool call/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/rationale/i)).toBeInTheDocument();
    const approve = screen.getByRole('button', { name: 'Approve' });
    expect(approve).toBeDisabled();
  });

  it('enables Approve once rationale has content', () => {
    render(
      <ApproveRationaleModal
        open onOpenChange={() => {}}
        sessionId="SES-1" toolCallId="0" onApproved={() => {}}
      />,
    );
    fireEvent.change(screen.getByLabelText(/rationale/i), { target: { value: 'looks safe' } });
    expect(screen.getByRole('button', { name: 'Approve' })).toBeEnabled();
  });

  it('POSTs decision=approve with rationale on submit, then calls onApproved + close', async () => {
    const fetchMock = makeFetch({ ok: true, body: { session_id: 'SES-1' } });
    globalThis.fetch = fetchMock;
    const onApproved = vi.fn();
    const onOpenChange = vi.fn();
    render(
      <ApproveRationaleModal
        open onOpenChange={onOpenChange}
        sessionId="SES-1" toolCallId="3"
        approver="alice" onApproved={onApproved}
      />,
    );
    fireEvent.change(screen.getByLabelText(/rationale/i), { target: { value: 'low risk; previously reviewed' } });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));
    await waitFor(() => expect(onApproved).toHaveBeenCalledTimes(1));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const call = fetchMock.mock.calls[0];
    if (!call) throw new Error('expected fetch call');
    const [url, init] = call;
    expect(url).toBe('/api/v1/sessions/SES-1/approvals/3');
    expect(init?.method).toBe('POST');
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      decision: 'approve',
      approver: 'alice',
      rationale: 'low risk; previously reviewed',
    });
  });

  it('renders rationale templates as clickable chips that fill the textarea', () => {
    render(
      <ApproveRationaleModal
        open onOpenChange={() => {}}
        sessionId="SES-1" toolCallId="0"
        templates={['Low risk', 'Pre-approved by oncall']}
        onApproved={() => {}}
      />,
    );
    const chip = screen.getByRole('button', { name: 'Pre-approved by oncall' });
    fireEvent.click(chip);
    expect(screen.getByLabelText(/rationale/i)).toHaveValue('Pre-approved by oncall');
  });

  it('shows error envelope when API rejects', async () => {
    globalThis.fetch = makeFetch({
      ok: false,
      status: 409,
      body: { error: { code: 'session_closed', message: 'already closed', details: {} } },
    });
    render(
      <ApproveRationaleModal
        open onOpenChange={() => {}}
        sessionId="SES-1" toolCallId="0" onApproved={() => {}}
      />,
    );
    fireEvent.change(screen.getByLabelText(/rationale/i), { target: { value: 'go' } });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert')).toHaveTextContent(/session_closed: already closed/);
  });
});
