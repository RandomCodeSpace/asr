import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '../_helpers/render';
import { SessionCanvas } from '@/canvas/SessionCanvas';

const fullBundle = {
  session: {
    id: 'SES-1', status: 'in_progress',
    created_at: '2026-05-15T14:16:30Z',
    updated_at: '2026-05-15T14:17:00Z',
    deleted_at: null,
    agents_run: [], tool_calls: [], findings: { title: 'Payments latency spike' },
    pending_intervention: null, user_inputs: [],
    parent_session_id: null, dedup_rationale: null,
    extra_fields: { env: 'prod', sev: 2, reporter: 'u1@platform' }, version: 1,
  },
  agents_run: [
    { agent: 'intake', started_at: '2026-05-15T14:16:30Z', ended_at: '2026-05-15T14:16:32Z',
      summary: 'Triage observed elevated p99 latency.', confidence: 0.92,
      confidence_rationale: null, signal: null },
  ],
  tool_calls: [],
  events: [],
  agent_definitions: {},
  vm_seq: 1,
};

describe('<SessionCanvas>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(fullBundle), {
        status: 200, headers: { 'content-type': 'application/json' },
      }),
    );
    // @ts-expect-error -- jsdom does not provide EventSource
    global.EventSource = class {
      constructor(public url: string) {}
      onopen: ((e: Event) => void) | null = null;
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: Event) => void) | null = null;
      close() {}
      addEventListener() {}
    };
  });

  it('renders empty state when sid is null', () => {
    render(<SessionCanvas activeSid={null} />);
    expect(screen.getByText(/Select a session/i)).toBeInTheDocument();
  });

  it('shows loading state when fetching', () => {
    render(<SessionCanvas activeSid="SES-1" />);
    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  it('renders CanvasHead with session id once loaded', async () => {
    render(<SessionCanvas activeSid="SES-1" />);
    await waitFor(() => expect(screen.getByText(/SES-1/)).toBeInTheDocument());
  });

  it('renders Transcript with the agent turn once loaded', async () => {
    render(<SessionCanvas activeSid="SES-1" />);
    await waitFor(() => expect(screen.getByText('intake')).toBeInTheDocument());
    expect(screen.getByText(/Triage observed/)).toBeInTheDocument();
  });

  it('renders the title from findings.title', async () => {
    render(<SessionCanvas activeSid="SES-1" />);
    await waitFor(() => expect(screen.getByText(/Payments latency spike/)).toBeInTheDocument());
  });
});
