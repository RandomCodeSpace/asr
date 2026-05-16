import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../_helpers/render';
import { App } from '@/App';

const uiHints = {
  brand_name: 'Test Brand',
  brand_logo_url: null,
  approval_rationale_templates: [],
  hitl_question_templates: {},
  environments: ['dev'],
};

const sessions: unknown[] = [];
const agents: unknown[] = [];

describe('<App>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/config/ui-hints')) {
        return Promise.resolve(new Response(JSON.stringify(uiHints), {
          status: 200, headers: { 'content-type': 'application/json' },
        }));
      }
      if (url.includes('/sessions') && !url.includes('/recent/events') && !url.includes('/events')) {
        return Promise.resolve(new Response(JSON.stringify(sessions), {
          status: 200, headers: { 'content-type': 'application/json' },
        }));
      }
      if (url.includes('/agents')) {
        return Promise.resolve(new Response(JSON.stringify(agents), {
          status: 200, headers: { 'content-type': 'application/json' },
        }));
      }
      return Promise.resolve(new Response('{}', { status: 200 }));
    });
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

  it('renders the brand name from useUiHints once loaded', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('Test Brand')).toBeInTheDocument());
  });

  it('renders an empty-state SessionsRail when no sessions', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/No sessions yet/i)).toBeInTheDocument());
  });

  it('renders Statusbar with version', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/ui v/)).toBeInTheDocument());
  });

  it('renders the canvas empty state when no session selected', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/Select a session/i)).toBeInTheDocument());
  });
});
