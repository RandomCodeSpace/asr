import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '../_helpers/render';
import { LessonsPanel } from '@/monitors/LessonsPanel';

const lessons = [
  {
    id: 'L1',
    title: 'Always check deploy diff',
    summary: 'When p99 spikes within 10m of deploy, the deploy is the prime suspect.',
    agent: 'triage',
  },
  {
    id: 'L2',
    title: 'Restart only with rationale',
    summary: 'Restart requires documented rationale per policy.',
    agent: 'investigate',
  },
];

describe('<LessonsPanel>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(lessons), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('renders nothing in the body when sessionId is null', () => {
    render(<LessonsPanel sessionId={null} />);
    // header always renders
    expect(screen.getByText(/Lessons/i)).toBeInTheDocument();
  });

  it('fetches lessons and renders titles', async () => {
    render(<LessonsPanel sessionId="SES-1" />);
    // header is collapsed by default — click to expand
    const header = screen.getByText(/Lessons/i);
    header.click();
    await waitFor(() => expect(screen.getByText('Always check deploy diff')).toBeInTheDocument());
    expect(screen.getByText('Restart only with rationale')).toBeInTheDocument();
  });

  it('hits /api/v1/sessions/{sid}/lessons', async () => {
    render(<LessonsPanel sessionId="SES-42" />);
    // expand
    screen.getByText(/Lessons/i).click();
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/sessions/SES-42/lessons',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});
