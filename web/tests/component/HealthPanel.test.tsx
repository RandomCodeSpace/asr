import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '../_helpers/render';
import { HealthPanel } from '@/monitors/HealthPanel';

describe('<HealthPanel>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'ok', uptime_seconds: 3600 }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
  });

  it('renders header', () => {
    render(<HealthPanel />);
    expect(screen.getByText(/System Health/i)).toBeInTheDocument();
  });

  it('shows status label after fetch when expanded', async () => {
    render(<HealthPanel />);
    screen.getByText(/System Health/i).click();
    await waitFor(() => expect(screen.getByText(/ok/i)).toBeInTheDocument());
  });

  it('hits /health (NOT /api/v1/health)', async () => {
    render(<HealthPanel />);
    screen.getByText(/System Health/i).click();
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('shows degraded color when status=degraded', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'degraded' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    const { container } = render(<HealthPanel />);
    screen.getByText(/System Health/i).click();
    await waitFor(() => expect(screen.getByText(/degraded/i)).toBeInTheDocument());
    expect(container.querySelector('[data-health-dot]')).toHaveAttribute('data-status', 'degraded');
  });
});
