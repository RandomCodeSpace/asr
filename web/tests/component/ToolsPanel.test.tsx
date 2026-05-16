import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '../_helpers/render';
import { ToolsPanel } from '@/monitors/ToolsPanel';

const tools = [
  { name: 'obs:get_logs', description: 'Fetch service logs', risk: 'low' },
  { name: 'rem:restart_service', description: 'Restart a service', risk: 'high' },
];

describe('<ToolsPanel>', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(tools), { status: 200, headers: { 'content-type': 'application/json' } }),
    );
  });

  it('renders header always', () => {
    render(<ToolsPanel />);
    expect(screen.getByText(/Tool Catalog/i)).toBeInTheDocument();
  });

  it('fetches tools and renders names when expanded', async () => {
    render(<ToolsPanel />);
    screen.getByText(/Tool Catalog/i).click();
    await waitFor(() => expect(screen.getByText('obs:get_logs')).toBeInTheDocument());
    expect(screen.getByText('rem:restart_service')).toBeInTheDocument();
  });

  it('hits /api/v1/tools', async () => {
    render(<ToolsPanel />);
    screen.getByText(/Tool Catalog/i).click();
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/tools',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});
