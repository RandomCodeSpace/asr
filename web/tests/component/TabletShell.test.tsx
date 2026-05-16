import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { TabletShell } from '@/shell/TabletShell';

const ORIGINAL_FETCH = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn(() => Promise.resolve({
    ok: true, status: 200, statusText: 'OK',
    clone() { return this; },
    json: () => Promise.resolve({}),
  } as unknown as Response));
});
afterEach(() => { globalThis.fetch = ORIGINAL_FETCH; });

describe('<TabletShell>', () => {
  it('renders SessionsRail + Monitors button + empty SessionCanvas hint', () => {
    render(
      <TabletShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(screen.getAllByText(/Sessions/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /open monitors/i })).toBeInTheDocument();
    expect(screen.getByText(/Select a session/i)).toBeInTheDocument();
  });

  it('uses a 180px sessions rail grid template (tablet shell marker)', () => {
    const { container } = render(
      <TabletShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(container.querySelector('[data-shell="tablet"]')).toBeInTheDocument();
  });

  it('opens the monitors sheet when the button is clicked', () => {
    render(
      <TabletShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(document.querySelector('[data-monitors-sheet]')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /open monitors/i }));
    expect(document.querySelector('[data-monitors-sheet]')).toBeInTheDocument();
  });
});
