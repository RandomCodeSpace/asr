import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { MobileShell } from '@/shell/MobileShell';

const ORIGINAL_FETCH = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn(() => Promise.resolve({
    ok: true, status: 200, statusText: 'OK',
    clone() { return this; },
    json: () => Promise.resolve({}),
  } as unknown as Response));
});
afterEach(() => { globalThis.fetch = ORIGINAL_FETCH; });

describe('<MobileShell>', () => {
  it('renders 3-tab bottom nav and canvas by default', () => {
    render(
      <MobileShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(screen.getByRole('navigation', { name: /mobile/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open sessions/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /show canvas/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open monitors/i })).toBeInTheDocument();
    expect(screen.getByText(/Select a session/i)).toBeInTheDocument();
  });

  it('opens sessions sheet on tap', () => {
    render(
      <MobileShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(document.querySelector('[data-mobile-sheet="sessions"]')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /open sessions/i }));
    expect(document.querySelector('[data-mobile-sheet="sessions"]')).toBeInTheDocument();
  });

  it('opens monitors sheet on tap and closes via the sheet Close button', () => {
    render(
      <MobileShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /open monitors/i }));
    expect(document.querySelector('[data-mobile-sheet="monitors"]')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(document.querySelector('[data-mobile-sheet="monitors"]')).not.toBeInTheDocument();
  });

  it('stamps data-shell="mobile" for layout snapshots', () => {
    const { container } = render(
      <MobileShell
        sessions={[]} activeSid={null} onSelectSession={() => {}}
        queue={[]} agentsByName={{}} toolCalls={[]}
      />,
    );
    expect(container.querySelector('[data-shell="mobile"]')).toBeInTheDocument();
  });
});
