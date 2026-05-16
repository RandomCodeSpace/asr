import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { OtherSessionsPanel } from '@/monitors/OtherSessionsPanel';
import type { SessionSummary } from '@/state/useSessionList';

const sessions: SessionSummary[] = [
  { id: 'SES-1', status: 'in_progress', label: 'A', created_at: 't0', updated_at: 't1', active_agent: 'triage' },
  { id: 'SES-2', status: 'awaiting_input', label: 'B paused', created_at: 't0', updated_at: 't2' },
  { id: 'SES-3', status: 'resolved', label: 'C done', created_at: 't0', updated_at: 't3' },
];

describe('<OtherSessionsPanel>', () => {
  it('excludes the active session from the list', () => {
    render(<OtherSessionsPanel sessions={sessions} activeSid="SES-2" onSelect={() => {}} />);
    expect(screen.queryByText('SES-2')).not.toBeInTheDocument();
    expect(screen.getByText('SES-1')).toBeInTheDocument();
    expect(screen.getByText('SES-3')).toBeInTheDocument();
  });

  it('renders all sessions when activeSid is null', () => {
    render(<OtherSessionsPanel sessions={sessions} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText('SES-1')).toBeInTheDocument();
    expect(screen.getByText('SES-2')).toBeInTheDocument();
    expect(screen.getByText('SES-3')).toBeInTheDocument();
  });

  it('shows count in the header', () => {
    render(<OtherSessionsPanel sessions={sessions} activeSid="SES-1" onSelect={() => {}} />);
    // count = 2 (excluding active)
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('shows label and active agent', () => {
    render(<OtherSessionsPanel sessions={sessions} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText('A')).toBeInTheDocument();
    expect(screen.getByText(/triage/)).toBeInTheDocument();
  });

  it('shows empty message when no other sessions', () => {
    render(<OtherSessionsPanel sessions={[sessions[0]!]} activeSid="SES-1" onSelect={() => {}} />);
    expect(screen.getByText(/Only this session is active/i)).toBeInTheDocument();
  });

  it('calls onSelect on tile click', () => {
    const onSelect = vi.fn();
    render(<OtherSessionsPanel sessions={sessions} activeSid="SES-2" onSelect={onSelect} />);
    fireEvent.click(screen.getByText('SES-1').closest('[data-tile]')!);
    expect(onSelect).toHaveBeenCalledWith('SES-1');
  });
});
