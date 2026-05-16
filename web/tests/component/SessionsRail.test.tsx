import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { SessionsRail } from '@/shell/SessionsRail';
import type { SessionSummary } from '@/state/useSessionList';

const sessions: SessionSummary[] = [
  { id: 'SES-1', status: 'in_progress', label: 'Foo running', created_at: 't0', updated_at: 't1' },
  { id: 'SES-2', status: 'awaiting_input', label: 'Bar paused', created_at: 't0', updated_at: 't2' },
  { id: 'SES-3', status: 'resolved', label: 'Baz done', created_at: 't0', updated_at: 't3' },
];

describe('<SessionsRail>', () => {
  it('renders the header with total count', () => {
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText(/Sessions/)).toBeInTheDocument();
    expect(screen.getByText(/3 total/)).toBeInTheDocument();
  });

  it('groups sessions into Active and Recent', () => {
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText(/Active/)).toBeInTheDocument();
    expect(screen.getByText(/Recent/)).toBeInTheDocument();
    // SES-1 + SES-2 in Active (in_progress + awaiting_input), SES-3 in Recent (resolved)
  });

  it('renders rows with id, label, and status', () => {
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText('SES-1')).toBeInTheDocument();
    expect(screen.getByText('Foo running')).toBeInTheDocument();
  });

  it('marks the active session row with data-active="true"', () => {
    render(<SessionsRail sessions={sessions} activeSid="SES-2" onSelect={() => {}} />);
    const row = screen.getByText('SES-2').closest('[data-row]');
    expect(row).toHaveAttribute('data-active', 'true');
  });

  it('shows pending dot on awaiting_input rows', () => {
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={() => {}} />);
    const row = screen.getByText('SES-2').closest('[data-row]');
    expect(row?.querySelector('[data-pending-dot]')).not.toBeNull();
    const otherRow = screen.getByText('SES-1').closest('[data-row]');
    expect(otherRow?.querySelector('[data-pending-dot]')).toBeNull();
  });

  it('calls onSelect with sid when a row is clicked', () => {
    const onSelect = vi.fn();
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={onSelect} />);
    fireEvent.click(screen.getByText('SES-2').closest('[data-row]')!);
    expect(onSelect).toHaveBeenCalledWith('SES-2');
  });

  it('filters rows when filter input has text', () => {
    render(<SessionsRail sessions={sessions} activeSid={null} onSelect={() => {}} />);
    const filter = screen.getByPlaceholderText(/Filter/i);
    fireEvent.change(filter, { target: { value: 'Bar' } });
    expect(screen.queryByText('SES-1')).not.toBeInTheDocument();
    expect(screen.getByText('SES-2')).toBeInTheDocument();
    expect(screen.queryByText('SES-3')).not.toBeInTheDocument();
  });

  it('shows empty state when no sessions', () => {
    render(<SessionsRail sessions={[]} activeSid={null} onSelect={() => {}} />);
    expect(screen.getByText(/No sessions yet/i)).toBeInTheDocument();
  });
});
