import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../_helpers/render';
import { ApprovalsQueuePanel } from '@/monitors/ApprovalsQueuePanel';
import type { SessionSummary } from '@/state/useSessionList';

const queue: SessionSummary[] = [
  { id: 'SES-3', status: 'awaiting_input', label: 'oldest', created_at: 't0', updated_at: '2026-05-15T14:00:00Z' },
  { id: 'SES-2', status: 'awaiting_input', label: 'newer', created_at: 't0', updated_at: '2026-05-15T14:05:00Z' },
];

describe('<ApprovalsQueuePanel>', () => {
  it('renders the count in header', () => {
    render(<ApprovalsQueuePanel queue={queue} onSelect={() => {}} />);
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('renders each queue item with id + label', () => {
    render(<ApprovalsQueuePanel queue={queue} onSelect={() => {}} />);
    expect(screen.getByText('SES-3')).toBeInTheDocument();
    expect(screen.getByText('oldest')).toBeInTheDocument();
    expect(screen.getByText('SES-2')).toBeInTheDocument();
  });

  it('shows empty state when queue=[]', () => {
    render(<ApprovalsQueuePanel queue={[]} onSelect={() => {}} />);
    expect(screen.getByText(/No approvals waiting/i)).toBeInTheDocument();
  });

  it('calls onSelect with sid on row click', () => {
    const onSelect = vi.fn();
    render(<ApprovalsQueuePanel queue={queue} onSelect={onSelect} />);
    fireEvent.click(screen.getByText('SES-3').closest('[data-row]')!);
    expect(onSelect).toHaveBeenCalledWith('SES-3');
  });
});
